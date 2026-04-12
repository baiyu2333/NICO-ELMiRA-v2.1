#!/usr/bin/env python3
"""
Grasp Refinement State for ELMiRA v2 - Visual Servoing

After the robot's hand reaches the initial approach pose (hovering above
the detected object position), this state:

1. Takes a second camera image (with the hand visible near the object)
2. Calls the mllm_refine_grasp service to get a correction offset
3. Applies the correction to the original coordinates
4. Plans the final descent (grasp pose + lift) with corrected coordinates
5. Outputs the corrected trajectory for execution

This implements a simple 1-pass visual servoing loop that dramatically
improves grasp accuracy by using closed-loop visual feedback.
"""

import json
import rospy
import smach
import smach_ros

from geometry_msgs.msg import Pose, Point, Quaternion
from elmira.srv import (
    PromptVisionLLM,
    InverseKinematics,
)


class GraspRefinement(smach.StateMachine):
    """
    Visual servoing refinement for grasp actions.
    
    After the arm reaches the approach pose, this state machine:
    1. Calls GPT vision to estimate correction offset
    2. Applies correction to original coordinates 
    3. Plans descent + grasp + lift trajectory with corrected coords
    4. Outputs the final trajectory for execution
    
    Input keys:
        real_x, real_y: Original detected coordinates (meters)
        table_z: Table height (meters)
        planning_group: Which arm (l_arm / r_arm)
        motion_init_pose: Initial pose dict for IK solver
        target_object: Object name for GPT prompt context
        
    Output keys:
        joint_trajectory: Final trajectory (descent → close → lift → init)
        hand_action: Will be set to "close"
    """
    
    def __init__(self):
        super(GraspRefinement, self).__init__(
            input_keys=[
                "real_x", "real_y", "table_z",
                "planning_group", "motion_init_pose",
                "target_object",
            ],
            output_keys=["joint_trajectory", "hand_action", "planning_group"],
            outcomes=["succeeded", "aborted", "preempted"],
        )
        
        with self:
            # Step 1: Call GPT vision refinement service
            smach.StateMachine.add(
                "CALL_REFINEMENT",
                CallRefinementService(),
                transitions={
                    "succeeded": "PLAN_REFINED_GRASP",
                    "failed": "PLAN_REFINED_GRASP",  # Fall back to original coords
                },
            )
            
            # Step 2: Plan the final grasp trajectory with corrected coords
            smach.StateMachine.add(
                "PLAN_REFINED_GRASP",
                PlanRefinedGrasp(),
                transitions={
                    "succeeded": "SOLVE_REFINED_IK",
                },
            )
            
            # Step 3: Solve IK for refined poses
            def ik_request_cb(userdata, request):
                initial_position = userdata.motion_init_pose[userdata.planning_group]
                request.initial_position.joint_name = initial_position["names"]
                request.initial_position.position = initial_position["positions"]
                return request
            
            def ik_response_cb(userdata, response):
                trajectory = []
                for i, position in enumerate(response.positions):
                    step = {
                        userdata.planning_group: {
                            "names": position.joint_name,
                            "positions": position.position,
                        }
                    }
                    # Step 0 = refined grasp pose, Step 1 = lift
                    # Insert hand close after reaching grasp pose
                    if i == 0:
                        step["hand_action"] = "close"
                        step["planning_group"] = userdata.planning_group
                    
                    trajectory.append(step)
                
                # Append return to init pose
                userdata.joint_trajectory = trajectory + [userdata.motion_init_pose]
                userdata.hand_action = "close"
                return "succeeded"
            
            smach.StateMachine.add(
                "SOLVE_REFINED_IK",
                smach_ros.ServiceState(
                    "inverse_kinematics",
                    InverseKinematics,
                    input_keys=["planning_group", "motion_init_pose", "refined_poses"],
                    request_slots=["planning_group", "poses"],
                    output_keys=["joint_trajectory", "hand_action"],
                    request_cb=ik_request_cb,
                    response_cb=ik_response_cb,
                ),
                remapping={
                    "poses": "refined_poses",
                },
                transitions={
                    "succeeded": "succeeded",
                },
            )


class CallRefinementService(smach.State):
    """Call the mllm_refine_grasp service and parse the correction offsets."""
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed"],
            input_keys=["real_x", "real_y", "target_object"],
            output_keys=["x_correction", "y_correction"],
        )
    
    def execute(self, userdata):
        # Set the target object in ROS param so the MLLM gateway knows what we're grasping
        target_name = userdata.target_object
        if isinstance(target_name, (list, tuple)):
            target_name = target_name[0] if target_name else "the object"
        rospy.set_param("/mllm_refine_target", str(target_name))
        
        rospy.loginfo(
            f"GraspRefinement: Calling refinement for '{target_name}' "
            f"at ({userdata.real_x:.3f}, {userdata.real_y:.3f})"
        )
        
        try:
            # Wait for refinement service
            rospy.wait_for_service("mllm_refine_grasp", timeout=5.0)
            
            refine_srv = rospy.ServiceProxy("mllm_refine_grasp", PromptVisionLLM)
            response = refine_srv()
            
            # Parse response
            data = json.loads(response.response)
            x_off = float(data.get("x_offset_meters", 0.0))
            y_off = float(data.get("y_offset_meters", 0.0))
            confidence = float(data.get("confidence", 0.0))
            description = data.get("description", "")
            
            rospy.loginfo(
                f"GraspRefinement: Correction x={x_off:.4f}m, y={y_off:.4f}m "
                f"(confidence={confidence:.2f}): {description}"
            )
            
            userdata.x_correction = x_off
            userdata.y_correction = y_off
            return "succeeded"
            
        except rospy.ROSException as e:
            rospy.logwarn(f"GraspRefinement: Service timeout: {e}")
            userdata.x_correction = 0.0
            userdata.y_correction = 0.0
            return "failed"
        except Exception as e:
            rospy.logwarn(f"GraspRefinement: Error: {e}")
            userdata.x_correction = 0.0
            userdata.y_correction = 0.0
            return "failed"


class PlanRefinedGrasp(smach.State):
    """
    Plan the refined grasp trajectory (descent + lift) using corrected coordinates.
    
    Applies the correction offsets from the refinement service to the original
    coordinates, then generates the final descent and lift poses.
    """
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded"],
            input_keys=[
                "real_x", "real_y", "table_z",
                "planning_group",
                "x_correction", "y_correction",
            ],
            output_keys=["refined_poses"],
        )
    
    def execute(self, userdata):
        # Apply correction
        refined_x = userdata.real_x + userdata.x_correction
        refined_y = userdata.real_y + userdata.y_correction
        table_z = userdata.table_z
        
        # Clamp to robot workspace limits (same as ActionTrajectory)
        MAX_REACH_X = 0.32
        MIN_REACH_X = 0.10
        MAX_REACH_Y = 0.25
        
        orig_rx, orig_ry = refined_x, refined_y
        refined_x = max(MIN_REACH_X, min(MAX_REACH_X, refined_x))
        refined_y = max(-MAX_REACH_Y, min(MAX_REACH_Y, refined_y))
        if orig_rx != refined_x or orig_ry != refined_y:
            rospy.logwarn(
                f"GraspRefinement: Clamped refined coords from "
                f"({orig_rx:.3f}, {orig_ry:.3f}) to ({refined_x:.3f}, {refined_y:.3f})"
            )
        
        is_right = "r_" in userdata.planning_group
        
        rospy.loginfo(
            f"GraspRefinement: Original ({userdata.real_x:.3f}, {userdata.real_y:.3f}) "
            f"→ Refined ({refined_x:.3f}, {refined_y:.3f}) "
            f"[correction: ({userdata.x_correction:.4f}, {userdata.y_correction:.4f})]"
        )
        
        poses = []
        
        # 1. Refined grasp pose (descend to object)
        grasp_pose = Pose()
        grasp_pose.position = Point(
            refined_x,
            refined_y,
            table_z + 0.02,  # Just above table
        )
        if is_right:
            grasp_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
        else:
            grasp_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
        poses.append(grasp_pose)
        
        # 2. Lift pose (raise object after grasp)
        lift_pose = Pose()
        lift_pose.position = Point(
            refined_x,
            refined_y,
            table_z + 0.12,  # Lift up
        )
        if is_right:
            lift_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
        else:
            lift_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
        poses.append(lift_pose)
        
        userdata.refined_poses = poses
        return "succeeded"
