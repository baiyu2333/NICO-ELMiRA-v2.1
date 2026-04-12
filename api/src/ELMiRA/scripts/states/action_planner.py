import numpy as np
import rospy
import smach
import smach_ros

from geometry_msgs.msg import Pose, Point, Quaternion
from elmira.srv import (
    DetectObjects,
    CoordinateTransfer,
    InverseKinematics,
    CheckLLMObjectVisibility,
    DetectWithMLLM,
)

# v2 MLLM configuration - read from ROS params
def get_mllm_config():
    """Get MLLM configuration from ROS params."""
    use_mllm = rospy.get_param("/use_mllm", False)
    use_mllm_grounding = rospy.get_param("/use_mllm_grounding", False)
    
    if use_mllm:
        visibility_service = "mllm_visibility"
        detection_service = "mllm_detect" if use_mllm_grounding else "object_detector"
    else:
        visibility_service = "llm_object_visibility"
        detection_service = "object_detector"
    
    return {
        "use_mllm": use_mllm,
        "use_mllm_grounding": use_mllm_grounding,
        "visibility_service": visibility_service,
        "detection_service": detection_service,
    }


class ObjectSelector(smach.State):
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=["succeeded", "object_not_found", "object_out_of_reach"],
            input_keys=["objects", "target_object"],
            output_keys=["image_x", "image_y", "system_message"],
        )
        self.workspace = np.array(
            [
                [0.0396, 0.7160],
                [0.2021, 0.3444],
                [0.7646, 0.3278],
                [0.9448, 0.7313],
                [0.8162, 0.8069],
                [0.6391, 0.8632],
                [0.4380, 0.8757],
                [0.2599, 0.8375],
                [0.1328, 0.7771],
            ]
        )

    def within_workspace(self, x, y):
        cross_products = np.array(
            [
                (x - self.workspace[i - 1][0])
                * (self.workspace[i][1] - self.workspace[i - 1][1])
                - (self.workspace[i][0] - self.workspace[i - 1][0])
                * (y - self.workspace[i - 1][1])
                for i in range(len(self.workspace))
            ],
        )
        return np.logical_or(np.all(cross_products <= 0), np.all(cross_products >= 0))

    def execute(self, userdata):
        if len(userdata.objects) == 0:
            rospy.logwarn(
                f"SYSTEM: Could not find {userdata.target_object} in the image."
            )
            userdata.system_message = (
                f"SYSTEM: Could not find {userdata.target_object} in the image."
            )
            return "object_not_found"
        high_to_low = np.argsort([-obj.score for obj in userdata.objects])
        for obj in np.array(userdata.objects)[high_to_low]:
            bottom_x = obj.center_x
            bottom_y = obj.center_y + obj.height / 2
            if self.within_workspace(bottom_x, bottom_y):
                userdata.image_x = bottom_x
                userdata.image_y = bottom_y
                return "succeeded"
        rospy.logwarn(
            f"SYSTEM: All {len(userdata.objects)} detected candidates for {userdata.target_object} in the image are out of reach."
        )
        userdata.system_message = f"SYSTEM: All {len(userdata.objects)} detected candidates for {userdata.target_object} in the image or out of reach."
        return "object_out_of_reach"


class ActionTrajectory(smach.State):
    """Generates target poses for robot actions.
    
    Supports bimanual manipulation - arm selection based on target Y coordinate:
    - target_y < 0: Right arm (r_arm)
    - target_y >= 0: Left arm (l_arm)
    
    Supported action types:
    - touch: Touch object with hand
    - show: Point at object
    - push, push_left, push_right: Push object in direction
    - grasp: Pre-grasp approach, lower, and close hand signal
    - place: Lower to table, open hand, and retreat
    - open_hand: Signal to open gripper
    - close_hand: Signal to close gripper
    """
    
    # Real-world workspace limits (meters) based on NICO arm URDF
    # The arm has ~35cm reach from shoulder. Shoulder is ~3cm forward from torso center.
    # Practical reachable range (avoiding full extension):
    MAX_REACH_X = 0.32   # Max forward reach (meters)
    MIN_REACH_X = 0.10   # Min forward reach
    MAX_REACH_Y = 0.25   # Max left/right reach from center
    
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=["succeeded", "unknown_action", "hand_action"],
            input_keys=["action_type", "target_x", "target_y", "target_z"],
            output_keys=[
                "planning_group",
                "poses",
                "hand_action",  # "open", "close", or None
            ],
        )
    
    def _clamp_coordinates(self, x, y):
        """Clamp target coordinates to the robot's reachable workspace."""
        orig_x, orig_y = x, y
        x = max(self.MIN_REACH_X, min(self.MAX_REACH_X, x))
        y = max(-self.MAX_REACH_Y, min(self.MAX_REACH_Y, y))
        if orig_x != x or orig_y != y:
            rospy.logwarn(
                f"ActionTrajectory: Clamped coordinates from "
                f"({orig_x:.3f}, {orig_y:.3f}) to ({x:.3f}, {y:.3f}) "
                f"[workspace limits: x=[{self.MIN_REACH_X}, {self.MAX_REACH_X}], "
                f"y=[{-self.MAX_REACH_Y}, {self.MAX_REACH_Y}]]"
            )
        return x, y

    def execute(self, userdata):
        # Select arm based on target Y coordinate:
        # - target_y < 0: right side of table -> right arm
        # - target_y >= 0: left side of table -> left arm
        # NOTE: Left hand (wrist/fingers) is non-functional, but left arm
        # (shoulder/elbow) works and can be used for pointing/pushing
        is_right = userdata.target_y < 0
        
        # Set default values for output keys (will be overwritten for grasp/place actions)
        userdata.planning_group = "r_arm" if is_right else "l_arm"
        userdata.hand_action = None  # No hand action by default
        
        # Log and clamp coordinates to reachable workspace
        rospy.loginfo(
            f"ActionTrajectory: action={userdata.action_type}, "
            f"raw target=({userdata.target_x:.3f}, {userdata.target_y:.3f}, {userdata.target_z:.3f}), "
            f"arm={'right' if is_right else 'left'}"
        )
        target_x, target_y = self._clamp_coordinates(userdata.target_x, userdata.target_y)
        target_z = userdata.target_z
        
        # calculate target for action
        target_poses = []
        if userdata.action_type == "touch":
            target_pose = Pose()
            target_pose.position = Point(
                target_x - 0.0, target_y, target_z
            )
            if is_right:  # x, y, z, w
                target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
            else:
                target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
            target_poses.append(target_pose)
        elif userdata.action_type == "show":
            target_pose = Pose()
            # NOTE: Reduced X offset from -0.08 to -0.04 for better reachability
            target_pose.position = Point(
                target_x - 0.04, target_y, target_z + 0.03
            )
            if is_right:  # x, y, z, w
                target_pose.orientation = Quaternion(-1.0, 0, 0, 0.0)
            else:
                target_pose.orientation = Quaternion(1.0, 0, 0, 0.0)
            target_poses.append(target_pose)
        elif userdata.action_type == "push":
            for offset in [
                (-0.04, 0.0, 0.0),
                (0.03, 0.0, 0.0),
                (0.06, 0.0, 0.0),
                (0.06, 0.0, 0.10),
            ]:
                target_pose = Pose()
                target_pose.position = Point(
                    target_x + offset[0],
                    target_y + offset[1],
                    target_z + offset[2],
                )
                if is_right:  # x, y, z, w
                    target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
                else:
                    target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
                target_poses.append(target_pose)
        elif userdata.action_type == "push_left":
            # 08, 06
            for offset in [
                (0.04, -0.10, 0.10),
                (0.04, -0.10, 0.0),
                (0.04, 0.04, 0.0),
                (0.04, 0.04, 0.10),
            ]:
                target_pose = Pose()
                target_pose.position = Point(
                    target_x + offset[0],
                    target_y + offset[1],
                    target_z + offset[2],
                )
                if is_right:  # x, y, z, w
                    target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
                else:
                    target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
                target_poses.append(target_pose)
        elif userdata.action_type == "push_right":
            for offset in [
                (0.04, 0.10, 0.10),
                (0.04, 0.10, 0.0),
                (0.04, -0.04, 0.0),
                (0.04, -0.04, 0.10),
            ]:
                target_pose = Pose()
                target_pose.position = Point(
                    target_x + offset[0],
                    target_y + offset[1],
                    target_z + offset[2],
                )
                if is_right:  # x, y, z, w
                    target_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
                else:
                    target_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
                target_poses.append(target_pose)
        elif userdata.action_type == "grasp":
            # Grasp Phase 1: ONLY the approach pose (hover above object).
            # The descent, hand-close, and lift are deferred to the visual
            # servoing refinement loop (GraspRefinement state) which will
            # re-plan them with corrected coordinates after a second look.
            approach_pose = Pose()
            approach_pose.position = Point(
                target_x - 0.02,  # Slightly behind
                target_y,
                target_z + 0.08,  # Above object
            )
            if is_right:
                approach_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
            else:
                approach_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
            target_poses.append(approach_pose)
            
            # No hand_action here - it will be set by the refinement state
            userdata.hand_action = None
            
        elif userdata.action_type == "place":
            # Place sequence: lower to table, open hand, retreat
            # 1. Pre-place pose (above target)
            preplace_pose = Pose()
            preplace_pose.position = Point(
                target_x,
                target_y,
                target_z + 0.10,  # Above placement
            )
            if is_right:
                preplace_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
            else:
                preplace_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
            target_poses.append(preplace_pose)
            
            # 2. Place pose (at table level)
            place_pose = Pose()
            place_pose.position = Point(
                target_x,
                target_y,
                target_z + 0.02,  # Just above table
            )
            if is_right:
                place_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
            else:
                place_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
            target_poses.append(place_pose)
            
            # Signal hand to open after reaching place pose
            userdata.hand_action = "open"
            
            # 3. Retreat pose (move back and up)
            retreat_pose = Pose()
            retreat_pose.position = Point(
                target_x - 0.05,  # Move back
                target_y,
                target_z + 0.12,  # Lift up
            )
            if is_right:
                retreat_pose.orientation = Quaternion(-0.7071068, 0, 0, 0.7071068)
            else:
                retreat_pose.orientation = Quaternion(0.7071068, 0, 0, 0.7071068)
            target_poses.append(retreat_pose)
            
        elif userdata.action_type == "open_hand":
            # Pure hand action - no arm movement
            userdata.hand_action = "open"
            userdata.planning_group = "r_hand" if is_right else "l_hand"
            userdata.poses = []
            return "hand_action"
            
        elif userdata.action_type == "close_hand":
            # Pure hand action - no arm movement
            userdata.hand_action = "close"
            userdata.planning_group = "r_hand" if is_right else "l_hand"
            userdata.poses = []
            return "hand_action"
        else:
            rospy.loginfo(f"Action '{userdata.action_type}' not defined")
            userdata.system_message = (
                f"SYSTEM: Action '{userdata.action_type}' not defined"
            )
            return "unknown_action"
        
        # set output userdata
        userdata.planning_group = "r_arm" if is_right else "l_arm"
        userdata.poses = target_poses
        
        return "succeeded"


class ActionPlanner(smach.StateMachine):
    """Locates target object and plans motion trajectory for the requested action.
    
    Supports both v1 (OWLv2) and v2 (MLLM) detection paths based on ROS params:
    - /use_mllm: Use MLLM gateway
    - /use_mllm_grounding: Use MLLM for object detection (vs OWLv2)
    
    Supports bimanual manipulation with grasp/place actions.
    Hand control is signaled via hand_action output key.
    """

    def __init__(
        self,
    ):
        super(ActionPlanner, self).__init__(
            input_keys=["action_type", "target_object", "table_z", "motion_init_pose"],
            output_keys=["joint_trajectory", "system_message", "real_x", "real_y", "hand_action", "planning_group"],
            outcomes=[
                "succeeded",
                "preempted",
                "aborted",
                "system_out",
            ],
        )
        
        # Get MLLM configuration
        mllm_config = get_mllm_config()
        detection_service = mllm_config["detection_service"]
        detection_srv_type = DetectWithMLLM if mllm_config["use_mllm_grounding"] else DetectObjects
        
        rospy.loginfo(f"ActionPlanner using detection service: {detection_service}")
        
        # Open the container
        with self:
            # detect object in image space
            smach.StateMachine.add(
                "OBJECT_DETECTION",
                smach_ros.ServiceState(
                    detection_service,
                    detection_srv_type,
                    request_slots=["texts"],
                    response_slots=["objects"],
                ),
                transitions={
                    "succeeded": "CHOOSE_TARGET_OBJECT",
                },
                remapping={
                    "texts": "target_object",
                },
            )
            # process dected objects
            smach.StateMachine.add(
                "CHOOSE_TARGET_OBJECT",
                ObjectSelector(),
                transitions={
                    "succeeded": "COORDINATE_TRANSFER",
                    "object_not_found": "system_out",  # TODO combine not found and out of reach?
                    "object_out_of_reach": "system_out",
                },
            )
            # image to real
            smach.StateMachine.add(
                "COORDINATE_TRANSFER",
                smach_ros.ServiceState(
                    "image_to_real",
                    CoordinateTransfer,
                    request_slots=["image_x", "image_y"],
                    response_slots=["real_x", "real_y"],
                ),
                transitions={"succeeded": "PLAN_ACTION_TARGETS"},
            )
            # set action target
            # Generates target poses for arm movement and sets hand_action for grasp/place
            smach.StateMachine.add(
                "PLAN_ACTION_TARGETS",
                ActionTrajectory(),
                transitions={
                    "succeeded": "SOLVE_IK",
                    "unknown_action": "system_out",
                    "hand_action": "succeeded",  # Pure hand actions skip IK
                },
                remapping={
                    "target_x": "real_x",
                    "target_y": "real_y",
                    "target_z": "table_z",
                },
            )

            # callback to create initial pose
            def ik_request_callback(userdata, request):
                initial_position = userdata.motion_init_pose[userdata.planning_group]
                request.initial_position.joint_name = initial_position["names"]
                request.initial_position.position = initial_position["positions"]
                return request

            # callback to post-process detected objects
            def ik_response_callback(userdata, response):
                trajectory = []
                for i, position in enumerate(response.positions):
                    step = {
                        userdata.planning_group: {
                            "names": position.joint_name,
                            "positions": position.position,
                        }
                    }
                    
                    # Target poses logic from ActionTrajectory:
                    # grasp: [0] approach only (refinement handles the rest)
                    # place: [0] preplace, [1] place, [2] retreat
                    
                    action = str(userdata.action_type).lower()
                    # Grasp no longer embeds hand_action here - handled by GraspRefinement
                        
                    if action in ["place", "drop", "release", "open"] and i == 1:
                        step["hand_action"] = "open"
                        step["planning_group"] = userdata.planning_group
                        
                    trajectory.append(step)
                
                action = str(userdata.action_type).lower()
                if action in ["grasp", "grab", "pick", "take"]:
                    # For grasp: don't append init pose - refinement state handles conclusion
                    userdata.joint_trajectory = trajectory
                else:
                    userdata.joint_trajectory = trajectory + [userdata.motion_init_pose]
                return "succeeded"

            # IK Solver
            smach.StateMachine.add(
                "SOLVE_IK",
                smach_ros.ServiceState(
                    "inverse_kinematics",
                    InverseKinematics,
                    input_keys=["planning_group", "motion_init_pose", "action_type"],
                    request_slots=["planning_group", "poses"],
                    output_keys=["joint_trajectory"],
                    request_cb=ik_request_callback,
                    response_cb=ik_response_callback,
                ),
                transitions={
                    "succeeded": "succeeded",
                },
            )


class ConcurrentPlanAndVerify(smach.Concurrence):
    """Starts Action Planning and LLM Vision Verification in parallel to save time.
    
    Supports both v1 and v2 visibility services based on ROS params:
    - v1: Uses llm_object_visibility (GPT-4o Assistant API)
    - v2: Uses mllm_visibility (unified MLLM gateway)
    
    Supports bimanual manipulation - passes through hand_action and planning_group.
    """

    def __init__(
        self,
    ):
        super(ConcurrentPlanAndVerify, self).__init__(
            input_keys=[
                "action_type",
                "target_object",
                "llm_input",
                "table_z",
                "motion_init_pose",
            ],
            output_keys=["joint_trajectory", "system_message", "real_x", "real_y", "hand_action", "planning_group"],
            outcomes=[
                "succeeded",
                "preempted",
                "aborted",
                "system_out",
            ],
            default_outcome="system_out",
            child_termination_cb=self.child_termination_cb,
            outcome_cb=self.outcome_cb,
        )
        
        # Get MLLM configuration for visibility service
        mllm_config = get_mllm_config()
        visibility_service = mllm_config["visibility_service"]
        rospy.loginfo(f"ConcurrentPlanAndVerify using visibility service: {visibility_service}")
        
        # Open the container
        with self:
            smach.Concurrence.add(
                "ACTION_PLANNER",
                ActionPlanner(),
            )

            def object_visible_response_cb(userdata, response):
                if not response.object_visible:
                    rospy.logwarn(response.system_message)
                    return "object_not_visible"
                return "succeeded"

            smach.Concurrence.add(
                "CHECK_OBJECT_VISIBILITY",
                smach_ros.ServiceState(
                    visibility_service,
                    CheckLLMObjectVisibility,
                    request_slots=["prompt"],
                    response_slots=["system_message"],
                    response_cb=object_visible_response_cb,
                    outcomes=["object_not_visible"],
                ),
                remapping={
                    "prompt": "llm_input",
                },
            )

    def child_termination_cb(self, outcome_map):
        # Terminate early when either system returns a failure message
        if (
            outcome_map["ACTION_PLANNER"] == "system_out"
            or outcome_map["CHECK_OBJECT_VISIBILITY"] == "object_not_visible"
        ):
            return True
        return False

    def outcome_cb(self, outcome_map):
        # Terminate early when either system returns a failure message
        if outcome_map["ACTION_PLANNER"] == "system_out":
            self.get_children()["CHECK_OBJECT_VISIBILITY"].recall_preempt()
            return "system_out"
        elif outcome_map["CHECK_OBJECT_VISIBILITY"] == "object_not_visible":
            self.get_children()["ACTION_PLANNER"].recall_preempt()
            return "system_out"
        elif (
            outcome_map["ACTION_PLANNER"] == "succeeded"
            and outcome_map["CHECK_OBJECT_VISIBILITY"] == "succeeded"
        ):
            return "succeeded"
