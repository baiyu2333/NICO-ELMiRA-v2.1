import numpy as np
import rospy
import smach
import smach_ros

from geometry_msgs.msg import Pose, Point, Quaternion
from elmira.srv import (
    CoordinateTransfer,
    InverseKinematics,
    CheckLLMObjectVisibility,
    DetectWithMLLM,
)

from utils.constants import (
    LEFT_HAND_FUNCTIONAL,
    HAND_REQUIRED_ACTIONS,
    clamp_to_workspace,
    get_arm_orientation,
    get_grasp_orientation,
    get_point_orientation,
    get_pre_grasp_orientation,
    get_touch_orientation,
    LLM_DETECT_SERVICE,
    LLM_VISIBILITY_SERVICE,
)


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
            in_ws = self.within_workspace(bottom_x, bottom_y)
            rospy.loginfo(
                f"ObjectSelector: '{userdata.target_object}' candidate "
                f"center=({obj.center_x:.3f}, {obj.center_y:.3f}), "
                f"size=({obj.width:.3f}x{obj.height:.3f}), "
                f"bottom=({bottom_x:.3f}, {bottom_y:.3f}), "
                f"score={obj.score:.3f}, in_workspace={in_ws}"
            )
            if in_ws:
                userdata.image_x = bottom_x
                userdata.image_y = bottom_y
                return "succeeded"
        rospy.logwarn(
            f"SYSTEM: All {len(userdata.objects)} detected candidates for "
            f"{userdata.target_object} in the image are out of reach. "
            f"Workspace Y range: [{self.workspace[:, 1].min():.3f}, {self.workspace[:, 1].max():.3f}], "
            f"X range: [{self.workspace[:, 0].min():.3f}, {self.workspace[:, 0].max():.3f}]"
        )
        userdata.system_message = f"SYSTEM: All {len(userdata.objects)} detected candidates for {userdata.target_object} in the image are out of reach."
        return "object_out_of_reach"


class ActionTrajectory(smach.State):
    """Generates target poses for robot actions.
    
    Arm selection:
    - Explicit user hand choice is honored when present.
    - Otherwise, select arm by target Y coordinate:
      target_y < 0 -> right arm, target_y >= 0 -> left arm.
    
    Supported action types:
    - touch: Touch object with hand
    - show: Point at object
    - push, push_left, push_right: Push object in direction
    - grasp: Pre-grasp approach, lower, and close hand signal
    - place: Lower to table, open hand, and retreat
    - open_hand: Signal to open gripper
    - close_hand: Signal to close gripper
    """
    
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "unknown_action", "hand_action"],
            input_keys=["action_type", "target_x", "target_y", "target_z", "requested_hand"],
            output_keys=[
                "planning_group",
                "poses",
                "hand_action",  # "open", "close", or None
            ],
        )

    def execute(self, userdata):
        action_lower = str(userdata.action_type).lower()
        is_grasp_action = action_lower in ["grasp", "grab", "pick", "take"]
        requested_hand = getattr(userdata, "requested_hand", None)
        
        if requested_hand in ("left", "right"):
            if (
                requested_hand == "left"
                and action_lower in HAND_REQUIRED_ACTIONS
                and not LEFT_HAND_FUNCTIONAL
            ):
                is_right = True
                rospy.logwarn(
                    "ActionTrajectory: User requested LEFT hand, but left hand is disabled; "
                    "falling back to RIGHT arm"
                )
            else:
                is_right = requested_hand == "right"
                rospy.loginfo(
                    f"ActionTrajectory: Honoring explicit {requested_hand} hand request"
                )
        elif not LEFT_HAND_FUNCTIONAL and action_lower in HAND_REQUIRED_ACTIONS:
            is_right = True
            if userdata.target_y >= 0:
                rospy.logwarn(
                    f"ActionTrajectory: Object is on LEFT side (y={userdata.target_y:.3f}) "
                    f"but forcing RIGHT arm because left hand is disabled"
                )
        else:
            is_right = userdata.target_y < 0
        
        userdata.planning_group = "r_arm" if is_right else "l_arm"
        userdata.hand_action = None
        
        # Clamp coordinates to reachable workspace
        target_x, target_y, was_clamped = clamp_to_workspace(
            userdata.target_x, userdata.target_y
        )
        if was_clamped:
            dx = target_x - userdata.target_x
            dy = target_y - userdata.target_y
            reasons = []
            if dx != 0:
                reasons.append(f"X {'too far' if dx < 0 else 'too close'} (shifted {abs(dx):.3f}m)")
            if dy != 0:
                reasons.append(
                    f"Y {'too far left' if dy < 0 else 'too far right'} "
                    f"(shifted {abs(dy):.3f}m)"
                )
            rospy.logwarn(
                f"ActionTrajectory: Target clamped! "
                f"({userdata.target_x:.3f}, {userdata.target_y:.3f}) "
                f"→ ({target_x:.3f}, {target_y:.3f}). "
                f"Reason: {', '.join(reasons)}. "
                f"r_shoulder_z limit is ±0.8rad — lateral reach is limited."
            )
        target_z = userdata.target_z
        base_target_z = target_z
        if action_lower == "touch" and is_right:
            touch_z_offset = float(rospy.get_param("/elmira/right_touch_z_offset", 0.0))
            table_z_min = float(rospy.get_param("/elmira/table_z_min", 0.45))
            table_z_max = float(rospy.get_param("/elmira/table_z_max", 0.85))
            target_z = max(table_z_min, min(table_z_max, target_z + touch_z_offset))
            rospy.loginfo(
                "ActionTrajectory: right touch z adjusted "
                f"base_z={base_target_z:.3f}, offset={touch_z_offset:.3f}, "
                f"target_z={target_z:.3f}"
            )
        
        rospy.loginfo(
            f"ActionTrajectory: action={action_lower}, "
            f"target=({target_x:.3f}, {target_y:.3f}, {target_z:.3f}), "
            f"arm={'right' if is_right else 'left'}"
        )
        
        # calculate target for action
        target_poses = []
        if action_lower == "touch":
            target_pose = Pose()
            target_pose.position = Point(target_x, target_y, target_z)
            target_pose.orientation = get_touch_orientation(is_right)
            target_poses.append(target_pose)
        elif action_lower == "show":
            target_pose = Pose()
            target_pose.position = Point(target_x - 0.04, target_y, target_z + 0.03)
            target_pose.orientation = get_point_orientation(is_right)
            target_poses.append(target_pose)
        elif action_lower == "push":
            for offset in [(-0.04, 0.0, 0.0), (0.03, 0.0, 0.0), (0.06, 0.0, 0.0), (0.06, 0.0, 0.10)]:
                target_pose = Pose()
                target_pose.position = Point(target_x + offset[0], target_y + offset[1], target_z + offset[2])
                target_pose.orientation = get_arm_orientation(is_right)
                target_poses.append(target_pose)
        elif action_lower == "push_left":
            for offset in [(-0.04, -0.10, 0.10), (0.04, -0.10, 0.0), (0.04, 0.04, 0.0), (0.04, 0.04, 0.10)]:
                target_pose = Pose()
                target_pose.position = Point(target_x + offset[0], target_y + offset[1], target_z + offset[2])
                target_pose.orientation = get_arm_orientation(is_right)
                target_poses.append(target_pose)
        elif action_lower == "push_right":
            for offset in [(0.04, 0.10, 0.10), (0.04, 0.10, 0.0), (0.04, -0.04, 0.0), (0.04, -0.04, 0.10)]:
                target_pose = Pose()
                target_pose.position = Point(target_x + offset[0], target_y + offset[1], target_z + offset[2])
                target_pose.orientation = get_arm_orientation(is_right)
                target_poses.append(target_pose)
        elif is_grasp_action:
            table_z_min = float(rospy.get_param("/elmira/table_z_min", 0.45))
            table_z_max = float(rospy.get_param("/elmira/table_z_max", 0.85))
            hover_z_offset = max(
                0.04,
                min(0.12, float(rospy.get_param("/elmira/grasp_hover_z_offset", 0.06))),
            )
            pre_grasp_backoff = max(
                0.00,
                min(0.08, float(rospy.get_param("/elmira/grasp_pre_grasp_x_backoff", 0.035))),
            )
            approach_backoff = max(
                0.00,
                min(0.05, float(rospy.get_param("/elmira/grasp_approach_x_backoff", 0.015))),
            )
            pre_grasp_z = max(table_z_min, min(table_z_max, target_z + hover_z_offset + 0.03))
            approach_z = max(table_z_min, min(table_z_max, target_z + hover_z_offset))
            pre_grasp_orientation = get_pre_grasp_orientation(is_right)
            grasp_orientation = get_grasp_orientation(is_right)

            # Stage 1: move to a conservative pre-grasp pose with the right hand
            # already opened/aligned by the inline hand action below.
            pre_grasp_pose = Pose()
            pre_grasp_pose.position = Point(
                max(0.10, target_x - pre_grasp_backoff),
                target_y,
                pre_grasp_z,
            )
            pre_grasp_pose.orientation = pre_grasp_orientation
            target_poses.append(pre_grasp_pose)

            # Stage 2: small approach near the object. The final close/lift is
            # handled by GraspRefinement so the robot does not keep refining away
            # from a good first pose.
            approach_pose = Pose()
            approach_pose.position = Point(
                max(0.10, target_x - approach_backoff),
                target_y,
                approach_z,
            )
            approach_pose.orientation = grasp_orientation
            target_poses.append(approach_pose)

            userdata.hand_action = None
            
        elif action_lower == "place":
            orient = get_arm_orientation(is_right)
            # 1. Pre-place (above target)
            preplace_pose = Pose()
            preplace_pose.position = Point(target_x, target_y, target_z + 0.10)
            preplace_pose.orientation = orient
            target_poses.append(preplace_pose)
            # 2. Place (at table level)
            place_pose = Pose()
            place_pose.position = Point(target_x, target_y, target_z + 0.02)
            place_pose.orientation = orient
            target_poses.append(place_pose)
            # Signal hand to open after placing
            userdata.hand_action = "open"
            # 3. Retreat
            retreat_pose = Pose()
            retreat_pose.position = Point(target_x - 0.05, target_y, target_z + 0.12)
            retreat_pose.orientation = orient
            target_poses.append(retreat_pose)
            
        elif action_lower == "open_hand":
            userdata.hand_action = "open"
            userdata.planning_group = "r_hand" if is_right else "l_hand"
            userdata.poses = []
            return "hand_action"
        elif action_lower == "close_hand":
            userdata.hand_action = "close"
            userdata.planning_group = "r_hand" if is_right else "l_hand"
            userdata.poses = []
            return "hand_action"
        else:
            rospy.logwarn(f"Action '{action_lower}' not defined")
            userdata.system_message = f"SYSTEM: Action '{action_lower}' not defined"
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

    def __init__(self):
        super(ActionPlanner, self).__init__(
            input_keys=[
                "action_type",
                "target_object",
                "table_z",
                "motion_init_pose",
                "requested_hand",
            ],
            output_keys=["joint_trajectory", "system_message", "real_x", "real_y", "hand_action", "planning_group"],
            outcomes=["succeeded", "preempted", "aborted", "system_out"],
        )
        
        rospy.loginfo(f"ActionPlanner using detection service: {LLM_DETECT_SERVICE}")
        
        # Open the container
        with self:
            # detect object in image space
            smach.StateMachine.add(
                "OBJECT_DETECTION",
                smach_ros.ServiceState(
                    LLM_DETECT_SERVICE,
                    DetectWithMLLM,
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
                    if action == "touch" and userdata.planning_group == "r_arm" and i == 0:
                        step["hand_action"] = "touch_wrist"
                        step["planning_group"] = userdata.planning_group

                    if action in ["grasp", "grab", "pick", "take"] and userdata.planning_group == "r_arm":
                        step["planning_group"] = userdata.planning_group
                        if i == 0:
                            step["hand_action"] = "open"
                        elif i == len(response.positions) - 1:
                            step["hand_action"] = "touch_wrist"
                        
                    if action in ["place", "drop", "release", "open"] and i == 1:
                        step["hand_action"] = "open"
                        step["planning_group"] = userdata.planning_group
                        
                    trajectory.append(step)
                
                action = str(userdata.action_type).lower()
                if action in ["grasp", "grab", "pick", "take"]:
                    # For grasp: don't append init pose - refinement state handles conclusion
                    userdata.joint_trajectory = trajectory
                else:
                    # Reset only the active arm. Appending the full init pose also
                    # commands the opposite arm, which is unsafe during right-arm
                    # touch/reach tuning on the swapped-hand setup.
                    reset_pose = {}
                    if "head" in userdata.motion_init_pose:
                        reset_pose["head"] = userdata.motion_init_pose["head"]
                    if userdata.planning_group in userdata.motion_init_pose:
                        reset_pose[userdata.planning_group] = userdata.motion_init_pose[userdata.planning_group]
                    userdata.joint_trajectory = trajectory + [reset_pose]
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

    def __init__(self):
        super(ConcurrentPlanAndVerify, self).__init__(
            input_keys=[
                "action_type",
                "target_object",
                "llm_input",
                "table_z",
                "motion_init_pose",
                "requested_hand",
            ],
            output_keys=["joint_trajectory", "system_message", "real_x", "real_y", "hand_action", "planning_group"],
            outcomes=["succeeded", "preempted", "aborted", "system_out"],
            default_outcome="system_out",
            child_termination_cb=self.child_termination_cb,
            outcome_cb=self.outcome_cb,
        )
        
        rospy.loginfo(f"ConcurrentPlanAndVerify using visibility: {LLM_VISIBILITY_SERVICE}")
        
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
                    LLM_VISIBILITY_SERVICE,
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
