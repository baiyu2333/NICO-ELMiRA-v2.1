import rospy
import smach


def _normalize_requested_hand(value):
    """Return 'left', 'right', or None from an LLM hand/arm field."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("left", "l", "l_arm", "l_hand", "left_arm", "left_hand"):
        return "left"
    if text in ("right", "r", "r_arm", "r_hand", "right_arm", "right_hand"):
        return "right"
    return None


def _infer_requested_hand_from_prompt(prompt):
    """Recover explicit user hand choice when the LLM omits the hand field."""
    text = str(prompt or "").lower()
    left_markers = (
        "left hand",
        "left arm",
        "with your left",
        "with his left",
        "use your left",
        "using your left",
    )
    right_markers = (
        "right hand",
        "right arm",
        "with your right",
        "with his right",
        "use your right",
        "using your right",
    )
    has_left = any(marker in text for marker in left_markers)
    has_right = any(marker in text for marker in right_markers)
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


class ActionParser(smach.State):
    def __init__(self):
        # Your state initialization goes here
        smach.State.__init__(
            self,
            outcomes=["initial_pose", "speak", "act", "describe", "quit"],
            input_keys=["llm_actions", "action_index", "motion_init_pose", "llm_input"],
            output_keys=[
                "action",
                "tts_text",
                "action_type",
                "target_object",
                "llm_input",
                "joint_trajectory",
                "planning_group",
                "hand_action",
                "requested_hand",
            ],
        )

    def execute(self, userdata):
        next_action = userdata.llm_actions[userdata.action_index]
        rospy.loginfo(f"Action: {next_action['action']}")
        userdata.action = next_action["action"]
        if next_action["action"] == "speak":
            rospy.loginfo(f"Text: {next_action['text']}")
            userdata.tts_text = next_action["text"]
            userdata.requested_hand = None
            return "speak"
        elif next_action["action"] == "act":
            rospy.loginfo(
                f"Type: {next_action['type']}, Object: {next_action['object']}"
            )
            original_prompt = getattr(userdata, "llm_input", "")
            userdata.llm_input = f"{next_action['object']}"
            userdata.action_type = next_action["type"]
            userdata.target_object = [next_action["object"]]
            requested_hand = _normalize_requested_hand(
                next_action.get("hand", next_action.get("side", next_action.get("arm")))
            )
            if requested_hand is None:
                requested_hand = _infer_requested_hand_from_prompt(original_prompt)
            userdata.requested_hand = requested_hand
            if requested_hand:
                rospy.loginfo(f"ActionParser: User requested {requested_hand} hand")
            return "act"
        elif next_action["action"] == "describe":
            userdata.requested_hand = None
            return "describe"
        elif next_action["action"] == "quit":
            userdata.requested_hand = None
            return "quit"
        elif next_action["action"] == "initial_pose":
            userdata.joint_trajectory = [userdata.motion_init_pose]
            userdata.planning_group = None  # No specific arm for initial pose
            userdata.hand_action = None     # No hand action for initial pose
            userdata.action_type = "initial_pose"
            userdata.requested_hand = None
            return "initial_pose"
