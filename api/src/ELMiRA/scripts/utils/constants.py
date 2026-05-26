"""
ELMiRA shared constants — single source of truth for workspace limits,
arm orientations, and hardware configuration.
"""

from geometry_msgs.msg import Quaternion

try:
    import rospy
except Exception:  # pragma: no cover - allows offline preview/compile contexts
    rospy = None

# ─── Robot Hardware Status ────────────────────────────────────────────
# The new left hand is controlled outside the main pypot motor config because
# it uses XL-320/SEED protocol. Keep only Protocol 1.0 joints in arm movement.
LEFT_HAND_FUNCTIONAL = False  # Left gripper is unreliable; keep execution on right hand.

# Actions that require a working hand (grasp/close/open)
HAND_REQUIRED_ACTIONS = frozenset([
    "grasp", "grab", "pick", "take",
    "place", "drop", "release",
    "open_hand", "close_hand",
])

# The active motor config does not expose old hand sensor objects. The new left
# XL-320 hand is command-only here, so a 0 reading should not be treated as a
# failed grasp.
PALM_SENSOR_AVAILABLE = {
    "left": False,
    "right": False,
}


def palm_sensor_available(side: str) -> bool:
    return PALM_SENSOR_AVAILABLE.get(str(side).lower(), False)

# Joints that can be commanded through /nico/motion/setAngle when the
# corresponding motors are enabled in the active PyPot config. The current
# swapped physical right hand is controlled separately via XL-320 IDs 31/33/34-37,
# so r_wrist_z/r_wrist_x only represent the old SR-SEED wrist if IDs 23/25 are
# actually enabled at Motion.py startup.
COMMANDABLE_JOINTS = {
    "head": frozenset(["head_z", "head_y"]),
    "l_arm": frozenset(["l_shoulder_z", "l_shoulder_y", "l_arm_x", "l_elbow_y"]),
    "r_arm": frozenset([
        "r_shoulder_z",
        "r_shoulder_y",
        "r_arm_x",
        "r_elbow_y",
        "r_wrist_z",
        "r_wrist_x",
    ]),
}

JOINT_MOTOR_IDS = {
    "r_wrist_z": 23,
    "r_wrist_x": 25,
    "r_indexfingers_x": 29,
}


def _runtime_disabled_motor_ids():
    """Return Motion.py disabled motor IDs if ROS params are available."""
    if rospy is None:
        return set()
    try:
        if rospy.has_param("/nico/motion/disabledMotorIds"):
            raw = rospy.get_param("/nico/motion/disabledMotorIds")
        else:
            raw = []
    except Exception:
        return set()
    if raw is None:
        return set()
    if isinstance(raw, str):
        import re

        return {int(match) for match in re.findall(r"-?\d+", raw)}
    try:
        return {int(item) for item in raw}
    except Exception:
        return set()


def filter_commandable_joints(group: str, names, positions):
    """Drop joints that are not present in the active pypot motor config."""
    allowed = COMMANDABLE_JOINTS.get(group)
    if allowed is None:
        return list(names), list(positions), []
    disabled_ids = _runtime_disabled_motor_ids()

    filtered_names = []
    filtered_positions = []
    dropped = []
    for name, position in zip(names, positions):
        motor_id = JOINT_MOTOR_IDS.get(name)
        if name in allowed and (motor_id is None or motor_id not in disabled_ids):
            filtered_names.append(name)
            filtered_positions.append(position)
        else:
            dropped.append(name)
    return filtered_names, filtered_positions, dropped

# ─── Workspace Limits (meters) ───────────────────────────────────────
# Based on NICO arm URDF and IK reachability testing (EvoIK solver).
#
# The right arm's lateral (Y) range is ASYMMETRIC because r_shoulder_z
# has a hardware limit of ±0.8 rad (±46°). IK test results at table_z=0.68:
#
#   X=0.10: Y reachable [-0.30, -0.25]  (mostly right side only)
#   X=0.15: Y reachable [-0.30, -0.25]
#   X=0.20: Y reachable [-0.30, -0.20]
#   X=0.25: Y reachable [-0.30, +0.20]  (wider range when arm extends forward)
#   X=0.30: Y reachable [-0.30, +0.20]
#
# Conservative limits (with ~2-5cm safety margin from tested extremes):
MAX_REACH_X = 0.32    # Max forward reach
MIN_REACH_X = 0.10    # Min forward reach (too close = self-collision risk)
MAX_REACH_Y_LEFT = 0.18   # Max left reach (positive Y) — only reachable at X≥0.25
MAX_REACH_Y_RIGHT = -0.25  # Max right reach (negative Y) — reachable at all X values

# Legacy alias for backward compatibility
MAX_REACH_Y = MAX_REACH_Y_LEFT


def clamp_to_workspace(x: float, y: float) -> tuple:
    """
    Clamp target coordinates to the robot's reachable workspace.

    The Y limit is asymmetric: the right arm can reach further to the
    right (negative Y) than to the left (positive Y). Additionally,
    the left-side reach shrinks when X is small (arm close to body).

    Returns (clamped_x, clamped_y, was_clamped).
    """
    clamped_x = max(MIN_REACH_X, min(MAX_REACH_X, x))

    # Y limits are asymmetric — right side has more range
    # Also restrict left reach when X is small (arm can't extend sideways
    # when close to body)
    if clamped_x < 0.25:
        # At short forward reach, left-side (positive Y) is very limited
        effective_y_left = 0.05
    else:
        effective_y_left = MAX_REACH_Y_LEFT

    clamped_y = max(MAX_REACH_Y_RIGHT, min(effective_y_left, y))
    was_clamped = (clamped_x != x or clamped_y != y)
    return clamped_x, clamped_y, was_clamped


# ─── Arm Orientations (Quaternion) ───────────────────────────────────
# End-effector orientations for right and left arms.
# These produce a forward-reaching horizontal 'crab claw' hand.
ARM_ORIENTATION_RIGHT = Quaternion(0.7071068, 0, 0, 0.7071068)
ARM_ORIENTATION_LEFT = Quaternion(0.7071068, 0, 0, 0.7071068)

# Pointing orientation — end-effector pointing forward/down
POINT_ORIENTATION_RIGHT = Quaternion(-1.0, 0, 0, 0.0)
POINT_ORIENTATION_LEFT = Quaternion(1.0, 0, 0, 0.0)

# Touch orientation — use the forward/down end-effector attitude so the wrist
# does not lift into an upward check-mark shape for table-object contact.
TOUCH_ORIENTATION_RIGHT = POINT_ORIENTATION_RIGHT
TOUCH_ORIENTATION_LEFT = POINT_ORIENTATION_LEFT


def get_arm_orientation(is_right: bool) -> Quaternion:
    """Get the standard manipulation orientation for the given arm."""
    return ARM_ORIENTATION_RIGHT if is_right else ARM_ORIENTATION_LEFT


def get_point_orientation(is_right: bool) -> Quaternion:
    """Get the pointing orientation for the given arm."""
    return POINT_ORIENTATION_RIGHT if is_right else POINT_ORIENTATION_LEFT


def get_touch_orientation(is_right: bool) -> Quaternion:
    """Get a downward/right-hand touch orientation for table-object contact."""
    return TOUCH_ORIENTATION_RIGHT if is_right else TOUCH_ORIENTATION_LEFT


def get_pre_grasp_orientation(is_right: bool) -> Quaternion:
    """Get a conservative pre-grasp orientation before hand close."""
    return get_touch_orientation(is_right) if is_right else get_arm_orientation(is_right)


def get_grasp_orientation(is_right: bool) -> Quaternion:
    """Get the final grasp/contact orientation."""
    return get_touch_orientation(is_right) if is_right else get_arm_orientation(is_right)


# ─── ROS Service Names (v2 only) ─────────────────────────────────────
LLM_CHAT_SERVICE = "mllm_chat"
LLM_VISION_SERVICE = "mllm_vision"
LLM_VISIBILITY_SERVICE = "mllm_visibility"
LLM_DETECT_SERVICE = "mllm_detect"
