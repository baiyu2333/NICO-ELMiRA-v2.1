"""
ELMiRA shared constants — single source of truth for workspace limits,
arm orientations, and hardware configuration.
"""

from geometry_msgs.msg import Quaternion

# ─── Robot Hardware Status ────────────────────────────────────────────
# Left hand (wrist/fingers) is PHYSICALLY BROKEN.
# Left arm (shoulder/elbow) can still move but MUST NOT be used for
# grasp/place actions that require hand control.
LEFT_HAND_FUNCTIONAL = True  # New XL-320 4-finger hand installed

# Actions that require a working hand (grasp/close/open)
HAND_REQUIRED_ACTIONS = frozenset([
    "grasp", "grab", "pick", "take",
    "place", "drop", "release",
    "open_hand", "close_hand",
])

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


def get_arm_orientation(is_right: bool) -> Quaternion:
    """Get the standard manipulation orientation for the given arm."""
    return ARM_ORIENTATION_RIGHT if is_right else ARM_ORIENTATION_LEFT


def get_point_orientation(is_right: bool) -> Quaternion:
    """Get the pointing orientation for the given arm."""
    return POINT_ORIENTATION_RIGHT if is_right else POINT_ORIENTATION_LEFT


# ─── ROS Service Names (v2 only) ─────────────────────────────────────
LLM_CHAT_SERVICE = "mllm_chat"
LLM_VISION_SERVICE = "mllm_vision"
LLM_VISIBILITY_SERVICE = "mllm_visibility"
LLM_DETECT_SERVICE = "mllm_detect"
