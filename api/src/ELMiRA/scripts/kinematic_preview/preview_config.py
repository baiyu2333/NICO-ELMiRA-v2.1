#!/usr/bin/env python3
"""Approximate setup values for NICO kinematic previews.

These values are coarse measurements from the current Week 10 setup. They are
used for visualization and logging only, not final calibration and not direct
robot control.
"""


# Coordinate convention:
# x = forward from NICO toward the object, meters
# y = robot left/right, meters; negative y is robot right side
# z = upward from ground, meters
GROUND_Z_M = 0.0
TABLE_SURFACE_Z_M = 0.08
DEFAULT_RIGHT_HAND_HEIGHT_FROM_GROUND_M = 0.24
MEASURED_RIGHT_HAND_HEIGHT_MIN_M = 0.23
MEASURED_RIGHT_HAND_HEIGHT_MAX_M = 0.25
RED_OBJECT_DISTANCE_FROM_NICO_M = 0.20
DEFAULT_TARGET_Y_M = -0.10
DEFAULT_OBJECT_CENTER_Z_M = 0.10
MIN_SAFE_Z_M = 0.085
PRE_GRASP_Z_M = 0.12
GRASP_ATTEMPT_Z_M = 0.10
LIFT_Z_M = 0.18

HEIGHT_WARNING = "Warning: estimated hand height is too close to or below table surface."
PREVIEW_NOTE = (
    "Coarse kinematic estimate only. No ROS motion command, no physics, "
    "no collision checking, and no contact simulation."
)
SAFETY_NOTE = "Preview only. No real robot motion is commanded."


def measured_setup_values():
    return {
        "ground_z_m": GROUND_Z_M,
        "table_surface_z_m": TABLE_SURFACE_Z_M,
        "default_right_hand_height_from_ground_m": DEFAULT_RIGHT_HAND_HEIGHT_FROM_GROUND_M,
        "measured_right_hand_height_min_m": MEASURED_RIGHT_HAND_HEIGHT_MIN_M,
        "measured_right_hand_height_max_m": MEASURED_RIGHT_HAND_HEIGHT_MAX_M,
        "red_object_distance_from_nico_m": RED_OBJECT_DISTANCE_FROM_NICO_M,
        "default_target_y_m": DEFAULT_TARGET_Y_M,
        "default_object_center_z_m": DEFAULT_OBJECT_CENTER_Z_M,
        "min_safe_z_m": MIN_SAFE_Z_M,
        "pre_grasp_z_m": PRE_GRASP_Z_M,
        "grasp_attempt_z_m": GRASP_ATTEMPT_Z_M,
        "lift_z_m": LIFT_Z_M,
    }

