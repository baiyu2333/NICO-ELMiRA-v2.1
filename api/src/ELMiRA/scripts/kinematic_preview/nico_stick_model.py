#!/usr/bin/env python3
"""Simplified offline NICO upper-body stick model.

This module intentionally avoids ROS and robot motor APIs. It is only a
forward-kinematics style pose preview from predefined joint-angle templates.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

try:
    from kinematic_preview.preview_config import (
        HEIGHT_WARNING,
        MIN_SAFE_Z_M,
        TABLE_SURFACE_Z_M,
    )
except Exception:  # Allows direct local imports during standalone checks.
    from preview_config import HEIGHT_WARNING, MIN_SAFE_Z_M, TABLE_SURFACE_Z_M


Point3 = Tuple[float, float, float]


@dataclass
class NICOStickModel:
    """Small upper-body model for report/demo previews, not full simulation."""

    torso_height: float = 0.55
    shoulder_half_width: float = 0.18
    neck_height: float = 0.08
    head_radius: float = 0.07
    upper_arm_length: float = 0.19
    lower_arm_length: float = 0.18
    hand_length: float = 0.08
    right_shoulder_offset: Point3 = field(default_factory=lambda: (0.0, -0.18, 0.52))
    left_shoulder_offset: Point3 = field(default_factory=lambda: (0.0, 0.18, 0.52))

    @staticmethod
    def _add(a: Point3, b: Point3) -> Point3:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _scale(v: Point3, length: float) -> Point3:
        return (v[0] * length, v[1] * length, v[2] * length)

    @staticmethod
    def _direction(pitch_deg: float, roll_deg: float) -> Point3:
        """Return a unit-ish arm direction from coarse pitch/roll angles.

        Coordinate convention:
        - x: forward from torso
        - y: robot left, so right side is negative y
        - z: up

        A pitch of 0 points downward. Positive pitch moves the hand forward.
        Roll gives a small lateral offset for visualization.
        """
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        base = (math.sin(pitch), 0.0, -math.cos(pitch))

        # Rotate around the x axis for a simple lateral shoulder roll.
        x = base[0]
        y = base[1] * math.cos(roll) - base[2] * math.sin(roll)
        z = base[1] * math.sin(roll) + base[2] * math.cos(roll)
        norm = math.sqrt(x * x + y * y + z * z) or 1.0
        return (x / norm, y / norm, z / norm)

    @staticmethod
    def _xyz_dict(point: Point3) -> Dict[str, float]:
        return {"x": round(point[0], 4), "y": round(point[1], 4), "z": round(point[2], 4)}

    def forward_kinematics(
        self,
        joint_angles: Dict[str, float],
        min_safe_z_m: float = MIN_SAFE_Z_M,
    ) -> Dict[str, object]:
        """Compute approximate upper-body and right-arm coordinates.

        Required template angles are:
        right_shoulder_pitch, right_shoulder_roll, right_elbow_pitch,
        right_wrist_pitch. Missing values default to 0.
        """
        shoulder_pitch = float(joint_angles.get("right_shoulder_pitch", 0.0))
        shoulder_roll = float(joint_angles.get("right_shoulder_roll", 0.0))
        elbow_pitch = float(joint_angles.get("right_elbow_pitch", 0.0))
        wrist_pitch = float(joint_angles.get("right_wrist_pitch", 0.0))

        torso_base = (0.0, 0.0, 0.0)
        torso_top = (0.0, 0.0, self.torso_height)
        neck = (0.0, 0.0, self.torso_height + self.neck_height)
        head_center = (0.0, 0.0, self.torso_height + self.neck_height + self.head_radius)
        right_shoulder = self.right_shoulder_offset
        left_shoulder = self.left_shoulder_offset

        upper_dir = self._direction(shoulder_pitch, shoulder_roll)
        lower_dir = self._direction(shoulder_pitch + elbow_pitch, shoulder_roll * 0.65)
        hand_dir = self._direction(
            shoulder_pitch + elbow_pitch + wrist_pitch,
            shoulder_roll * 0.45,
        )

        right_elbow = self._add(right_shoulder, self._scale(upper_dir, self.upper_arm_length))
        right_wrist = self._add(right_elbow, self._scale(lower_dir, self.lower_arm_length))
        right_hand = self._add(right_wrist, self._scale(hand_dir, self.hand_length))

        # Passive left arm pose is included only to make the figure recognisable.
        left_elbow = self._add(left_shoulder, (0.01, 0.03, -self.upper_arm_length))
        left_wrist = self._add(left_elbow, (0.01, 0.02, -self.lower_arm_length))
        left_hand = self._add(left_wrist, (0.0, 0.01, -self.hand_length))

        coordinates = {
            "torso_base": torso_base,
            "torso_top": torso_top,
            "neck": neck,
            "head_center": head_center,
            "right_shoulder": right_shoulder,
            "right_elbow": right_elbow,
            "right_wrist": right_wrist,
            "right_hand": right_hand,
            "left_shoulder": left_shoulder,
            "left_elbow": left_elbow,
            "left_wrist": left_wrist,
            "left_hand": left_hand,
        }

        above_table_safety_limit = right_hand[2] >= min_safe_z_m
        height_warning = "" if above_table_safety_limit else HEIGHT_WARNING

        return {
            "coordinate_system": "x forward, y left, z up; meters; coarse preview only",
            "coordinates": {
                name: self._xyz_dict(point) for name, point in coordinates.items()
            },
            "right_shoulder_xyz": self._xyz_dict(right_shoulder),
            "right_elbow_xyz": self._xyz_dict(right_elbow),
            "estimated_right_wrist_xyz": self._xyz_dict(right_wrist),
            "estimated_right_hand_xyz": self._xyz_dict(right_hand),
            "table_surface_z_m": TABLE_SURFACE_Z_M,
            "min_safe_z_m": min_safe_z_m,
            "above_table_safety_limit": above_table_safety_limit,
            "height_warning": height_warning,
            "link_lengths": {
                "upper_arm_m": self.upper_arm_length,
                "lower_arm_m": self.lower_arm_length,
                "hand_m": self.hand_length,
            },
        }

    @staticmethod
    def coordinate_tuple(coordinates: Dict[str, Dict[str, float]], name: str) -> Point3:
        point = coordinates[name]
        return (float(point["x"]), float(point["y"]), float(point["z"]))

    @staticmethod
    def polyline(coordinates: Dict[str, Dict[str, float]], names: Iterable[str]):
        return [NICOStickModel.coordinate_tuple(coordinates, name) for name in names]
