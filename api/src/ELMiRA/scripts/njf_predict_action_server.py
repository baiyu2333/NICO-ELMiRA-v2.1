#!/usr/bin/env python3
"""
NJF-inspired visual Jacobian refinement service for NICO grasping.

This is not a full Neural Jacobian Field implementation. It is a small,
bounded visual-servo controller with the same service boundary a trained NJF
model can later use: visual/metric error in, safe correction out.
"""

import math

import rospy

from elmira.srv import NJFPredictAction, NJFPredictActionResponse
from utils.constants import clamp_to_workspace


class NJFPredictActionServer:
    def __init__(self):
        self.max_correction_m = float(rospy.get_param("~max_correction_m", 0.025))
        self.max_joint_delta = float(rospy.get_param("~max_joint_delta", 0.08))
        self.target_error_threshold = float(
            rospy.get_param("~target_error_threshold", 0.005)
        )
        self.image_error_threshold = float(
            rospy.get_param("~image_error_threshold", 0.02)
        )
        self.metric_gain = float(rospy.get_param("~metric_gain", 0.75))
        self.image_gain_m_per_unit = float(
            rospy.get_param("~image_gain_m_per_unit", 0.08)
        )
        self.service_name = rospy.get_param("~service_name", "njf_predict_action")

        rospy.Service(self.service_name, NJFPredictAction, self.handle_predict)
        rospy.loginfo(
            "NJF visual Jacobian server ready on %s "
            "(max_correction=%.3fm, max_joint_delta=%.3frad)",
            self.service_name,
            self.max_correction_m,
            self.max_joint_delta,
        )

    @staticmethod
    def _finite(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    @staticmethod
    def _clamp(value, limit):
        limit = abs(float(limit))
        return max(-limit, min(limit, float(value)))

    def _metric_error_from_request(self, request):
        x_error = self._finite(request.x_error_m)
        y_error = self._finite(request.y_error_m)
        image_u = self._finite(request.image_error_u)
        image_v = self._finite(request.image_error_v)

        if abs(x_error) < self.target_error_threshold:
            x_error = 0.0
        if abs(y_error) < self.target_error_threshold:
            y_error = 0.0

        # If metric error is unavailable, use normalized image error as a local
        # visual-servo proxy. u controls lateral correction, v controls forward.
        if x_error == 0.0 and abs(image_v) >= self.image_error_threshold:
            x_error = image_v * self.image_gain_m_per_unit
        if y_error == 0.0 and abs(image_u) >= self.image_error_threshold:
            y_error = image_u * self.image_gain_m_per_unit

        return x_error, y_error

    def _joint_delta_names(self, planning_group):
        prefix = "r" if str(planning_group).startswith("r") else "l"
        return [
            f"{prefix}_shoulder_z",
            f"{prefix}_shoulder_y",
            f"{prefix}_arm_x",
            f"{prefix}_elbow_y",
        ]

    def _predict_joint_delta(self, planning_group, x_corr, y_corr):
        """Small pseudo-Jacobian joint delta for future direct joint control."""
        side = -1.0 if str(planning_group).startswith("r") else 1.0
        raw = [
            1.2 * y_corr * side,
            -0.8 * x_corr,
            0.7 * x_corr * side,
            -0.5 * x_corr,
        ]
        return [self._clamp(delta, self.max_joint_delta) for delta in raw]

    def handle_predict(self, request):
        planning_group = str(request.planning_group or "r_arm")
        if planning_group not in ("l_arm", "r_arm", "l_hand", "r_hand"):
            return NJFPredictActionResponse(
                success=False,
                message=f"unsupported planning_group '{planning_group}'",
                x_correction_m=0.0,
                y_correction_m=0.0,
                joint_delta_names=[],
                joint_deltas=[],
                confidence=0.0,
            )

        try:
            x_error, y_error = self._metric_error_from_request(request)
            x_corr = self._clamp(self.metric_gain * x_error, self.max_correction_m)
            y_corr = self._clamp(self.metric_gain * y_error, self.max_correction_m)

            # Reject correction if adding it to the current estimate would be
            # outside broad table reach limits. The final planner also clamps,
            # but this prevents the service from suggesting obviously bad moves.
            current_x = self._finite(rospy.get_param("/elmira/current_refine_x", 0.0))
            current_y = self._finite(rospy.get_param("/elmira/current_refine_y", 0.0))
            if current_x or current_y:
                checked_x, checked_y, was_clamped = clamp_to_workspace(
                    current_x + x_corr, current_y + y_corr
                )
                if was_clamped:
                    x_corr = checked_x - current_x
                    y_corr = checked_y - current_y

            if abs(x_corr) < self.target_error_threshold:
                x_corr = 0.0
            if abs(y_corr) < self.target_error_threshold:
                y_corr = 0.0

            joint_names = self._joint_delta_names(planning_group)
            joint_deltas = self._predict_joint_delta(planning_group, x_corr, y_corr)
            confidence = 0.8 if (x_corr or y_corr) else 1.0

            return NJFPredictActionResponse(
                success=True,
                message=(
                    "visual Jacobian correction "
                    f"x={x_corr:.4f}m y={y_corr:.4f}m"
                ),
                x_correction_m=float(x_corr),
                y_correction_m=float(y_corr),
                joint_delta_names=joint_names,
                joint_deltas=[float(v) for v in joint_deltas],
                confidence=float(confidence),
            )
        except Exception as exc:
            rospy.logwarn("NJF visual Jacobian prediction failed: %s", exc)
            return NJFPredictActionResponse(
                success=False,
                message=f"fallback zero correction: {exc}",
                x_correction_m=0.0,
                y_correction_m=0.0,
                joint_delta_names=[],
                joint_deltas=[],
                confidence=0.0,
            )


def main():
    rospy.init_node("njf_predict_action_server")
    NJFPredictActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
