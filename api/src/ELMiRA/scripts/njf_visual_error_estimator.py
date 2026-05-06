#!/usr/bin/env python3
"""
Color-marker visual error estimator for NJF-inspired grasp refinement.

This node detects one object marker and one hand/gripper marker in the camera
image using configurable HSV ranges. It publishes a debug image and writes the
normalized hand-to-object error into ROS params consumed by grasp_refinement.py
and njf_predict_action_server.py.
"""

import math

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class NJFVisualErrorEstimator:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/nico/vision/left")
        self.debug_topic = rospy.get_param(
            "~debug_image_topic", "/elmira/njf_visual_error/debug_image"
        )
        self.status_topic = rospy.get_param(
            "~status_topic", "/elmira/njf_visual_error/status"
        )
        self.object_hsv_lower = self._parse_hsv(
            rospy.get_param("~object_hsv_lower", [40, 80, 50])
        )
        self.object_hsv_upper = self._parse_hsv(
            rospy.get_param("~object_hsv_upper", [85, 255, 255])
        )
        self.hand_hsv_lower = self._parse_hsv(
            rospy.get_param("~hand_hsv_lower", [100, 80, 50])
        )
        self.hand_hsv_upper = self._parse_hsv(
            rospy.get_param("~hand_hsv_upper", [130, 255, 255])
        )
        self.min_area = float(rospy.get_param("~min_area", 80.0))
        self.publish_debug_image = self._parse_bool(
            rospy.get_param("~publish_debug_image", True)
        )
        self.meters_per_norm_error = float(
            rospy.get_param("~meters_per_norm_error", 0.0)
        )

        self._bridge = CvBridge()
        self._debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)
        self._status_pub = rospy.Publisher(self.status_topic, String, queue_size=5)
        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)

        self._set_error_params(False, False, False, 0.0, 0.0)
        rospy.loginfo(
            "NJFVisualErrorEstimator: image=%s object_hsv=%s..%s "
            "hand_hsv=%s..%s min_area=%.1f",
            self.image_topic,
            self.object_hsv_lower.tolist(),
            self.object_hsv_upper.tolist(),
            self.hand_hsv_lower.tolist(),
            self.hand_hsv_upper.tolist(),
            self.min_area,
        )

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _parse_hsv(value):
        if isinstance(value, str):
            cleaned = value.strip().strip("[]()")
            parts = [p.strip() for p in cleaned.replace(";", ",").split(",")]
            values = [int(float(p)) for p in parts if p]
        else:
            values = [int(float(v)) for v in value]

        if len(values) != 3:
            raise ValueError(f"HSV param must contain 3 values, got {value}")
        h = max(0, min(179, values[0]))
        s = max(0, min(255, values[1]))
        v = max(0, min(255, values[2]))
        return np.array([h, s, v], dtype=np.uint8)

    @staticmethod
    def _finite(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def _detect_marker(self, hsv_image, lower, upper):
        mask = cv2.inRange(hsv_image, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contour_result = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contour_result[-2]
        if not contours:
            return None, 0.0, mask

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_area:
            return None, area, mask

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None, area, mask

        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        return center, area, mask

    def _set_error_params(
        self,
        valid,
        hand_visible,
        object_visible,
        image_error_u,
        image_error_v,
    ):
        image_error_u = max(-1.0, min(1.0, self._finite(image_error_u)))
        image_error_v = max(-1.0, min(1.0, self._finite(image_error_v)))

        rospy.set_param("/elmira/njf_visual_error_valid", bool(valid))
        rospy.set_param("/elmira/njf_hand_visible", bool(hand_visible))
        rospy.set_param("/elmira/njf_object_visible", bool(object_visible))
        rospy.set_param("/elmira/njf_image_error_u", float(image_error_u))
        rospy.set_param("/elmira/njf_image_error_v", float(image_error_v))
        rospy.set_param("/elmira/njf_visual_error_stamp", rospy.Time.now().to_sec())

        if valid and self.meters_per_norm_error > 0.0:
            rospy.set_param(
                "/elmira/njf_x_error_m",
                float(image_error_v * self.meters_per_norm_error),
            )
            rospy.set_param(
                "/elmira/njf_y_error_m",
                float(image_error_u * self.meters_per_norm_error),
            )
        else:
            rospy.set_param("/elmira/njf_x_error_m", 0.0)
            rospy.set_param("/elmira/njf_y_error_m", 0.0)

    def _draw_debug(
        self,
        image,
        object_center,
        hand_center,
        object_area,
        hand_area,
        valid,
        image_error_u,
        image_error_v,
    ):
        debug = image.copy()
        if object_center is not None:
            cv2.circle(debug, object_center, 8, (0, 255, 0), -1)
            cv2.putText(
                debug,
                f"object {object_area:.0f}",
                (object_center[0] + 10, object_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        if hand_center is not None:
            cv2.circle(debug, hand_center, 8, (255, 0, 0), -1)
            cv2.putText(
                debug,
                f"hand {hand_area:.0f}",
                (hand_center[0] + 10, hand_center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        if valid:
            cv2.arrowedLine(debug, hand_center, object_center, (0, 255, 255), 2)

        status = (
            f"valid={valid} err_u={image_error_u:.3f} "
            f"err_v={image_error_v:.3f}"
        )
        cv2.putText(
            debug,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if valid else (0, 0, 255),
            2,
        )
        return debug

    def _image_cb(self, msg):
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(
                5, f"NJFVisualErrorEstimator: image conversion failed: {exc}"
            )
            self._set_error_params(False, False, False, 0.0, 0.0)
            return

        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        object_center, object_area, _ = self._detect_marker(
            hsv, self.object_hsv_lower, self.object_hsv_upper
        )
        hand_center, hand_area, _ = self._detect_marker(
            hsv, self.hand_hsv_lower, self.hand_hsv_upper
        )

        object_visible = object_center is not None
        hand_visible = hand_center is not None
        valid = object_visible and hand_visible

        if valid:
            image_error_u = (object_center[0] - hand_center[0]) / max(width / 2.0, 1.0)
            image_error_v = (object_center[1] - hand_center[1]) / max(height / 2.0, 1.0)
        else:
            image_error_u = 0.0
            image_error_v = 0.0

        self._set_error_params(
            valid, hand_visible, object_visible, image_error_u, image_error_v
        )

        status = (
            f"valid={valid} hand={hand_visible} object={object_visible} "
            f"err_u={image_error_u:.3f} err_v={image_error_v:.3f}"
        )
        self._status_pub.publish(String(data=status))
        if valid:
            rospy.loginfo_throttle(2.0, f"NJFVisualErrorEstimator: {status}")
        else:
            rospy.logwarn_throttle(2.0, f"NJFVisualErrorEstimator: {status}")

        if self.publish_debug_image:
            debug = self._draw_debug(
                image,
                object_center,
                hand_center,
                object_area,
                hand_area,
                valid,
                image_error_u,
                image_error_v,
            )
            debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self._debug_pub.publish(debug_msg)


def main():
    rospy.init_node("njf_visual_error_estimator")
    NJFVisualErrorEstimator()
    rospy.spin()


if __name__ == "__main__":
    main()
