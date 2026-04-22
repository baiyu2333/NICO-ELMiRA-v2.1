#!/usr/bin/env python3
"""
Workspace Bounding Box Visualizer for ELMiRA

Subscribes to the camera topic and draws the workspace polygon on the image.
This helps you see EXACTLY where objects need to be placed for the robot to reach them.

Usage (after roscore and camera are running):
    python3 ~/catkin_ws/src/NICO-software/api/src/ELMiRA/scripts/visualize_workspace.py

Then view in another terminal:
    rqt_image_view /elmira/workspace_debug
"""

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image


# ─── Same ROS Image → Numpy converter used by the rest of the project ─────
def ros_image_to_numpy(msg):
    dtype = np.uint8
    n_channels = 3 if msg.encoding in ('bgr8', 'rgb8') else 1
    img_buf = np.frombuffer(msg.data, dtype=dtype)
    try:
        img = img_buf.reshape((msg.height, msg.width, n_channels))
        if msg.encoding == 'rgb8':
            img = img[:, :, ::-1]
        return img
    except ValueError as e:
        rospy.logwarn(f"Image reshape failed: {e}")
        return None


def numpy_to_ros_image(img):
    """Convert numpy BGR image back to ROS Image message."""
    msg = Image()
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = img.tobytes()
    return msg


# This is the EXACT same workspace polygon used in ObjectSelector
# Coordinates are normalized (0-1) relative to image width/height
WORKSPACE_POLYGON = np.array([
    [0.0396, 0.7160],
    [0.2021, 0.3444],
    [0.7646, 0.3278],
    [0.9448, 0.7313],
    [0.8162, 0.8069],
    [0.6391, 0.8632],
    [0.4380, 0.8757],
    [0.2599, 0.8375],
    [0.1328, 0.7771],
])


def main():
    rospy.init_node("workspace_visualizer", anonymous=True)

    camera_topic = rospy.get_param("~camera_topic", "/nico/vision/right")
    debug_pub = rospy.Publisher(
        "/elmira/workspace_debug",
        Image,
        queue_size=1,
    )

    rospy.loginfo(f"Workspace visualizer: subscribing to {camera_topic}")
    rospy.loginfo("View at: rqt_image_view /elmira/workspace_debug")

    rate = rospy.Rate(2)  # 2 Hz

    while not rospy.is_shutdown():
        try:
            img_msg = rospy.wait_for_message(camera_topic, Image, timeout=5.0)
        except rospy.ROSException:
            rospy.logwarn_throttle(10, f"No camera image from {camera_topic}")
            rate.sleep()
            continue

        cv_image = ros_image_to_numpy(img_msg)
        if cv_image is None:
            rate.sleep()
            continue

        h, w = cv_image.shape[:2]

        # Convert normalized polygon to pixel coordinates
        pts = (WORKSPACE_POLYGON * np.array([w, h])).astype(np.int32)

        # Draw filled semi-transparent polygon (green = valid area)
        overlay = cv_image.copy()
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        cv2.addWeighted(overlay, 0.25, cv_image, 0.75, 0, cv_image)

        # Draw polygon border (bright green)
        cv2.polylines(cv_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw vertex labels
        for i, (px, py) in enumerate(pts):
            cv2.circle(cv_image, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(cv_image, f"P{i}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw info text
        cv2.putText(cv_image, "GREEN = Reachable Workspace", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(cv_image, "Place objects INSIDE the green area", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(cv_image, f"Image: {w}x{h}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Publish debug image
        debug_msg = numpy_to_ros_image(cv_image)
        debug_msg.header.stamp = rospy.Time.now()
        debug_pub.publish(debug_msg)

        rate.sleep()


if __name__ == "__main__":
    main()
