#!/usr/bin/env python3
"""
Record NICO image/joint data in a Neural Jacobian Field compatible layout.

This node is intentionally passive: it does not command the robot. Run it while
ELMiRA or manual test scripts move the robot to collect image/action sequences.
"""

import json
import tempfile
import threading
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from nicomsg.msg import sff
from sensor_msgs.msg import Image, JointState


DEFAULT_JOINT_NAMES = [
    "l_shoulder_z",
    "l_shoulder_y",
    "l_arm_x",
    "l_elbow_y",
    "l_wrist_z",
    "l_wrist_x",
    "l_thumb_z",
    "l_thumb_x",
    "l_indexfingers_x",
    "l_middlefingers_x",
]

XL320_MOTOR_NAMES = {
    31: "l_wrist_z",
    33: "l_wrist_x",
    34: "l_thumb_z",
    35: "l_thumb_x",
    36: "l_indexfingers_x",
    37: "l_middlefingers_x",
}


class NJFDataRecorder:
    def __init__(self):
        self.output_dir = Path(
            rospy.get_param("~output_dir", "/tmp/nico_njf_dataset")
        ).expanduser()
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.image_topic = rospy.get_param("~image_topic", "/nico/vision/left")
        self.left_joint_topic = rospy.get_param(
            "~left_joint_topic", "/left/open_manipulator_p/joint_states"
        )
        self.right_joint_topic = rospy.get_param(
            "~right_joint_topic", "/right/open_manipulator_p/joint_states"
        )
        self.head_joint_topic = rospy.get_param(
            "~head_joint_topic", "/NICOL/joint_states"
        )
        self.xl320_topic = rospy.get_param("~xl320_topic", "/nico/motion/xl320_cmd")
        self.sample_rate_hz = float(rospy.get_param("~sample_rate_hz", 2.0))
        self.sample_idx = int(rospy.get_param("~sample_idx", 0))
        self.camera_idx = int(rospy.get_param("~camera_idx", 0))
        self.joint_names = list(rospy.get_param("~joint_names", DEFAULT_JOINT_NAMES))

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_image_shape = None
        self._joint_positions = {name: 0.0 for name in self.joint_names}
        self._frames = []
        self._frame_idx = 0

        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)
        rospy.Subscriber(self.left_joint_topic, JointState, self._joint_state_cb)
        rospy.Subscriber(self.right_joint_topic, JointState, self._joint_state_cb)
        rospy.Subscriber(self.head_joint_topic, JointState, self._joint_state_cb)
        rospy.Subscriber(self.xl320_topic, sff, self._xl320_cb)

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.sample_rate_hz, 0.1)), self._record_frame
        )
        rospy.on_shutdown(self._write_transforms)

        rospy.loginfo(
            "NJFDataRecorder: recording %s at %.2f Hz into %s",
            self.image_topic,
            self.sample_rate_hz,
            self.output_dir,
        )

    def _image_cb(self, msg):
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5, f"NJFDataRecorder: image conversion failed: {exc}")
            return

        with self._lock:
            self._latest_image = image.copy()
            self._latest_image_shape = image.shape[:2]

    def _joint_state_cb(self, msg):
        with self._lock:
            for name, position in zip(msg.name, msg.position):
                if name in self._joint_positions:
                    self._joint_positions[name] = float(position)

    def _xl320_cb(self, msg):
        try:
            motor_id = int(msg.param1)
            register = int(msg.param2)
            value = float(msg.param3)
        except (TypeError, ValueError):
            return

        if register != 30 or motor_id not in XL320_MOTOR_NAMES:
            return

        # XL-320 goal position register is 0..1023 for roughly -150..+150 deg.
        position_deg = (value / 1023.0) * 300.0 - 150.0
        joint_name = XL320_MOTOR_NAMES[motor_id]
        with self._lock:
            if joint_name in self._joint_positions:
                self._joint_positions[joint_name] = position_deg

    def _record_frame(self, _event):
        with self._lock:
            if self._latest_image is None:
                rospy.logwarn_throttle(
                    10, f"NJFDataRecorder: waiting for image on {self.image_topic}"
                )
                return

            frame_idx = self._frame_idx
            image = self._latest_image.copy()
            joint_pos = [self._joint_positions[name] for name in self.joint_names]
            self._frame_idx += 1

        file_name = f"{self.sample_idx:05d}_{frame_idx:05d}.png"
        rel_path = f"images/{file_name}"
        abs_path = self.images_dir / file_name
        if not cv2.imwrite(str(abs_path), image):
            rospy.logwarn(f"NJFDataRecorder: failed to write {abs_path}")
            return

        self._frames.append(
            {
                "file_path": rel_path,
                "time": frame_idx / self.sample_rate_hz,
                "sample_idx": self.sample_idx,
                "camera_idx": self.camera_idx,
                "joint_pos": joint_pos,
            }
        )
        if frame_idx % 10 == 0:
            self._write_transforms()
            rospy.loginfo(f"NJFDataRecorder: recorded {frame_idx + 1} frames")

    def _camera_metadata(self):
        with self._lock:
            shape = self._latest_image_shape
        height, width = shape if shape is not None else (480, 640)

        fl_x = float(rospy.get_param("~fl_x", width))
        fl_y = float(rospy.get_param("~fl_y", width))
        cx = float(rospy.get_param("~cx", width / 2.0))
        cy = float(rospy.get_param("~cy", height / 2.0))
        transform = rospy.get_param(
            "~camera_transform_matrix",
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        return {
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": int(width),
            "h": int(height),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "transform_matrix": transform,
        }

    def _write_transforms(self):
        data = {
            "camera_model": "OPENCV",
            "joint_names": self.joint_names,
            "cameras": [self._camera_metadata()],
            "frames": self._frames,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "transforms.json"
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=str(self.output_dir), suffix=".json"
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(target)


def main():
    rospy.init_node("njf_data_recorder")
    NJFDataRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
