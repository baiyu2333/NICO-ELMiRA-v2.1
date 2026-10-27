"""ROS 1 capture adapter. No motion publishers, motor imports or motion services.

The only services exposed are capture/reset. A next-view request is text for a
supervisor, never a PoseStamped on a topic that could trigger an actuator.
"""
import json
import os
import socket
import threading
import xmlrpc.client
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .common import LOG_ROOT, read_json, sha256, transform, write_json
from .geometry import GuardedGeometry
from .masks import segment, validate_mask
from .session import ObserveMoveObserve, validate_observation


def ros_status():
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(3)
    try:
        master = xmlrpc.client.ServerProxy(os.environ.get("ROS_MASTER_URI", "http://localhost:11311"))
        code, _, topics = master.getPublishedTopics("/acf_rgb_read_only_check", "")
        if code != 1:
            raise RuntimeError("ROS master rejected read-only topic query")
        return {"status": "available", "topics": [t for t in topics if t[1] in
                ("sensor_msgs/Image", "sensor_msgs/JointState", "sensor_msgs/CameraInfo", "tf2_msgs/TFMessage")],
                "real_robot_motion_commanded": False}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc), "real_robot_motion_commanded": False}
    finally:
        socket.setdefaulttimeout(previous)


def decode_image(message):
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(message.encoding)
    if channels is None:
        raise ValueError("Supported Image encodings: bgr8/rgb8/mono8")
    height, width, step = int(message.height), int(message.width), int(message.step)
    if min(height, width) <= 0 or step < width*channels or len(message.data) != height*step:
        raise ValueError("Malformed image dimensions/stride/data")
    rows = np.frombuffer(message.data, np.uint8).reshape(height, step)
    image = rows[:, :width*channels].reshape(height, width, channels).copy()
    if channels == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if message.encoding == "rgb8" else image


def transform_message(message):
    t, q = message.transform.translation, message.transform.rotation
    from scipy.spatial.transform import Rotation
    quat = np.array([q.x, q.y, q.z, q.w])
    if not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1) > .001:
        raise ValueError("Invalid TF quaternion")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return transform(T)


class ReadOnlyNode:
    def __init__(self, config, output):
        import rospy
        import tf2_ros
        from sensor_msgs.msg import Image, JointState, CameraInfo
        from std_srvs.srv import Trigger
        self.rospy, self.config = rospy, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.images, self.masks, self.joints = deque(maxlen=3), deque(maxlen=8), deque(maxlen=200)
        self.info = None
        self.records, self.last_stamp = [], None
        self.calibration = Calibration.load(config["calibration"]) if config.get("calibration") else None
        if self.calibration:
            write_json(self.output / "calibration.json", self.calibration.record)
        self.geometry, self.loop = None, None
        if self.calibration and config.get("volume_verified") is True and config.get("volume_evidence"):
            self.geometry = GuardedGeometry(self.calibration.rectified_K, self.calibration.size,
                                            config["volume_center_m"], config["half_extent_m"])
        if self.geometry and config.get("candidates"):
            if any(candidate.get("world_frame") != config["world_frame"]
                   or candidate.get("optical_frame") != config["optical_frame"] for candidate in config["candidates"]):
                raise ValueError("Candidate poses must use the configured world and optical frames")
            self.loop = ObserveMoveObserve(config["volume_center_m"], config["head_limits"], config["candidates"],
                                           K=self.calibration.rectified_K, image_size=self.calibration.size)
        self.encoder = self.memory = None
        self.view_scorer = None
        if config.get("recognition_evaluation"):
            from .recognition import RGBEncoder
            from .vendor.acf3d.memory import LifelongPrototypeMemory
            report_path = Path(config["recognition_evaluation"])
            report = read_json(report_path)
            if report.get("feature_mode", "masked_rgb_visual_branch_only") != "masked_rgb_visual_branch_only":
                raise ValueError("This encoder is an offline diagnostic baseline, not a ROS recognition model")
            self.encoder = RGBEncoder(config["checkpoint"])
            if report["encoder_sha256"] != self.encoder.checkpoint_sha256:
                raise ValueError("Recognition model does not match real-object evaluation")
            self.memory = LifelongPrototypeMemory.load(report_path.parent / "real_object_memory.npz")
            if report.get("memory_sha256") != sha256(report_path.parent / "real_object_memory.npz"):
                raise ValueError("Recognition memory differs from evaluated memory")
            if self.loop:
                from .view_planner import RGBViewScorer
                self.view_scorer = RGBViewScorer(self.geometry, self.encoder, self.memory)
                self.loop.view_scorer = self.view_scorer
        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.tf)
        self.subscribers = [
            rospy.Subscriber(config["image_topic"], Image, self.image_callback, queue_size=1),
            rospy.Subscriber(config["mask_topic"], Image, self.mask_callback, queue_size=1),
            rospy.Subscriber(config["joint_topic"], JointState, self.joint_callback, queue_size=5),
            rospy.Subscriber(config["camera_info_topic"], CameraInfo, self.info_callback, queue_size=1)]
        self.services = [rospy.Service("~capture", Trigger, self.capture),
                         rospy.Service("~reset_loop", Trigger, self.reset_loop)]
        self.timer = rospy.Timer(rospy.Duration(.2), self.check_timeout)
        write_json(self.output / "config.json", config)

    def image_callback(self, message):
        with self.lock:
            self.images.append(message)

    def mask_callback(self, message):
        with self.lock:
            self.masks.append(message)

    def info_callback(self, message):
        with self.lock:
            self.info = message

    def joint_callback(self, message):
        with self.lock:
            self.joints.append(message)
        if self.loop and self.capture_lock.acquire(blocking=False):
            try:
                self.loop.feedback(self.head_record(message), self.rospy.Time.now().to_sec())
            except (ValueError, KeyError):
                pass
            finally:
                self.capture_lock.release()

    @staticmethod
    def head_record(message):
        positions = dict(zip(message.name, message.position))
        return {"timestamp_sec": message.header.stamp.to_sec(),
                "positions_rad": {j: positions[j] for j in ("head_z", "head_y")}}

    def check_timeout(self, _event):
        if self.loop and self.capture_lock.acquire(blocking=False):
            try:
                if self.loop.phase in ("waiting_for_move", "awaiting_new_image") and self.rospy.Time.now().to_sec()-self.loop.request_time > self.loop.timeout_sec:
                    self.loop.phase = "timeout"
                    write_json(self.output / "loop_status.json", {"status": "timeout", "real_robot_motion_commanded": False})
            finally:
                self.capture_lock.release()

    def reset_loop(self, _request):
        from std_srvs.srv import TriggerResponse
        with self.capture_lock:
            if self.loop:
                self.loop = ObserveMoveObserve(self.config["volume_center_m"], self.config["head_limits"], self.config["candidates"],
                                               K=self.calibration.rectified_K, image_size=self.calibration.size,
                                               view_scorer=self.view_scorer)
        return TriggerResponse(True, "Read-only viewpoint loop reset. No robot motion commanded.")

    def update_geometry(self, record, bgr, mask, now):
        stamp, T = validate_observation(record, self.calibration, self.last_stamp)
        if self.loop:
            self.loop.check_observation(T, stamp, record["head_feedback"], now)
        rectified, rectified_mask = self.calibration.rectify(bgr, mask)
        record["geometry"] = self.geometry.update_checked(rectified, rectified_mask, T)
        if not record["geometry"]["accepted"]:
            record["next_view"] = {"status": "observation_rejected", "real_robot_motion_commanded": False}
            return
        self.geometry.export(self.output / "reconstruction.npz")
        if self.view_scorer:
            self.view_scorer.add_observation(rectified, rectified_mask, T)
        if self.loop:
            record["next_view"] = self.loop.observe(T, stamp, record["head_feedback"], now)
            write_json(self.output / "loop_status.json", record["next_view"])

    def capture(self, _request):
        from std_srvs.srv import TriggerResponse
        if not self.capture_lock.acquire(blocking=False):
            return TriggerResponse(False, "Capture already processing")
        try:
            record = self.capture_snapshot()
            return TriggerResponse(True, json.dumps(record, allow_nan=False))
        except Exception as exc:
            return TriggerResponse(False, "Capture unavailable: " + str(exc))
        finally:
            self.capture_lock.release()

    def capture_snapshot(self):
        with self.lock:
            image = self.images[-1] if self.images else None
            masks, joints, info = list(self.masks), list(self.joints), self.info
        if image is None:
            raise ValueError("No image received")
        stamp, now = image.header.stamp.to_sec(), self.rospy.Time.now().to_sec()
        if stamp <= 0 or not 0 <= now-stamp <= .25 or (self.last_stamp is not None and stamp <= self.last_stamp):
            raise ValueError("Need a fresh, new timestamped image")
        bgr = decode_image(image)
        if bgr.ndim != 3:
            raise ValueError("RGB camera image required")
        index = len(self.records)
        frame = self.output / ("frame_%04d.png" % index)
        if not cv2.imwrite(str(frame), bgr):
            raise OSError("Cannot save RGB frame")
        c = self.config
        record = {"image": frame.name, "timestamp_sec": stamp, "timestamp_source": "ROS image header (publisher time, not verified exposure time)",
                  "image_sha256": sha256(frame),
                  "camera_id": c["camera_id"], "optical_frame": c["optical_frame"],
                  "original_image_frame_id": image.header.frame_id, "world_frame": c["world_frame"],
                  "units": "metres", "object_static": c.get("object_static", False),
                  "real_robot_motion_commanded": False, "missing": []}
        if joints:
            joint = min(joints, key=lambda j: abs(j.header.stamp.to_sec()-stamp))
            if abs(joint.header.stamp.to_sec()-stamp) <= .05:
                try:
                    record["head_feedback"] = self.head_record(joint)
                except KeyError:
                    record["missing"].append("Both head joints not present")
        if "head_feedback" not in record:
            record["missing"].append("Synchronized head feedback unavailable")
        try:
            if image.header.frame_id != c["optical_frame"] and not (
                    not image.header.frame_id and c.get("image_frame_binding_verified") is True
                    and c.get("image_frame_binding_evidence")):
                raise ValueError("Image optical frame missing/mismatched; verify physical camera-to-frame binding")
            if not self.calibration:
                raise ValueError("Verified calibration unavailable (raw image still saved)")
            if c["camera_id"] != self.calibration.record["camera_id"] or c["optical_frame"] != self.calibration.record["optical_frame"]:
                raise ValueError("Configured camera differs from calibration")
            if info is not None:
                record["camera_info"] = {"K": list(info.K), "D": list(info.D), "width": info.width, "height": info.height,
                                         "distortion_model": info.distortion_model, "frame_id": info.header.frame_id}
                if ((info.width, info.height) != self.calibration.size or info.header.frame_id != c["optical_frame"]
                        or info.distortion_model != self.calibration.model
                        or np.asarray(info.D).shape != self.calibration.D.shape
                        or not np.allclose(info.D, self.calibration.D)
                        or not np.allclose(np.asarray(info.K).reshape(3, 3), self.calibration.K)):
                    raise ValueError("CameraInfo conflicts with verified calibration")
            rectified = self.calibration.rectify(bgr)
            if c.get("pose_source") == "fixed_metric_chessboard_pnp":
                from .calibration import board_pose
                if c.get("board_fixed_and_order_verified") is not True:
                    raise ValueError("Verify fixed board origin, corner ordering and metric square size first")
                record.update(board_pose(rectified, self.calibration.rectified_K,
                                         c["board_columns"], c["board_rows"], c["square_m"]))
            else:
                tf = self.tf.lookup_transform(c["world_frame"], c["optical_frame"], image.header.stamp,
                                              self.rospy.Duration(.2))
                record.update(T_wc=transform_message(tf).tolist(), pose_source="tf_calibrated")
            record.update(pose_timestamp_sec=stamp, pose_verified=c.get("pose_verified", False),
                          pose_evidence=c.get("pose_evidence", ""))
        except Exception as exc:
            record["missing"].append(str(exc))
        matching = [m for m in masks if abs(m.header.stamp.to_sec()-stamp) < 1e-6
                    and m.header.frame_id == image.header.frame_id]
        mask = None
        try:
            if matching:
                mask = validate_mask(decode_image(matching[-1]), bgr.shape)
                record.update(mask_reviewed=c.get("mask_verified", False), mask_evidence=c.get("mask_evidence", ""))
            elif c.get("initial_box_xywh"):
                mask, details = segment(bgr, c["initial_box_xywh"])
                record.update(details)
            else:
                raise ValueError("No binary mask with matching image timestamp/frame; capture RGB for later review")
            mask_path = self.output / ("mask_%04d.png" % index)
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError("Cannot save mask")
            record["mask"] = mask_path.name
            record["mask_sha256"] = sha256(mask_path)
            if self.encoder:
                if not record.get("mask_reviewed"):
                    raise ValueError("Recognition requires reviewed/validated mask source")
                label, similarity, margin, _, _ = self.memory.predict(self.encoder.encode(bgr, mask))
                record["recognition"] = {"label": label, "similarity": similarity, "margin": margin,
                                          "mode": "real_object_rgb_gallery", "not_a_probability": True}
            if self.geometry:
                self.update_geometry(record, bgr, mask, now)
        except Exception as exc:
            record["missing"].append(str(exc))
        self.last_stamp = stamp
        self.records.append(record)
        write_json(frame.with_suffix(".json"), record)
        manifest = {k: c.get(k) for k in ("calibration", "volume_verified", "volume_evidence", "volume_center_m", "half_extent_m")}
        manifest["calibration"] = "calibration.json" if self.calibration else None
        manifest.update(observations=self.records, real_robot_motion_commanded=False)
        write_json(self.output / "session.json", manifest)
        return record


def run(config_path, output=None):
    status = ros_status()
    if status["status"] != "available":
        print(json.dumps(status, indent=2))
        return 0
    import rospy
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    for field in ("calibration", "checkpoint", "recognition_evaluation"):
        if config.get(field):
            config[field] = str((config_path.parent / config[field]).resolve())
    output = output or LOG_ROOT / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
    rospy.init_node("acf_single_rgb", anonymous=False)
    node = ReadOnlyNode(config, output)
    print("Read-only ROS 1 capture ready: /acf_single_rgb/capture. Output: " + str(node.output))
    rospy.spin()
    return 0
