#!/usr/bin/env python3
"""Opt-in isolated ROS1 smoke test. Starts NO NICO, motion or robot nodes.

Uses a fresh localhost master port and synthetic camera/mask/head/TF publishers.
All subprocess groups belong to this test and are stopped in finally.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import xmlrpc.client
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from active_vision.common import LOG_ROOT, write_json


def main():
    processes = []
    done = threading.Event()
    thread = None
    report = {"test": "isolated ROS1 synthetic adapter smoke", "real_robot_motion_commanded": False}
    with tempfile.TemporaryDirectory(prefix="acf_ros1_test_") as directory:
        root = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        uri = "http://127.0.0.1:%d" % port
        env = dict(os.environ, ROS_MASTER_URI=uri, ROS_HOSTNAME="127.0.0.1", ROS_LOG_DIR=str(root / "ros_logs"))
        env.pop("ROS_IP", None)
        os.environ.update({k: env[k] for k in ("ROS_MASTER_URI", "ROS_HOSTNAME", "ROS_LOG_DIR")})
        os.environ.pop("ROS_IP", None)
        with (root / "processes.log").open("w+") as logs:
            try:
                master = subprocess.Popen(["roscore", "-p", str(port)], env=env, stdout=logs,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(master)
                deadline = time.monotonic()+12
                while time.monotonic() < deadline:
                    try:
                        if xmlrpc.client.ServerProxy(uri).getPid("/acf_isolated_test")[0] == 1:
                            break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError("Isolated ROS master did not start")
                import rospy
                import tf2_ros
                from sensor_msgs.msg import Image, JointState, CameraInfo
                from geometry_msgs.msg import TransformStamped
                from std_srvs.srv import Trigger
                rospy.init_node("acf_synthetic_source", disable_signals=True)
                K = [[200, 0, 120], [0, 200, 100], [0, 0, 1]]
                calibration = {"K": K, "D": [0]*4, "width": 240, "height": 200,
                               "distortion_model": "plumb_bob", "camera_id": "SYNTHETIC_TEST_CAMERA",
                               "optical_frame": "test_optical", "verified": True,
                               "verification_evidence": "synthetic analytic fixture only, not NICO"}
                write_json(root / "calibration.json", calibration)
                config = {"image_topic": "/acf_test/image", "mask_topic": "/acf_test/mask",
                          "joint_topic": "/acf_test/joints", "camera_info_topic": "/acf_test/info",
                          "camera_id": "SYNTHETIC_TEST_CAMERA", "optical_frame": "test_optical",
                          "world_frame": "test_world", "calibration": str(root / "calibration.json"),
                          "pose_verified": True, "pose_evidence": "synthetic analytic TF fixture",
                          "mask_verified": True, "mask_evidence": "synthetic circle silhouette",
                          "object_static": True, "volume_verified": True, "volume_evidence": "synthetic sphere",
                          "volume_center_m": [0, 0, .5], "half_extent_m": .08}
                write_json(root / "config.json", config)
                node = subprocess.Popen([sys.executable, str(SCRIPTS / "active_vision_cli.py"), "ros-node",
                                         "--config", str(root / "config.json"), "--output", str(root / "capture")],
                                        env=env, stdout=logs, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(node)
                rospy.wait_for_service("/acf_single_rgb/capture", timeout=15)
                pubs = [rospy.Publisher(topic, cls, queue_size=2) for topic, cls in
                        [("/acf_test/image", Image), ("/acf_test/mask", Image),
                         ("/acf_test/joints", JointState), ("/acf_test/info", CameraInfo)]]
                broadcaster = tf2_ros.TransformBroadcaster()
                position = [-.05]

                def publish():
                    while not done.is_set():
                        stamp = rospy.Time.now()
                        x = position[0]
                        tf = TransformStamped()
                        tf.header.stamp, tf.header.frame_id = stamp, "test_world"
                        tf.child_frame_id = "test_optical"
                        tf.transform.translation.x, tf.transform.rotation.w = x, 1
                        broadcaster.sendTransform(tf)
                        mask = np.zeros((200, 240), np.uint8)
                        cv2.circle(mask, (int(120-200*x/.5), 100), 20, 255, -1)
                        bgr = np.repeat(mask[:, :, None], 3, axis=2)
                        for pub, array, encoding in ((pubs[0], bgr, "bgr8"), (pubs[1], mask, "mono8")):
                            msg = Image()
                            msg.header.stamp, msg.header.frame_id = stamp, "test_optical"
                            msg.height, msg.width, msg.encoding = 200, 240, encoding
                            msg.step = 720 if encoding == "bgr8" else 240
                            msg.data = array.tobytes()
                            pub.publish(msg)
                        joint = JointState()
                        joint.header.stamp = stamp
                        joint.name, joint.position = ["head_z", "head_y"], [0, -.5]
                        pubs[2].publish(joint)
                        info = CameraInfo()
                        info.header.stamp, info.header.frame_id = stamp, "test_optical"
                        info.width, info.height = 240, 200
                        info.K, info.D, info.distortion_model = np.array(K).ravel().tolist(), [0]*4, "plumb_bob"
                        pubs[3].publish(info)
                        done.wait(.05)

                thread = threading.Thread(target=publish, daemon=True)
                thread.start()
                time.sleep(1)
                capture = rospy.ServiceProxy("/acf_single_rgb/capture", Trigger)
                results = []
                for x in (-.05, .05):
                    position[0] = x
                    time.sleep(.4)
                    response = capture()
                    if not response.success:
                        raise AssertionError(response.message)
                    record = json.loads(response.message)
                    if record["missing"]:
                        raise AssertionError(record["missing"])
                    results.append(record)
                assert results[0]["geometry"]["status"] == "awaiting_multiview"
                assert results[1]["geometry"]["status"] == "coarse_multiview"
                assert results[1]["head_feedback"]["positions_rad"] == {"head_z": 0, "head_y": -.5}
                done.set()
                thread.join(timeout=2)
                time.sleep(.4)
                stale = capture()
                assert not stale.success and "fresh" in stale.message
                report.update(status="passed", captured_frames=len(results),
                              synchronized_head_feedback=True, exact_time_tf=True,
                              two_view_geometry=results[1]["geometry"], stale_image_rejected=True,
                              isolated_master=uri, note="Synthetic ROS messages only; not hardware validation")
            except Exception as exc:
                logs.flush()
                logs.seek(0)
                report.update(status="failed", error=str(exc), process_output=logs.read()[-12000:])
            finally:
                done.set()
                if thread:
                    thread.join(timeout=2)
                for process in reversed(processes):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=2)
                if "rospy" in sys.modules:
                    sys.modules["rospy"].signal_shutdown("isolated test complete")
    write_json(LOG_ROOT / "ros1_smoke_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
