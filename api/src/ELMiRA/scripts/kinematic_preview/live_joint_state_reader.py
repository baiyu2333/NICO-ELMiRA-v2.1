#!/usr/bin/env python3
"""Read NICO joint states and map them to the simplified preview model.

This script is read-only. It subscribes to joint-state topics when ROS is
available and never publishes or commands robot motion.
"""

import argparse
import json
import math
import os
import signal
import socket
import sys
import xmlrpc.client
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kinematic_preview.nico_stick_model import NICOStickModel
from kinematic_preview.render_stick_nico import render_preview
from utils.kinematic_trial_logger import find_api_root, preview_paths, write_json


class HardTimeout(Exception):
    pass


DEFAULT_TOPIC_CANDIDATES = [
    "/joint_states",
    "/right/open_manipulator_p/joint_states",
    "/NICOL/joint_states",
]

DEFAULT_MAPPING = {
    "right_shoulder_pitch": [
        {"source": "right_shoulder_pitch", "scale": 1.0, "offset": 0.0, "abs": False},
        {"source": "r_arm_x", "scale": 0.45, "offset": 0.0, "abs": True},
        {"source": "r_shoulder_y", "scale": 1.0, "offset": 0.0, "abs": True},
    ],
    "right_shoulder_roll": [
        {"source": "right_shoulder_roll", "scale": 1.0, "offset": 0.0, "abs": False},
        {"source": "r_shoulder_z", "scale": -0.4, "offset": 0.0, "abs": False},
    ],
    "right_elbow_pitch": [
        {"source": "right_elbow_pitch", "scale": 1.0, "offset": 0.0, "abs": False},
        {"source": "r_elbow_y", "scale": 0.45, "offset": 0.0, "abs": True},
    ],
    "right_wrist_pitch": [
        {"source": "right_wrist_pitch", "scale": 1.0, "offset": 0.0, "abs": False},
        {"source": "r_wrist_x", "scale": 0.2, "offset": 0.0, "abs": False},
    ],
}


def _to_degrees(value: float) -> float:
    value = float(value)
    if abs(value) <= 6.5:
        return math.degrees(value)
    return value


def load_mapping(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_MAPPING
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else DEFAULT_MAPPING


def ros_master_available(timeout_s: float = 1.0) -> bool:
    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        master = xmlrpc.client.ServerProxy(master_uri)
        code, _message, _pid = master.getPid("/kinematic_live_joint_state_reader_probe")
        return code == 1
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(old_timeout)


def map_joint_angles(message: Any, mapping: Dict[str, Any]) -> Dict[str, Any]:
    names = list(getattr(message, "name", []))
    positions = list(getattr(message, "position", []))
    values = {name: _to_degrees(pos) for name, pos in zip(names, positions)}
    mapped: Dict[str, float] = {}
    source_names: Dict[str, str] = {}
    missing = []

    for simplified, candidates in mapping.items():
        selected = None
        for item in candidates:
            if isinstance(item, str):
                item = {"source": item, "scale": 1.0, "offset": 0.0, "abs": False}
            source = item.get("source")
            if source in values:
                raw = values[source]
                if item.get("abs"):
                    raw = abs(raw)
                mapped[simplified] = round(raw * float(item.get("scale", 1.0)) + float(item.get("offset", 0.0)), 3)
                source_names[simplified] = source
                selected = source
                break
        if selected is None:
            missing.append(simplified)

    status = "live" if not missing else ("partial" if mapped else "unavailable")
    return {
        "status": status,
        "mapped_joint_angles_deg": mapped,
        "source_joint_names": source_names,
        "missing_simplified_joints": missing,
        "raw_joint_count": len(names),
    }


def read_once(topic_candidates: Iterable[str], timeout_s: float, mapping: Dict[str, Any]) -> Dict[str, Any]:
    if not ros_master_available(timeout_s=1.0):
        return {
            "status": "unavailable",
            "message": "ROS master unavailable; cannot read /joint_states.",
            "real_robot_motion_commanded": False,
        }

    try:
        import rospy
        from sensor_msgs.msg import JointState
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"ROS Python modules unavailable: {exc}",
            "real_robot_motion_commanded": False,
        }

    if not rospy.core.is_initialized():
        rospy.init_node("kinematic_live_joint_state_reader", anonymous=True, disable_signals=True)

    errors = []
    for topic in topic_candidates:
        try:
            msg = rospy.wait_for_message(topic, JointState, timeout=timeout_s)
            mapped = map_joint_angles(msg, mapping)
            mapped.update(
                {
                    "source_topic": topic,
                    "latest_update_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "real_robot_motion_commanded": False,
                    "message": "Read joint states only; no robot motion commanded.",
                }
            )
            return mapped
        except Exception as exc:
            errors.append(f"{topic}: {exc}")

    return {
        "status": "timeout",
        "message": "No joint-state message received before timeout.",
        "errors": errors[-3:],
        "real_robot_motion_commanded": False,
    }


def maybe_render_live_pose(record: Dict[str, Any]) -> Dict[str, Any]:
    angles = record.get("mapped_joint_angles_deg") or {}
    if not angles:
        return record
    api_root = find_api_root(__file__)
    paths = preview_paths(api_root)
    live_image = paths["log_dir"] / "latest_live_preview.png"
    live_json = paths["log_dir"] / "latest_live_pose.json"
    model = NICOStickModel()
    preview = model.forward_kinematics(angles)
    render_preview(
        preview,
        live_image,
        target_object="live pose",
        target_object_xyz=None,
        template_name="live_joint_state",
    )
    record["estimated_right_hand_xyz"] = preview.get("estimated_right_hand_xyz", {})
    record["estimated_right_wrist_xyz"] = preview.get("estimated_right_wrist_xyz", {})
    record["preview_image_path"] = str(live_image)
    record["preview_json_path"] = str(live_json)
    record["above_table_safety_limit"] = preview.get("above_table_safety_limit")
    record["height_warning"] = preview.get("height_warning", "")
    write_json(record, live_json)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Read live right-arm joint states for preview.")
    parser.add_argument("--once", action="store_true", help="Read one joint-state message and exit")
    parser.add_argument("--topic", action="append", help="JointState topic to try; may be repeated")
    parser.add_argument("--timeout", type=float, default=3.0, help="Timeout per topic in seconds")
    parser.add_argument("--mapping-json", default=None, help="Optional JSON mapping file")
    parser.add_argument("--render", action="store_true", help="Render latest_live_preview.png when data is available")
    args = parser.parse_args()

    if not args.once:
        parser.error("Only --once mode is currently implemented for safe dashboard use")

    topics = args.topic or DEFAULT_TOPIC_CANDIDATES
    hard_timeout_s = max(2, int(args.timeout * len(topics) + 3))

    def _hard_timeout(_signum, _frame):
        raise HardTimeout()

    signal.signal(signal.SIGALRM, _hard_timeout)
    signal.alarm(hard_timeout_s)
    try:
        record = read_once(topics, args.timeout, load_mapping(args.mapping_json))
        if args.render:
            record = maybe_render_live_pose(record)
    except HardTimeout:
        record = {
            "status": "timeout",
            "message": f"Live joint-state read exceeded {hard_timeout_s}s hard timeout.",
            "real_robot_motion_commanded": False,
        }
    except Exception as exc:
        record = {
            "status": "unavailable",
            "message": f"Live joint-state read failed: {exc}",
            "real_robot_motion_commanded": False,
        }
    finally:
        signal.alarm(0)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
