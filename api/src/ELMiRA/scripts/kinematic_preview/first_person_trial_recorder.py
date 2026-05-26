#!/usr/bin/env python3
"""Read-only first-person camera trial recorder for NICO actions.

The recorder subscribes to camera image topics and saves a short frame
sequence plus trial metadata. It never publishes ROS messages and never
commands robot motion.
"""

import argparse
import csv
import json
import os
import signal
import socket
import sys
import time
import xmlrpc.client
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.kinematic_trial_logger import find_api_root


LOG_DIR_NAME = "first_person_trials"
TRIAL_CSV_FIELDS = [
    "timestamp_start",
    "timestamp_end",
    "trial_id",
    "image_topic",
    "frame_rate",
    "duration_sec",
    "frame_count",
    "frames_dir",
    "action_type",
    "target_object",
    "x_m",
    "y_m",
    "z_m",
    "robot_touch_z_offset_m",
    "robot_touch_target_z_m",
    "trial_change_summary",
    "key_frame_index",
    "key_frame_path",
    "offset_direction",
    "offset_size",
    "contact_status",
    "outcome",
    "failure_reason",
    "notes",
    "label_source",
    "real_robot_motion_commanded_by_recorder",
    "status",
    "message",
]

STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def api_root() -> Path:
    return find_api_root(__file__)


def log_root() -> Path:
    root = api_root() / "logs" / LOG_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def latest_record_path() -> Path:
    return log_root() / "latest_trial_record.json"


def trials_csv_path() -> Path:
    return log_root() / "first_person_trials.csv"


def make_trial_id(value: str = "") -> str:
    value = (value or "").strip()
    if value:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return "T" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return str(value)


def write_json(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(record), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def ros_param_value(name: str, default: Any = "unknown", timeout_s: float = 0.5) -> Any:
    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        master = xmlrpc.client.ServerProxy(master_uri)
        code, _message, value = master.getParam("/first_person_trial_recorder_probe", name)
        return value if code == 1 else default
    except Exception:
        return default
    finally:
        socket.setdefaulttimeout(old_timeout)


def as_float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def robot_touch_tuning_fields(action_type: str) -> Dict[str, Any]:
    if str(action_type or "").lower() != "touch":
        return {
            "robot_touch_z_offset_m": "not_applicable",
            "robot_touch_target_z_m": "not_applicable",
            "trial_change_summary": "No touch-specific robot Z tuning was applied for this trial.",
        }

    table_z_base = as_float_or_none(ros_param_value("/elmira/table_z_base"))
    global_z_offset = as_float_or_none(ros_param_value("/elmira/offset_z"))
    touch_z_offset = as_float_or_none(ros_param_value("/elmira/right_touch_z_offset"))
    if table_z_base is None or global_z_offset is None or touch_z_offset is None:
        return {
            "robot_touch_z_offset_m": "unknown",
            "robot_touch_target_z_m": "unknown",
            "trial_change_summary": "Touch Z tuning was unavailable when this camera-only trial was logged.",
        }

    target_z = table_z_base + global_z_offset + touch_z_offset
    return {
        "robot_touch_z_offset_m": round(touch_z_offset, 4),
        "robot_touch_target_z_m": round(target_z, 4),
        "trial_change_summary": (
            f"Touch trial used right_touch_z_offset={touch_z_offset:.3f} m, "
            f"giving robot-frame target_z={target_z:.3f} m."
        ),
    }


def ensure_trial_change_summary(record: Dict[str, Any]) -> None:
    if not record.get("trial_change_summary"):
        record.update(robot_touch_tuning_fields(str(record.get("action_type", ""))))


def upsert_trial_csv(record: Dict[str, Any]) -> None:
    path = trials_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]

    row = {field: csv_value(record.get(field, "")) for field in TRIAL_CSV_FIELDS}
    trial_id = str(record.get("trial_id", ""))
    replaced = False
    for index, existing in enumerate(rows):
        if existing.get("trial_id") == trial_id:
            rows[index] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_trial_record(record: Dict[str, Any]) -> Dict[str, Path]:
    ensure_trial_change_summary(record)
    trial_id = make_trial_id(record.get("trial_id", ""))
    record["trial_id"] = trial_id
    trial_dir = log_root() / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    record_path = trial_dir / "trial_record.json"
    record["trial_record_path"] = str(record_path)
    write_json(record, record_path)
    write_json(record, latest_record_path())
    upsert_trial_csv(record)
    return {
        "trial_dir": trial_dir,
        "record_path": record_path,
        "latest_record_path": latest_record_path(),
        "csv_path": trials_csv_path(),
    }


def ros_master_available(timeout_s: float = 1.0) -> bool:
    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        master = xmlrpc.client.ServerProxy(master_uri)
        code, _message, _pid = master.getPid("/first_person_trial_recorder_probe")
        return code == 1
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(old_timeout)


def list_image_topics(timeout_s: float = 2.0) -> Dict[str, Any]:
    if not ros_master_available(timeout_s=1.0):
        return {
            "status": "unavailable",
            "message": "ROS master unavailable; no image topics listed.",
            "topics": [],
            "real_robot_motion_commanded_by_recorder": False,
        }
    try:
        import rospy
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"ROS Python modules unavailable: {exc}",
            "topics": [],
            "real_robot_motion_commanded_by_recorder": False,
        }
    try:
        if not rospy.core.is_initialized():
            rospy.init_node("first_person_trial_topic_lister", anonymous=True, disable_signals=True)
        deadline = time.time() + timeout_s
        topics: List[Tuple[str, str]] = []
        while time.time() < deadline:
            topics = rospy.get_published_topics()
            if topics:
                break
            time.sleep(0.1)
        image_topics = [
            {"topic": topic, "type": msg_type}
            for topic, msg_type in topics
            if msg_type in ("sensor_msgs/Image", "sensor_msgs/CompressedImage")
        ]
        return {
            "status": "available" if image_topics else "unavailable",
            "message": "Image topics listed." if image_topics else "No sensor_msgs image topics found.",
            "topics": image_topics,
            "real_robot_motion_commanded_by_recorder": False,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"Image topic listing failed: {exc}",
            "topics": [],
            "real_robot_motion_commanded_by_recorder": False,
        }


def import_image_dependencies() -> Dict[str, Any]:
    deps: Dict[str, Any] = {}
    try:
        import rospy
        from sensor_msgs.msg import CompressedImage, Image

        deps["rospy"] = rospy
        deps["Image"] = Image
        deps["CompressedImage"] = CompressedImage
    except Exception as exc:
        deps["ros_error"] = exc
        return deps

    try:
        import cv2

        deps["cv2"] = cv2
    except Exception as exc:
        deps["cv2_error"] = exc

    try:
        import numpy as np

        deps["np"] = np
    except Exception as exc:
        deps["np_error"] = exc

    try:
        from cv_bridge import CvBridge

        deps["bridge"] = CvBridge()
    except Exception as exc:
        deps["bridge_error"] = exc
    return deps


def convert_image_message(msg: Any, deps: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    cv2 = deps.get("cv2")
    np = deps.get("np")
    compressed_cls = deps.get("CompressedImage")
    image_cls = deps.get("Image")
    bridge = deps.get("bridge")

    if compressed_cls is not None and isinstance(msg, compressed_cls):
        if cv2 is None or np is None:
            return None, "OpenCV/numpy unavailable for CompressedImage conversion."
        array = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            return None, "CompressedImage decode failed."
        return image, ""

    if image_cls is not None and isinstance(msg, image_cls):
        if bridge is not None:
            try:
                return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), ""
            except Exception as exc:
                return None, f"cv_bridge conversion failed: {exc}"
        return None, "cv_bridge unavailable for sensor_msgs/Image conversion."

    return None, "Unsupported image message type."


def wait_for_camera_frame(topic: str, deps: Dict[str, Any], timeout_s: float) -> Tuple[Optional[Any], str]:
    rospy = deps.get("rospy")
    image_cls = deps.get("Image")
    compressed_cls = deps.get("CompressedImage")
    if rospy is None:
        return None, f"ROS Python modules unavailable: {deps.get('ros_error')}"

    message_types = [compressed_cls] if topic.endswith("/compressed") else [image_cls, compressed_cls]
    errors = []
    for msg_type in message_types:
        if msg_type is None:
            continue
        try:
            msg = rospy.wait_for_message(topic, msg_type, timeout=timeout_s)
            return convert_image_message(msg, deps)
        except Exception as exc:
            errors.append(str(exc))
    return None, "; ".join(errors[-2:]) or "No image frame received."


def build_record(
    trial_id: str,
    topic: str,
    fps: float,
    action_type: str,
    target_object: str,
    x_m: float,
    y_m: float,
    z_m: float,
    duration_sec: Optional[float],
) -> Dict[str, Any]:
    trial_id = make_trial_id(trial_id)
    frames_dir = log_root() / trial_id / "frames"
    return {
        "timestamp_start": utc_now(),
        "timestamp_end": "unknown",
        "trial_id": trial_id,
        "image_topic": topic,
        "frame_rate": fps,
        "duration_sec": duration_sec if duration_sec is not None else "manual_stop",
        "frame_count": 0,
        "frames_dir": str(frames_dir),
        "action_type": action_type,
        "target_object": target_object,
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        **robot_touch_tuning_fields(action_type),
        "key_frame_index": "unknown",
        "key_frame_path": "unknown",
        "offset_direction": "unknown",
        "offset_size": "unknown",
        "contact_status": "unknown",
        "outcome": "unknown",
        "failure_reason": "unknown",
        "notes": "",
        "label_source": "manual_human_annotation_pending",
        "real_robot_motion_commanded_by_recorder": False,
        "status": "recording",
        "message": "Recording camera frames only; no robot motion commanded.",
    }


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def record_trial(
    topic: str,
    trial_id: str,
    action_type: str,
    target_object: str,
    x_m: float,
    y_m: float,
    z_m: float,
    fps: float,
    record_seconds: Optional[float],
) -> Dict[str, Any]:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    fps = max(0.2, float(fps or 5.0))
    interval = 1.0 / fps
    record = build_record(trial_id, topic, fps, action_type, target_object, x_m, y_m, z_m, record_seconds)
    frames_dir = Path(record["frames_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not ros_master_available(timeout_s=1.0):
        record.update(
            {
                "timestamp_end": utc_now(),
                "duration_sec": 0,
                "status": "unavailable",
                "message": "ROS master unavailable; first-person recording not started.",
            }
        )
        write_trial_record(record)
        return record

    deps = import_image_dependencies()
    if deps.get("ros_error"):
        record.update(
            {
                "timestamp_end": utc_now(),
                "duration_sec": 0,
                "status": "unavailable",
                "message": f"ROS Python modules unavailable: {deps.get('ros_error')}",
            }
        )
        write_trial_record(record)
        return record
    if deps.get("cv2") is None:
        record.update(
            {
                "timestamp_end": utc_now(),
                "duration_sec": 0,
                "status": "unavailable",
                "message": f"OpenCV unavailable for saving frames: {deps.get('cv2_error')}",
            }
        )
        write_trial_record(record)
        return record

    rospy = deps["rospy"]
    if not rospy.core.is_initialized():
        rospy.init_node("first_person_trial_recorder", anonymous=True, disable_signals=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    start_time = time.time()
    frame_paths: List[str] = []
    last_error = ""
    while not STOP_REQUESTED:
        elapsed = time.time() - start_time
        if record_seconds is not None and elapsed >= record_seconds:
            break
        frame_start = time.time()
        frame, error = wait_for_camera_frame(topic, deps, timeout_s=min(1.0, max(0.2, interval)))
        if frame is not None:
            frame_path = frames_dir / f"frame_{len(frame_paths):03d}.jpg"
            ok = deps["cv2"].imwrite(str(frame_path), frame)
            if ok:
                frame_paths.append(str(frame_path))
            else:
                last_error = f"Failed to write {frame_path}"
        else:
            last_error = error
        sleep_time = interval - (time.time() - frame_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    end_time = time.time()
    record.update(
        {
            "timestamp_end": utc_now(),
            "duration_sec": round(end_time - start_time, 3),
            "frame_count": len(frame_paths),
            "frame_paths": frame_paths,
            "status": "recorded" if frame_paths else "unavailable",
            "message": (
                f"Recorded {len(frame_paths)} first-person frames; no robot motion commanded."
                if frame_paths
                else f"No frames recorded from {topic}. {last_error}".strip()
            ),
        }
    )
    write_trial_record(record)
    return record


def resolve_record_path(path_value: str = "") -> Path:
    value = (path_value or "").strip()
    if not value:
        return latest_record_path()
    path = Path(value).expanduser()
    return path if path.is_absolute() else api_root() / path


def update_latest_if_same(record: Dict[str, Any], path: Path) -> None:
    latest = latest_record_path()
    if latest.exists():
        latest_data = read_json(latest)
        if latest_data.get("trial_id") == record.get("trial_id"):
            write_json(record, latest)
    elif path.name == latest.name:
        write_json(record, latest)


def save_key_frame(record_path: Path, frame_index: int) -> Dict[str, Any]:
    record = read_json(record_path)
    frames = record.get("frame_paths")
    if not isinstance(frames, list) or not frames:
        frames_dir = Path(str(record.get("frames_dir", "")))
        frames = [str(path) for path in sorted(frames_dir.glob("frame_*.jpg"))] if frames_dir.exists() else []
    if not frames:
        record["status"] = "unavailable"
        record["message"] = "No trial frames available for key-frame selection."
        write_trial_record(record)
        return record
    index = max(0, min(int(frame_index or 0), len(frames) - 1))
    record["key_frame_index"] = index
    record["key_frame_path"] = frames[index]
    record["label_source"] = "manual_human_annotation_started"
    record["message"] = f"Saved key frame {index}."
    write_trial_record(record)
    update_latest_if_same(record, record_path)
    return record


def save_manual_label(
    record_path: Path,
    offset_direction: str,
    offset_size: str,
    contact_status: str,
    outcome: str,
    failure_reason: str,
    notes: str,
) -> Dict[str, Any]:
    record = read_json(record_path)
    record.update(
        {
            "offset_direction": offset_direction or "unknown",
            "offset_size": offset_size or "unknown",
            "contact_status": contact_status or "unknown",
            "outcome": outcome or "unknown",
            "failure_reason": failure_reason or "unknown",
            "notes": notes or "",
            "label_source": "manual_human_annotation",
            "message": "Manual trial label saved.",
        }
    )
    write_trial_record(record)
    update_latest_if_same(record, record_path)
    return record


def print_json(record: Dict[str, Any]) -> None:
    print(json.dumps(json_safe(record), indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only first-person trial recorder.")
    parser.add_argument("--list-image-topics", action="store_true", help="List available sensor_msgs image topics")
    parser.add_argument("--record-seconds", type=float, default=None, help="Record for a fixed duration")
    parser.add_argument("--topic", default="/nico/vision/left", help="Image or compressed image topic")
    parser.add_argument("--trial-id", default="", help="Trial ID; autogenerated if empty")
    parser.add_argument("--action-type", default="point", choices=["point", "reach", "touch", "grasp_attempt"])
    parser.add_argument("--target-object", default="red object")
    parser.add_argument("--x", type=float, default=0.20)
    parser.add_argument("--y", type=float, default=-0.08)
    parser.add_argument("--z", type=float, default=0.14)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--save-key-frame", action="store_true")
    parser.add_argument("--save-manual-label", action="store_true")
    parser.add_argument("--trial-record", default="")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--offset-direction", default="unknown")
    parser.add_argument("--offset-size", default="unknown")
    parser.add_argument("--contact-status", default="unknown")
    parser.add_argument("--outcome", default="unknown")
    parser.add_argument("--failure-reason", default="unknown")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if args.list_image_topics:
        print_json(list_image_topics())
        return 0
    if args.save_key_frame:
        print_json(save_key_frame(resolve_record_path(args.trial_record), args.frame_index))
        return 0
    if args.save_manual_label:
        print_json(
            save_manual_label(
                resolve_record_path(args.trial_record),
                args.offset_direction,
                args.offset_size,
                args.contact_status,
                args.outcome,
                args.failure_reason,
                args.notes,
            )
        )
        return 0

    print_json(
        record_trial(
            topic=args.topic,
            trial_id=args.trial_id,
            action_type=args.action_type,
            target_object=args.target_object,
            x_m=args.x,
            y_m=args.y,
            z_m=args.z,
            fps=args.fps,
            record_seconds=args.record_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
