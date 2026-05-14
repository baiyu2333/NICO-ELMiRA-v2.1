#!/usr/bin/env python3
"""Read-only live joint-state recorder for NICO kinematic replay.

The recorder subscribes to joint-state topics and writes timestamped frames. It
does not publish, command motors, create trajectories, or move the robot.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kinematic_preview.live_joint_state_reader import (
    DEFAULT_MAPPING,
    DEFAULT_TOPIC_CANDIDATES,
    map_joint_angles,
    ros_master_available,
)
from kinematic_preview.nico_stick_model import NICOStickModel
from utils.kinematic_trial_logger import find_api_root, preview_paths, write_json


TRIAL_CSV_FIELDS = [
    "recording_id",
    "instruction",
    "target_object",
    "action_type",
    "start_time",
    "end_time",
    "duration_sec",
    "frame_count",
    "joint_state_available",
    "replay_available",
    "outcome",
    "notes",
    "real_robot_motion_commanded",
    "recording_json_path",
]


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def recording_id(label: str) -> str:
    safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label.strip())
    safe_label = safe_label.strip("_") or "joint_recording"
    return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{safe_label}_{uuid.uuid4().hex[:6]}"


def ros_time_from_msg(message: Any) -> Optional[float]:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    try:
        return float(stamp.to_sec())
    except Exception:
        return None


class JointTrajectoryRecorder:
    def __init__(
        self,
        label: str,
        instruction: str,
        target_object: str,
        action_type: str,
        arm_used: str = "right",
        pre_roll_sec: float = 3.0,
        topic_candidates: Optional[List[str]] = None,
    ) -> None:
        self.label = label
        self.instruction = instruction
        self.target_object = target_object
        self.action_type = action_type
        self.arm_used = arm_used
        self.pre_roll_sec = max(0.0, float(pre_roll_sec))
        self.topic_candidates = topic_candidates or list(DEFAULT_TOPIC_CANDIDATES)
        self.model = NICOStickModel()
        self.recording = False
        self.stop_requested = False
        self.recording_id = recording_id(label)
        self.timestamp_start = ""
        self.timestamp_end = ""
        self.wall_start = 0.0
        self.frames: List[Dict[str, Any]] = []
        self.raw_joint_names = set()
        self.pre_roll = deque()
        self.source_topics = set()

    def start_recording(self) -> None:
        self.recording = True
        self.timestamp_start = utc_now()
        self.wall_start = time.time()
        for frame in list(self.pre_roll):
            copied = dict(frame)
            copied["t_relative_sec"] = round(copied.get("wall_time", self.wall_start) - self.wall_start, 4)
            copied.pop("wall_time", None)
            self.frames.append(copied)

    def stop_recording(self) -> None:
        self.recording = False
        self.stop_requested = True
        self.timestamp_end = utc_now()

    def _build_frame(self, message: Any, topic: str) -> Dict[str, Any]:
        names = list(getattr(message, "name", []))
        positions = list(getattr(message, "position", []))
        raw_positions = {name: float(pos) for name, pos in zip(names, positions)}
        mapped = map_joint_angles(message, DEFAULT_MAPPING)
        mapped_angles = mapped.get("mapped_joint_angles_deg", {})
        preview = self.model.forward_kinematics(mapped_angles) if mapped_angles else {}
        self.raw_joint_names.update(names)
        self.source_topics.add(topic)
        wall_time = time.time()
        return {
            "wall_time": wall_time,
            "t_relative_sec": round(wall_time - self.wall_start, 4) if self.wall_start else 0.0,
            "source_topic": topic,
            "ros_timestamp": ros_time_from_msg(message),
            "raw_joint_positions": raw_positions,
            "mapped_right_arm_joint_angles": mapped_angles,
            "mapping_status": mapped.get("status", "unavailable"),
            "estimated_right_hand_xyz": preview.get("estimated_right_hand_xyz", {}),
            "estimated_right_wrist_xyz": preview.get("estimated_right_wrist_xyz", {}),
            "above_table_safety_limit": preview.get("above_table_safety_limit"),
            "height_warning": preview.get("height_warning", ""),
        }

    def joint_state_callback(self, message: Any, topic: str) -> None:
        frame = self._build_frame(message, topic)
        if self.recording:
            clean = dict(frame)
            clean.pop("wall_time", None)
            self.frames.append(clean)
        else:
            cutoff = time.time() - self.pre_roll_sec
            self.pre_roll.append(frame)
            while self.pre_roll and self.pre_roll[0].get("wall_time", 0.0) < cutoff:
                self.pre_roll.popleft()

    def record(self, duration_sec: Optional[float]) -> Dict[str, Any]:
        if not ros_master_available(timeout_s=1.0):
            self.timestamp_start = utc_now()
            self.timestamp_end = self.timestamp_start
            return self._build_record(
                joint_state_available=False,
                notes="Live joint recording unavailable: ROS master is not available.",
            )

        try:
            import rospy
            from sensor_msgs.msg import JointState
        except Exception as exc:
            self.timestamp_start = utc_now()
            self.timestamp_end = self.timestamp_start
            return self._build_record(
                joint_state_available=False,
                notes=f"Live joint recording unavailable: ROS Python modules unavailable: {exc}",
            )

        if not rospy.core.is_initialized():
            rospy.init_node("kinematic_joint_trajectory_recorder", anonymous=True, disable_signals=True)

        for topic in self.topic_candidates:
            try:
                rospy.Subscriber(
                    topic,
                    JointState,
                    self.joint_state_callback,
                    callback_args=topic,
                    queue_size=50,
                )
            except Exception:
                pass

        self.start_recording()
        deadline = time.time() + float(duration_sec) if duration_sec else None
        rate = rospy.Rate(30)
        while not self.stop_requested and not rospy.is_shutdown():
            if deadline is not None and time.time() >= deadline:
                break
            rate.sleep()
        self.stop_recording()
        return self._build_record(joint_state_available=bool(self.frames), notes="")

    def _build_record(self, joint_state_available: bool, notes: str) -> Dict[str, Any]:
        if not self.timestamp_start:
            self.timestamp_start = utc_now()
        if not self.timestamp_end:
            self.timestamp_end = utc_now()
        if not notes and not joint_state_available:
            notes = "Live joint recording unavailable: /joint_states not available or no frames received."
        duration = max(0.0, time.time() - self.wall_start) if self.wall_start else 0.0
        if self.frames:
            duration = max(frame.get("t_relative_sec", 0.0) for frame in self.frames)
        sample_rate = round(len(self.frames) / duration, 3) if duration > 0 and self.frames else 0.0
        return {
            "recording_id": self.recording_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "duration_sec": round(duration, 3),
            "label": self.label,
            "instruction": self.instruction,
            "target_object": self.target_object,
            "action_type": self.action_type,
            "arm_used": self.arm_used,
            "sample_rate_hz": sample_rate,
            "raw_joint_names": sorted(self.raw_joint_names),
            "source_topics": sorted(self.source_topics),
            "joint_state_frames": self.frames,
            "mapped_right_arm_frames": [
                {
                    "t_relative_sec": frame.get("t_relative_sec"),
                    "mapped_right_arm_joint_angles": frame.get("mapped_right_arm_joint_angles", {}),
                    "estimated_right_hand_xyz": frame.get("estimated_right_hand_xyz", {}),
                    "estimated_right_wrist_xyz": frame.get("estimated_right_wrist_xyz", {}),
                }
                for frame in self.frames
            ],
            "joint_state_available": joint_state_available,
            "real_robot_motion_commanded": False,
            "notes": notes
            or "Read-only joint-state recording. Real movement, if any, came from the existing ELMiRA pipeline.",
        }


def write_recording(record: Dict[str, Any]) -> Dict[str, Path]:
    api_root = find_api_root(__file__)
    paths = preview_paths(api_root)
    recordings_dir = paths["log_dir"] / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    json_path = recordings_dir / f"{record['recording_id']}.json"
    csv_path = recordings_dir / f"{record['recording_id']}.csv"
    latest_recording = paths["log_dir"] / "latest_recording.json"
    latest_summary = paths["log_dir"] / "latest_recorded_trial_summary.json"
    trials_csv = paths["log_dir"] / "recorded_motion_trials.csv"

    write_json(record, json_path)
    write_json(record, latest_recording)
    write_frame_csv(record, csv_path)
    summary = build_trial_summary(record, json_path)
    write_json(summary, latest_summary)
    append_trial_summary(summary, trials_csv)
    return {
        "recording_json": json_path,
        "recording_csv": csv_path,
        "latest_recording": latest_recording,
        "latest_summary": latest_summary,
        "recorded_motion_trials_csv": trials_csv,
    }


def write_frame_csv(record: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "recording_id",
        "frame_index",
        "t_relative_sec",
        "source_topic",
        "ros_timestamp",
        "mapped_right_arm_joint_angles",
        "estimated_right_hand_xyz",
        "estimated_right_wrist_xyz",
        "raw_joint_positions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, frame in enumerate(record.get("joint_state_frames", [])):
            writer.writerow(
                {
                    "recording_id": record.get("recording_id"),
                    "frame_index": idx,
                    "t_relative_sec": frame.get("t_relative_sec"),
                    "source_topic": frame.get("source_topic"),
                    "ros_timestamp": frame.get("ros_timestamp"),
                    "mapped_right_arm_joint_angles": json.dumps(
                        frame.get("mapped_right_arm_joint_angles", {}),
                        sort_keys=True,
                    ),
                    "estimated_right_hand_xyz": json.dumps(
                        frame.get("estimated_right_hand_xyz", {}),
                        sort_keys=True,
                    ),
                    "estimated_right_wrist_xyz": json.dumps(
                        frame.get("estimated_right_wrist_xyz", {}),
                        sort_keys=True,
                    ),
                    "raw_joint_positions": json.dumps(
                        frame.get("raw_joint_positions", {}),
                        sort_keys=True,
                    ),
                }
            )
    return path


def build_trial_summary(record: Dict[str, Any], json_path: Path) -> Dict[str, Any]:
    return {
        "recording_id": record.get("recording_id"),
        "instruction": record.get("instruction"),
        "target_object": record.get("target_object"),
        "action_type": record.get("action_type"),
        "start_time": record.get("timestamp_start"),
        "end_time": record.get("timestamp_end"),
        "duration_sec": record.get("duration_sec"),
        "frame_count": len(record.get("joint_state_frames", [])),
        "joint_state_available": record.get("joint_state_available", False),
        "replay_available": bool(record.get("joint_state_frames")),
        "outcome": "unknown",
        "notes": record.get("notes", ""),
        "real_robot_motion_commanded": False,
        "recording_json_path": str(json_path),
    }


def append_trial_summary(summary: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in TRIAL_CSV_FIELDS})
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Record NICO joint states for kinematic replay.")
    parser.add_argument("--record-seconds", type=float, default=None, help="Fixed recording duration")
    parser.add_argument("--label", default="right_arm_point_red_object_trial")
    parser.add_argument("--instruction", default="Point to the red object with your right hand.")
    parser.add_argument("--target-object", default="red object")
    parser.add_argument(
        "--action-type",
        default="point",
        choices=["point", "reach", "touch", "pre_grasp", "grasp_attempt"],
    )
    parser.add_argument("--pre-roll-sec", type=float, default=3.0)
    args = parser.parse_args()

    recorder = JointTrajectoryRecorder(
        label=args.label,
        instruction=args.instruction,
        target_object=args.target_object,
        action_type=args.action_type,
        pre_roll_sec=args.pre_roll_sec,
    )

    def _stop(_signum, _frame):
        recorder.stop_recording()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    record = recorder.record(args.record_seconds)
    paths = write_recording(record)
    output = {
        "status": "recorded" if record.get("joint_state_available") else "unavailable",
        "recording_id": record.get("recording_id"),
        "frame_count": len(record.get("joint_state_frames", [])),
        "recording_json_path": str(paths["recording_json"]),
        "recording_csv_path": str(paths["recording_csv"]),
        "latest_recording_path": str(paths["latest_recording"]),
        "trial_summary_path": str(paths["latest_summary"]),
        "real_robot_motion_commanded": False,
        "notes": record.get("notes", ""),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
