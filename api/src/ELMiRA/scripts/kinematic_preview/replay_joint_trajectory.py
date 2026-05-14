#!/usr/bin/env python3
"""Replay recorded NICO joint-state frames in the stick-figure preview.

This module is read-only. It loads recorded joint states, renders selected
frames, and never publishes or commands robot motion.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kinematic_preview.nico_stick_model import NICOStickModel
from kinematic_preview.preview_config import (
    DEFAULT_OBJECT_CENTER_Z_M,
    DEFAULT_TARGET_Y_M,
    RED_OBJECT_DISTANCE_FROM_NICO_M,
)
from kinematic_preview.render_stick_nico import render_preview
from utils.kinematic_trial_logger import find_api_root, preview_paths, write_json


def resolve_recording_path(path_value: str) -> Path:
    api_root = find_api_root(__file__)
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return api_root / path


def load_recording(path_value: str) -> Dict[str, Any]:
    path = resolve_recording_path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Recording is not a JSON object: {path}")
    data["_recording_path"] = str(path)
    return data


def get_frames(recording: Dict[str, Any]):
    frames = recording.get("joint_state_frames") or []
    return frames if isinstance(frames, list) else []


def get_frame_count(recording: Dict[str, Any]) -> int:
    return len(get_frames(recording))


def clamp_frame_index(recording: Dict[str, Any], index: int) -> int:
    count = get_frame_count(recording)
    if count <= 0:
        return 0
    return max(0, min(int(index), count - 1))


def get_frame(recording: Dict[str, Any], index: int) -> Dict[str, Any]:
    frames = get_frames(recording)
    if not frames:
        raise IndexError("Recording has no joint-state frames to replay.")
    return frames[clamp_frame_index(recording, index)]


def frame_joint_angles(frame: Dict[str, Any]) -> Dict[str, float]:
    angles = frame.get("mapped_right_arm_joint_angles") or frame.get("mapped_joint_angles_deg") or {}
    if not isinstance(angles, dict):
        return {}
    return {str(key): float(value) for key, value in angles.items()}


def default_target_xyz() -> Dict[str, float]:
    return {
        "x": RED_OBJECT_DISTANCE_FROM_NICO_M,
        "y": DEFAULT_TARGET_Y_M,
        "z": DEFAULT_OBJECT_CENTER_Z_M,
    }


def render_frame(recording_path: str, frame_index: int) -> Dict[str, Any]:
    api_root = find_api_root(__file__)
    paths = preview_paths(api_root)
    replay_dir = paths["log_dir"] / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    output_image = replay_dir / "latest_replay_frame.png"
    output_summary = replay_dir / "latest_replay_summary.json"

    recording = load_recording(recording_path)
    count = get_frame_count(recording)
    if count <= 0:
        summary = {
            "status": "unavailable",
            "message": "Recording has no joint-state frames to replay.",
            "recording_id": recording.get("recording_id"),
            "recording_path": recording.get("_recording_path"),
            "frame_count": 0,
            "real_robot_motion_commanded": False,
        }
        write_json(summary, output_summary)
        return summary

    index = clamp_frame_index(recording, frame_index)
    frame = get_frame(recording, index)
    angles = frame_joint_angles(frame)
    model = NICOStickModel()
    preview = model.forward_kinematics(angles) if angles else {}

    image_path: Optional[Path] = None
    if preview.get("coordinates"):
        image_path = render_preview(
            preview,
            output_image,
            target_object=recording.get("target_object", "target object"),
            target_object_xyz=recording.get("target_object_xyz") or default_target_xyz(),
            template_name=f"replay frame {index}",
        )

    summary = {
        "status": "rendered" if image_path else "partial",
        "recording_id": recording.get("recording_id"),
        "recording_path": recording.get("_recording_path"),
        "frame_index": index,
        "frame_count": count,
        "t_relative_sec": frame.get("t_relative_sec"),
        "ros_timestamp": frame.get("ros_timestamp"),
        "mapped_right_arm_joint_angles": angles,
        "estimated_right_hand_xyz": preview.get(
            "estimated_right_hand_xyz", frame.get("estimated_right_hand_xyz", {})
        ),
        "estimated_right_wrist_xyz": preview.get(
            "estimated_right_wrist_xyz", frame.get("estimated_right_wrist_xyz", {})
        ),
        "above_table_safety_limit": preview.get(
            "above_table_safety_limit", frame.get("above_table_safety_limit")
        ),
        "height_warning": preview.get("height_warning", frame.get("height_warning", "")),
        "preview_image_path": str(image_path) if image_path else "",
        "summary_json_path": str(output_summary),
        "real_robot_motion_commanded": False,
    }
    write_json(summary, output_summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded NICO joint-state frame.")
    parser.add_argument("--recording", required=True, help="Recording JSON path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to render")
    parser.add_argument("--export-gif", action="store_true", help="Reserved for future use")
    args = parser.parse_args()

    summary = render_frame(args.recording, args.frame)
    if args.export_gif:
        summary["gif_status"] = "not implemented; frame replay image generated instead"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
