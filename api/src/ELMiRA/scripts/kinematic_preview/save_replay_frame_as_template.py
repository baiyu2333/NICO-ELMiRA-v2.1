#!/usr/bin/env python3
"""Save one replay frame as a reusable right-arm action template.

The saved template is for future preview/planning work only. It is not sent to
the real robot and does not create a trajectory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kinematic_preview.nico_stick_model import NICOStickModel
from kinematic_preview.replay_joint_trajectory import (
    clamp_frame_index,
    frame_joint_angles,
    get_frame,
    get_frame_count,
    load_recording,
)
from utils.kinematic_trial_logger import write_json


CAPTURED_TEMPLATE_PATH = SCRIPT_DIR / "templates" / "captured_right_arm_templates.json"
COMMANDABLE_RIGHT_ARM_JOINTS = [
    "r_shoulder_z",
    "r_shoulder_y",
    "r_arm_x",
    "r_elbow_y",
    "r_wrist_z",
    "r_wrist_x",
]


def load_captured_templates() -> Dict[str, Any]:
    if not CAPTURED_TEMPLATE_PATH.exists():
        return {
            "note": "Captured preview templates. Do not send directly to real motors.",
            "templates": {},
        }
    with CAPTURED_TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        data = {}
    data.setdefault("note", "Captured preview templates. Do not send directly to real motors.")
    data.setdefault("templates", {})
    if not isinstance(data["templates"], dict):
        data["templates"] = {}
    return data


def save_frame_as_template(
    recording_path: str,
    frame_index: int,
    template_name: str,
    action_type: str,
    target_object: str,
    hand_state: str = "neutral",
) -> Dict[str, Any]:
    recording = load_recording(recording_path)
    count = get_frame_count(recording)
    if count <= 0:
        raise ValueError("Recording has no joint-state frames to save as a template.")

    index = clamp_frame_index(recording, frame_index)
    frame = get_frame(recording, index)
    joint_angles_deg = frame_joint_angles(frame)
    preview = NICOStickModel().forward_kinematics(joint_angles_deg) if joint_angles_deg else {}
    raw_positions = frame.get("raw_joint_positions") or {}
    commandable_positions = {
        joint_name: float(raw_positions[joint_name])
        for joint_name in COMMANDABLE_RIGHT_ARM_JOINTS
        if joint_name in raw_positions
    }

    name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in template_name.strip())
    name = name.strip("_")
    if not name:
        raise ValueError("Template name is required.")

    template = {
        "name": name,
        "description": (
            f"Captured from replay frame {index} of recording "
            f"{recording.get('recording_id', 'unknown')}."
        ),
        "source": "replay_frame_from_real_robot_joint_recording",
        "recording_id": recording.get("recording_id"),
        "recording_path": recording.get("_recording_path"),
        "frame_index": index,
        "action_type": action_type,
        "arm_used": "right",
        "joint_angles_deg": joint_angles_deg,
        "commandable_right_arm_joint_positions_rad": commandable_positions,
        "direct_execution_ready": all(
            joint_name in commandable_positions for joint_name in COMMANDABLE_RIGHT_ARM_JOINTS
        ),
        "estimated_right_hand_xyz": preview.get(
            "estimated_right_hand_xyz", frame.get("estimated_right_hand_xyz", {})
        ),
        "estimated_right_wrist_xyz": preview.get(
            "estimated_right_wrist_xyz", frame.get("estimated_right_wrist_xyz", {})
        ),
        "hand_state": hand_state,
        "target_object": target_object or recording.get("target_object", "target object"),
        "safety_note": "Captured from replay. Not automatically sent to robot.",
        "real_robot_motion_commanded": False,
    }

    data = load_captured_templates()
    data["templates"][name] = template
    write_json(data, CAPTURED_TEMPLATE_PATH)

    return {
        "status": "saved",
        "template_name": name,
        "template_path": str(CAPTURED_TEMPLATE_PATH),
        "frame_index": index,
        "recording_id": recording.get("recording_id"),
        "estimated_right_hand_xyz": template["estimated_right_hand_xyz"],
        "real_robot_motion_commanded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Save a replay frame as a preview template.")
    parser.add_argument("--recording", required=True, help="Recording JSON path")
    parser.add_argument("--frame", type=int, required=True, help="Frame index to save")
    parser.add_argument("--template-name", required=True, help="New template name")
    parser.add_argument(
        "--action-type",
        default="point",
        choices=["point", "reach", "touch", "pre_grasp", "grasp_attempt", "lift"],
    )
    parser.add_argument("--target-object", default="red object")
    parser.add_argument("--hand-state", default="neutral", choices=["open", "close", "neutral"])
    args = parser.parse_args()

    try:
        summary = save_frame_as_template(
            args.recording,
            args.frame,
            args.template_name,
            args.action_type,
            args.target_object,
            args.hand_state,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "message": str(exc),
                    "real_robot_motion_commanded": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
