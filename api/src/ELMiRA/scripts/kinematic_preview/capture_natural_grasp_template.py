#!/usr/bin/env python3
"""Capture the current NICO right-arm pose as a natural grasp template.

This module is read-only. It subscribes to joint-state topics, estimates a
coarse stick-model pose, and writes template/log files. It never publishes motor
commands or calls motion services.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kinematic_preview.live_joint_state_reader import (
    DEFAULT_MAPPING,
    map_joint_angles,
    ros_master_available,
)
from kinematic_preview.nico_stick_model import NICOStickModel
from kinematic_preview.render_stick_nico import render_preview
from utils.kinematic_trial_logger import find_api_root, write_json


API_ROOT = find_api_root(__file__)
LOG_DIR = API_ROOT / "logs" / "grasp_templates"
LATEST_CAPTURE_JSON = LOG_DIR / "latest_captured_template.json"
LATEST_XL320_COMMAND_JSON = LOG_DIR / "latest_xl320_command.json"
LATEST_PREVIEW_IMAGE = LOG_DIR / "latest_captured_template_preview.png"
LATEST_SEED_PLAN_JSON = LOG_DIR / "latest_captured_template_seed_plan.json"
CAPTURED_TEMPLATES_CSV = LOG_DIR / "captured_grasp_templates.csv"
CAPTURED_TEMPLATE_PATH = SCRIPT_DIR / "templates" / "captured_grasp_templates.json"

RIGHT_JOINT_TOPIC = "/right/open_manipulator_p/joint_states"
GENERIC_JOINT_TOPIC = "/joint_states"
RIGHT_ARM_JOINTS = [
    "r_shoulder_z",
    "r_shoulder_y",
    "r_arm_x",
    "r_elbow_y",
    "r_wrist_z",
    "r_wrist_x",
]
RIGHT_BASE_JOINTS = ["r_shoulder_z", "r_shoulder_y", "r_arm_x", "r_elbow_y"]
XL320_WRIST_IDS = [31, 33]
XL320_FINGER_IDS = [34, 35, 36, 37]

CSV_FIELDS = [
    "timestamp",
    "template_name",
    "status",
    "action_stage",
    "target_object",
    "right_arm_joint_names",
    "right_arm_joint_positions_rad",
    "mapped_joint_angles_deg",
    "xl320_wrist_state",
    "xl320_finger_state",
    "latest_xl320_command_state",
    "commanded_wrist_z_deg",
    "commanded_wrist_x_deg",
    "commanded_finger_state",
    "estimated_right_hand_xyz",
    "estimated_right_wrist_xyz",
    "direct_execution_ready",
    "ik_seed_ready",
    "real_robot_motion_commanded",
    "notes",
    "json_path",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if value is None:
        return ""
    return str(value)


def append_csv(record: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        row = {field: _csv_value(record.get(field, "")) for field in CSV_FIELDS}
        row["json_path"] = str(LATEST_CAPTURE_JSON)
        writer.writerow(row)
    return path


def safe_template_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip()).strip("_")
    if not name:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"captured_natural_grasp_{timestamp}"
    return name


def load_templates() -> Dict[str, Any]:
    if not CAPTURED_TEMPLATE_PATH.exists():
        return {
            "note": "Captured natural grasp templates. Read-only capture; do not send directly to motors.",
            "templates": {},
        }
    try:
        data = json.loads(CAPTURED_TEMPLATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault(
        "note",
        "Captured natural grasp templates. Read-only capture; do not send directly to motors.",
    )
    data.setdefault("templates", {})
    if not isinstance(data["templates"], dict):
        data["templates"] = {}
    return data


def save_template(template: Dict[str, Any], allow_overwrite: bool) -> Tuple[bool, str]:
    data = load_templates()
    name = template["name"]
    if name in data["templates"] and not allow_overwrite:
        return False, f"Template '{name}' already exists. Enable overwrite to replace it."
    data["templates"][name] = template
    write_json(data, CAPTURED_TEMPLATE_PATH)
    return True, ""


def _read_joint_state(topic: str, timeout_s: float) -> Dict[str, Any]:
    try:
        import rospy
        from sensor_msgs.msg import JointState

        msg = rospy.wait_for_message(topic, JointState, timeout=timeout_s)
        names = list(msg.name)
        positions = [float(value) for value in msg.position]
        return {
            "status": "success",
            "topic": topic,
            "message": "",
            "joint_names": names,
            "joint_positions": positions,
            "joint_position_map": {name: pos for name, pos in zip(names, positions)},
            "ros_timestamp": getattr(getattr(msg, "header", None), "stamp", None).to_sec()
            if getattr(getattr(msg, "header", None), "stamp", None)
            else None,
            "_message": msg,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "topic": topic,
            "message": str(exc),
            "joint_names": [],
            "joint_positions": [],
            "joint_position_map": {},
        }


def read_current_joint_states(timeout_s: float) -> Dict[str, Any]:
    if not ros_master_available(timeout_s=1.0):
        return {
            "status": "unavailable",
            "message": "ROS master unavailable; cannot capture current pose.",
            "real_robot_motion_commanded": False,
        }
    try:
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node("capture_natural_grasp_template", anonymous=True, disable_signals=True)
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"ROS init failed: {exc}",
            "real_robot_motion_commanded": False,
        }

    right_state = _read_joint_state(RIGHT_JOINT_TOPIC, timeout_s)
    generic_state = _read_joint_state(GENERIC_JOINT_TOPIC, max(0.4, min(timeout_s, 1.0)))
    raw_map = {}
    if generic_state.get("joint_position_map"):
        raw_map.update(generic_state["joint_position_map"])
    if right_state.get("joint_position_map"):
        raw_map.update(right_state["joint_position_map"])
    status = "success" if right_state.get("status") == "success" else "partial" if raw_map else "unavailable"
    return {
        "status": status,
        "message": "" if raw_map else "No joint states received.",
        "right_state": right_state,
        "generic_state": generic_state,
        "combined_joint_position_map": raw_map,
        "real_robot_motion_commanded": False,
    }


def extract_right_arm(raw_map: Dict[str, float]) -> Dict[str, Any]:
    positions = {
        joint_name: float(raw_map[joint_name])
        for joint_name in RIGHT_ARM_JOINTS
        if joint_name in raw_map
    }
    missing = [joint_name for joint_name in RIGHT_ARM_JOINTS if joint_name not in positions]
    return {
        "joint_names": list(positions.keys()),
        "joint_positions_rad": positions,
        "missing_joints": missing,
        "base_joints_available": all(joint_name in positions for joint_name in RIGHT_BASE_JOINTS),
        "full_six_joint_seed_available": all(joint_name in positions for joint_name in RIGHT_ARM_JOINTS),
    }


def _xl320_name_candidates(motor_id: int) -> List[str]:
    return [
        f"xl320_{motor_id}",
        f"XL320_{motor_id}",
        f"id_{motor_id}",
        f"motor_{motor_id}",
        str(motor_id),
    ]


def extract_xl320(raw_map: Dict[str, float], motor_ids: List[int]) -> Dict[str, Any]:
    values = {}
    source_names = {}
    missing = []
    for motor_id in motor_ids:
        selected = None
        for name in _xl320_name_candidates(motor_id):
            if name in raw_map:
                selected = name
                break
        if selected is None:
            missing.append(motor_id)
        else:
            values[str(motor_id)] = float(raw_map[selected])
            source_names[str(motor_id)] = selected
    return {
        "status": "available" if not missing else ("partial" if values else "unavailable"),
        "positions": values,
        "source_names": source_names,
        "missing_motor_ids": missing,
        "note": (
            "Captured from joint-state names matching XL-320 IDs."
            if values
            else "No read-only XL-320 state topic was available; only arm joint state was captured."
        ),
    }


def read_latest_xl320_command_state() -> Dict[str, Any]:
    if not LATEST_XL320_COMMAND_JSON.exists():
        return {
            "status": "unavailable",
            "message": "No latest XL-320 command log exists yet.",
            "path": str(LATEST_XL320_COMMAND_JSON),
            "value_type": "last_commanded_not_sensor_feedback",
            "note": "XL-320 values are last commanded values, not sensor feedback.",
        }
    try:
        data = json.loads(LATEST_XL320_COMMAND_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"Could not read latest XL-320 command log: {exc}",
            "path": str(LATEST_XL320_COMMAND_JSON),
            "value_type": "last_commanded_not_sensor_feedback",
            "note": "XL-320 values are last commanded values, not sensor feedback.",
        }
    if not isinstance(data, dict):
        data = {}
    data.setdefault("status", "available")
    data.setdefault("path", str(LATEST_XL320_COMMAND_JSON))
    data.setdefault("value_type", "last_commanded_not_sensor_feedback")
    data.setdefault("note", "XL-320 values are last commanded values, not sensor feedback.")
    return data


def _command_entry(command_state: Dict[str, Any], motor_id: int) -> Dict[str, Any]:
    latest = command_state.get("latest_by_motor_id") or {}
    entry = latest.get(str(motor_id)) or {}
    return entry if isinstance(entry, dict) else {}


def _entry_deg(entry: Dict[str, Any]) -> Optional[float]:
    value = entry.get("approximate_degree_value")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_last_commanded_xl320(command_state: Dict[str, Any]) -> Dict[str, Any]:
    wrist_z = _command_entry(command_state, 31)
    wrist_x = _command_entry(command_state, 33)
    finger_entries = {str(motor_id): _command_entry(command_state, motor_id) for motor_id in XL320_FINGER_IDS}
    finger_positions = {
        motor_id: {
            "semantic_label": entry.get("semantic_label"),
            "raw_value": entry.get("raw_value"),
            "approximate_degree_value": entry.get("approximate_degree_value"),
            "source_action": entry.get("source_action"),
            "timestamp": entry.get("timestamp"),
        }
        for motor_id, entry in finger_entries.items()
        if entry
    }
    source_actions = sorted(
        {
            str(entry.get("source_action"))
            for entry in list(finger_entries.values()) + [wrist_z, wrist_x]
            if entry and entry.get("source_action")
        }
    )
    return {
        "latest_xl320_command_state": command_state,
        "commanded_wrist_z_deg": _entry_deg(wrist_z),
        "commanded_wrist_x_deg": _entry_deg(wrist_x),
        "commanded_finger_state": {
            "status": "available" if finger_positions else "unavailable",
            "positions": finger_positions,
            "source_actions": source_actions,
            "value_type": "last_commanded_not_sensor_feedback",
            "note": "XL-320 values are last commanded values, not sensor feedback.",
        },
        "xl320_value_type": "last_commanded_not_sensor_feedback",
        "xl320_feedback_available": False,
        "xl320_note": "XL-320 values are last commanded values, not sensor feedback.",
    }


def estimate_pose(joint_message: Any, raw_map: Dict[str, float]) -> Dict[str, Any]:
    if joint_message is not None:
        mapped = map_joint_angles(joint_message, DEFAULT_MAPPING)
    else:
        class _Msg:
            pass

        msg = _Msg()
        msg.name = list(raw_map.keys())
        msg.position = list(raw_map.values())
        mapped = map_joint_angles(msg, DEFAULT_MAPPING)
    angles = mapped.get("mapped_joint_angles_deg") or {}
    if not angles:
        return {"mapped": mapped, "preview": {}}
    preview = NICOStickModel().forward_kinematics(angles)
    render_preview(
        preview,
        LATEST_PREVIEW_IMAGE,
        target_object="red object",
        target_object_xyz=None,
        template_name="captured_natural_grasp",
    )
    return {"mapped": mapped, "preview": preview}


def build_template(
    template_name: str,
    target_object: str,
    action_stage: str,
    notes: str,
    timeout_s: float,
) -> Dict[str, Any]:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    name = safe_template_name(template_name)
    joint_state = read_current_joint_states(timeout_s)
    raw_map = joint_state.get("combined_joint_position_map", {})
    right_arm = extract_right_arm(raw_map)
    wrist_state = extract_xl320(raw_map, XL320_WRIST_IDS)
    finger_state = extract_xl320(raw_map, XL320_FINGER_IDS)
    latest_xl320_command = read_latest_xl320_command_state()
    commanded_xl320 = summarize_last_commanded_xl320(latest_xl320_command)
    right_msg = (joint_state.get("right_state") or {}).get("_message")
    estimate = estimate_pose(right_msg, raw_map)
    mapped = estimate.get("mapped", {})
    preview = estimate.get("preview", {})

    status = "success" if right_arm.get("base_joints_available") else joint_state.get("status", "unavailable")
    hand_state = "close" if str(action_stage).lower() == "close" else "neutral"
    commandable_positions = {
        joint_name: right_arm["joint_positions_rad"][joint_name]
        for joint_name in RIGHT_ARM_JOINTS
        if joint_name in right_arm["joint_positions_rad"]
    }
    ik_seed_names = [joint for joint in RIGHT_ARM_JOINTS if joint in commandable_positions]
    ik_seed_positions = [commandable_positions[joint] for joint in ik_seed_names]

    template = {
        "name": name,
        "description": f"Natural grasp pose captured from current NICO joint states at stage {action_stage}.",
        "source": "current_real_robot_joint_state_capture",
        "timestamp": timestamp,
        "action_type": "grasp_attempt",
        "action_stage": action_stage,
        "arm_used": "right",
        "target_object": target_object or "red object",
        "hand_state": hand_state,
        "joint_angles_deg": mapped.get("mapped_joint_angles_deg", {}),
        "source_joint_names": mapped.get("source_joint_names", {}),
        "raw_right_arm_joint_positions_rad": right_arm.get("joint_positions_rad", {}),
        "commandable_right_arm_joint_positions_rad": commandable_positions,
        "missing_right_arm_joints": right_arm.get("missing_joints", []),
        "ik_seed_joint_names": ik_seed_names,
        "ik_seed_joint_positions": ik_seed_positions,
        "ik_seed_ready": bool(ik_seed_names),
        "direct_replay_template": right_arm.get("base_joints_available", False),
        "direct_execution_ready": right_arm.get("base_joints_available", False),
        "xl320_wrist_ids": XL320_WRIST_IDS,
        "xl320_wrist_state": wrist_state,
        "xl320_finger_ids": XL320_FINGER_IDS,
        "xl320_finger_state": finger_state,
        "latest_xl320_command_state": commanded_xl320["latest_xl320_command_state"],
        "commanded_wrist_z_deg": commanded_xl320["commanded_wrist_z_deg"],
        "commanded_wrist_x_deg": commanded_xl320["commanded_wrist_x_deg"],
        "commanded_finger_state": commanded_xl320["commanded_finger_state"],
        "xl320_value_type": commanded_xl320["xl320_value_type"],
        "xl320_feedback_available": commanded_xl320["xl320_feedback_available"],
        "xl320_note": commanded_xl320["xl320_note"],
        "estimated_right_hand_xyz": preview.get("estimated_right_hand_xyz", {}),
        "estimated_right_wrist_xyz": preview.get("estimated_right_wrist_xyz", {}),
        "preview_image_path": str(LATEST_PREVIEW_IMAGE) if preview else "",
        "safety_note": "Captured template only. Not automatically sent to robot.",
        "real_robot_motion_commanded": False,
        "notes": notes or "",
    }
    record = {
        "timestamp": timestamp,
        "status": status,
        "template_name": name,
        "action_stage": action_stage,
        "target_object": target_object or "red object",
        "joint_state_status": joint_state.get("status"),
        "joint_state_message": joint_state.get("message", ""),
        "right_arm_joint_names": right_arm.get("joint_names", []),
        "right_arm_joint_positions_rad": right_arm.get("joint_positions_rad", {}),
        "mapped_joint_angles_deg": mapped.get("mapped_joint_angles_deg", {}),
        "xl320_wrist_state": wrist_state,
        "xl320_finger_state": finger_state,
        "latest_xl320_command_state": commanded_xl320["latest_xl320_command_state"],
        "commanded_wrist_z_deg": commanded_xl320["commanded_wrist_z_deg"],
        "commanded_wrist_x_deg": commanded_xl320["commanded_wrist_x_deg"],
        "commanded_finger_state": commanded_xl320["commanded_finger_state"],
        "xl320_value_type": commanded_xl320["xl320_value_type"],
        "xl320_feedback_available": commanded_xl320["xl320_feedback_available"],
        "xl320_note": commanded_xl320["xl320_note"],
        "estimated_right_hand_xyz": template["estimated_right_hand_xyz"],
        "estimated_right_wrist_xyz": template["estimated_right_wrist_xyz"],
        "preview_image_path": template["preview_image_path"],
        "template": template,
        "direct_execution_ready": template["direct_execution_ready"],
        "ik_seed_ready": template["ik_seed_ready"],
        "real_robot_motion_commanded": False,
        "notes": notes or "",
    }
    return record


def capture_current_pose(
    template_name: str,
    target_object: str,
    action_stage: str,
    notes: str,
    allow_overwrite: bool,
    timeout_s: float,
) -> Dict[str, Any]:
    record = build_template(template_name, target_object, action_stage, notes, timeout_s)
    if record.get("status") in ("success", "partial") and record.get("template", {}).get("ik_seed_ready"):
        saved, message = save_template(record["template"], allow_overwrite)
        if not saved:
            record["status"] = "blocked_existing_template"
            record["message"] = message
        else:
            record["message"] = "Captured natural grasp template saved."
    else:
        record["message"] = record.get("joint_state_message") or "No right-arm joint state available; template not saved."
    write_json(record, LATEST_CAPTURE_JSON)
    append_csv(record, CAPTURED_TEMPLATES_CSV)
    record["latest_json_path"] = str(LATEST_CAPTURE_JSON)
    record["templates_path"] = str(CAPTURED_TEMPLATE_PATH)
    record["csv_path"] = str(CAPTURED_TEMPLATES_CSV)
    record["summary_text"] = summary_text(record)
    write_json(record, LATEST_CAPTURE_JSON)
    return record


def get_template(template_name: str) -> Dict[str, Any]:
    data = load_templates()
    templates = data.get("templates", {})
    name = safe_template_name(template_name)
    if not name and templates:
        name = sorted(templates.keys())[-1]
    if name not in templates:
        available = ", ".join(sorted(templates.keys()))
        raise KeyError(f"Captured grasp template '{name}' not found. Available: {available}")
    return dict(templates[name])


def preview_template(template_name: str) -> Dict[str, Any]:
    template = get_template(template_name)
    angles = template.get("joint_angles_deg") or {}
    if not angles:
        return {
            "status": "unavailable",
            "message": "Template has no mapped joint_angles_deg for stick preview.",
            "template_name": template.get("name"),
            "real_robot_motion_commanded": False,
        }
    preview = NICOStickModel().forward_kinematics(angles)
    render_preview(
        preview,
        LATEST_PREVIEW_IMAGE,
        target_object=template.get("target_object", "red object"),
        target_object_xyz=None,
        template_name=template.get("name", "captured_grasp_template"),
    )
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "success",
        "mode": "preview_captured_template",
        "template_name": template.get("name"),
        "template": template,
        "estimated_right_hand_xyz": preview.get("estimated_right_hand_xyz", {}),
        "estimated_right_wrist_xyz": preview.get("estimated_right_wrist_xyz", {}),
        "preview_image_path": str(LATEST_PREVIEW_IMAGE),
        "real_robot_motion_commanded": False,
        "message": "Preview generated from captured template. No robot motion commanded.",
    }
    write_json(record, LATEST_CAPTURE_JSON)
    return record


def seed_plan(template_name: str) -> Dict[str, Any]:
    template = get_template(template_name)
    seed_names = template.get("ik_seed_joint_names") or list(
        (template.get("commandable_right_arm_joint_positions_rad") or {}).keys()
    )
    seed_positions_map = template.get("commandable_right_arm_joint_positions_rad") or {}
    seed_positions = template.get("ik_seed_joint_positions") or [
        seed_positions_map[name] for name in seed_names if name in seed_positions_map
    ]
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "success" if seed_names else "unavailable",
        "mode": "captured_template_as_ik_seed_plan_only",
        "template_name": template.get("name"),
        "target_object": template.get("target_object"),
        "action_stage": template.get("action_stage"),
        "planning_group": "r_arm",
        "captured_template_used_as_seed": True,
        "ik_initial_seed": {
            "joint_names": seed_names,
            "joint_positions": seed_positions,
            "source": "captured_natural_grasp_template",
        },
        "message": "Plan-only seed prepared. IK and robot motion were not executed.",
        "real_robot_motion_commanded": False,
        "template_path": str(CAPTURED_TEMPLATE_PATH),
    }
    write_json(record, LATEST_SEED_PLAN_JSON)
    return record


def summary_text(record: Dict[str, Any]) -> str:
    template = record.get("template") or {}
    lines = [
        f"Status: {record.get('status')}",
        f"Template: {record.get('template_name', template.get('name', 'unknown'))}",
        f"Stage: {record.get('action_stage', template.get('action_stage', 'unknown'))}",
        f"Target: {record.get('target_object', template.get('target_object', 'red object'))}",
        f"Right arm joints: {record.get('right_arm_joint_names', list((template.get('raw_right_arm_joint_positions_rad') or {}).keys()))}",
        f"Estimated right_hand_xyz: {record.get('estimated_right_hand_xyz', template.get('estimated_right_hand_xyz', {}))}",
        f"Estimated right_wrist_xyz: {record.get('estimated_right_wrist_xyz', template.get('estimated_right_wrist_xyz', {}))}",
        f"XL-320 wrist state: {record.get('xl320_wrist_state', template.get('xl320_wrist_state', {})).get('status', 'unknown') if isinstance(record.get('xl320_wrist_state', template.get('xl320_wrist_state', {})), dict) else 'unknown'}",
        f"XL-320 finger state: {record.get('xl320_finger_state', template.get('xl320_finger_state', {})).get('status', 'unknown') if isinstance(record.get('xl320_finger_state', template.get('xl320_finger_state', {})), dict) else 'unknown'}",
        f"Last commanded XL-320 wrist_z deg: {record.get('commanded_wrist_z_deg', template.get('commanded_wrist_z_deg'))}",
        f"Last commanded XL-320 wrist_x deg: {record.get('commanded_wrist_x_deg', template.get('commanded_wrist_x_deg'))}",
        "XL-320 command note: last commanded values, not sensor feedback.",
        f"Direct replay template: {template.get('direct_replay_template', False)}",
        f"IK seed ready: {record.get('ik_seed_ready', template.get('ik_seed_ready', False))}",
        f"Motion commanded: {record.get('real_robot_motion_commanded', False)}",
        f"Message: {record.get('message', '')}",
        f"Latest JSON: {LATEST_CAPTURE_JSON}",
        f"Templates JSON: {CAPTURED_TEMPLATE_PATH}",
        f"CSV: {CAPTURED_TEMPLATES_CSV}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture or preview a natural grasp template without motion.")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Capture current right-arm pose")
    capture.add_argument("--template-name", default="")
    capture.add_argument("--target-object", default="red object")
    capture.add_argument(
        "--action-stage",
        default="pre_grasp",
        choices=["pre_grasp", "approach", "close", "lift"],
    )
    capture.add_argument("--notes", default="")
    capture.add_argument("--timeout", type=float, default=2.5)
    capture.add_argument("--allow-overwrite", action="store_true")

    preview = sub.add_parser("preview", help="Preview captured template")
    preview.add_argument("--template-name", required=True)

    seed = sub.add_parser("seed-plan", help="Prepare captured template as an IK seed, plan-only")
    seed.add_argument("--template-name", required=True)

    args = parser.parse_args()
    try:
        if args.command == "capture":
            record = capture_current_pose(
                args.template_name,
                args.target_object,
                args.action_stage,
                args.notes,
                args.allow_overwrite,
                args.timeout,
            )
        elif args.command == "preview":
            record = preview_template(args.template_name)
            record["summary_text"] = summary_text(record)
        else:
            record = seed_plan(args.template_name)
            record["summary_text"] = (
                f"Status: {record.get('status')}\n"
                f"Template: {record.get('template_name')}\n"
                f"Planning group: {record.get('planning_group')}\n"
                f"Seed joints: {record.get('ik_initial_seed', {}).get('joint_names', [])}\n"
                f"Captured template used as seed: {record.get('captured_template_used_as_seed')}\n"
                f"Motion commanded: {record.get('real_robot_motion_commanded')}\n"
                f"Message: {record.get('message')}\n"
                f"Seed plan JSON: {LATEST_SEED_PLAN_JSON}"
            )
        print(json.dumps(_json_safe(record), indent=2, sort_keys=True))
        return 0 if record.get("status") not in ("failed",) else 1
    except Exception as exc:
        record = {
            "status": "failed",
            "message": str(exc),
            "real_robot_motion_commanded": False,
        }
        print(json.dumps(record, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
