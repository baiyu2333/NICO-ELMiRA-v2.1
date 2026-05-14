#!/usr/bin/env python3
"""Logging helpers for offline NICO kinematic action previews."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


LOG_DIR_NAME = "kinematic_preview"
LATEST_PREVIEW_JSON = "latest_preview.json"
LATEST_PREVIEW_SUMMARY = "latest_preview_summary.md"
PREVIEW_TRIALS_CSV = "preview_trials.csv"

CSV_FIELDS = [
    "timestamp",
    "template_name",
    "action_type",
    "arm_used",
    "hand_state",
    "target_object",
    "target_object_xyz",
    "grasp_offset_if_available",
    "joint_angles_deg",
    "estimated_right_hand_xyz",
    "estimated_right_wrist_xyz",
    "table_surface_z_m",
    "default_right_hand_height_from_ground_m",
    "red_object_distance_from_nico_m",
    "min_safe_z_m",
    "above_table_safety_limit",
    "height_warning",
    "preview_image_path",
    "real_robot_motion_commanded",
    "safety_note",
    "preview_note",
    "notes",
    "record_json",
]


def find_api_root(start_file: str) -> Path:
    current = Path(start_file).resolve()
    for candidate in [current] + list(current.parents):
        if candidate.name == "api" and (candidate / "src" / "ELMiRA").exists():
            return candidate
    for candidate in current.parents:
        if candidate.name == "api":
            return candidate
        if (candidate / "src" / "ELMiRA").exists():
            return candidate
    raise RuntimeError("Could not locate api root")


def ensure_log_dir(api_root: Optional[Path] = None) -> Path:
    root = api_root or find_api_root(__file__)
    log_dir = root / "logs" / LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def preview_paths(api_root: Optional[Path] = None) -> Dict[str, Path]:
    log_dir = ensure_log_dir(api_root)
    return {
        "log_dir": log_dir,
        "latest_json": log_dir / LATEST_PREVIEW_JSON,
        "latest_summary": log_dir / LATEST_PREVIEW_SUMMARY,
        "latest_image": log_dir / "latest_preview.png",
        "trials_csv": log_dir / PREVIEW_TRIALS_CSV,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return str(value)


def write_json(record: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(record), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def append_csv(record: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: _csv_value(record.get(field, "")) for field in CSV_FIELDS}
    row["record_json"] = json.dumps(_json_safe(record), sort_keys=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def write_summary(record: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Kinematic Preview Summary",
        "",
        f"Timestamp: {record.get('timestamp', '')}",
        f"Template: {record.get('template_name', '')}",
        f"Action type: {record.get('action_type', '')}",
        f"Arm used: {record.get('arm_used', '')}",
        f"Hand state: {record.get('hand_state', '')}",
        "",
        "## Target",
        "",
        f"Target object: {record.get('target_object', '')}",
        f"Target object xyz: `{json.dumps(record.get('target_object_xyz', {}), sort_keys=True)}`",
        "",
        "## Preview Result",
        "",
        f"Joint angles deg: `{json.dumps(record.get('joint_angles_deg', {}), sort_keys=True)}`",
        f"Estimated right hand xyz: `{json.dumps(record.get('estimated_right_hand_xyz', {}), sort_keys=True)}`",
        f"Estimated right wrist xyz: `{json.dumps(record.get('estimated_right_wrist_xyz', {}), sort_keys=True)}`",
        f"Above table safety limit: {record.get('above_table_safety_limit')}",
        f"Height warning: {record.get('height_warning') or 'none'}",
        "",
        "## Measured Setup Values",
        "",
        f"Table surface z: {record.get('table_surface_z_m')} m",
        f"Default right hand height: {record.get('default_right_hand_height_from_ground_m')} m",
        f"Red object distance from NICO: {record.get('red_object_distance_from_nico_m')} m",
        f"Minimum safe z: {record.get('min_safe_z_m')} m",
        "",
        "## Safety",
        "",
        f"Real robot motion commanded: {record.get('real_robot_motion_commanded')}",
        f"Safety note: {record.get('safety_note', '')}",
        f"Preview note: {record.get('preview_note', '')}",
        "",
        f"Preview image path: `{record.get('preview_image_path', '')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_record(
    template: Dict[str, Any],
    preview: Dict[str, Any],
    preview_image_path: Path,
    measured_values: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    measured_values = measured_values or {}
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "template_name": template.get("name", "unknown"),
        "action_type": template.get("action_type", "unknown"),
        "arm_used": template.get("arm_used", "right"),
        "hand_state": template.get("hand_state", "neutral"),
        "target_object": template.get("target_object", "target object"),
        "target_object_xyz": template.get("target_object_xyz"),
        "grasp_offset_if_available": template.get("grasp_offset"),
        "joint_angles_deg": template.get("joint_angles_deg", template.get("joint_angles", {})),
        "estimated_right_hand_xyz": preview.get("estimated_right_hand_xyz", {}),
        "estimated_right_wrist_xyz": preview.get("estimated_right_wrist_xyz", {}),
        "table_surface_z_m": measured_values.get("table_surface_z_m", preview.get("table_surface_z_m")),
        "default_right_hand_height_from_ground_m": measured_values.get(
            "default_right_hand_height_from_ground_m"
        ),
        "red_object_distance_from_nico_m": measured_values.get("red_object_distance_from_nico_m"),
        "min_safe_z_m": measured_values.get("min_safe_z_m", preview.get("min_safe_z_m")),
        "above_table_safety_limit": preview.get("above_table_safety_limit"),
        "height_warning": preview.get("height_warning", ""),
        "preview_image_path": str(preview_image_path),
        "real_robot_motion_commanded": False,
        "safety_note": template.get("safety_note", "Preview only. No real robot motion is commanded."),
        "preview_note": measured_values.get(
            "preview_note",
            "Coarse kinematic estimate only. No ROS motion command, no physics, no contact simulation.",
        ),
        "notes": notes
        or "Coarse kinematic estimate only. No ROS motion command, no physics, no contact simulation.",
        "coordinate_system": preview.get("coordinate_system"),
        "coordinates": preview.get("coordinates", {}),
    }


def log_preview_result(record: Dict[str, Any], api_root: Optional[Path] = None) -> Dict[str, Path]:
    paths = preview_paths(api_root)
    write_json(record, paths["latest_json"])
    write_summary(record, paths["latest_summary"])
    append_csv(record, paths["trials_csv"])
    return paths
