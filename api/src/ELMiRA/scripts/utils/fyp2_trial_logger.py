#!/usr/bin/env python3
"""Small logging helpers for the FYP2 AP2 monitoring demo."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


LOG_DIR_NAME = "fyp2_monitoring"

LATEST_NJF_PROBE = "latest_njf_probe.json"
LATEST_NJF_SERVICE_CHECK = "latest_njf_service_check.json"
LATEST_ROS_CHECK = "latest_ros_check.json"
LATEST_ROBOT_EVIDENCE = "latest_robot_evidence.json"
LATEST_CSV = "latest_njf_probe.csv"
MONITORING_SUMMARY = "monitoring_summary.md"

CSV_FIELDS = [
    "timestamp",
    "session_id",
    "method_name",
    "implementation_level",
    "mode",
    "status",
    "robot_required",
    "ros_required",
    "gpu_required",
    "user_instruction",
    "target_object",
    "planning_group",
    "physical_robot_used",
    "physical_motion_executed",
    "objective_id",
    "objective_measure",
    "experiment_stage",
    "experiment_design",
    "evaluation_metrics",
    "success_criteria",
    "ai_training_status",
    "ai_testing_status",
    "safety_note",
    "limitation",
    "relevance_to_nico",
    "next_action",
    "log_json_path",
    "record_json",
]


def find_api_root(start_file: str) -> Path:
    """Locate the repository's api directory without relying on cwd."""
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


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def _latest_records(log_dir: Path, current_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    file_map = {
        "njf_probe": LATEST_NJF_PROBE,
        "service_check": LATEST_NJF_SERVICE_CHECK,
        "ros_check": LATEST_ROS_CHECK,
        "robot_evidence": LATEST_ROBOT_EVIDENCE,
    }
    for key, filename in file_map.items():
        data = _read_json_if_exists(log_dir / filename)
        if data:
            records[key] = data

    mode = str(current_record.get("mode", ""))
    if mode == "dry-run":
        records["njf_probe"] = current_record
    elif mode == "service-check":
        records["service_check"] = current_record
    elif mode == "ros-check":
        records["ros_check"] = current_record
    elif mode == "robot-evidence":
        records["robot_evidence"] = current_record
    return records


def _format_list(values: Any) -> str:
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict)):
        return "\n".join(f"- {item}" for item in values)
    return f"- {values}" if values else "- Not recorded"


def _record_line(record: Optional[Dict[str, Any]], label: str) -> str:
    if not record:
        return f"- {label}: not recorded yet"
    status = record.get("status", "unknown")
    mode = record.get("mode", "unknown")
    timestamp = record.get("timestamp", "unknown time")
    return f"- {label}: {status} ({mode}, {timestamp})"


def write_monitoring_summary(record: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _latest_records(path.parent, record)
    probe = records.get("njf_probe") or record
    ros_check = records.get("ros_check")
    robot_evidence = records.get("robot_evidence")
    service_check = records.get("service_check")

    generated_at = datetime.now().isoformat(timespec="seconds")
    content = f"""# FYP2 Monitoring Update

Generated: {generated_at}

{_record_line(probe, "NJF dry-run feasibility probe")}
{_record_line(ros_check, "ROS check")}
{_record_line(robot_evidence, "Robot evidence, no motion")}
{_record_line(service_check, "Optional NJF service check")}

## What the NJF Probe Tests

The current implementation level is: {probe.get("implementation_level", "not recorded")}.

Target object: {probe.get("target_object", "not recorded")}

Input visual error: `{json.dumps(_json_safe(probe.get("input_visual_error", {})), sort_keys=True)}`

Predicted bounded correction: `{json.dumps(_json_safe(probe.get("predicted_correction", {})), sort_keys=True)}`

Safety note: {probe.get("safety_note", "No unsafe motion commanded by NJF probe.")}

## Evaluation Design

Objective ID: {probe.get("objective_id", "not recorded")}

Objective measure: {probe.get("objective_measure", "not recorded")}

Experiment stage: {probe.get("experiment_stage", "not recorded")}

Experiment design: {probe.get("experiment_design", "not recorded")}

Evaluation metrics:
{_format_list(probe.get("evaluation_metrics"))}

Success criteria:
{_format_list(probe.get("success_criteria"))}

## AI Training and Testing Status

Training status: {probe.get("ai_training_status", "not recorded")}

Testing status: {probe.get("ai_testing_status", "not recorded")}

## Current Limitations

{probe.get("limitation", "not recorded")}

Service check status: {(service_check or {}).get("status", "not recorded")}

ROS check status: {(ros_check or {}).get("status", "not recorded")}

Robot evidence status: {(robot_evidence or {}).get("status", "not recorded")}

## Next Actions for Week 10-12

{probe.get("next_action", "Run controlled NICO data recording and physical manipulation trials after monitoring feedback.")}
"""
    path.write_text(content, encoding="utf-8")
    return path


def monitoring_paths(api_root: Optional[Path] = None) -> Dict[str, Path]:
    log_dir = ensure_log_dir(api_root)
    return {
        "log_dir": log_dir,
        "latest_njf_probe": log_dir / LATEST_NJF_PROBE,
        "latest_njf_service_check": log_dir / LATEST_NJF_SERVICE_CHECK,
        "latest_ros_check": log_dir / LATEST_ROS_CHECK,
        "latest_robot_evidence": log_dir / LATEST_ROBOT_EVIDENCE,
        "latest_csv": log_dir / LATEST_CSV,
        "monitoring_summary": log_dir / MONITORING_SUMMARY,
    }
