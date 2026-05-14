#!/usr/bin/env python3
"""FYP2 AP2 NJF feasibility probe and no-motion evidence checks."""

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils.fyp2_trial_logger import (
    append_csv,
    find_api_root,
    monitoring_paths,
    write_json,
    write_monitoring_summary,
)


API_ROOT = find_api_root(__file__)
PATHS = monitoring_paths(API_ROOT)
CHECK_TIMEOUT_S = 3

IMPLEMENTATION_LEVEL = "NJF-inspired feasibility probe / ROS service boundary"
LIMITATION = (
    "This is not full trained Neural Jacobian Field deployment. Full "
    "NICO-specific NJF requires robot-specific image-action data collection, "
    "calibration, and physical validation."
)
RELEVANCE_TO_NICO = (
    "Provides a safe service boundary and structured logging path for future "
    "NICO manipulation refinement experiments."
)
NEXT_ACTION = (
    "Run controlled NICO data recording and physical manipulation trials after "
    "monitoring feedback."
)

EVALUATION_METRICS = [
    "probe_status",
    "log_generation_success",
    "ros_availability",
    "njf_service_availability",
    "physical_robot_evidence",
    "physical_motion_executed_false_for_probe",
    "future_controlled_trial_success_rate",
]
SUCCESS_CRITERIA = [
    "dry_run_generates_json_csv_markdown",
    "dashboard_displays_latest_probe_result",
    "ros_check_does_not_crash",
    "robot_evidence_check_commands_no_motion",
    "service_check_reports_available_or_unavailable_clearly",
    "physical_trials_scheduled_for_week_10_12",
]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def session_id() -> str:
    return "ap2-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def common_record(mode: str) -> Dict[str, Any]:
    return {
        "timestamp": timestamp(),
        "session_id": session_id(),
        "method_name": "Neural Jacobian Fields",
        "implementation_level": IMPLEMENTATION_LEVEL,
        "mode": mode,
        "robot_required": False,
        "ros_required": False,
        "gpu_required": False,
        "user_instruction": "Refine the grasp position for the red block.",
        "target_object": "red block",
        "planning_group": "r_arm",
        "input_visual_error": {
            "image_error_u": -0.18,
            "image_error_v": 0.12,
            "x_error_m": 0.012,
            "y_error_m": -0.010,
        },
        "predicted_correction": {
            "x_correction_m": 0.009,
            "y_correction_m": -0.0075,
            "bounded": True,
        },
        "status": "not_run",
        "physical_robot_used": False,
        "physical_motion_executed": False,
        "safety_note": "No unsafe motion commanded by NJF probe.",
        "limitation": LIMITATION,
        "relevance_to_nico": RELEVANCE_TO_NICO,
        "next_action": NEXT_ACTION,
        "objective_id": "O1/O2/O3",
        "objective_measure": (
            "O1: NJF feasibility probe executed; O2: structured manipulation "
            "log generated; O3: ROS/robot evidence workflow prepared for "
            "controlled trials."
        ),
        "experiment_stage": "AP2 monitoring-stage feasibility validation",
        "experiment_design": (
            "dry-run validation -> ROS check -> robot evidence check without "
            "motion -> optional NJF service check -> controlled physical NICO "
            "trials in Week 10-12"
        ),
        "evaluation_metrics": EVALUATION_METRICS,
        "success_criteria": SUCCESS_CRITERIA,
        "ai_training_status": (
            "VLM/LLM modules are used as pretrained reasoning components. No "
            "full NICO-specific Neural Jacobian Field training is claimed at "
            "AP2 monitoring stage."
        ),
        "ai_testing_status": (
            "Current testing focuses on deterministic NJF-inspired dry-run, "
            "ROS service-boundary validation, structured logs, and preparation "
            "for future NICO image-action data collection."
        ),
    }


def run_ros_command(command: str, timeout_s: int = CHECK_TIMEOUT_S) -> Dict[str, Any]:
    shell_command = (
        "source activate.bash 2>/dev/null || true; "
        "source devel/setup.bash 2>/dev/null || true; "
        f"{command}"
    )
    try:
        completed = subprocess.run(
            ["bash", "-lc", shell_command],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
            "timeout": True,
        }
    except Exception as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timeout": False,
        }


def split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def possible_camera_topics(topics: List[str]) -> List[str]:
    markers = ("camera", "image", "vision", "rgb", "depth")
    return [topic for topic in topics if any(marker in topic.lower() for marker in markers)]


def robot_topics(topics: List[str]) -> List[str]:
    markers = ("nico", "joint_states", "open_manipulator", "speech_asr", "mllm")
    return [topic for topic in topics if any(marker in topic.lower() for marker in markers)]


def build_dry_run_record() -> Dict[str, Any]:
    record = common_record("dry-run")
    record.update(
        {
            "status": "success",
            "robot_required": False,
            "ros_required": False,
            "gpu_required": False,
            "physical_robot_used": False,
            "physical_motion_executed": False,
            "dry_run_label": "dry-run / feasibility probe",
        }
    )
    return record


def build_ros_check_record() -> Dict[str, Any]:
    record = common_record("ros-check")
    record.update({"robot_required": False, "ros_required": True})

    nodes = run_ros_command("rosnode list", CHECK_TIMEOUT_S)
    topics = run_ros_command("rostopic list", CHECK_TIMEOUT_S)
    node_lines = split_lines(nodes.get("stdout", ""))
    topic_lines = split_lines(topics.get("stdout", ""))

    if nodes["timeout"] or topics["timeout"]:
        status = "timeout"
    elif nodes["returncode"] != 0 and topics["returncode"] != 0:
        status = "unavailable"
    elif nodes["returncode"] == 0 and topics["returncode"] == 0:
        status = "success"
    else:
        status = "partial"

    record.update(
        {
            "status": status,
            "roscore_available": nodes["returncode"] == 0 or topics["returncode"] == 0,
            "rosnode_count": len(node_lines),
            "rostopic_count": len(topic_lines),
            "rosnodes": node_lines[:80],
            "rostopics": topic_lines[:120],
            "possible_camera_topics": possible_camera_topics(topic_lines),
            "ros_check_stdout": {
                "rosnode_list": nodes.get("stdout", ""),
                "rostopic_list": topics.get("stdout", ""),
            },
            "ros_check_stderr": {
                "rosnode_list": nodes.get("stderr", ""),
                "rostopic_list": topics.get("stderr", ""),
            },
            "physical_robot_used": False,
            "physical_motion_executed": False,
        }
    )
    return record


def build_robot_evidence_record() -> Dict[str, Any]:
    record = common_record("robot-evidence")
    record.update({"robot_required": True, "ros_required": True})

    topics = run_ros_command("rostopic list", CHECK_TIMEOUT_S)
    nodes = run_ros_command("rosnode list", CHECK_TIMEOUT_S)
    topic_lines = split_lines(topics.get("stdout", ""))
    node_lines = split_lines(nodes.get("stdout", ""))
    evidence_topics = robot_topics(topic_lines)
    camera_topics = possible_camera_topics(topic_lines)

    if topics["timeout"] or nodes["timeout"]:
        status = "timeout"
    elif topics["returncode"] != 0 and nodes["returncode"] != 0:
        status = "unavailable"
    elif evidence_topics or camera_topics:
        status = "success"
    else:
        status = "partial"

    record.update(
        {
            "status": status,
            "physical_robot_used": bool(evidence_topics or camera_topics),
            "physical_motion_executed": False,
            "robot_evidence_topics": evidence_topics[:120],
            "possible_camera_topics": camera_topics,
            "rosnodes": node_lines[:80],
            "rostopics": topic_lines[:120],
            "safety_note": (
                "Robot evidence check is passive. It records ROS/topic evidence "
                "and commands no robot motion."
            ),
        }
    )
    return record


def parse_float_from_output(output: str, field: str) -> float:
    match = re.search(rf"{re.escape(field)}\s*:\s*([-+]?\d+(?:\.\d+)?)", output)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def build_service_check_record() -> Dict[str, Any]:
    record = common_record("service-check")
    record.update({"robot_required": False, "ros_required": True})

    services = run_ros_command("rosservice list", CHECK_TIMEOUT_S)
    service_lines = split_lines(services.get("stdout", ""))
    service_name = "/njf_predict_action"
    if services["timeout"]:
        record.update({"status": "timeout", "available_services": service_lines[:120]})
        return record
    if services["returncode"] != 0:
        record.update(
            {
                "status": "unavailable",
                "service_check_stderr": services.get("stderr", ""),
                "available_services": service_lines[:120],
            }
        )
        return record
    if service_name not in service_lines:
        record.update(
            {
                "status": "unavailable",
                "service_name": service_name,
                "message": "NJF service unavailable or not built.",
                "available_services": service_lines[:120],
            }
        )
        return record

    request = """planning_group: 'r_arm'
x_error_m: 0.012
y_error_m: -0.010
image_error_u: -0.18
image_error_v: 0.12
joint_names: []
joint_positions: []
prefer_joint_delta: false"""
    call = run_ros_command(
        "rosservice call /njf_predict_action " + shlex.quote(request),
        CHECK_TIMEOUT_S,
    )

    if call["timeout"]:
        status = "timeout"
    elif call["returncode"] == 0:
        status = "success"
    elif "Unable to load type" in call.get("stderr", "") or "Cannot load" in call.get("stderr", ""):
        status = "not built"
    else:
        status = "failed"

    stdout = call.get("stdout", "")
    record.update(
        {
            "status": status,
            "service_name": service_name,
            "service_stdout": stdout,
            "service_stderr": call.get("stderr", ""),
            "predicted_correction": {
                "x_correction_m": parse_float_from_output(stdout, "x_correction_m"),
                "y_correction_m": parse_float_from_output(stdout, "y_correction_m"),
                "bounded": True,
            },
            "physical_robot_used": False,
            "physical_motion_executed": False,
        }
    )
    return record


def build_summary_record() -> Dict[str, Any]:
    record = common_record("summary")
    record.update(
        {
            "status": "success",
            "physical_robot_used": False,
            "physical_motion_executed": False,
            "message": "Monitoring summary regenerated from latest available records.",
        }
    )
    return record


def record_json_path(mode: str) -> Path:
    if mode == "dry-run":
        return PATHS["latest_njf_probe"]
    if mode == "service-check":
        return PATHS["latest_njf_service_check"]
    if mode == "ros-check":
        return PATHS["latest_ros_check"]
    if mode == "robot-evidence":
        return PATHS["latest_robot_evidence"]
    return PATHS["log_dir"] / "latest_summary_request.json"


def persist_record(record: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(record.get("mode", "summary"))
    json_path = record_json_path(mode)
    record["log_json_path"] = str(json_path)
    record["csv_log_path"] = str(PATHS["latest_csv"])
    record["markdown_summary_path"] = str(PATHS["monitoring_summary"])

    if mode != "summary":
        write_json(record, json_path)
    append_csv(record, PATHS["latest_csv"])
    write_monitoring_summary(record, PATHS["monitoring_summary"])
    return record


def build_record(mode: str) -> Dict[str, Any]:
    if mode == "dry-run":
        return build_dry_run_record()
    if mode == "ros-check":
        return build_ros_check_record()
    if mode == "service-check":
        return build_service_check_record()
    if mode == "robot-evidence":
        return build_robot_evidence_record()
    if mode == "summary":
        return build_summary_record()
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["dry-run", "ros-check", "service-check", "robot-evidence", "summary"],
        required=True,
    )
    args = parser.parse_args()

    record = persist_record(build_record(args.mode))
    print(json.dumps(record, indent=2, sort_keys=True))
    print("Wrote monitoring logs:")
    if args.mode != "summary":
        print(f"- JSON: {record['log_json_path']}")
    print(f"- CSV: {record['csv_log_path']}")
    print(f"- Markdown: {record['markdown_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
