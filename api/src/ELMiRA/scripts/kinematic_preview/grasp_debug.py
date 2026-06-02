#!/usr/bin/env python3
"""Stage-by-stage grasp debugging for ELMiRA/NICO.

This module is deliberately conservative:
- planning, detection, and coordinate checks command no robot motion;
- wrist/hand/arm stage execution requires an explicit confirmation flag;
- only the physical right XL-320 wrist/hand IDs are used for hand-stage tests.
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


try:
    from utils.fyp2_trial_logger import find_api_root, write_json
except Exception:
    def find_api_root(start_file: str) -> Path:
        current = Path(start_file).resolve()
        for candidate in [current] + list(current.parents):
            if candidate.name == "api" and (candidate / "src" / "ELMiRA").exists():
                return candidate
        raise RuntimeError("Could not locate api root")

    def write_json(record: Dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return path


API_ROOT = find_api_root(__file__)
LOG_DIR = API_ROOT / "logs" / "grasp_debug"
LATEST_PLAN_JSON = LOG_DIR / "latest_grasp_plan.json"
TRIALS_CSV = LOG_DIR / "grasp_debug_trials.csv"

LLM_DETECT_SERVICE = "mllm_detect"
COORDINATE_TRANSFER_SERVICE = "image_to_real"
XL320_TOPIC = "/nico/motion/xl320_cmd"
RIGHT_WRIST_IDS = [31, 33]
RIGHT_FINGER_IDS = [34, 35, 36, 37]
RIGHT_TOUCH_WRIST_POSITIONS_DEG = {31: 80.0, 33: -45.0}
RIGHT_HAND_OPEN_DEG = -150.0
RIGHT_HAND_CLOSE_DEG = 90.0
RIGHT_HAND_CLOSE_DEG_BY_ID = {34: 150.0, 35: 150.0, 36: 150.0, 37: 70.0}
RIGHT_XL320_POSITION_MAX_RAW = 1500
RIGHT_XL320_COMMAND_REPEATS = 4
RIGHT_XL320_COMMAND_INTERVAL_SEC = 0.04

MIN_REACH_X = 0.10
MAX_REACH_X = 0.32
MAX_REACH_Y_RIGHT = -0.25
MAX_REACH_Y_LEFT = 0.18

ORIENTATIONS = {
    "pre_grasp": {"x": -1.0, "y": 0.0, "z": 0.0, "w": 0.0, "name": "right touch/point orientation"},
    "approach": {"x": -1.0, "y": 0.0, "z": 0.0, "w": 0.0, "name": "right touch/point orientation"},
    "lift_retreat": {"x": -1.0, "y": 0.0, "z": 0.0, "w": 0.0, "name": "right touch/point orientation"},
}


CSV_FIELDS = [
    "timestamp",
    "stage",
    "action_type",
    "target_object",
    "status",
    "selected_arm",
    "real_x",
    "real_y",
    "target_z",
    "workspace_clamped",
    "wrist_ids",
    "finger_ids",
    "real_robot_motion_commanded",
    "execution_status",
    "failure_reason",
    "json_path",
]


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
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
        row["json_path"] = str(LATEST_PLAN_JSON)
        writer.writerow(row)
    return path


def log_record(record: Dict[str, Any]) -> Dict[str, Any]:
    write_json(record, LATEST_PLAN_JSON)
    append_csv(record, TRIALS_CSV)
    record["latest_json_path"] = str(LATEST_PLAN_JSON)
    record["trials_csv_path"] = str(TRIALS_CSV)
    return record


def ros_master_available(timeout_s: float = 1.0) -> bool:
    try:
        import rosgraph

        master = rosgraph.Master("/elmira_grasp_debug_check")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                master.getSystemState()
                return True
            except Exception:
                time.sleep(0.05)
    except Exception:
        return False
    return False


def init_ros_node(name: str) -> Tuple[bool, str]:
    if not ros_master_available(timeout_s=1.0):
        return False, "ROS master unavailable."
    try:
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node(name, anonymous=True, disable_signals=True)
        return True, ""
    except Exception as exc:
        return False, f"ROS init failed: {exc}"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def invalid_image_coords(image_x: Any, image_y: Any) -> bool:
    try:
        x_value = float(image_x)
        y_value = float(image_y)
    except (TypeError, ValueError):
        return True
    # The red object should never be represented by the exact top-left image
    # corner. In the dashboard this usually means Gradio sent empty Number
    # fields as 0.0.
    return abs(x_value) < 1e-9 and abs(y_value) < 1e-9


def get_ros_float_param(name: str, default: float) -> float:
    try:
        import rospy

        return _safe_float(rospy.get_param(name, default), default)
    except Exception:
        return default


def get_ros_int_param(name: str, default: int) -> int:
    try:
        import rospy

        return int(rospy.get_param(name, default))
    except Exception:
        return default


def get_ros_int_list_param(name: str, default: List[int]) -> List[int]:
    try:
        import rospy

        raw = rospy.get_param(name, default)
        if isinstance(raw, str):
            import re

            values = [int(match) for match in re.findall(r"-?\d+", raw)]
            return values or list(default)
        return [int(item) for item in raw]
    except Exception:
        return list(default)


def get_ros_float_map_param(name: str, default: Dict[int, float]) -> Dict[int, float]:
    try:
        import re
        import rospy

        raw = rospy.get_param(name, default)
        if isinstance(raw, dict):
            return {int(key): float(value) for key, value in raw.items()}
        if isinstance(raw, str):
            parsed = {
                int(key): float(value)
                for key, value in re.findall(r"(-?\d+)\s*:\s*(-?\d+(?:\.\d+)?)", raw)
            }
            return parsed or dict(default)
    except Exception:
        pass
    return dict(default)


def current_xl320_config() -> Dict[str, Any]:
    wrist_z_id = get_ros_int_param("/elmira/right_xl320_wrist_z_id", RIGHT_WRIST_IDS[0])
    wrist_x_id = get_ros_int_param("/elmira/right_xl320_wrist_x_id", RIGHT_WRIST_IDS[1])
    finger_ids = get_ros_int_list_param("/elmira/right_xl320_finger_ids", RIGHT_FINGER_IDS)
    wrist_z_deg = get_ros_float_param("/elmira/right_touch_wrist_z_deg", RIGHT_TOUCH_WRIST_POSITIONS_DEG[31])
    wrist_x_deg = get_ros_float_param("/elmira/right_touch_wrist_x_deg", RIGHT_TOUCH_WRIST_POSITIONS_DEG[33])
    return {
        "wrist_ids": [wrist_z_id, wrist_x_id],
        "wrist_positions_deg": {str(wrist_z_id): wrist_z_deg, str(wrist_x_id): wrist_x_deg},
        "finger_ids": finger_ids,
        "open_deg": get_ros_float_param("/elmira/right_hand_open_deg", RIGHT_HAND_OPEN_DEG),
        "close_deg": get_ros_float_param("/elmira/right_hand_close_deg", RIGHT_HAND_CLOSE_DEG),
        "close_deg_by_id": get_ros_float_map_param(
            "/elmira/right_hand_close_deg_by_id",
            RIGHT_HAND_CLOSE_DEG_BY_ID,
        ),
        "position_max_raw": max(
            1023,
            min(
                4095,
                int(get_ros_float_param("/elmira/right_xl320_position_max_raw", RIGHT_XL320_POSITION_MAX_RAW)),
            ),
        ),
    }


def clamp_to_workspace(x: float, y: float) -> Tuple[float, float, bool, Dict[str, Any]]:
    clamped_x = max(MIN_REACH_X, min(MAX_REACH_X, float(x)))
    effective_y_left = 0.05 if clamped_x < 0.25 else MAX_REACH_Y_LEFT
    clamped_y = max(MAX_REACH_Y_RIGHT, min(effective_y_left, float(y)))
    return (
        clamped_x,
        clamped_y,
        clamped_x != float(x) or clamped_y != float(y),
        {
            "x": [MIN_REACH_X, MAX_REACH_X],
            "y": [MAX_REACH_Y_RIGHT, effective_y_left],
        },
    )


def _deg_to_xl320_raw(deg: float, position_max_raw: int = RIGHT_XL320_POSITION_MAX_RAW) -> int:
    max_raw = max(1023, min(4095, int(position_max_raw)))
    raw = int((float(deg) + 150.0) / 300.0 * float(max_raw))
    return max(0, min(max_raw, raw))


def detect_target(target_object: str, timeout_s: float) -> Dict[str, Any]:
    ok, reason = init_ros_node("elmira_grasp_debug_detect")
    if not ok:
        return {"status": "unavailable", "message": reason, "objects": [], "selected_target": None}
    try:
        import rospy
        from elmira.srv import DetectWithMLLM, DetectWithMLLMRequest

        rospy.wait_for_service(LLM_DETECT_SERVICE, timeout=timeout_s)
        request = DetectWithMLLMRequest()
        request.texts = [target_object]
        request.confidence_threshold = 0.5
        response = rospy.ServiceProxy(LLM_DETECT_SERVICE, DetectWithMLLM)(request)
        objects = []
        for obj in getattr(response, "objects", []):
            objects.append(
                {
                    "label": getattr(obj, "label", target_object),
                    "score": float(getattr(obj, "score", 0.0)),
                    "center_x": float(getattr(obj, "center_x", 0.0)),
                    "center_y": float(getattr(obj, "center_y", 0.0)),
                    "width": float(getattr(obj, "width", 0.0)),
                    "height": float(getattr(obj, "height", 0.0)),
                    "bbox": {
                        "center_x": float(getattr(obj, "center_x", 0.0)),
                        "center_y": float(getattr(obj, "center_y", 0.0)),
                        "width": float(getattr(obj, "width", 0.0)),
                        "height": float(getattr(obj, "height", 0.0)),
                    },
                    "bottom_x": float(getattr(obj, "center_x", 0.0)),
                    "bottom_y": float(getattr(obj, "center_y", 0.0)) + float(getattr(obj, "height", 0.0)) / 2.0,
                }
            )
        objects.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        selected = objects[0] if objects else None
        return {
            "status": "success" if selected else "partial",
            "message": "" if selected else "Detector returned no objects.",
            "objects": objects,
            "selected_target": selected,
            "service_success": bool(getattr(response, "success", selected is not None)),
            "error_message": getattr(response, "error_message", ""),
            "latency_ms": float(getattr(response, "latency_ms", 0.0)),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"Detection unavailable: {exc}",
            "objects": [],
            "selected_target": None,
        }


def coordinate_transfer(image_x: Optional[float], image_y: Optional[float], timeout_s: float) -> Dict[str, Any]:
    if image_x is None or image_y is None:
        return {"status": "unavailable", "message": "image_x/image_y not provided."}
    ok, reason = init_ros_node("elmira_grasp_debug_coordinate")
    if not ok:
        return {"status": "unavailable", "message": reason}
    try:
        import rospy
        from elmira.srv import CoordinateTransfer, CoordinateTransferRequest

        rospy.wait_for_service(COORDINATE_TRANSFER_SERVICE, timeout=timeout_s)
        request = CoordinateTransferRequest()
        request.image_x = float(image_x)
        request.image_y = float(image_y)
        response = rospy.ServiceProxy(COORDINATE_TRANSFER_SERVICE, CoordinateTransfer)(request)
        return {
            "status": "success",
            "real_x": float(response.real_x),
            "real_y": float(response.real_y),
            "image_x": float(image_x),
            "image_y": float(image_y),
            "message": "",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"Coordinate transfer unavailable: {exc}",
            "image_x": image_x,
            "image_y": image_y,
        }


def build_grasp_plan(
    action_type: str,
    target_object: str,
    real_x: float,
    real_y: float,
    target_z: float,
) -> Dict[str, Any]:
    clamped_x, clamped_y, was_clamped, limits = clamp_to_workspace(real_x, real_y)
    table_z_min = get_ros_float_param("/elmira/table_z_min", 0.45)
    table_z_max = get_ros_float_param("/elmira/table_z_max", 0.85)
    target_z_clamped = max(table_z_min, min(table_z_max, float(target_z)))
    hover_z_offset = max(0.04, min(0.12, get_ros_float_param("/elmira/grasp_hover_z_offset", 0.06)))
    pre_grasp_backoff = max(0.0, min(0.08, get_ros_float_param("/elmira/grasp_pre_grasp_x_backoff", 0.035)))
    approach_backoff = max(0.0, min(0.05, get_ros_float_param("/elmira/grasp_approach_x_backoff", 0.015)))
    contact_z_offset = max(-0.12, min(hover_z_offset, get_ros_float_param("/elmira/grasp_contact_z_offset", get_ros_float_param("/elmira/right_touch_z_offset", -0.075))))
    lift_z_offset = max(max(0.05, hover_z_offset), min(0.30, get_ros_float_param("/elmira/grasp_lift_z_offset", 0.12)))

    pre_grasp_pose = {
        "x": max(0.10, clamped_x - pre_grasp_backoff),
        "y": clamped_y,
        "z": max(table_z_min, min(table_z_max, target_z_clamped + hover_z_offset + 0.03)),
    }
    approach_pose = {
        "x": max(0.10, clamped_x - approach_backoff),
        "y": clamped_y,
        "z": max(table_z_min, min(table_z_max, target_z_clamped + hover_z_offset)),
    }
    contact_pose = {
        "x": clamped_x,
        "y": clamped_y,
        "z": max(table_z_min, min(table_z_max, target_z_clamped + contact_z_offset)),
    }
    lift_pose = {
        "x": clamped_x,
        "y": clamped_y,
        "z": max(table_z_min, min(table_z_max, target_z_clamped + lift_z_offset)),
    }
    xl320 = current_xl320_config()
    planned_steps = [
        {
            "stage": "pre_grasp",
            "pose": pre_grasp_pose,
            "orientation": ORIENTATIONS["pre_grasp"],
            "hand_action": "open",
            "note": "Right hand opens before/at conservative pre-grasp.",
            "will_command_wrist_ids": xl320["wrist_ids"],
        },
        {
            "stage": "approach",
            "pose": approach_pose,
            "orientation": ORIENTATIONS["approach"],
            "hand_action": "touch_wrist",
            "note": "Small approach. Wrist alignment should command XL-320 IDs 31/33.",
            "will_command_wrist_ids": xl320["wrist_ids"],
        },
        {
            "stage": "wrist_alignment",
            "pose": None,
            "orientation": "XL-320 wrist posture",
            "hand_action": "touch_wrist",
            "note": "Right XL-320 wrist only: IDs 31/33.",
            "will_command_wrist_ids": xl320["wrist_ids"],
        },
        {
            "stage": "close_hand",
            "pose": contact_pose,
            "orientation": ORIENTATIONS["approach"],
            "hand_action": "close",
            "note": "Close fingers after reaching contact pose.",
            "will_command_wrist_ids": xl320["wrist_ids"],
            "will_command_finger_ids": xl320["finger_ids"],
        },
        {
            "stage": "lift_retreat",
            "pose": lift_pose,
            "orientation": ORIENTATIONS["lift_retreat"],
            "hand_action": None,
            "note": "Small lift/retreat after close.",
            "will_command_wrist_ids": [],
        },
    ]
    z_warning = ""
    if float(target_z) <= 0.20:
        z_warning = (
            "Input target_z looks like physical table-frame z. ELMiRA IK uses robot-frame table_z "
            "around 0.70 m; execution stages are blocked below table_z_min."
        )
    return {
        "action_type": action_type,
        "target_object": target_object,
        "selected_arm": "right",
        "input_real_x": real_x,
        "input_real_y": real_y,
        "real_x": clamped_x,
        "real_y": clamped_y,
        "target_z": target_z_clamped,
        "target_z_input": target_z,
        "target_z_was_clamped": target_z_clamped != float(target_z),
        "workspace_clamped": was_clamped,
        "workspace_limits": limits,
        "z_warning": z_warning,
        "table_z_min": table_z_min,
        "table_z_max": table_z_max,
        "hover_z_offset": hover_z_offset,
        "contact_z_offset": contact_z_offset,
        "lift_z_offset": lift_z_offset,
        "pre_grasp_backoff": pre_grasp_backoff,
        "approach_backoff": approach_backoff,
        "planned_steps": planned_steps,
        "xl320_config": xl320,
        "wrist_ids_31_33_will_be_commanded": xl320["wrist_ids"] == [31, 33],
    }


def publish_xl320_commands(commands: List[Tuple[int, int, int]], timeout_s: float = 2.0) -> Tuple[bool, str]:
    ok, reason = init_ros_node("elmira_grasp_debug_xl320")
    if not ok:
        return False, reason
    try:
        import rospy
        import nicomsg.msg

        publisher = rospy.Publisher(XL320_TOPIC, nicomsg.msg.sff, queue_size=1)
        deadline = time.time() + timeout_s
        while publisher.get_num_connections() == 0 and time.time() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.05)
        if publisher.get_num_connections() == 0:
            return False, f"{XL320_TOPIC} has no subscriber."
        repeat_count = max(
            1,
            int(get_ros_float_param("/elmira/right_xl320_command_repeats", RIGHT_XL320_COMMAND_REPEATS)),
        )
        interval_sec = max(
            0.005,
            get_ros_float_param(
                "/elmira/right_xl320_command_interval_sec",
                RIGHT_XL320_COMMAND_INTERVAL_SEC,
            ),
        )
        for motor_id, register, value in commands:
            per_command_repeats = repeat_count if int(register) == 30 else min(2, repeat_count)
            for _ in range(per_command_repeats):
                msg = nicomsg.msg.sff()
                msg.param1 = str(int(motor_id))
                msg.param2 = float(register)
                msg.param3 = float(value)
                publisher.publish(msg)
                rospy.sleep(interval_sec)
        rospy.sleep(0.15)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def execute_wrist_only(safety_confirmed: bool) -> Dict[str, Any]:
    if not safety_confirmed:
        return {
            "execution_status": "blocked_by_safety",
            "failure_reason": "Safety checkbox not confirmed.",
            "real_robot_motion_commanded": False,
        }
    xl320 = current_xl320_config()
    wrist_positions = {int(k): v for k, v in xl320["wrist_positions_deg"].items()}
    commands = []
    for motor_id in xl320["wrist_ids"]:
        commands.append((motor_id, 24, 1))
    for motor_id in xl320["wrist_ids"]:
        commands.append((motor_id, 32, 120))
    for motor_id, deg in wrist_positions.items():
        commands.append((motor_id, 30, _deg_to_xl320_raw(deg, xl320["position_max_raw"])))
    ok, reason = publish_xl320_commands(commands)
    return {
        "execution_status": "executed" if ok else "unavailable",
        "failure_reason": reason,
        "real_robot_motion_commanded": ok,
        "commanded_xl320_ids": xl320["wrist_ids"] if ok else [],
        "position_max_raw": xl320["position_max_raw"],
        "commands": commands if ok else [],
        "note": "Wrist-only stage commands XL-320 IDs 31/33 by default; no arm/finger motion.",
    }


def execute_hand_only(hand_action: str, safety_confirmed: bool) -> Dict[str, Any]:
    if not safety_confirmed:
        return {
            "execution_status": "blocked_by_safety",
            "failure_reason": "Safety checkbox not confirmed.",
            "real_robot_motion_commanded": False,
        }
    xl320 = current_xl320_config()
    commands = []
    for motor_id in xl320["finger_ids"]:
        commands.append((motor_id, 24, 1))
    for motor_id in xl320["finger_ids"]:
        commands.append((motor_id, 32, 150))
    target_by_motor = {}
    for motor_id in xl320["finger_ids"]:
        target_deg = (
            xl320["open_deg"]
            if hand_action == "open"
            else xl320["close_deg_by_id"].get(motor_id, xl320["close_deg"])
        )
        raw = _deg_to_xl320_raw(target_deg, xl320["position_max_raw"])
        commands.append((motor_id, 30, raw))
        target_by_motor[motor_id] = {"deg": target_deg, "raw": raw}
    ok, reason = publish_xl320_commands(commands)
    return {
        "execution_status": "executed" if ok else "unavailable",
        "failure_reason": reason,
        "real_robot_motion_commanded": ok,
        "commanded_xl320_ids": xl320["finger_ids"] if ok else [],
        "target_by_motor": target_by_motor,
        "position_max_raw": xl320["position_max_raw"],
        "commands": commands if ok else [],
        "command_repeats": get_ros_float_param("/elmira/right_xl320_command_repeats", RIGHT_XL320_COMMAND_REPEATS),
        "note": f"{hand_action} fingers only; no arm or wrist motion. Commands are repeated to avoid missed XL-320 writes.",
    }


def execute_arm_stage(stage: str, plan: Dict[str, Any], safety_confirmed: bool) -> Dict[str, Any]:
    if not safety_confirmed:
        return {
            "execution_status": "blocked_by_safety",
            "failure_reason": "Safety checkbox not confirmed.",
            "real_robot_motion_commanded": False,
        }
    if plan.get("target_z", 0.0) < plan.get("table_z_min", 0.45):
        return {
            "execution_status": "blocked_by_safety",
            "failure_reason": "Planned z is below safe table_z_min.",
            "real_robot_motion_commanded": False,
        }
    pose = None
    for step in plan.get("planned_steps", []):
        if step.get("stage") == stage:
            pose = step.get("pose")
            break
    if not pose:
        return {
            "execution_status": "failed",
            "failure_reason": f"No pose found for stage {stage}.",
            "real_robot_motion_commanded": False,
        }
    try:
        from kinematic_preview.safe_preset_debug import execute_cartesian_xyz

        result = execute_cartesian_xyz(
            pose["x"],
            pose["y"],
            pose["z"],
            "touch_forward",
            safety_confirmed=True,
            path_time=3.5 if stage == "approach" else 3.0,
        )
        result["debug_stage"] = stage
        return result
    except Exception as exc:
        return {
            "execution_status": "failed",
            "failure_reason": str(exc),
            "real_robot_motion_commanded": False,
        }


def summary_text(record: Dict[str, Any]) -> str:
    detection = record.get("detection", {})
    coordinate = record.get("coordinate_transfer", {})
    plan = record.get("plan", {})
    selected = detection.get("selected_target") or {}
    lines = [
        f"Stage: {record.get('stage')}",
        f"Status: {record.get('status')}",
        f"Action: {record.get('action_type')}",
        f"Target: {record.get('target_object')}",
        f"Detected target: {selected.get('label', 'not available')}",
        f"Image coordinates: ({selected.get('bottom_x', coordinate.get('image_x', 'n/a'))}, {selected.get('bottom_y', coordinate.get('image_y', 'n/a'))})",
        f"Bounding box: {selected.get('bbox', 'not available')}",
        f"Real coordinates: x={plan.get('real_x', record.get('real_x'))}, y={plan.get('real_y', record.get('real_y'))}, z={plan.get('target_z', record.get('target_z'))}",
        f"Workspace clamped: {plan.get('workspace_clamped', False)}",
        f"Selected arm: {plan.get('selected_arm', 'right')}",
        f"Wrist IDs 31/33 commanded by wrist stage: {plan.get('wrist_ids_31_33_will_be_commanded', False)}",
        f"XL-320 wrist IDs: {plan.get('xl320_config', {}).get('wrist_ids', RIGHT_WRIST_IDS)}",
        f"XL-320 finger IDs: {plan.get('xl320_config', {}).get('finger_ids', RIGHT_FINGER_IDS)}",
        f"XL-320 position max raw: {plan.get('xl320_config', {}).get('position_max_raw', RIGHT_XL320_POSITION_MAX_RAW)}",
        "Planned pose/action sequence:",
    ]
    for step in plan.get("planned_steps", []):
        lines.append(
            f"- {step.get('stage')}: pose={step.get('pose')}, "
            f"orientation={step.get('orientation')}, hand_action={step.get('hand_action')}"
        )
    if plan.get("z_warning"):
        lines.append(f"Warning: {plan.get('z_warning')}")
    execution = record.get("execution", {})
    if execution:
        lines.append(
            f"Execution: {execution.get('execution_status')} "
            f"(motion commanded: {execution.get('real_robot_motion_commanded', False)})"
        )
        if execution.get("failure_reason"):
            lines.append(f"Execution note: {execution.get('failure_reason')}")
    lines.append(f"Log: {LATEST_PLAN_JSON}")
    return "\n".join(lines)


def build_record(args: argparse.Namespace) -> Dict[str, Any]:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    target_object = args.target_object or "red object"
    detection = {}
    coordinate = {}

    image_x = args.image_x
    image_y = args.image_y
    if args.stage == "detect":
        detection = detect_target(target_object, args.timeout)
        selected = detection.get("selected_target") or {}
        image_x = selected.get("bottom_x", image_x)
        image_y = selected.get("bottom_y", image_y)
    elif args.stage == "coordinate" and invalid_image_coords(image_x, image_y):
        detection = detect_target(target_object, args.timeout)
        selected = detection.get("selected_target") or {}
        image_x = selected.get("bottom_x", image_x)
        image_y = selected.get("bottom_y", image_y)
    elif args.use_detection:
        detection = detect_target(target_object, args.timeout)
        selected = detection.get("selected_target") or {}
        image_x = selected.get("bottom_x", image_x)
        image_y = selected.get("bottom_y", image_y)

    real_x = args.real_x
    real_y = args.real_y
    if args.stage == "coordinate" or (args.use_coordinate and image_x is not None and image_y is not None):
        coordinate = coordinate_transfer(image_x, image_y, args.timeout)
        if coordinate.get("status") == "success":
            real_x = coordinate.get("real_x", real_x)
            real_y = coordinate.get("real_y", real_y)

    if real_x is None:
        real_x = 0.20
    if real_y is None:
        real_y = -0.08

    plan = build_grasp_plan(args.action_type, target_object, float(real_x), float(real_y), float(args.target_z))
    execution: Dict[str, Any] = {}
    if args.stage == "execute-pre-grasp":
        execution = execute_arm_stage("pre_grasp", plan, args.confirm)
    elif args.stage == "execute-approach":
        execution = execute_arm_stage("approach", plan, args.confirm)
    elif args.stage == "align-wrist":
        execution = execute_wrist_only(args.confirm)
    elif args.stage == "open-hand":
        execution = execute_hand_only("open", args.confirm)
    elif args.stage == "close-hand":
        execution = execute_hand_only("close", args.confirm)
    elif args.stage == "full-grasp":
        execution = {
            "execution_status": "blocked_by_debug_mode",
            "failure_reason": "Full grasp is intentionally not automated here. Verify each stage first, then use the normal ELMiRA pipeline.",
            "real_robot_motion_commanded": False,
        }

    status = "success"
    if args.stage == "detect":
        status = detection.get("status", "unavailable")
    elif args.stage == "coordinate":
        status = coordinate.get("status", "unavailable")
    elif execution:
        status = execution.get("execution_status", "unknown")

    record = {
        "timestamp": timestamp,
        "stage": args.stage,
        "action_type": args.action_type,
        "target_object": target_object,
        "status": status,
        "detection": detection,
        "coordinate_transfer": coordinate,
        "image_x": image_x,
        "image_y": image_y,
        "real_x": plan.get("real_x"),
        "real_y": plan.get("real_y"),
        "target_z": plan.get("target_z"),
        "selected_arm": "right",
        "workspace_clamped": plan.get("workspace_clamped", False),
        "plan": plan,
        "execution": execution,
        "wrist_ids": plan.get("xl320_config", {}).get("wrist_ids", RIGHT_WRIST_IDS),
        "finger_ids": plan.get("xl320_config", {}).get("finger_ids", RIGHT_FINGER_IDS),
        "real_robot_motion_commanded": bool(execution.get("real_robot_motion_commanded", False)),
        "execution_status": execution.get("execution_status", "not_executed"),
        "failure_reason": execution.get("failure_reason", ""),
        "safety_note": "Right arm/hand only. No left arm. Full grasp not automated in debug mode.",
    }
    record["summary_text"] = summary_text(record)
    return log_record(record)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug ELMiRA grasp stages without automatic full grasp.")
    parser.add_argument(
        "--stage",
        choices=[
            "detect",
            "coordinate",
            "plan",
            "execute-pre-grasp",
            "align-wrist",
            "open-hand",
            "close-hand",
            "execute-approach",
            "full-grasp",
        ],
        required=True,
    )
    parser.add_argument("--action-type", default="grasp")
    parser.add_argument("--target-object", default="red object")
    parser.add_argument("--image-x", type=float, default=None)
    parser.add_argument("--image-y", type=float, default=None)
    parser.add_argument("--real-x", type=float, default=None)
    parser.add_argument("--real-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=0.70)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--use-detection", action="store_true")
    parser.add_argument("--use-coordinate", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    record = build_record(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record.get("status") not in ("failed",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
