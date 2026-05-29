#!/usr/bin/env python3
"""Dump the ELMiRA grasp/touch IK pipeline without commanding motion."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from kinematic_preview.grasp_debug import (
        ORIENTATIONS,
        RIGHT_FINGER_IDS,
        RIGHT_WRIST_IDS,
        build_grasp_plan,
        coordinate_transfer,
        current_xl320_config,
        detect_target,
        invalid_image_coords,
        ros_master_available,
    )
except Exception as exc:  # pragma: no cover - import failure is reported in trace.
    raise RuntimeError(f"Could not import grasp debug helpers: {exc}") from exc

try:
    from utils.fyp2_trial_logger import find_api_root, write_json
except Exception:
    from utils.kinematic_trial_logger import find_api_root, write_json

try:
    from utils.constants import COMMANDABLE_JOINTS, JOINT_MOTOR_IDS, filter_commandable_joints
except Exception:
    COMMANDABLE_JOINTS = {"r_arm": frozenset(["r_shoulder_z", "r_shoulder_y", "r_arm_x", "r_elbow_y"])}
    JOINT_MOTOR_IDS = {"r_wrist_z": 23, "r_wrist_x": 25}

    def filter_commandable_joints(group: str, names: Iterable[str], positions: Iterable[float]):
        allowed = COMMANDABLE_JOINTS.get(group, set())
        kept_names, kept_positions, dropped = [], [], []
        for name, position in zip(names, positions):
            if name in allowed:
                kept_names.append(name)
                kept_positions.append(position)
            else:
                dropped.append(name)
        return kept_names, kept_positions, dropped


API_ROOT = find_api_root(__file__)
LOG_DIR = API_ROOT / "logs" / "ik_trace"
LATEST_TRACE_JSON = LOG_DIR / "latest_ik_trace.json"
TRIALS_CSV = LOG_DIR / "ik_trace_trials.csv"
CAPTURED_TEMPLATE_PATH = SCRIPT_DIR / "templates" / "captured_grasp_templates.json"

DEFAULT_RIGHT_ARM_NAMES = [
    "r_shoulder_z",
    "r_shoulder_y",
    "r_arm_x",
    "r_elbow_y",
    "r_wrist_z",
    "r_wrist_x",
]
DEFAULT_RIGHT_ARM_POSITIONS = [-0.157, 0.0, -0.8203, -1.57, -1.39, 0.0]
IK_SERVICE = "inverse_kinematics"
RIGHT_JOINT_STATE_TOPIC = "/right/open_manipulator_p/joint_states"

CSV_FIELDS = [
    "timestamp",
    "action_type",
    "target_object",
    "status",
    "selected_arm",
    "image_x",
    "image_y",
    "real_x",
    "real_y",
    "target_z",
    "workspace_clamped",
    "ik_available",
    "ik_succeeded",
    "captured_template_used_as_seed",
    "seed_template_name",
    "seed_joint_names",
    "ik_returned_r_wrist_z",
    "ik_returned_r_wrist_x",
    "dropped_joints",
    "final_joint_names",
    "hand_action_sequence",
    "xl320_wrist_ids_will_be_commanded",
    "finger_ids_will_be_commanded",
    "real_robot_motion_commanded",
    "diagnosis",
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
        row["json_path"] = str(LATEST_TRACE_JSON)
        writer.writerow(row)
    return path


def log_record(record: Dict[str, Any]) -> Dict[str, Any]:
    write_json(record, LATEST_TRACE_JSON)
    append_csv(record, TRIALS_CSV)
    record["latest_json_path"] = str(LATEST_TRACE_JSON)
    record["trials_csv_path"] = str(TRIALS_CSV)
    return record


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


def _pose_dict(stage: str, pose: Dict[str, float], orientation: Dict[str, Any], hand_action: Optional[str], sent_to_initial_ik: bool) -> Dict[str, Any]:
    return {
        "stage": stage,
        "position": {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
        },
        "orientation_quaternion": {
            "x": float(orientation.get("x", 0.0)),
            "y": float(orientation.get("y", 0.0)),
            "z": float(orientation.get("z", 0.0)),
            "w": float(orientation.get("w", 1.0)),
        },
        "orientation_name": orientation.get("name", "right touch/point orientation"),
        "pose_basis": "right_tcp / wrist-based IK target, not fingertip contact target",
        "approach_vector": {"x": 1.0, "y": 0.0, "z": 0.0},
        "grasp_offset_from_object_center": {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "note": "Offsets are relative to object-center estimate when real_x/real_y/target_z are known.",
        },
        "hand_action": hand_action,
        "sent_to_initial_ik_request": sent_to_initial_ik,
    }


def planned_target_poses(action_type: str, plan: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    action = str(action_type or "grasp").lower()
    if action == "touch":
        touch_pose = {
            "x": float(plan["real_x"]),
            "y": float(plan["real_y"]),
            "z": float(plan["target_z"]),
        }
        pose_record = _pose_dict(
            "touch",
            touch_pose,
            ORIENTATIONS["approach"],
            "touch_wrist",
            True,
        )
        return [pose_record], [pose_record]

    by_stage = {step.get("stage"): step for step in plan.get("planned_steps", [])}
    planned = []
    for stage, sent_to_initial_ik in [
        ("pre_grasp", True),
        ("approach", True),
        ("close_hand", False),
        ("lift_retreat", False),
    ]:
        step = by_stage.get(stage) or {}
        pose = step.get("pose")
        if not pose:
            continue
        normalized_stage = "touch" if stage == "close_hand" else stage
        planned.append(
            _pose_dict(
                normalized_stage,
                pose,
                step.get("orientation") or ORIENTATIONS["approach"],
                step.get("hand_action"),
                sent_to_initial_ik,
            )
        )
    return planned, [pose for pose in planned if pose.get("sent_to_initial_ik_request")]


def _disabled_motor_ids() -> List[int]:
    try:
        import rospy

        raw = rospy.get_param("/nico/motion/disabledMotorIds", [])
    except Exception:
        return []
    if isinstance(raw, str):
        import re

        return [int(match) for match in re.findall(r"-?\d+", raw)]
    try:
        return [int(item) for item in raw]
    except Exception:
        return []


def load_captured_seed_template(template_name: Optional[str]) -> Optional[Dict[str, Any]]:
    name = (template_name or "").strip()
    if not name:
        return None
    if not CAPTURED_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Captured template file not found: {CAPTURED_TEMPLATE_PATH}")
    data = json.loads(CAPTURED_TEMPLATE_PATH.read_text(encoding="utf-8"))
    templates = data.get("templates", {}) if isinstance(data, dict) else {}
    if name not in templates:
        available = ", ".join(sorted(templates.keys()))
        raise KeyError(f"Captured template '{name}' not found. Available: {available}")
    template = dict(templates[name])
    raw_seed_names = template.get("ik_seed_joint_names") or list(
        (template.get("commandable_right_arm_joint_positions_rad") or {}).keys()
    )
    raw_seed_positions = template.get("ik_seed_joint_positions")
    if not raw_seed_positions:
        positions_map = template.get("commandable_right_arm_joint_positions_rad") or {}
        raw_seed_positions = [
            positions_map[joint] for joint in raw_seed_names if joint in positions_map
        ]
    if not raw_seed_names or len(raw_seed_names) != len(raw_seed_positions):
        raise ValueError(f"Captured template '{name}' does not contain a usable IK seed.")

    # EvoIK was configured around a six-joint right-arm seed. The current
    # physical robot only publishes the four Protocol 1.0 arm joints; old SR
    # wrist joints r_wrist_z/r_wrist_x are disabled and later filtered out.
    # For plan-only IK tracing, pad the missing wrist seed entries with the
    # normal defaults so the IK request shape stays valid.
    seed_map = {
        joint: float(position)
        for joint, position in zip(raw_seed_names, raw_seed_positions)
    }
    default_map = dict(zip(DEFAULT_RIGHT_ARM_NAMES, DEFAULT_RIGHT_ARM_POSITIONS))
    padded_positions = [
        seed_map.get(joint_name, default_map[joint_name])
        for joint_name in DEFAULT_RIGHT_ARM_NAMES
    ]
    template["raw_resolved_seed_joint_names"] = list(raw_seed_names)
    template["raw_resolved_seed_joint_positions"] = [
        float(value) for value in raw_seed_positions
    ]
    template["resolved_seed_joint_names"] = list(DEFAULT_RIGHT_ARM_NAMES)
    template["resolved_seed_joint_positions"] = padded_positions
    template["seed_padding_note"] = (
        "Captured template provides the four active right-arm joints. "
        "Missing old SR wrist seed joints were padded with defaults for IK trace only; "
        "they remain disabled/filtered before real execution."
    )
    return template


def read_initial_position(timeout_s: float, seed_template_name: Optional[str] = None) -> Dict[str, Any]:
    if seed_template_name:
        try:
            template = load_captured_seed_template(seed_template_name)
            if template:
                return {
                    "source": "captured_natural_grasp_template",
                    "status": "success",
                    "message": "",
                    "joint_names": template["resolved_seed_joint_names"],
                    "joint_positions": template["resolved_seed_joint_positions"],
                    "captured_template_used_as_seed": True,
                    "seed_template_name": template.get("name", seed_template_name),
                    "captured_template": template,
                }
        except Exception as exc:
            return {
                "source": "captured_natural_grasp_template",
                "status": "unavailable",
                "message": f"Could not load captured seed template: {exc}",
                "joint_names": list(DEFAULT_RIGHT_ARM_NAMES),
                "joint_positions": list(DEFAULT_RIGHT_ARM_POSITIONS),
                "captured_template_used_as_seed": False,
                "seed_template_name": seed_template_name,
            }
    ok, reason = init_ros_node("elmira_ik_trace_initial")
    names = list(DEFAULT_RIGHT_ARM_NAMES)
    positions = list(DEFAULT_RIGHT_ARM_POSITIONS)
    if not ok:
        return {
            "source": "default_state_machine_seed",
            "status": "unavailable",
            "message": reason,
            "joint_names": names,
            "joint_positions": positions,
            "captured_template_used_as_seed": False,
        }
    try:
        import rospy
        from sensor_msgs.msg import JointState

        msg = rospy.wait_for_message(RIGHT_JOINT_STATE_TOPIC, JointState, timeout=timeout_s)
        current = {name: position for name, position in zip(msg.name, msg.position)}
        for index, joint_name in enumerate(names):
            if joint_name in current:
                positions[index] = float(current[joint_name])
        return {
            "source": RIGHT_JOINT_STATE_TOPIC,
            "status": "success",
            "message": "",
            "joint_names": names,
            "joint_positions": positions,
            "raw_joint_names": list(msg.name),
            "captured_template_used_as_seed": False,
        }
    except Exception as exc:
        return {
            "source": "default_state_machine_seed",
            "status": "partial",
            "message": f"Could not read current right-arm joint state: {exc}",
            "joint_names": names,
            "joint_positions": positions,
            "captured_template_used_as_seed": False,
        }


def call_ik(ik_request: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    ok, reason = init_ros_node("elmira_ik_trace_solver")
    if not ok:
        return {"status": "unavailable", "message": reason, "positions": []}
    try:
        import rospy
        from geometry_msgs.msg import Pose
        from elmira.msg import JointPosition as ElmiraJointPosition
        from elmira.srv import InverseKinematics, InverseKinematicsRequest

        rospy.wait_for_service(IK_SERVICE, timeout=timeout_s)
        request = InverseKinematicsRequest()
        request.planning_group = ik_request["planning_group"]
        request.initial_position = ElmiraJointPosition()
        request.initial_position.joint_name = list(ik_request["initial_position"]["joint_names"])
        request.initial_position.position = list(ik_request["initial_position"]["joint_positions"])
        for pose_info in ik_request.get("target_pose_list", []):
            pose = Pose()
            position = pose_info["position"]
            orientation = pose_info["orientation_quaternion"]
            pose.position.x = float(position["x"])
            pose.position.y = float(position["y"])
            pose.position.z = float(position["z"])
            pose.orientation.x = float(orientation["x"])
            pose.orientation.y = float(orientation["y"])
            pose.orientation.z = float(orientation["z"])
            pose.orientation.w = float(orientation["w"])
            request.poses.append(pose)
        response = rospy.ServiceProxy(IK_SERVICE, InverseKinematics)(request)
        positions = []
        for index, joint_position in enumerate(getattr(response, "positions", [])):
            names = list(joint_position.joint_name)
            values = [float(value) for value in joint_position.position]
            positions.append(
                {
                    "stage": ik_request.get("target_pose_list", [{}])[index].get("stage", f"pose_{index}"),
                    "joint_names": names,
                    "joint_positions": values,
                    "joint_positions_deg": [round(value * 180.0 / 3.141592653589793, 3) for value in values],
                    "includes_r_wrist_z": "r_wrist_z" in names,
                    "includes_r_wrist_x": "r_wrist_x" in names,
                }
            )
        return {
            "status": "success" if positions else "partial",
            "message": "" if positions else "IK returned no positions.",
            "positions": positions,
            "service": IK_SERVICE,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": f"IK service unavailable or failed: {exc}",
            "positions": [],
            "service": IK_SERVICE,
        }


def command_filter_trace(ik_response: Dict[str, Any], planning_group: str) -> Dict[str, Any]:
    disabled = _disabled_motor_ids()
    steps = []
    all_dropped = []
    for step in ik_response.get("positions", []):
        before_names = list(step.get("joint_names", []))
        before_positions = list(step.get("joint_positions", []))
        after_names, after_positions, dropped = filter_commandable_joints(
            planning_group,
            before_names,
            before_positions,
        )
        dropped_reasons = {}
        allowed = COMMANDABLE_JOINTS.get(planning_group, frozenset())
        for name in dropped:
            motor_id = JOINT_MOTOR_IDS.get(name)
            if name not in allowed:
                reason = "not in COMMANDABLE_JOINTS"
            elif motor_id in disabled:
                reason = f"disabled motor ID {motor_id}"
            else:
                reason = "filtered by active commandable-joint rules"
            dropped_reasons[name] = reason
        all_dropped.extend(dropped)
        steps.append(
            {
                "stage": step.get("stage"),
                "joints_before_filtering": before_names,
                "positions_before_filtering": before_positions,
                "joints_after_filtering": after_names,
                "positions_after_filtering": after_positions,
                "dropped_joints": dropped,
                "dropped_reasons": dropped_reasons,
            }
        )
    return {
        "disabled_motor_ids": disabled,
        "steps": steps,
        "dropped_joints": sorted(set(all_dropped)),
    }


def final_command_trace(
    action_type: str,
    filter_trace: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    action = str(action_type or "grasp").lower()
    if action == "touch":
        hand_actions = ["touch_wrist"]
    else:
        hand_actions = [
            step.get("hand_action")
            for step in plan.get("planned_steps", [])
            if step.get("hand_action")
        ]
    command_steps = []
    for step in filter_trace.get("steps", []):
        command_steps.append(
            {
                "stage": step.get("stage"),
                "final_joint_names": step.get("joints_after_filtering", []),
                "final_joint_positions": step.get("positions_after_filtering", []),
            }
        )
    xl320 = plan.get("xl320_config") or current_xl320_config()
    return {
        "command_steps_if_executed": command_steps,
        "hand_action_sequence": hand_actions,
        "xl320_wrist_ids_31_33_will_be_commanded": "touch_wrist" in hand_actions and xl320.get("wrist_ids") == [31, 33],
        "xl320_wrist_ids": xl320.get("wrist_ids", RIGHT_WRIST_IDS),
        "fingers_34_37_will_open_close": any(action in ("open", "close") for action in hand_actions)
        and xl320.get("finger_ids") == [34, 35, 36, 37],
        "xl320_finger_ids": xl320.get("finger_ids", RIGHT_FINGER_IDS),
        "real_robot_motion_commanded": False,
        "note": "This trace does not call MoveRobot, SetJointPosition, xl320_cmd, or hand control.",
    }


def build_diagnosis(record: Dict[str, Any]) -> str:
    final_command = record.get("final_command", {})
    filter_trace = record.get("command_filtering", {})
    ik_response = record.get("ik_response", {})
    planned = record.get("planned_target_poses_before_ik", [])
    basis_note = "right_tcp / wrist-based IK target" if planned else "no target pose"
    dropped = filter_trace.get("dropped_joints", [])
    response_positions = ik_response.get("positions", [])
    wrist_returned = any(
        step.get("includes_r_wrist_z") or step.get("includes_r_wrist_x")
        for step in response_positions
    )
    if ik_response.get("status") != "success":
        return "IK trace is incomplete because IK service was unavailable or returned no positions."
    if dropped:
        return (
            f"IK returns wrist-capable joints={wrist_returned}, but command filtering drops {dropped}. "
            f"The arm target is {basis_note}, not a fingertip/palm contact target, so the physical XL-320 hand may still look unnatural."
        )
    if final_command.get("xl320_wrist_ids_31_33_will_be_commanded"):
        return (
            f"IK command path keeps the filtered arm joints and a separate touch_wrist stage can command XL-320 wrist IDs 31/33. "
            f"The target is still {basis_note}; unnatural grasp can come from wrist-target IK plus separate hand orientation timing."
        )
    return (
        f"Final command does not include a confirmed XL-320 wrist alignment stage. "
        f"The target is {basis_note}, so grasp posture may be controlled mainly by shoulder/elbow IK."
    )


def build_trace(args: argparse.Namespace) -> Dict[str, Any]:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    target_object = args.target_object or "red object"
    detection = {}
    coordinate = {}
    image_x = args.image_x
    image_y = args.image_y

    if args.use_detection or invalid_image_coords(image_x, image_y):
        detection = detect_target(target_object, args.timeout)
        selected = detection.get("selected_target") or {}
        image_x = selected.get("bottom_x", image_x)
        image_y = selected.get("bottom_y", image_y)

    if args.use_coordinate and image_x is not None and image_y is not None:
        coordinate = coordinate_transfer(image_x, image_y, args.timeout)

    real_x = coordinate.get("real_x", args.real_x)
    real_y = coordinate.get("real_y", args.real_y)
    if real_x is None:
        real_x = 0.20
    if real_y is None:
        real_y = -0.08

    plan = build_grasp_plan(args.action_type, target_object, float(real_x), float(real_y), float(args.target_z))
    planned_poses, ik_target_poses = planned_target_poses(args.action_type, plan)
    initial_position = read_initial_position(args.timeout, getattr(args, "seed_template", None))
    ik_request = {
        "planning_group": "r_arm",
        "initial_position": initial_position,
        "target_pose_list": ik_target_poses,
        "captured_template_used_as_seed": bool(initial_position.get("captured_template_used_as_seed")),
        "seed_template_name": initial_position.get("seed_template_name"),
        "seed_joint_names": list(initial_position.get("joint_names", [])),
        "seed_joint_positions": list(initial_position.get("joint_positions", [])),
        "motion_commanded": False,
    }
    ik_response = call_ik(ik_request, args.timeout) if ik_target_poses else {
        "status": "partial",
        "message": "No IK target poses generated.",
        "positions": [],
    }
    filtering = command_filter_trace(ik_response, "r_arm")
    final_command = final_command_trace(args.action_type, filtering, plan)
    selected = detection.get("selected_target") or {}
    record = {
        "timestamp": timestamp,
        "trace_type": "ik_trace_dump_no_motion",
        "action_type": args.action_type,
        "target_object": target_object,
        "status": "success" if ik_response.get("status") == "success" else "partial",
        "object_detection_result": {
            "target_object": target_object,
            "status": detection.get("status", "not_run"),
            "message": detection.get("message", ""),
            "selected_target": selected,
            "bounding_box": selected.get("bbox"),
            "image_center": {
                "x": selected.get("center_x"),
                "y": selected.get("center_y"),
            },
            "image_bottom_point_used_by_object_selector": {
                "x": selected.get("bottom_x", image_x),
                "y": selected.get("bottom_y", image_y),
            },
            "detection_score": selected.get("score"),
            "all_candidates": detection.get("objects", []),
        },
        "coordinate_transfer_result": {
            "status": coordinate.get("status", "not_run"),
            "message": coordinate.get("message", ""),
            "image_x": image_x,
            "image_y": image_y,
            "real_x": real_x,
            "real_y": real_y,
            "table_z": plan.get("target_z"),
            "target_z": plan.get("target_z"),
            "selected_arm": "right",
            "x_y_were_clamped": plan.get("workspace_clamped", False),
            "clamped_x": plan.get("real_x"),
            "clamped_y": plan.get("real_y"),
            "workspace_limits": plan.get("workspace_limits"),
        },
        "planned_target_poses_before_ik": planned_poses,
        "ik_request": ik_request,
        "ik_response": ik_response,
        "command_filtering": filtering,
        "final_command": final_command,
        "captured_template_used_as_seed": bool(initial_position.get("captured_template_used_as_seed")),
        "seed_template_name": initial_position.get("seed_template_name"),
        "seed_joint_names": list(initial_position.get("joint_names", [])),
        "seed_joint_positions": list(initial_position.get("joint_positions", [])),
        "selected_arm": "right",
        "image_x": image_x,
        "image_y": image_y,
        "real_x": plan.get("real_x"),
        "real_y": plan.get("real_y"),
        "target_z": plan.get("target_z"),
        "workspace_clamped": plan.get("workspace_clamped", False),
        "ik_available": ik_response.get("status") != "unavailable",
        "ik_succeeded": ik_response.get("status") == "success",
        "ik_returned_r_wrist_z": any(step.get("includes_r_wrist_z") for step in ik_response.get("positions", [])),
        "ik_returned_r_wrist_x": any(step.get("includes_r_wrist_x") for step in ik_response.get("positions", [])),
        "dropped_joints": filtering.get("dropped_joints", []),
        "final_joint_names": [
            step.get("final_joint_names", [])
            for step in final_command.get("command_steps_if_executed", [])
        ],
        "hand_action_sequence": final_command.get("hand_action_sequence", []),
        "xl320_wrist_ids_will_be_commanded": final_command.get("xl320_wrist_ids_31_33_will_be_commanded", False),
        "finger_ids_will_be_commanded": final_command.get("fingers_34_37_will_open_close", False),
        "real_robot_motion_commanded": False,
        "safety_note": "Trace only. No robot motion, no xl320 wrist/finger command, no SetJointPosition call.",
    }
    record["diagnosis"] = build_diagnosis(record)
    record["summary_text"] = summary_text(record)
    return log_record(record)


def summary_text(record: Dict[str, Any]) -> str:
    coord = record.get("coordinate_transfer_result", {})
    planned = record.get("planned_target_poses_before_ik", [])
    ik_response = record.get("ik_response", {})
    filtering = record.get("command_filtering", {})
    final_command = record.get("final_command", {})
    lines = [
        "IK Trace Dump, No Motion",
        f"Status: {record.get('status')}",
        f"Action: {record.get('action_type')}",
        f"Target: {record.get('target_object')}",
        f"Image bottom point: ({coord.get('image_x')}, {coord.get('image_y')})",
        f"Real target: x={coord.get('real_x')}, y={coord.get('real_y')}, z={coord.get('target_z')}",
        f"Workspace clamped: {coord.get('x_y_were_clamped')}",
        "Target poses sent/planned:",
    ]
    for pose in planned:
        sent = "sent to initial IK" if pose.get("sent_to_initial_ik_request") else "planned but not in initial IK"
        lines.append(
            f"- {pose.get('stage')}: pos={pose.get('position')}, "
            f"orientation={pose.get('orientation_quaternion')} ({pose.get('orientation_name')}), {sent}"
        )
    lines.extend(
        [
            f"IK status: {ik_response.get('status')} {ik_response.get('message', '')}".strip(),
            f"Captured template used as seed: {record.get('captured_template_used_as_seed')}",
            f"Seed template: {record.get('seed_template_name')}",
            f"Seed joints: {record.get('seed_joint_names', [])}",
            f"IK returned r_wrist_z: {record.get('ik_returned_r_wrist_z')}",
            f"IK returned r_wrist_x: {record.get('ik_returned_r_wrist_x')}",
            f"Dropped joints: {filtering.get('dropped_joints', [])}",
            f"Hand actions if executed: {final_command.get('hand_action_sequence', [])}",
            f"XL-320 wrist IDs 31/33 will be commanded by hand stage: {final_command.get('xl320_wrist_ids_31_33_will_be_commanded')}",
            f"Fingers 34-37 will open/close: {final_command.get('fingers_34_37_will_open_close')}",
            f"Diagnosis: {record.get('diagnosis')}",
            f"Log: {LATEST_TRACE_JSON}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump IK planning trace without commanding robot motion.")
    parser.add_argument("--action-type", default="grasp", choices=["grasp", "touch"])
    parser.add_argument("--target-object", default="red object")
    parser.add_argument("--image-x", type=float, default=None)
    parser.add_argument("--image-y", type=float, default=None)
    parser.add_argument("--real-x", type=float, default=None)
    parser.add_argument("--real-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=0.70)
    parser.add_argument("--seed-template", default="")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--use-detection", action="store_true")
    parser.add_argument("--use-coordinate", action="store_true")
    args = parser.parse_args()

    record = build_trace(args)
    print(json.dumps(_json_safe(record), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
