#!/usr/bin/env python3
"""Safe preset preview/debug helpers for right-arm action templates.

Execution uses only fixed right-arm presets through the existing nicoros
JointController service. There is no arbitrary joint-angle UI and no trajectory
generation in this module.
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kinematic_preview.action_templates import get_template
from kinematic_preview.live_joint_state_reader import ros_master_available
from kinematic_preview.nico_stick_model import NICOStickModel
from kinematic_preview.render_stick_nico import render_preview
from utils.constants import filter_commandable_joints
from utils.kinematic_trial_logger import find_api_root, preview_paths, write_json


PRESET_TO_TEMPLATE = {
    "reset_pose": "right_arm_reset_pose",
    "right_arm_point_pose": "right_arm_point_red_object",
    "right_arm_reach_pose": "right_arm_reach_forward",
    "right_arm_pre_grasp_pose": "right_arm_pre_grasp",
    "right_arm_touch_forward": "right_arm_touch_forward",
    "right_arm_touch_side": "right_arm_touch_side",
    "close_hand_pose": "right_arm_close_hand",
    "lift_pose": "right_arm_lift_pose",
    "open_right_hand": "right_arm_pre_grasp",
    "close_right_hand": "right_arm_close_hand",
}

ARM_SERVICE = "/right/open_manipulator_p/goal_joint_space_path"
IK_SERVICE = "/inverse_kinematics"
HAND_OPEN_TOPIC = "/nico/motion/openHand"
HAND_CLOSE_TOPIC = "/nico/motion/closeHand"
XL320_TOPIC = "/nico/motion/xl320_cmd"
XL320_RIGHT_HAND_MOTOR_IDS = [34, 35, 36, 37]
XL320_RIGHT_WRIST_POSITIONS_DEG = {
    31: 80.0,
    33: -45.0,
}
XL320_HAND_OPEN_DEG = -150.0
XL320_HAND_CLOSE_DEG = 80.0
XL320_HAND_CLOSE_DEG_BY_ID = {37: 130.0}
XL320_COMMAND_REPEATS = 4
XL320_COMMAND_INTERVAL_SEC = 0.04
COMMANDABLE_RIGHT_ARM_JOINTS = [
    "r_shoulder_z",
    "r_shoulder_y",
    "r_arm_x",
    "r_elbow_y",
    "r_wrist_z",
    "r_wrist_x",
]
REQUIRED_RIGHT_ARM_BASE_JOINTS = [
    "r_shoulder_z",
    "r_shoulder_y",
    "r_arm_x",
    "r_elbow_y",
]
RIGHT_ARM_IK_INITIAL_NAMES = [
    "r_shoulder_z",
    "r_shoulder_y",
    "r_arm_x",
    "r_elbow_y",
    "r_wrist_z",
    "r_wrist_x",
]
RIGHT_ARM_IK_INITIAL_POSITIONS = [-0.157, 0.0, -0.8203, -1.57, -1.39, 0.0]
XYZ_LIMITS = {
    "x": (0.10, 0.32),
    "y": (-0.25, 0.05),
    "z": (0.60, 0.78),
}

# Fixed conservative execution presets in radians, based around the existing
# ELMiRA right-arm default pose from state_machine.py. These are deliberately
# small preset poses, not user-editable trajectories.
EXECUTION_PRESETS_RAD = {
    "reset_pose": {
        "joint_names": list(COMMANDABLE_RIGHT_ARM_JOINTS),
        "positions": [-0.157, 0.0, -0.8203, -1.57, -1.39, 0.0],
        "path_time": 3.0,
        "description": "Return right arm to the known default/safe pose.",
    },
    "right_arm_point_pose": {
        "joint_names": list(COMMANDABLE_RIGHT_ARM_JOINTS),
        "positions": [-0.22, 0.0, -0.68, -1.20, -1.39, -0.35],
        "path_time": 3.0,
        "description": "Conservative right-arm point pose.",
    },
    "right_arm_reach_pose": {
        "joint_names": list(COMMANDABLE_RIGHT_ARM_JOINTS),
        "positions": [-0.20, 0.0, -0.58, -1.05, -1.39, -0.20],
        "path_time": 3.0,
        "description": "Conservative right-arm forward reach pose.",
    },
    "right_arm_pre_grasp_pose": {
        "joint_names": list(COMMANDABLE_RIGHT_ARM_JOINTS),
        "positions": [-0.24, 0.0, -0.64, -1.12, -1.39, -0.30],
        "path_time": 3.0,
        "description": "Conservative right-arm pre-grasp pose.",
    },
    "right_arm_touch_forward": {
        "joint_names": list(COMMANDABLE_RIGHT_ARM_JOINTS),
        "positions": [-0.20, 0.0, -0.56, -0.96, -1.39, -0.55],
        "path_time": 3.5,
        "description": "Experimental forward touch pose with wrist angled downward. Preview first; execute only with safety confirmation.",
    },
    "right_arm_touch_side": {
        "joint_names": list(COMMANDABLE_RIGHT_ARM_JOINTS),
        "positions": [-0.28, 0.0, -0.58, -1.00, -1.10, -0.45],
        "path_time": 3.5,
        "description": "Experimental side touch pose with wrist angled downward from robot-right side. Preview first; execute only with safety confirmation.",
    },
    "lift_pose": {
        "joint_names": list(COMMANDABLE_RIGHT_ARM_JOINTS),
        "positions": [-0.157, 0.0, -0.72, -1.36, -1.39, 0.0],
        "path_time": 3.0,
        "description": "Conservative right-arm lift/retreat pose.",
    },
}

HAND_PRESETS = {
    "open_right_hand": {
        "topic": XL320_TOPIC,
        "action": "open",
        "description": "Open physical right hand through XL-320 hand wrapper.",
    },
    "close_right_hand": {
        "topic": XL320_TOPIC,
        "action": "close",
        "description": "Close physical right hand through XL-320 hand wrapper.",
    },
}

CSV_FIELDS = [
    "timestamp",
    "mode",
    "preset_name",
    "arm_used",
    "hand_state",
    "safety_confirmed",
    "real_robot_motion_commanded",
    "execution_status",
    "failure_reason",
    "notes",
    "record_json",
]


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def append_preset_csv(record: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: _csv_value(record.get(field, "")) for field in CSV_FIELDS}
    row["record_json"] = json.dumps(record, sort_keys=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def log_preset_debug(record: Dict[str, Any]) -> Dict[str, Path]:
    paths = preview_paths(find_api_root(__file__))
    latest_json = paths["log_dir"] / "latest_preset_debug.json"
    trials_csv = paths["log_dir"] / "preset_debug_trials.csv"
    write_json(record, latest_json)
    append_preset_csv(record, trials_csv)
    return {"latest_json": latest_json, "trials_csv": trials_csv, "latest_image": paths["latest_image"]}


def _ros_unavailable_record(
    timestamp: str,
    mode: str,
    preset_name: str,
    template: Dict[str, Any],
    safety_confirmed: bool,
    reason: str,
) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "mode": mode,
        "preset_name": preset_name,
        "arm_used": "right",
        "hand_state": template.get("hand_state", "neutral"),
        "safety_confirmed": safety_confirmed,
        "real_robot_motion_commanded": False,
        "execution_status": "unavailable",
        "failure_reason": reason,
        "notes": "Start ROS/nicoros joint_controller before using direct preset execution.",
    }


def motion_setangle_subscriber_ready(timeout_s: float = 2.0) -> bool:
    """Return True only when Motion subscribes to /nico/motion/setAngle."""
    try:
        import rospy
        import nicomsg.msg
    except Exception:
        return False
    publisher = rospy.Publisher("/nico/motion/setAngle", nicomsg.msg.sff, queue_size=1)
    deadline = time.time() + timeout_s
    while time.time() < deadline and not rospy.is_shutdown():
        if publisher.get_num_connections() > 0:
            return True
        rospy.sleep(0.05)
    return False


def _deg_to_xl320_raw(deg: float) -> int:
    raw = int((float(deg) + 150.0) / 300.0 * 1023.0)
    return max(0, min(1023, raw))


def _ros_float_map_param(name: str, default: Dict[int, float]) -> Dict[int, float]:
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


def resolve_right_arm_execution_preset(preset_name: str, template: Dict[str, Any]) -> Dict[str, Any]:
    if preset_name in EXECUTION_PRESETS_RAD:
        return EXECUTION_PRESETS_RAD[preset_name]

    captured = template.get("commandable_right_arm_joint_positions_rad") or {}
    if not isinstance(captured, dict):
        captured = {}
    missing = [joint_name for joint_name in REQUIRED_RIGHT_ARM_BASE_JOINTS if joint_name not in captured]
    if missing:
        raise ValueError(
            "Captured template does not include required right-arm base joints: "
            + ", ".join(missing)
            + ". Capture a natural grasp template again while right-arm joint states are publishing."
        )
    joint_names = [
        joint_name for joint_name in COMMANDABLE_RIGHT_ARM_JOINTS if joint_name in captured
    ]
    return {
        "joint_names": joint_names,
        "positions": [float(captured[joint_name]) for joint_name in joint_names],
        "path_time": float(template.get("path_time", 3.0)),
        "description": f"Captured direct template: {template.get('description', preset_name)}",
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def clamp_xyz_target(x: float, y: float, z: float) -> Dict[str, Any]:
    clamped = {
        "x": _clamp(x, *XYZ_LIMITS["x"]),
        "y": _clamp(y, *XYZ_LIMITS["y"]),
        "z": _clamp(z, *XYZ_LIMITS["z"]),
    }
    return {
        "target": clamped,
        "was_clamped": any(abs(clamped[key] - float(value)) > 1e-9 for key, value in {"x": x, "y": y, "z": z}.items()),
        "limits": XYZ_LIMITS,
    }


def right_arm_initial_position():
    names = list(RIGHT_ARM_IK_INITIAL_NAMES)
    positions = list(RIGHT_ARM_IK_INITIAL_POSITIONS)
    try:
        import rospy
        from sensor_msgs.msg import JointState
        msg = rospy.wait_for_message("/right/open_manipulator_p/joint_states", JointState, timeout=1.5)
        current = {name: pos for name, pos in zip(msg.name, msg.position)}
        for idx, joint_name in enumerate(names):
            if joint_name in current:
                positions[idx] = float(current[joint_name])
    except Exception:
        pass
    return names, positions


def apply_orientation_mode(pose: Any, orientation_mode: str) -> str:
    """Set a named experimental end-effector orientation on a geometry Pose."""
    mode = str(orientation_mode or "point").strip().lower()
    if mode in ("point", "touch_forward"):
        # Forward/down orientation already used by ELMiRA's show action.
        pose.orientation.x = -1.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 0.0
        return mode
    if mode == "touch_side":
        # Candidate side-contact orientation. Physical execution still depends
        # on which wrist joints are exposed by the active motor config.
        pose.orientation.x = -0.7071068
        pose.orientation.y = 0.0
        pose.orientation.z = 0.7071068
        pose.orientation.w = 0.0
        return mode

    # Existing reach/manipulation orientation.
    pose.orientation.x = 0.7071068
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 0.7071068
    return "reach"


def execute_cartesian_xyz(
    x: float,
    y: float,
    z: float,
    orientation_mode: str,
    safety_confirmed: bool,
    path_time: float = 3.0,
) -> Dict[str, Any]:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    template = {"hand_state": "neutral"}
    if not safety_confirmed:
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": "cartesian_xyz",
            "arm_used": "right",
            "hand_state": "neutral",
            "safety_confirmed": False,
            "real_robot_motion_commanded": False,
            "execution_status": "blocked_by_safety",
            "failure_reason": "Safety checkbox not confirmed.",
            "notes": "Confirm the robot workspace is clear before Cartesian IK execution.",
        }
    if not ros_master_available(timeout_s=1.0):
        return _ros_unavailable_record(
            timestamp,
            "execute",
            "cartesian_xyz",
            template,
            safety_confirmed,
            "ROS master unavailable.",
        )

    try:
        import rospy
        from geometry_msgs.msg import Pose
        from elmira.msg import JointPosition as ElmiraJointPosition
        from elmira.srv import InverseKinematics, InverseKinematicsRequest
        from open_manipulator_msgs.msg import JointPosition
        from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
    except Exception as exc:
        return _ros_unavailable_record(
            timestamp,
            "execute",
            "cartesian_xyz",
            template,
            safety_confirmed,
            f"ROS/IK message modules unavailable: {exc}",
        )

    if not rospy.core.is_initialized():
        rospy.init_node("kinematic_cartesian_xyz_control", anonymous=True, disable_signals=True)

    try:
        rospy.wait_for_service(IK_SERVICE, timeout=3.0)
    except Exception as exc:
        return _ros_unavailable_record(
            timestamp,
            "execute",
            "cartesian_xyz",
            template,
            safety_confirmed,
            f"IK service unavailable: {exc}",
        )
    try:
        rospy.wait_for_service(ARM_SERVICE, timeout=3.0)
    except Exception as exc:
        return _ros_unavailable_record(
            timestamp,
            "execute",
            "cartesian_xyz",
            template,
            safety_confirmed,
            f"Right-arm joint service unavailable: {exc}",
        )
    if not motion_setangle_subscriber_ready(timeout_s=2.0):
        return _ros_unavailable_record(
            timestamp,
            "execute",
            "cartesian_xyz",
            template,
            safety_confirmed,
            "Motion node is not subscribed to /nico/motion/setAngle.",
        )

    target_info = clamp_xyz_target(x, y, z)
    target = target_info["target"]
    pose = Pose()
    pose.position.x = target["x"]
    pose.position.y = target["y"]
    pose.position.z = target["z"]
    applied_orientation_mode = apply_orientation_mode(pose, orientation_mode)

    initial_names, initial_positions = right_arm_initial_position()
    try:
        ik_request = InverseKinematicsRequest()
        ik_request.planning_group = "r_arm"
        ik_request.poses = [pose]
        ik_request.initial_position = ElmiraJointPosition()
        ik_request.initial_position.joint_name = initial_names
        ik_request.initial_position.position = initial_positions
        ik_response = rospy.ServiceProxy(IK_SERVICE, InverseKinematics)(ik_request)
        if not getattr(ik_response, "positions", None):
            raise RuntimeError("IK service returned no joint positions.")
        ik_joint_position = ik_response.positions[0]
        ik_values = {
            name: float(position)
            for name, position in zip(ik_joint_position.joint_name, ik_joint_position.position)
        }
        missing = [name for name in REQUIRED_RIGHT_ARM_BASE_JOINTS if name not in ik_values]
        if missing:
            raise RuntimeError("IK result missing required right-arm joints: " + ", ".join(missing))

        requested_names = [name for name in COMMANDABLE_RIGHT_ARM_JOINTS if name in ik_values]
        requested_positions = [ik_values[name] for name in requested_names]
        command_names, command_positions, dropped = filter_commandable_joints(
            "r_arm", requested_names, requested_positions
        )
        missing_required_after_filter = [
            name for name in REQUIRED_RIGHT_ARM_BASE_JOINTS if name not in command_names
        ]
        if missing_required_after_filter:
            raise RuntimeError(
                "Active Motion config filtered required right-arm joints: "
                + ", ".join(missing_required_after_filter)
            )

        arm_request = SetJointPositionRequest()
        arm_request.planning_group = "r_arm"
        arm_request.joint_position = JointPosition()
        arm_request.joint_position.joint_name = command_names
        arm_request.joint_position.position = command_positions
        arm_request.path_time = float(path_time)
        arm_response = rospy.ServiceProxy(ARM_SERVICE, SetJointPosition)(arm_request)
        success = bool(getattr(arm_response, "is_planned", True))
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": "cartesian_xyz",
            "arm_used": "right",
            "hand_state": "neutral",
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": success,
            "execution_status": "executed" if success else "failed",
            "failure_reason": "" if success else "Right-arm service returned failure.",
            "target_xyz": target,
            "target_was_clamped": target_info["was_clamped"],
            "orientation_mode": applied_orientation_mode,
            "joint_names": command_names,
            "joint_positions_rad": arm_request.joint_position.position,
            "joint_positions_deg": [round(math.degrees(value), 2) for value in arm_request.joint_position.position],
            "dropped_joints": dropped,
            "path_time": arm_request.path_time,
            "ros_service": ARM_SERVICE,
            "ik_service": IK_SERVICE,
            "notes": (
                "Experimental Cartesian IK control using constrained ELMiRA IK coordinates. "
                "If SR wrist IDs 23/25 are disabled, physical wrist alignment is handled separately by the XL-320 hand wrapper."
            ),
        }
    except Exception as exc:
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": "cartesian_xyz",
            "arm_used": "right",
            "hand_state": "neutral",
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": False,
            "execution_status": "failed",
            "failure_reason": str(exc),
            "target_xyz": target,
            "target_was_clamped": target_info["was_clamped"],
            "orientation_mode": orientation_mode,
            "notes": "Cartesian IK failed before confirmed motion.",
        }


def execute_right_arm_preset(preset_name: str, timestamp: str, template: Dict[str, Any], safety_confirmed: bool) -> Dict[str, Any]:
    try:
        preset = resolve_right_arm_execution_preset(preset_name, template)
    except Exception as exc:
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": preset_name,
            "arm_used": "right",
            "hand_state": template.get("hand_state", "neutral"),
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": False,
            "execution_status": "unavailable",
            "failure_reason": str(exc),
            "notes": "Only fixed presets or captured replay templates with raw right-arm joint positions are executable.",
        }
    if not ros_master_available(timeout_s=1.0):
        return _ros_unavailable_record(
            timestamp,
            "execute",
            preset_name,
            template,
            safety_confirmed,
            "ROS master unavailable.",
        )

    try:
        import rospy
        from open_manipulator_msgs.msg import JointPosition
        from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
    except Exception as exc:
        return _ros_unavailable_record(
            timestamp,
            "execute",
            preset_name,
            template,
            safety_confirmed,
            f"ROS message/service modules unavailable: {exc}",
        )

    if not rospy.core.is_initialized():
        rospy.init_node("kinematic_safe_preset_debug", anonymous=True, disable_signals=True)

    try:
        rospy.wait_for_service(ARM_SERVICE, timeout=3.0)
        if not motion_setangle_subscriber_ready(timeout_s=2.0):
            return _ros_unavailable_record(
                timestamp,
                "execute",
                preset_name,
                template,
                safety_confirmed,
                "Motion node is not subscribed to /nico/motion/setAngle. The joint service is up, but physical motor bridge is not ready.",
            )
        command_names, command_positions, dropped = filter_commandable_joints(
            "r_arm", preset["joint_names"], preset["positions"]
        )
        missing_required_after_filter = [
            name for name in REQUIRED_RIGHT_ARM_BASE_JOINTS if name not in command_names
        ]
        if missing_required_after_filter:
            return {
                "timestamp": timestamp,
                "mode": "execute",
                "preset_name": preset_name,
                "arm_used": "right",
                "hand_state": template.get("hand_state", "neutral"),
                "safety_confirmed": safety_confirmed,
                "real_robot_motion_commanded": False,
                "execution_status": "failed",
                "failure_reason": (
                    "Active Motion config filtered required right-arm joints: "
                    + ", ".join(missing_required_after_filter)
                ),
                "ros_service": ARM_SERVICE,
                "notes": "Right-arm preset blocked before confirmed execution.",
            }
        request = SetJointPositionRequest()
        request.planning_group = "r_arm"
        request.joint_position = JointPosition()
        request.joint_position.joint_name = command_names
        request.joint_position.position = command_positions
        request.path_time = float(preset.get("path_time", 3.0))
        proxy = rospy.ServiceProxy(ARM_SERVICE, SetJointPosition)
        response = proxy(request)
        success = bool(getattr(response, "is_planned", True))
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": preset_name,
            "arm_used": "right",
            "hand_state": template.get("hand_state", "neutral"),
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": success,
            "execution_status": "executed" if success else "failed",
            "failure_reason": "" if success else "Joint service returned failure.",
            "ros_service": ARM_SERVICE,
            "joint_names": command_names,
            "joint_positions_rad": command_positions,
            "joint_positions_deg": [round(math.degrees(value), 2) for value in command_positions],
            "dropped_joints": dropped,
            "path_time": request.path_time,
            "notes": preset.get("description", "Fixed right-arm preset executed."),
        }
    except Exception as exc:
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": preset_name,
            "arm_used": "right",
            "hand_state": template.get("hand_state", "neutral"),
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": False,
            "execution_status": "failed",
            "failure_reason": str(exc),
            "ros_service": ARM_SERVICE,
            "notes": "Right-arm preset service call failed before confirmed execution.",
        }


def execute_right_hand_preset(preset_name: str, timestamp: str, template: Dict[str, Any], safety_confirmed: bool) -> Dict[str, Any]:
    if preset_name not in HAND_PRESETS:
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": preset_name,
            "arm_used": "right",
            "hand_state": template.get("hand_state", "neutral"),
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": False,
            "execution_status": "unavailable",
            "failure_reason": "No fixed right-hand execution preset is defined for this button.",
            "notes": "Only open/close right-hand presets are executable.",
        }
    if not ros_master_available(timeout_s=1.0):
        return _ros_unavailable_record(
            timestamp,
            "execute",
            preset_name,
            template,
            safety_confirmed,
            "ROS master unavailable.",
        )

    try:
        import rospy
        import nicomsg.msg
    except Exception as exc:
        return _ros_unavailable_record(
            timestamp,
            "execute",
            preset_name,
            template,
            safety_confirmed,
            f"ROS/nicomsg modules unavailable: {exc}",
        )

    if not rospy.core.is_initialized():
        rospy.init_node("kinematic_safe_preset_debug", anonymous=True, disable_signals=True)

    preset = HAND_PRESETS[preset_name]
    try:
        publisher = rospy.Publisher(preset["topic"], nicomsg.msg.sff, queue_size=1)
        deadline = time.time() + 2.0
        while publisher.get_num_connections() == 0 and time.time() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.05)
        if publisher.get_num_connections() == 0:
            return {
                "timestamp": timestamp,
                "mode": "execute",
                "preset_name": preset_name,
                "arm_used": "right",
                "hand_state": "open" if "open" in preset_name else "close",
                "safety_confirmed": safety_confirmed,
                "real_robot_motion_commanded": False,
                "execution_status": "unavailable",
                "failure_reason": "Motion hand topic has no subscriber.",
                "ros_topic": preset["topic"],
                "notes": "Start physical direct control and wait until Motion is ready.",
            }
        action = preset.get("action", "open")
        close_by_id = _ros_float_map_param(
            "/elmira/right_hand_close_deg_by_id",
            XL320_HAND_CLOSE_DEG_BY_ID,
        )
        repeat_count = max(1, int(rospy.get_param("/elmira/right_xl320_command_repeats", XL320_COMMAND_REPEATS)))
        interval_sec = max(
            0.005,
            float(rospy.get_param("/elmira/right_xl320_command_interval_sec", XL320_COMMAND_INTERVAL_SEC)),
        )

        def publish_command(motor_id, register, value, repeats):
            for _ in range(repeats):
                msg = nicomsg.msg.sff()
                msg.param1 = str(motor_id)
                msg.param2 = float(register)
                msg.param3 = float(value)
                publisher.publish(msg)
                rospy.sleep(interval_sec)

        all_motor_ids = sorted(set(list(XL320_RIGHT_WRIST_POSITIONS_DEG.keys()) + XL320_RIGHT_HAND_MOTOR_IDS))
        for motor_id in all_motor_ids:
            publish_command(motor_id, 24.0, 1.0, min(2, repeat_count))
        for motor_id in all_motor_ids:
            publish_command(motor_id, 32.0, 150.0, min(2, repeat_count))
        for motor_id, wrist_deg in XL320_RIGHT_WRIST_POSITIONS_DEG.items():
            publish_command(motor_id, 30.0, float(_deg_to_xl320_raw(wrist_deg)), repeat_count)
        target_by_motor = {}
        for motor_id in XL320_RIGHT_HAND_MOTOR_IDS:
            target_deg = (
                XL320_HAND_OPEN_DEG
                if action == "open"
                else close_by_id.get(motor_id, XL320_HAND_CLOSE_DEG)
            )
            target_raw = _deg_to_xl320_raw(target_deg)
            publish_command(motor_id, 30.0, float(target_raw), repeat_count)
            target_by_motor[motor_id] = {"deg": target_deg, "raw": target_raw}
        rospy.sleep(0.2)
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": preset_name,
            "arm_used": "right",
            "hand_state": action,
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": True,
            "execution_status": "executed",
            "failure_reason": "",
            "ros_topic": preset["topic"],
            "xl320_motor_ids": all_motor_ids,
            "finger_motor_ids": XL320_RIGHT_HAND_MOTOR_IDS,
            "target_by_motor": target_by_motor,
            "command_repeats": repeat_count,
            "notes": preset.get("description", "Fixed right-hand preset published through Motion wrapper."),
        }
    except Exception as exc:
        return {
            "timestamp": timestamp,
            "mode": "execute",
            "preset_name": preset_name,
            "arm_used": "right",
            "hand_state": template.get("hand_state", "neutral"),
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": False,
            "execution_status": "failed",
            "failure_reason": str(exc),
            "notes": "Right-hand preset publish failed before confirmed execution.",
        }


def handle_preset(preset_name: str, mode: str, safety_confirmed: bool) -> Dict[str, Any]:
    template_name = PRESET_TO_TEMPLATE.get(preset_name, preset_name)
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    try:
        template = get_template(template_name)
    except Exception as exc:
        record = {
            "timestamp": timestamp,
            "mode": mode,
            "preset_name": preset_name,
            "arm_used": "right",
            "hand_state": "unknown",
            "safety_confirmed": safety_confirmed,
            "real_robot_motion_commanded": False,
            "execution_status": "failed",
            "failure_reason": str(exc),
            "notes": "Preset template not found.",
        }
        log_preset_debug(record)
        return record

    if mode == "execute":
        if not safety_confirmed:
            record = {
                "timestamp": timestamp,
                "mode": mode,
                "preset_name": preset_name,
                "arm_used": "right",
                "hand_state": template.get("hand_state", "neutral"),
                "safety_confirmed": safety_confirmed,
                "real_robot_motion_commanded": False,
                "execution_status": "blocked_by_safety",
                "failure_reason": "Safety checkbox not confirmed.",
                "notes": "Confirm the robot workspace is clear before direct preset execution.",
            }
        elif preset_name in HAND_PRESETS:
            record = execute_right_hand_preset(preset_name, timestamp, template, safety_confirmed)
        else:
            record = execute_right_arm_preset(preset_name, timestamp, template, safety_confirmed)
        paths = log_preset_debug(record)
        record["latest_json_path"] = str(paths["latest_json"])
        record["trials_csv_path"] = str(paths["trials_csv"])
        return record

    model = NICOStickModel()
    preview = model.forward_kinematics(template.get("joint_angles_deg", {}))
    paths = preview_paths(find_api_root(__file__))
    image_path = paths["log_dir"] / "latest_preset_preview.png"
    render_preview(
        preview,
        image_path,
        target_object=template.get("target_object", "target object"),
        target_object_xyz=template.get("target_object_xyz"),
        template_name=template.get("name", template_name),
    )
    record = {
        "timestamp": timestamp,
        "mode": "preview",
        "preset_name": preset_name,
        "template_name": template_name,
        "arm_used": "right",
        "hand_state": template.get("hand_state", "neutral"),
        "safety_confirmed": safety_confirmed,
        "real_robot_motion_commanded": False,
        "execution_status": "preview_only",
        "failure_reason": "",
        "estimated_right_hand_xyz": preview.get("estimated_right_hand_xyz", {}),
        "preview_image_path": str(image_path),
        "notes": "Preview only. No robot motion was commanded.",
    }
    paths = log_preset_debug(record)
    record["latest_json_path"] = str(paths["latest_json"])
    record["trials_csv_path"] = str(paths["trials_csv"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview or log safe right-arm preset debug actions.")
    parser.add_argument("--mode", choices=["preview", "execute"], required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--safety-confirmed", action="store_true")
    parser.add_argument("--cartesian", action="store_true", help="Execute constrained Cartesian IK target")
    parser.add_argument("--x", type=float, default=0.20)
    parser.add_argument("--y", type=float, default=-0.20)
    parser.add_argument("--z", type=float, default=0.70)
    parser.add_argument(
        "--orientation-mode",
        choices=["point", "reach", "touch_forward", "touch_side"],
        default="point",
    )
    parser.add_argument("--path-time", type=float, default=3.0)
    args = parser.parse_args()
    if args.cartesian:
        record = execute_cartesian_xyz(
            args.x,
            args.y,
            args.z,
            args.orientation_mode,
            args.safety_confirmed,
            args.path_time,
        )
        paths = log_preset_debug(record)
        record["latest_json_path"] = str(paths["latest_json"])
        record["trials_csv_path"] = str(paths["trials_csv"])
    else:
        record = handle_preset(args.preset, args.mode, args.safety_confirmed)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
