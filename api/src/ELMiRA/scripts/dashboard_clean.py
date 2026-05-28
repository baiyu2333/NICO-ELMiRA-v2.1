#!/usr/bin/env python3
"""Clean AP2 monitoring dashboard for the ELMiRA FYP2 demo."""

import atexit
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gradio as gr

    GRADIO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when Gradio is absent.
    gr = None
    GRADIO_IMPORT_ERROR = exc

try:
    import psutil
except Exception:  # pragma: no cover - psutil is available in the target env.
    psutil = None

from utils.fyp2_trial_logger import find_api_root, monitoring_paths


API_ROOT = find_api_root(__file__)
PATHS = monitoring_paths(API_ROOT)
LOG_NODES_PATH = API_ROOT / "launch_nodes.log"
LOG_SM_PATH = API_ROOT / "launch_sm.log"
LOG_DIRECT_CONTROL_PATH = API_ROOT / "logs" / "direct_robot_control.log"
LOG_IK_CONTROL_PATH = API_ROOT / "logs" / "direct_ik_solver.log"
PROBE_SCRIPT = API_ROOT / "src" / "ELMiRA" / "scripts" / "fyp2_njf_demo_probe.py"
KINEMATIC_PREVIEW_SCRIPT = (
    API_ROOT / "src" / "ELMiRA" / "scripts" / "kinematic_preview" / "preview_action.py"
)
LIVE_JOINT_READER_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "live_joint_state_reader.py"
)
JOINT_RECORDER_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "joint_trajectory_recorder.py"
)
REPLAY_JOINT_TRAJECTORY_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "replay_joint_trajectory.py"
)
SAVE_REPLAY_TEMPLATE_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "save_replay_frame_as_template.py"
)
CAPTURE_NATURAL_GRASP_TEMPLATE_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "capture_natural_grasp_template.py"
)
SAFE_PRESET_DEBUG_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "safe_preset_debug.py"
)
FIRST_PERSON_TRIAL_RECORDER_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "first_person_trial_recorder.py"
)
GRASP_DEBUG_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "grasp_debug.py"
)
IK_TRACE_DUMP_SCRIPT = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "ik_trace_dump.py"
)
KINEMATIC_TEMPLATE_FILE = (
    API_ROOT
    / "src"
    / "ELMiRA"
    / "scripts"
    / "kinematic_preview"
    / "templates"
    / "right_arm_templates.json"
)
KINEMATIC_CAPTURED_TEMPLATE_FILE = KINEMATIC_TEMPLATE_FILE.parent / "captured_right_arm_templates.json"
CAPTURED_GRASP_TEMPLATE_FILE = KINEMATIC_TEMPLATE_FILE.parent / "captured_grasp_templates.json"
KINEMATIC_LOG_DIR = API_ROOT / "logs" / "kinematic_preview"
FIRST_PERSON_LOG_DIR = API_ROOT / "logs" / "first_person_trials"
GRASP_DEBUG_LOG_DIR = API_ROOT / "logs" / "grasp_debug"
IK_TRACE_LOG_DIR = API_ROOT / "logs" / "ik_trace"
GRASP_TEMPLATE_LOG_DIR = API_ROOT / "logs" / "grasp_templates"

CHAT_UNAVAILABLE = (
    "Chat/action service unavailable; use NJF dry-run and robot evidence fallback."
)
ASR_UNAVAILABLE = "ASR unavailable in this environment"

launch_process_nodes: Optional[subprocess.Popen] = None
launch_process_sm: Optional[subprocess.Popen] = None
direct_control_process: Optional[subprocess.Popen] = None
direct_control_log_handle: Optional[Any] = None
direct_ik_process: Optional[subprocess.Popen] = None
direct_ik_log_handle: Optional[Any] = None
joint_record_process: Optional[subprocess.Popen] = None
joint_record_log_handle: Optional[Any] = None
first_person_record_process: Optional[subprocess.Popen] = None
first_person_record_log_handle: Optional[Any] = None
log_file_handles: List[Any] = []
runtime_secrets = set()
runtime_api_config = {"provider": "", "api_key": "", "env_var": ""}
launch_attempted = False
last_launch_started_at: Optional[float] = None
last_system_message = "Ready. Launch ELMiRA before sending robot commands."

recording_state = {
    "launch_attempted": False,
    "njf_dry_run_generated": False,
    "logs_generated": False,
    "robot_evidence_recorded": False,
    "stop_attempted": False,
}


def fallback_commands() -> str:
    return f"""Gradio dashboard is unavailable. Use terminal fallback commands:

cd {API_ROOT}
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode dry-run
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode ros-check
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode robot-evidence
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode summary
"""


def sanitize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    secret_names = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
    for name in secret_names:
        secret = os.environ.get(name, "")
        if secret:
            text = text.replace(secret, "[MASKED]")
    for secret in list(runtime_secrets):
        if secret:
            text = text.replace(secret, "[MASKED]")
    text = re.sub(r"sk-[A-Za-z0-9_\-]{12,}", "[MASKED]", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[MASKED]", text)
    text = re.sub(
        r"(OPENAI_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY)=['\"]?[^\\s'\"]+",
        r"\1=[MASKED]",
        text,
    )
    return text


def provider_env_var(provider: str) -> str:
    """Match the existing dashboard.py launch convention."""
    return "OPENAI_API_KEY" if str(provider).lower() == "openai" else "GOOGLE_API_KEY"


def mask_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        return "[not saved]"
    suffix = key[-4:] if len(key) >= 4 else "****"
    if key.startswith("sk-"):
        return f"sk-...{suffix}"
    return f"...{suffix}"


def save_runtime_api_key(provider: str, api_key: str) -> Dict[str, str]:
    key = (api_key or "").strip()
    selected_provider = provider or "OpenAI"
    if not key:
        return {"provider": selected_provider, "api_key": "", "env_var": ""}

    env_var = provider_env_var(selected_provider)
    runtime_secrets.add(key)
    runtime_api_config.update(
        {"provider": selected_provider, "api_key": key, "env_var": env_var}
    )
    os.environ[env_var] = key
    return dict(runtime_api_config)


def saved_key_message(config: Optional[Dict[str, str]]) -> str:
    config = config or {}
    provider = config.get("provider") or "OpenAI"
    api_key = config.get("api_key") or ""
    if not api_key:
        return "No API key saved."
    return f"API key saved for {provider}: {mask_api_key(api_key)}"


def save_api_key_handler(provider: str, api_key: str):
    config = save_runtime_api_key(provider, api_key)
    if not config.get("api_key"):
        return config, "No API key saved. Enter a key first.", gr.update(value="")
    return config, saved_key_message(config), gr.update(value="")


def clear_api_key_handler(provider: str):
    selected_provider = provider or runtime_api_config.get("provider") or "OpenAI"
    env_var = provider_env_var(selected_provider)
    runtime_api_config.update({"provider": selected_provider, "api_key": "", "env_var": ""})
    os.environ.pop(env_var, None)
    return dict(runtime_api_config), "No API key saved.", gr.update(value="")


def resolve_launch_api_key(
    provider: str,
    current_api_key: str,
    saved_api_key_state: Optional[Dict[str, str]],
) -> Tuple[str, Dict[str, str], str]:
    """Use textbox key if present, otherwise use saved runtime state."""
    if (current_api_key or "").strip():
        config = save_runtime_api_key(provider, current_api_key)
        return config["api_key"], config, saved_key_message(config)

    state = saved_api_key_state or {}
    saved_key = (state.get("api_key") or "").strip()
    if saved_key:
        config = save_runtime_api_key(state.get("provider") or provider, saved_key)
        return saved_key, config, saved_key_message(config)

    return "", {"provider": provider or "OpenAI", "api_key": "", "env_var": ""}, (
        "No API key saved. Please enter and save your API key first."
    )


def run_env_command(command: str, timeout_s: int = 5) -> Dict[str, Any]:
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
            "stdout": sanitize_text(completed.stdout.strip()),
            "stderr": sanitize_text(completed.stderr.strip()),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "returncode": None,
            "stdout": sanitize_text(stdout.strip()),
            "stderr": sanitize_text(stderr.strip()),
            "timeout": True,
        }
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc), "timeout": False}


def read_text_safe(path: Path, default: str = "Not available yet.") -> str:
    try:
        if Path(path).exists():
            return sanitize_text(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Unable to read {path}: {exc}"
    return default


def tail_file(path: Path, lines: int = 40) -> str:
    try:
        if not path.exists():
            return "Waiting for log file..."
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            content = handle.readlines()[-lines:]
        return sanitize_text("".join(content)) or "Log file exists but is empty."
    except Exception as exc:
        return f"Unable to read log: {exc}"


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def process_alive(process: Optional[subprocess.Popen]) -> bool:
    return process is not None and process.poll() is None


def check_ros_status() -> str:
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["name"]):
                if proc.info.get("name") in ("roscore", "rosmaster"):
                    return "Running"
        except Exception:
            pass
    result = run_env_command("rosnode list", timeout_s=2)
    if result["returncode"] == 0:
        return "Running"
    return "Stopped"


def elmira_pipeline_status() -> str:
    nodes_alive = process_alive(launch_process_nodes)
    sm_alive = process_alive(launch_process_sm)
    if nodes_alive and sm_alive:
        return "Running"
    if nodes_alive or sm_alive:
        return "Partial"
    return "Stopped"


def latest_njf_status() -> str:
    record = load_json_safe(PATHS["latest_njf_probe"])
    if not record:
        return "Not run"
    return str(record.get("status", "unknown"))


def checklist_markdown(physical_nico_shown: bool = False) -> str:
    items = [
        ("Physical NICO shown", physical_nico_shown),
        ("Launch attempted", recording_state["launch_attempted"]),
        ("NJF dry-run generated", recording_state["njf_dry_run_generated"]),
        ("Logs generated", recording_state["logs_generated"]),
        ("Robot evidence / ROS check recorded", recording_state["robot_evidence_recorded"]),
        ("Stop attempted", recording_state["stop_attempted"]),
    ]
    lines = ["### Recording Checklist"]
    for label, done in items:
        mark = "[x]" if done else "[ ]"
        lines.append(f"- {mark} {label}")
    return "\n".join(lines)


def demo_mode_text() -> str:
    return (
        "Real Robot Mode: Launch/Stop and existing chat/action pipeline may use robot.\n"
        "NJF Probe Mode: dry-run, ROS check, service check, and robot evidence check command no motion."
    )


def status_values(dummy_mode: bool) -> Tuple[str, str, str, str, str, str]:
    robot_mode = "Dummy" if dummy_mode else "Real Robot"
    return (
        f"ROS Core: {check_ros_status()}",
        f"ELMiRA Pipeline: {elmira_pipeline_status()}",
        f"Robot Mode: {robot_mode}",
        f"NJF Probe: {latest_njf_status()}",
        "Safety: No unsafe motion commanded by NJF probe",
        demo_mode_text(),
    )


def parse_mic_device(value: str) -> str:
    return (value or "").strip()


def launch_robot(
    provider: str,
    use_mllm_grounding: bool,
    use_dummy: bool,
    api_key: str,
    mic_device: str = "",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.03,
) -> str:
    global launch_process_nodes, launch_process_sm, launch_attempted, last_launch_started_at

    launch_attempted = True
    recording_state["launch_attempted"] = True

    if process_alive(launch_process_nodes) or process_alive(launch_process_sm):
        return (
            "launch already running\n"
            f"provider selected: {provider}\n"
            f"dummy mode: {str(use_dummy).lower()}\n"
            f"nodes log: {LOG_NODES_PATH}\n"
            f"state machine log: {LOG_SM_PATH}\n"
            "API key: [MASKED]"
        )

    if not api_key:
        return (
            "launch not started: API key is required for the selected provider\n"
            f"provider selected: {provider}\n"
            f"dummy mode: {str(use_dummy).lower()}\n"
            "API key: [MASKED]"
        )

    runtime_secrets.add(api_key)
    key_var = provider_env_var(provider)
    lower_provider = str(provider).lower()
    mic_device = parse_mic_device(mic_device)
    launch_env = os.environ.copy()
    launch_env[key_var] = api_key

    launch_prefix = (
        "source activate.bash 2>/dev/null || true; "
        "source devel/setup.bash 2>/dev/null || true; "
        "export CUDA_VISIBLE_DEVICES=''; "
        "export PYTHONUNBUFFERED=1; "
    )
    cmd_nodes = (
        launch_prefix
        + "roslaunch elmira init_nodes_v2.launch "
        + f"mllm_provider:={shlex.quote(lower_provider)} "
        + f"use_mllm_grounding:={'true' if use_mllm_grounding else 'false'} "
        + f"dummy:={'true' if use_dummy else 'false'} "
        + f"mic_device:={shlex.quote(mic_device)} "
        + f"offset_x:={float(offset_x)} "
        + f"offset_y:={float(offset_y)} "
        + f"offset_z:={float(offset_z)}"
    )
    cmd_sm = (
        "source activate.bash 2>/dev/null || true; "
        "source devel/setup.bash 2>/dev/null || true; "
        "export PYTHONUNBUFFERED=1; "
        "rosrun elmira state_machine.py"
    )

    try:
        LOG_NODES_PATH.parent.mkdir(parents=True, exist_ok=True)
        nodes_log = LOG_NODES_PATH.open("w", encoding="utf-8", buffering=1)
        sm_log = LOG_SM_PATH.open("w", encoding="utf-8", buffering=1)
        log_file_handles.extend([nodes_log, sm_log])

        launch_process_nodes = subprocess.Popen(
            ["bash", "-lc", cmd_nodes],
            stdout=nodes_log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(API_ROOT),
            env=launch_env,
            preexec_fn=os.setsid,
        )

        def delayed_state_machine_start() -> None:
            global launch_process_sm
            time.sleep(15)
            if not process_alive(launch_process_nodes):
                return
            launch_process_sm = subprocess.Popen(
                ["bash", "-lc", cmd_sm],
                stdout=sm_log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(API_ROOT),
                env=launch_env,
                preexec_fn=os.setsid,
            )

        threading.Thread(target=delayed_state_machine_start, daemon=True).start()
        last_launch_started_at = time.time()

        return (
            "launch started\n"
            f"provider selected: {provider}\n"
            f"dummy mode: {str(use_dummy).lower()}\n"
            f"nodes/status: nodes process pid {launch_process_nodes.pid}; state machine starts after node warmup\n"
            f"nodes log: {LOG_NODES_PATH}\n"
            f"state machine log: {LOG_SM_PATH}\n"
            "API key: [MASKED]"
        )
    except Exception as exc:
        return f"failed: launch could not start: {sanitize_text(exc)}"


def stop_process_group(process: Optional[subprocess.Popen], name: str) -> Tuple[str, bool]:
    if process is None:
        return f"{name}: not started", True
    if process.poll() is not None:
        return f"{name}: already stopped", True
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=4)
        return f"{name}: stopped", True
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=2)
            return f"{name}: killed after timeout", True
        except Exception as exc:
            return f"{name}: failed to stop ({exc})", False


def cleanup_known_elmira_processes() -> Tuple[str, bool]:
    """Stop ELMiRA ROS processes that may outlive this dashboard instance."""
    patterns = [
        "roslaunch elmira init_nodes_v2.launch",
        "rosrun elmira state_machine.py",
        str(API_ROOT / "src" / "ELMiRA" / "scripts" / "state_machine.py"),
        str(API_ROOT / "src" / "ELMiRA" / "scripts" / "speech_asr.py"),
        str(API_ROOT / "src" / "ELMiRA" / "scripts" / "v2" / "llm_api_v2.py"),
        str(API_ROOT / "src" / "ELMiRA" / "scripts" / "coordinate_transfer.py"),
        str(API_ROOT / "src" / "ELMiRA" / "scripts" / "ik_solver.py"),
        str(API_ROOT / "src" / "ELMiRA" / "scripts" / "njf_predict_action_server.py"),
        str(API_ROOT / "src" / "nicoros" / "scripts" / "Motion.py"),
        str(API_ROOT / "src" / "nicoros" / "scripts" / "JointController.py"),
        str(API_ROOT / "src" / "nicoros" / "scripts" / "TextToSpeech.py"),
        "__name:=nicovision",
    ]
    current_pgid = os.getpgrp()
    pgids = set()
    try:
        ps_result = subprocess.run(
            ["ps", "-eo", "pid=,pgid=,args="],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return f"cleanup scan failed: {exc}", False

    for line in ps_result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        _pid_s, pgid_s, args = parts
        if "dashboard_clean.py" in args:
            continue
        if any(pattern in args for pattern in patterns):
            try:
                pgid = int(pgid_s)
            except ValueError:
                continue
            if pgid != current_pgid:
                pgids.add(pgid)

    if not pgids:
        return "known ELMiRA nodes: none found", True

    failures = []
    for pgid in sorted(pgids):
        try:
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except Exception as exc:
            failures.append(f"{pgid}: SIGINT failed ({exc})")
    time.sleep(1.0)
    for pgid in sorted(pgids):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as exc:
            failures.append(f"{pgid}: SIGKILL failed ({exc})")

    if failures:
        return "known ELMiRA nodes: partial cleanup; " + "; ".join(failures), False
    return f"known ELMiRA nodes: stopped {len(pgids)} process groups", True


def stop_robot() -> str:
    global launch_process_nodes, launch_process_sm
    recording_state["stop_attempted"] = True
    messages = []
    ok_flags = []

    msg, ok = stop_process_group(launch_process_sm, "state machine")
    messages.append(msg)
    ok_flags.append(ok)
    msg, ok = stop_process_group(launch_process_nodes, "nodes")
    messages.append(msg)
    ok_flags.append(ok)

    msg, ok = cleanup_known_elmira_processes()
    messages.append(msg)
    ok_flags.append(ok)

    launch_process_sm = None
    launch_process_nodes = None
    for handle in list(log_file_handles):
        try:
            handle.close()
        except Exception:
            pass
    log_file_handles.clear()

    if all(ok_flags):
        label = "stopped"
    elif any(ok_flags):
        label = "partial"
    else:
        label = "failed"
    return label + "\n" + "\n".join(messages)


def refresh_status(dummy_mode: bool, physical_nico_shown: bool):
    return (*status_values(dummy_mode), checklist_markdown(physical_nico_shown))


def refresh_logs():
    return (
        tail_file(LOG_NODES_PATH),
        tail_file(LOG_SM_PATH),
        read_text_safe(PATHS["latest_njf_probe"], "No NJF dry-run JSON yet."),
        read_text_safe(PATHS["monitoring_summary"], "No monitoring summary yet."),
    )


def extract_chat_response(output: str) -> str:
    output = sanitize_text(output.strip())
    text_match = re.search(r"[\'\"]text[\'\"]\s*:\s*([\'\"])(.*?)\1", output, re.DOTALL)
    if text_match:
        return text_match.group(2).strip()
    response_match = re.search(r"response:\s*([\'\"]?)(.*)\1", output, re.DOTALL)
    if response_match:
        return response_match.group(2).strip().replace("\\n", "\n")
    return output[:600] if output else CHAT_UNAVAILABLE


def chat_interface(message: str, transcript: str):
    message = (message or "").strip()
    transcript = transcript or ""
    if not message:
        return transcript, ""
    if not launch_attempted:
        return (
            transcript
            + "\nROBOT: Launch ELMiRA first for real robot chat/action, or use NJF dry-run fallback.\n",
            "",
        )

    request = "prompt: " + json.dumps(message)
    result = run_env_command("rosservice call /mllm_chat " + shlex.quote(request), timeout_s=12)
    if result["timeout"] or result["returncode"] != 0:
        reply = CHAT_UNAVAILABLE
    else:
        reply = extract_chat_response(result.get("stdout", ""))
        if not reply or reply.startswith("ERROR"):
            reply = CHAT_UNAVAILABLE

    new_transcript = (transcript.rstrip() + f"\nUSER: {message}\nROBOT: {reply}\n").strip()
    lines = new_transcript.splitlines()
    if len(lines) > 60:
        lines = lines[-60:]
    return "\n".join(lines), ""


def listen_audio():
    topics = run_env_command("rostopic list", timeout_s=3)
    if topics["returncode"] != 0 or "/speech_asr/goal" not in topics.get("stdout", ""):
        return ASR_UNAVAILABLE
    command = (
        "rostopic pub /speech_asr/goal elmira/PerformASRActionGoal "
        "\"{header: {seq: 0, stamp: now, frame_id: ''}, "
        "goal_id: {stamp: now, id: ''}, "
        "goal: {detect_start: true, detect_stop: true, start_timeout: 5.0, "
        "min_duration: 1.0, max_duration: 10.0, min_period: 0.5, live_text: true}}\" -1"
    )
    result = run_env_command(command, timeout_s=5)
    if result["returncode"] == 0:
        return "ASR listen request sent. If no text appears, use typed command or NJF fallback."
    return ASR_UNAVAILABLE


def stop_listening():
    topics = run_env_command("rostopic list", timeout_s=3)
    if topics["returncode"] != 0 or "/speech_asr/cancel" not in topics.get("stdout", ""):
        return ASR_UNAVAILABLE
    result = run_env_command(
        'rostopic pub /speech_asr/cancel actionlib_msgs/GoalID "{}" -1',
        timeout_s=5,
    )
    if result["returncode"] == 0:
        return "ASR stop request sent."
    return ASR_UNAVAILABLE


def read_latest_json_for_display() -> str:
    return read_text_safe(PATHS["latest_njf_probe"], "No NJF dry-run JSON yet.")


def latest_csv_path() -> str:
    path = PATHS["latest_csv"]
    if path.exists():
        return str(path)
    return f"No CSV log yet. Expected path: {path}"


def evidence_summary() -> str:
    probe = load_json_safe(PATHS["latest_njf_probe"])
    ros_check = load_json_safe(PATHS["latest_ros_check"])
    robot = load_json_safe(PATHS["latest_robot_evidence"])
    service = load_json_safe(PATHS["latest_njf_service_check"])

    if not probe:
        return (
            "### FYP2 Manipulation Evaluation / NJF Probe\n"
            "Latest probe status: Not run\n\n"
            "Run NJF Dry-run Probe to generate JSON, CSV, and Markdown evidence."
        )

    lines = [
        "### FYP2 Manipulation Evaluation / NJF Probe",
        f"Latest probe status: {probe.get('status', 'unknown')}",
        f"Implementation level: {probe.get('implementation_level', 'not recorded')}",
        f"Target object: {probe.get('target_object', 'not recorded')}",
        f"Input visual error: `{json.dumps(probe.get('input_visual_error', {}), sort_keys=True)}`",
        f"Predicted bounded correction: `{json.dumps(probe.get('predicted_correction', {}), sort_keys=True)}`",
        f"Evaluation metrics: {', '.join(probe.get('evaluation_metrics', []))}",
        f"Success criteria: {', '.join(probe.get('success_criteria', []))}",
        f"AI testing status: {probe.get('ai_testing_status', 'not recorded')}",
        f"Limitation: {probe.get('limitation', 'not recorded')}",
        f"Next action: {probe.get('next_action', 'not recorded')}",
        f"ROS check: {(ros_check or {}).get('status', 'not recorded')}",
        f"Robot evidence: {(robot or {}).get('status', 'not recorded')}",
        f"Service check: {(service or {}).get('status', 'optional / not recorded')}",
        f"JSON log path: {PATHS['latest_njf_probe']}",
        f"CSV log path: {PATHS['latest_csv']}",
        f"Markdown summary path: {PATHS['monitoring_summary']}",
    ]
    return "\n\n".join(lines)


def run_probe_mode(mode: str, physical_nico_shown: bool):
    timeout_s = 8
    try:
        completed = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), "--mode", mode],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        if completed.returncode == 0:
            status_text = f"{mode}: completed"
        else:
            status_text = f"{mode}: failed\n{output[-1200:]}"
    except subprocess.TimeoutExpired:
        status_text = f"{mode}: timeout after {timeout_s} seconds"
    except Exception as exc:
        status_text = f"{mode}: failed ({exc})"

    if mode == "dry-run":
        recording_state["njf_dry_run_generated"] = True
        recording_state["logs_generated"] = True
    if mode in ("ros-check", "robot-evidence"):
        recording_state["robot_evidence_recorded"] = True
        recording_state["logs_generated"] = True
    if mode == "summary":
        recording_state["logs_generated"] = True

    nodes_tail, sm_tail, latest_json, summary = refresh_logs()
    return (
        status_text,
        evidence_summary(),
        latest_json,
        latest_csv_path(),
        summary,
        checklist_markdown(physical_nico_shown),
        f"NJF Probe: {latest_njf_status()}",
        nodes_tail,
        sm_tail,
    )


def show_latest_outputs(physical_nico_shown: bool):
    nodes_tail, sm_tail, latest_json, summary = refresh_logs()
    return (
        evidence_summary(),
        latest_json,
        latest_csv_path(),
        summary,
        checklist_markdown(physical_nico_shown),
        nodes_tail,
        sm_tail,
    )


def launch_button_handler(
    provider,
    api_key,
    offset_x,
    offset_y,
    offset_z,
    dummy_mode,
    use_mllm_grounding,
    mic_device,
    physical_nico_shown,
):
    status = launch_robot(
        provider,
        use_mllm_grounding,
        dummy_mode,
        api_key,
        mic_device,
        offset_x,
        offset_y,
        offset_z,
    )
    nodes_tail, sm_tail, _, _ = refresh_logs()
    return (
        status,
        gr.update(interactive=True),
        *status_values(dummy_mode),
        checklist_markdown(physical_nico_shown),
        nodes_tail,
        sm_tail,
    )


def stop_button_handler(dummy_mode, physical_nico_shown):
    status = stop_robot()
    nodes_tail, sm_tail, _, _ = refresh_logs()
    return (
        status,
        *status_values(dummy_mode),
        checklist_markdown(physical_nico_shown),
        nodes_tail,
        sm_tail,
    )


CLEAN_DASHBOARD_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}
.main-title {
    padding: 18px 0 4px 0;
}
.main-title h1 {
    margin: 0;
    font-size: 34px;
    line-height: 1.1;
    letter-spacing: 0;
}
.main-title p {
    margin: 8px 0 0 0;
    color: #5f6368;
    font-size: 15px;
}
.status-strip textarea {
    text-align: center;
    font-weight: 650;
}
.conversation textarea {
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 16px;
    line-height: 1.55;
}
.system-note textarea,
.voice-note textarea {
    font-size: 14px;
}
button.primary-voice {
    min-height: 44px;
}
"""


def main_status_values(dummy_mode: bool) -> Tuple[str, str, str]:
    ros_status, pipeline_status, robot_mode_status, *_ = status_values(dummy_mode)
    return (
        ros_status.replace("ROS Core: ", ""),
        pipeline_status.replace("ELMiRA Pipeline: ", ""),
        robot_mode_status.replace("Robot Mode: ", ""),
    )


def compact_launch_message(raw_status: str, provider: str, dummy_mode: bool) -> str:
    status = sanitize_text(raw_status)
    provider_text = str(provider or "OpenAI")
    mode_text = "Dummy" if dummy_mode else "Real Robot"
    if status.startswith("launch started"):
        return (
            f"Launch started successfully using saved {provider_text} key. Mode: {mode_text}. "
            "API key: [MASKED]."
        )
    if status.startswith("launch already running"):
        return (
            f"Launch already running. Provider: {provider_text}. Mode: {mode_text}. "
            "API key: [MASKED]."
        )
    if "API key is required" in status:
        return (
            f"Launch not started. Add the {provider_text} API key in Advanced Settings. "
            "API key: [MASKED]."
        )
    if status.startswith("failed"):
        return status.splitlines()[0]
    return status.splitlines()[0] if status else "Launch status unavailable."


def compact_stop_message(raw_status: str) -> str:
    status = sanitize_text(raw_status)
    first_line = status.splitlines()[0] if status else "unknown"
    if first_line in ("stopped", "partial", "failed"):
        return f"Stop System: {first_line}."
    return f"Stop System: {first_line}."


def launch_progress_message() -> str:
    if process_alive(launch_process_nodes) and process_alive(launch_process_sm):
        return "ELMiRA is running. You can use Talk to NICO or the backup text command."

    if process_alive(launch_process_nodes):
        elapsed = int(time.time() - last_launch_started_at) if last_launch_started_at else 0
        if elapsed < 15:
            return (
                f"Launching ELMiRA... nodes are starting. "
                f"State machine starts automatically in about {max(0, 15 - elapsed)}s."
            )
        return "Launching ELMiRA... nodes are running; waiting for state machine."

    if process_alive(launch_process_sm):
        return "ELMiRA state machine is running. Nodes are still stabilizing."

    if launch_attempted:
        return "ELMiRA is not running. Check Advanced / Demo Tools if launch stopped unexpectedly."

    return last_system_message


def compact_evidence_summary() -> str:
    probe = load_json_safe(PATHS["latest_njf_probe"])
    ros_check = load_json_safe(PATHS["latest_ros_check"])
    robot = load_json_safe(PATHS["latest_robot_evidence"])
    service = load_json_safe(PATHS["latest_njf_service_check"])

    def state(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return "not run"
        return str(record.get("status", "unknown"))

    if not probe:
        return (
            "NJF Feasibility Test Result\n"
            "Status: not run\n"
            "Purpose: No-motion feasibility test for manipulation refinement.\n"
            "Input: visual error\n"
            "Output: bounded correction\n"
            "Safety: Robot motion commanded: No\n"
            "Evidence generated: not yet"
        )

    visual_error = probe.get("input_visual_error") or {}
    correction = probe.get("predicted_correction") or {}
    image_u = visual_error.get("image_error_u", "n/a")
    image_v = visual_error.get("image_error_v", "n/a")
    x_error = visual_error.get("x_error_m", "n/a")
    y_error = visual_error.get("y_error_m", "n/a")
    x_corr = correction.get("x_correction_m", "n/a")
    y_corr = correction.get("y_correction_m", "n/a")

    motion = "No"
    if probe and probe.get("physical_motion_executed") not in (False, "false", "False", None):
        motion = str(probe.get("physical_motion_executed"))

    return (
        "NJF Feasibility Test Result\n"
        f"Status: {state(probe)}\n"
        "Purpose: No-motion feasibility test for manipulation refinement.\n"
        f"Input: visual error = ({image_u}, {image_v}), "
        f"position error = ({x_error}m, {y_error}m)\n"
        f"Output: bounded correction = ({x_corr}m, {y_corr}m)\n"
        f"Safety: Robot motion commanded: {motion}\n"
        "Evidence generated: latest_njf_probe.json, latest_njf_probe.csv, monitoring_summary.md\n"
        f"ROS check: {state(ros_check)}\n"
        f"Robot evidence: {state(robot)}\n"
        f"Service check: {state(service)}"
    )


def kinematic_template_names() -> List[str]:
    names = set()
    try:
        data = json.loads(KINEMATIC_TEMPLATE_FILE.read_text(encoding="utf-8"))
        templates = data.get("templates", {})
        if isinstance(templates, dict):
            names.update(templates.keys())
    except Exception:
        pass
    try:
        data = json.loads(KINEMATIC_CAPTURED_TEMPLATE_FILE.read_text(encoding="utf-8"))
        templates = data.get("templates", {})
        if isinstance(templates, dict):
            names.update(templates.keys())
    except Exception:
        pass
    try:
        data = json.loads(CAPTURED_GRASP_TEMPLATE_FILE.read_text(encoding="utf-8"))
        templates = data.get("templates", {})
        if isinstance(templates, dict):
            names.update(templates.keys())
    except Exception:
        pass
    return sorted(names) if names else ["right_arm_pre_grasp"]


def captured_direct_template_names() -> List[str]:
    try:
        data = json.loads(KINEMATIC_CAPTURED_TEMPLATE_FILE.read_text(encoding="utf-8"))
        templates = data.get("templates", {})
        if isinstance(templates, dict):
            ready = [
                name
                for name, template in templates.items()
                if isinstance(template, dict)
                and template.get("direct_execution_ready")
                and template.get("commandable_right_arm_joint_positions_rad")
            ]
            return sorted(ready)
    except Exception:
        pass
    return []


def refresh_captured_direct_templates():
    names = captured_direct_template_names()
    if names:
        return gr.update(choices=names, value=names[0]), (
            f"Captured direct templates available: {', '.join(names)}"
        )
    return gr.update(choices=[], value=None), (
        "No captured direct templates yet. Record an ELMiRA motion, save a replay frame as a template, then refresh."
    )


def latest_captured_grasp_template_path() -> str:
    return str(GRASP_TEMPLATE_LOG_DIR / "latest_captured_template.json")


def captured_natural_grasp_template_names() -> List[str]:
    try:
        data = json.loads(CAPTURED_GRASP_TEMPLATE_FILE.read_text(encoding="utf-8"))
        templates = data.get("templates", {})
        if isinstance(templates, dict):
            return sorted(templates.keys())
    except Exception:
        pass
    return []


def natural_grasp_capture_summary(record: Optional[Dict[str, Any]], fallback: str = "") -> str:
    if not record:
        return fallback or "No natural grasp template captured yet."
    if record.get("summary_text"):
        return str(record.get("summary_text"))
    template = record.get("template") or {}
    return (
        f"Status: {record.get('status', 'unknown')}\n"
        f"Template: {record.get('template_name', template.get('name', 'unknown'))}\n"
        f"Stage: {record.get('action_stage', template.get('action_stage', 'unknown'))}\n"
        f"Target: {record.get('target_object', template.get('target_object', 'red object'))}\n"
        f"Estimated right_hand_xyz: {record.get('estimated_right_hand_xyz', template.get('estimated_right_hand_xyz', {}))}\n"
        f"Estimated right_wrist_xyz: {record.get('estimated_right_wrist_xyz', template.get('estimated_right_wrist_xyz', {}))}\n"
        f"IK seed ready: {record.get('ik_seed_ready', template.get('ik_seed_ready', False))}\n"
        f"Motion commanded: {record.get('real_robot_motion_commanded', False)}\n"
        f"Message: {record.get('message', '')}\n"
        f"Latest JSON: {latest_captured_grasp_template_path()}"
    )


def run_capture_natural_grasp_template_dashboard(
    template_name: str,
    target_object: str,
    action_stage: str,
    notes: str,
    allow_overwrite: bool,
):
    command = [
        sys.executable,
        str(CAPTURE_NATURAL_GRASP_TEMPLATE_SCRIPT),
        "capture",
        "--template-name",
        template_name or "",
        "--target-object",
        target_object or "red object",
        "--action-stage",
        action_stage or "pre_grasp",
        "--notes",
        notes or "",
    ]
    if allow_overwrite:
        command.append("--allow-overwrite")
    try:
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
        if completed.returncode != 0 and not record:
            status = output[-1200:]
        else:
            status = natural_grasp_capture_summary(record, output[-1200:])
    except subprocess.TimeoutExpired:
        record = None
        status = "Natural grasp capture timed out."
    except Exception as exc:
        record = None
        status = f"Natural grasp capture failed: {sanitize_text(exc)}"
    image_path = (record or {}).get("preview_image_path", "")
    image_value = image_path if image_path and Path(image_path).exists() else None
    return status, image_value, latest_captured_grasp_template_path()


def run_preview_captured_natural_grasp_template_dashboard(template_name: str):
    command = [
        sys.executable,
        str(CAPTURE_NATURAL_GRASP_TEMPLATE_SCRIPT),
        "preview",
        "--template-name",
        template_name or "",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
        status = natural_grasp_capture_summary(record, output[-1200:])
    except subprocess.TimeoutExpired:
        record = None
        status = "Captured template preview timed out."
    except Exception as exc:
        record = None
        status = f"Captured template preview failed: {sanitize_text(exc)}"
    image_path = (record or {}).get("preview_image_path", "")
    image_value = image_path if image_path and Path(image_path).exists() else None
    return status, image_value, latest_captured_grasp_template_path()


def run_captured_template_seed_plan_dashboard(template_name: str):
    command = [
        sys.executable,
        str(CAPTURE_NATURAL_GRASP_TEMPLATE_SCRIPT),
        "seed-plan",
        "--template-name",
        template_name or "",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=6,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
        status = natural_grasp_capture_summary(record, output[-1200:])
    except subprocess.TimeoutExpired:
        status = "Captured template seed-plan timed out."
    except Exception as exc:
        status = f"Captured template seed-plan failed: {sanitize_text(exc)}"
    return status, None, str(GRASP_TEMPLATE_LOG_DIR / "latest_captured_template_seed_plan.json")


def kinematic_preview_summary(record: Optional[Dict[str, Any]], status: str) -> str:
    if not record:
        return status or "No kinematic preview has been generated yet."
    hand_xyz = record.get("estimated_right_hand_xyz", {})
    wrist_xyz = record.get("estimated_right_wrist_xyz", {})
    return (
        f"Template: {record.get('template_name', 'unknown')}\n"
        f"Action type: {record.get('action_type', 'unknown')}\n"
        f"Target object: {record.get('target_object', 'target object')}\n"
        f"Target xyz: {record.get('target_object_xyz', {})}\n"
        f"Estimated right hand xyz: {hand_xyz}\n"
        f"Estimated right wrist xyz: {wrist_xyz}\n"
        f"Hand state: {record.get('hand_state', 'neutral')}\n"
        f"Above table safety limit: {record.get('above_table_safety_limit')}\n"
        f"Height warning: {record.get('height_warning') or 'none'}\n"
        f"Safety: real robot motion commanded = {record.get('real_robot_motion_commanded', False)}\n"
        f"Note: {record.get('preview_note', record.get('notes', 'coarse kinematic estimate only'))}"
    )


def run_kinematic_preview_dashboard(
    template_name: str,
    target_object: str,
    target_x: float,
    target_y: float,
    target_z: float,
):
    command = [sys.executable, str(KINEMATIC_PREVIEW_SCRIPT), "--template", template_name]
    target = (target_object or "").strip()
    if target:
        command.extend(["--target-object", target])
    command.extend(
        [
            "--target-x",
            str(float(target_x)),
            "--target-y",
            str(float(target_y)),
            "--target-z",
            str(float(target_z)),
        ]
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=12,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        status = "Kinematic preview generated." if completed.returncode == 0 else output[-800:]
    except subprocess.TimeoutExpired:
        status = "Kinematic preview timed out."
    except Exception as exc:
        status = f"Kinematic preview failed: {sanitize_text(exc)}"

    json_path = KINEMATIC_LOG_DIR / "latest_preview.json"
    image_path = KINEMATIC_LOG_DIR / "latest_preview.png"
    record = load_json_safe(json_path)
    image_value = str(image_path) if image_path.exists() else None
    json_text = str(json_path) if json_path.exists() else "No JSON preview log yet."
    return kinematic_preview_summary(record, status), image_value, json_text


def live_pose_summary(record: Optional[Dict[str, Any]], status: str) -> Tuple[str, str, str]:
    if not record:
        return status, "", ""
    hand_xyz = record.get("estimated_right_hand_xyz", {})
    lines = [
        f"Joint-state status: {record.get('status', 'unknown')}",
        f"Source topic: {record.get('source_topic', 'not available')}",
        f"Mapped right-arm angles: {record.get('mapped_joint_angles_deg', {})}",
        f"Estimated right hand xyz: {hand_xyz}",
        f"Latest update: {record.get('latest_update_timestamp', 'not available')}",
        f"Motion commanded: {record.get('real_robot_motion_commanded', False)}",
    ]
    if record.get("missing_simplified_joints"):
        lines.append(f"Missing simplified joints: {record.get('missing_simplified_joints')}")
    if record.get("message"):
        lines.append(f"Message: {record.get('message')}")
    return "\n".join(lines), str(hand_xyz), str(record.get("latest_update_timestamp", ""))


def refresh_live_pose_dashboard():
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(LIVE_JOINT_READER_SCRIPT),
                "--once",
                "--render",
                "--timeout",
                "1.5",
            ],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=7,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        status = output[-1200:] if completed.returncode != 0 else "Live pose refreshed."
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
    except subprocess.TimeoutExpired:
        status = "Live pose refresh timed out."
        record = None
    except Exception as exc:
        status = f"Live pose unavailable: {sanitize_text(exc)}"
        record = None
    image_path = KINEMATIC_LOG_DIR / "latest_live_preview.png"
    image_value = str(image_path) if image_path.exists() else None
    summary, hand_xyz, timestamp = live_pose_summary(record, status)
    return summary, image_value, hand_xyz, timestamp


def latest_recording_path() -> Path:
    return KINEMATIC_LOG_DIR / "latest_recording.json"


def resolve_recording_path_for_dashboard(path_value: str) -> Path:
    value = (path_value or "").strip()
    if not value:
        return latest_recording_path()
    path = Path(value).expanduser()
    return path if path.is_absolute() else API_ROOT / path


def read_recording_for_dashboard(path_value: str) -> Optional[Dict[str, Any]]:
    return load_json_safe(resolve_recording_path_for_dashboard(path_value))


def recording_frame_count(record: Optional[Dict[str, Any]]) -> int:
    if not record:
        return 0
    frames = record.get("joint_state_frames") or []
    return len(frames) if isinstance(frames, list) else 0


def recording_status_summary(record: Optional[Dict[str, Any]], fallback: str = "") -> str:
    if not record:
        return fallback or "No joint recording loaded yet."
    count = recording_frame_count(record)
    return (
        f"Recording: {record.get('recording_id', 'unknown')}\n"
        f"Status: {'recorded' if record.get('joint_state_available') else 'unavailable'}\n"
        f"Frames: {count}\n"
        f"Instruction: {record.get('instruction', '')}\n"
        f"Target object: {record.get('target_object', '')}\n"
        f"Action type: {record.get('action_type', '')}\n"
        f"Duration: {record.get('duration_sec', 0)}s\n"
        f"Motion commanded by recorder: {record.get('real_robot_motion_commanded', False)}\n"
        f"Notes: {record.get('notes', '')}"
    )


def replay_frame_summary(record: Optional[Dict[str, Any]], frame_index: int) -> str:
    if not record:
        return "No recording loaded."
    frames = record.get("joint_state_frames") or []
    if not frames:
        return recording_status_summary(record, "Recording has no frames.")
    index = max(0, min(int(frame_index or 0), len(frames) - 1))
    frame = frames[index]
    return (
        f"Frame: {index} / {len(frames) - 1}\n"
        f"Time: {frame.get('t_relative_sec', 'n/a')}s\n"
        f"Estimated right_hand_xyz: {frame.get('estimated_right_hand_xyz', {})}\n"
        f"Estimated right_wrist_xyz: {frame.get('estimated_right_wrist_xyz', {})}\n"
        f"Mapped right-arm joint angles: {frame.get('mapped_right_arm_joint_angles', {})}\n"
        f"Motion commanded by replay: {record.get('real_robot_motion_commanded', False)}"
    )


def latest_recording_text() -> str:
    path = latest_recording_path()
    return str(path) if path.exists() else str(path)


def start_joint_recording_dashboard(
    label: str,
    instruction: str,
    target_object: str,
    action_type: str,
):
    global joint_record_process, joint_record_log_handle
    if process_alive(joint_record_process):
        return (
            "Joint recording is already running. Give NICO the action command, then click Stop Joint Recording.",
            latest_recording_text(),
        )

    KINEMATIC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = KINEMATIC_LOG_DIR / "joint_recording_process.log"
    try:
        if joint_record_log_handle:
            try:
                joint_record_log_handle.close()
            except Exception:
                pass
        joint_record_log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        joint_record_process = subprocess.Popen(
            [
                sys.executable,
                str(JOINT_RECORDER_SCRIPT),
                "--label",
                label or "right_arm_point_red_object_trial",
                "--instruction",
                instruction or "Point to the red object with your right hand.",
                "--target-object",
                target_object or "red object",
                "--action-type",
                action_type or "point",
            ],
            cwd=str(API_ROOT),
            stdout=joint_record_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        time.sleep(0.4)
        if joint_record_process.poll() is not None:
            record = load_json_safe(latest_recording_path())
            return (
                recording_status_summary(
                    record,
                    "Live joint recording unavailable: /joint_states not available.",
                ),
                latest_recording_text(),
            )
        return (
            "Joint recording started. Now command NICO through the normal chat pipeline, then click Stop Joint Recording.",
            latest_recording_text(),
        )
    except Exception as exc:
        return f"Joint recording failed to start: {sanitize_text(exc)}", latest_recording_text()


def stop_joint_recording_dashboard():
    global joint_record_process, joint_record_log_handle
    if not process_alive(joint_record_process):
        record = load_json_safe(latest_recording_path())
        return (
            recording_status_summary(record, "No active joint recording process."),
            latest_recording_text(),
            gr.update(maximum=max(0, recording_frame_count(record) - 1), value=0),
            replay_frame_summary(record, 0) if record else "No recording loaded.",
        )

    try:
        os.killpg(os.getpgid(joint_record_process.pid), signal.SIGINT)
        joint_record_process.wait(timeout=8)
        status = "Joint recording stopped and saved."
    except Exception:
        try:
            os.killpg(os.getpgid(joint_record_process.pid), signal.SIGTERM)
            joint_record_process.wait(timeout=3)
            status = "Joint recording stopped after timeout and saved if frames were available."
        except Exception as exc:
            status = f"Joint recording stop failed: {sanitize_text(exc)}"
    finally:
        joint_record_process = None
        if joint_record_log_handle:
            try:
                joint_record_log_handle.close()
            except Exception:
                pass
            joint_record_log_handle = None

    record = load_json_safe(latest_recording_path())
    frame_count = recording_frame_count(record)
    return (
        status + "\n" + recording_status_summary(record),
        latest_recording_text(),
        gr.update(maximum=max(0, frame_count - 1), value=0),
        replay_frame_summary(record, 0) if record else "No recording loaded.",
    )


def record_10_seconds_dashboard(
    label: str,
    instruction: str,
    target_object: str,
    action_type: str,
):
    if process_alive(joint_record_process):
        return (
            "Stop the active recording before starting a fixed 10-second recording.",
            latest_recording_text(),
            gr.update(),
            "Active recording is still running.",
        )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(JOINT_RECORDER_SCRIPT),
                "--record-seconds",
                "10",
                "--label",
                label or "right_arm_point_red_object_trial",
                "--instruction",
                instruction or "Point to the red object with your right hand.",
                "--target-object",
                target_object or "red object",
                "--action-type",
                action_type or "point",
            ],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        status = "10-second joint recording completed." if completed.returncode == 0 else output[-1000:]
    except subprocess.TimeoutExpired:
        status = "10-second joint recording timed out. Saved if frames were available."
    except Exception as exc:
        status = f"10-second joint recording failed: {sanitize_text(exc)}"
    record = load_json_safe(latest_recording_path())
    frame_count = recording_frame_count(record)
    return (
        status + "\n" + recording_status_summary(record),
        latest_recording_text(),
        gr.update(maximum=max(0, frame_count - 1), value=0),
        replay_frame_summary(record, 0) if record else "No recording loaded.",
    )


def load_latest_recording_dashboard(path_value: str):
    path = resolve_recording_path_for_dashboard(path_value)
    record = load_json_safe(path)
    count = recording_frame_count(record)
    return (
        str(path),
        gr.update(maximum=max(0, count - 1), value=0),
        replay_frame_summary(record, 0) if record else f"No recording found at {path}",
    )


def render_replay_frame_dashboard(path_value: str, frame_index: float):
    path = resolve_recording_path_for_dashboard(path_value)
    index = int(frame_index or 0)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPLAY_JOINT_TRAJECTORY_SCRIPT),
                "--recording",
                str(path),
                "--frame",
                str(index),
            ],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=12,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        summary = json.loads(completed.stdout or "{}") if completed.stdout else {}
        if completed.returncode != 0:
            frame_info = output[-1000:]
        else:
            frame_info = (
                f"Replay status: {summary.get('status', 'unknown')}\n"
                f"Frame: {summary.get('frame_index', index)} / {max(0, int(summary.get('frame_count', 1)) - 1)}\n"
                f"Time: {summary.get('t_relative_sec', 'n/a')}s\n"
                f"Estimated right_hand_xyz: {summary.get('estimated_right_hand_xyz', {})}\n"
                f"Mapped right-arm joint angles: {summary.get('mapped_right_arm_joint_angles', {})}\n"
                f"Motion commanded by replay: {summary.get('real_robot_motion_commanded', False)}"
            )
        image_path = summary.get("preview_image_path") if isinstance(summary, dict) else ""
    except subprocess.TimeoutExpired:
        frame_info = "Replay render timed out."
        image_path = ""
    except Exception as exc:
        frame_info = f"Replay render failed: {sanitize_text(exc)}"
        image_path = ""
    image_value = image_path if image_path and Path(image_path).exists() else None
    return image_value, frame_info, gr.update(value=index)


def previous_replay_frame(path_value: str, frame_index: float):
    new_index = max(0, int(frame_index or 0) - 1)
    image, info, _slider = render_replay_frame_dashboard(path_value, new_index)
    return image, info, gr.update(value=new_index)


def next_replay_frame(path_value: str, frame_index: float):
    record = read_recording_for_dashboard(path_value)
    max_index = max(0, recording_frame_count(record) - 1)
    new_index = min(max_index, int(frame_index or 0) + 1)
    image, info, _slider = render_replay_frame_dashboard(path_value, new_index)
    return image, info, gr.update(value=new_index)


def save_selected_frame_template_dashboard(
    path_value: str,
    frame_index: float,
    template_name: str,
    action_type: str,
    target_object: str,
):
    path = resolve_recording_path_for_dashboard(path_value)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SAVE_REPLAY_TEMPLATE_SCRIPT),
                "--recording",
                str(path),
                "--frame",
                str(int(frame_index or 0)),
                "--template-name",
                template_name or "captured_right_arm_template",
                "--action-type",
                action_type or "point",
                "--target-object",
                target_object or "red object",
            ],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        if completed.returncode != 0:
            return output[-1000:]
        summary = json.loads(completed.stdout or "{}")
        return (
            f"Template saved: {summary.get('template_name', template_name)}\n"
            f"Path: {summary.get('template_path', '')}\n"
            f"Frame: {summary.get('frame_index', int(frame_index or 0))}\n"
            "Direct execution ready: refresh captured templates in Safety Debug.\n"
            f"Motion commanded: {summary.get('real_robot_motion_commanded', False)}"
        )
    except subprocess.TimeoutExpired:
        return "Save template timed out."
    except Exception as exc:
        return f"Save template failed: {sanitize_text(exc)}"


def latest_first_person_trial_path() -> Path:
    return FIRST_PERSON_LOG_DIR / "latest_trial_record.json"


def latest_first_person_trial_text() -> str:
    path = latest_first_person_trial_path()
    return str(path)


def resolve_first_person_trial_path(path_value: str) -> Path:
    value = (path_value or "").strip()
    if not value:
        return latest_first_person_trial_path()
    path = Path(value).expanduser()
    return path if path.is_absolute() else API_ROOT / path


def read_first_person_trial(path_value: str = "") -> Optional[Dict[str, Any]]:
    return load_json_safe(resolve_first_person_trial_path(path_value))


def first_person_frame_paths(record: Optional[Dict[str, Any]]) -> List[str]:
    if not record:
        return []
    paths = record.get("frame_paths")
    if isinstance(paths, list) and paths:
        return [str(path) for path in paths]
    frames_dir = Path(str(record.get("frames_dir", "")))
    if frames_dir.exists():
        return [str(path) for path in sorted(frames_dir.glob("frame_*.jpg"))]
    return []


def first_person_trial_summary(record: Optional[Dict[str, Any]], fallback: str = "") -> str:
    if not record:
        return fallback or "No first-person trial loaded yet."
    robot_touch_target = record.get("robot_touch_target_z_m", "")
    robot_touch_offset = record.get("robot_touch_z_offset_m", "")
    robot_touch_line = ""
    if robot_touch_target not in ("", None, "unknown", "not_applicable"):
        robot_touch_line = (
            f"Robot touch target Z: {robot_touch_target} m "
            f"(offset {robot_touch_offset} m)\n"
        )
    change_summary = record.get("trial_change_summary", "")
    change_line = f"Change: {change_summary}\n" if change_summary else ""
    return (
        f"Trial: {record.get('trial_id', 'unknown')}\n"
        f"Status: {record.get('status', 'unknown')}\n"
        f"Frames: {record.get('frame_count', len(first_person_frame_paths(record)))}\n"
        f"Topic: {record.get('image_topic', '')}\n"
        f"Action: {record.get('action_type', '')}\n"
        f"Target: {record.get('target_object', '')} at "
        f"({record.get('x_m', '')}, {record.get('y_m', '')}, {record.get('z_m', '')}) m\n"
        f"{robot_touch_line}"
        f"{change_line}"
        f"Key frame: {record.get('key_frame_index', 'unknown')}\n"
        f"Outcome: {record.get('outcome', 'unknown')}\n"
        f"Recorder commanded robot motion: {record.get('real_robot_motion_commanded_by_recorder', False)}\n"
        f"Message: {record.get('message', '')}"
    )


def first_person_frame_info(record: Optional[Dict[str, Any]], frame_index: int) -> str:
    if not record:
        return "No first-person trial loaded."
    frames = first_person_frame_paths(record)
    if not frames:
        return first_person_trial_summary(record, "Trial has no saved frames.")
    index = max(0, min(int(frame_index or 0), len(frames) - 1))
    return (
        f"Trial: {record.get('trial_id', 'unknown')}\n"
        f"Frame: {index} / {len(frames) - 1}\n"
        f"Frame path: {frames[index]}\n"
        f"Action: {record.get('action_type', '')}\n"
        f"Target: {record.get('target_object', '')}\n"
        f"Change: {record.get('trial_change_summary', '')}\n"
        f"Offset: {record.get('offset_direction', 'unknown')} / {record.get('offset_size', 'unknown')}\n"
        f"Contact: {record.get('contact_status', 'unknown')}\n"
        f"Outcome: {record.get('outcome', 'unknown')}\n"
        f"Robot motion commanded by recorder: {record.get('real_robot_motion_commanded_by_recorder', False)}"
    )


def list_first_person_image_topics_dashboard() -> str:
    try:
        completed = subprocess.run(
            [sys.executable, str(FIRST_PERSON_TRIAL_RECORDER_SCRIPT), "--list-image-topics"],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else {}
        topics = record.get("topics") if isinstance(record, dict) else []
        if topics:
            topic_lines = [f"- {item.get('topic')} ({item.get('type')})" for item in topics[:12]]
            return "Image topics available:\n" + "\n".join(topic_lines)
        return record.get("message", output[-1000:] or "No image topics found.") if isinstance(record, dict) else output[-1000:]
    except subprocess.TimeoutExpired:
        return "Image topic listing timed out."
    except Exception as exc:
        return f"Image topic listing unavailable: {sanitize_text(exc)}"


def start_first_person_trial_recording_dashboard(
    image_topic: str,
    trial_id: str,
    action_type: str,
    target_object: str,
    x_m: float,
    y_m: float,
    z_m: float,
    fps: float,
):
    global first_person_record_process, first_person_record_log_handle
    if process_alive(first_person_record_process):
        return (
            "First-person recording is already running. Command NICO through ELMiRA, then click Stop Trial Recording.",
            latest_first_person_trial_text(),
        )

    FIRST_PERSON_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = FIRST_PERSON_LOG_DIR / "first_person_recording_process.log"
    try:
        if first_person_record_log_handle:
            try:
                first_person_record_log_handle.close()
            except Exception:
                pass
        first_person_record_log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        first_person_record_process = subprocess.Popen(
            [
                sys.executable,
                str(FIRST_PERSON_TRIAL_RECORDER_SCRIPT),
                "--topic",
                image_topic or "/nico/vision/left",
                "--trial-id",
                trial_id or "",
                "--action-type",
                action_type or "point",
                "--target-object",
                target_object or "red object",
                "--x",
                str(x_m if x_m is not None else 0.20),
                "--y",
                str(y_m if y_m is not None else -0.08),
                "--z",
                str(z_m if z_m is not None else 0.14),
                "--fps",
                str(fps if fps is not None else 5),
            ],
            cwd=str(API_ROOT),
            stdout=first_person_record_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        time.sleep(0.5)
        if first_person_record_process.poll() is not None:
            record = load_json_safe(latest_first_person_trial_path())
            return (
                first_person_trial_summary(
                    record,
                    "Live first-person recording unavailable: camera topic not available.",
                ),
                latest_first_person_trial_text(),
            )
        return (
            "First-person recording started. Now command NICO through the normal ELMiRA pipeline, then click Stop Trial Recording.",
            latest_first_person_trial_text(),
        )
    except Exception as exc:
        return f"First-person recording failed to start: {sanitize_text(exc)}", latest_first_person_trial_text()


def stop_first_person_trial_recording_dashboard():
    global first_person_record_process, first_person_record_log_handle
    if not process_alive(first_person_record_process):
        record = load_json_safe(latest_first_person_trial_path())
        frames = first_person_frame_paths(record)
        return (
            first_person_trial_summary(record, "No active first-person recording process."),
            latest_first_person_trial_text(),
            gr.update(maximum=max(0, len(frames) - 1), value=0),
            first_person_frame_info(record, 0) if record else "No trial loaded.",
        )

    try:
        os.killpg(os.getpgid(first_person_record_process.pid), signal.SIGINT)
        first_person_record_process.wait(timeout=8)
        status = "First-person recording stopped and saved."
    except Exception:
        try:
            os.killpg(os.getpgid(first_person_record_process.pid), signal.SIGTERM)
            first_person_record_process.wait(timeout=3)
            status = "First-person recording stopped after timeout and saved if frames were available."
        except Exception as exc:
            status = f"First-person recording stop failed: {sanitize_text(exc)}"
    finally:
        first_person_record_process = None
        if first_person_record_log_handle:
            try:
                first_person_record_log_handle.close()
            except Exception:
                pass
            first_person_record_log_handle = None

    record = load_json_safe(latest_first_person_trial_path())
    frames = first_person_frame_paths(record)
    return (
        status + "\n" + first_person_trial_summary(record),
        latest_first_person_trial_text(),
        gr.update(maximum=max(0, len(frames) - 1), value=0),
        first_person_frame_info(record, 0) if record else "No trial loaded.",
    )


def record_first_person_10_seconds_dashboard(
    image_topic: str,
    trial_id: str,
    action_type: str,
    target_object: str,
    x_m: float,
    y_m: float,
    z_m: float,
    duration_sec: float,
    fps: float,
):
    if process_alive(first_person_record_process):
        return (
            "Stop the active first-person recording before starting a fixed recording.",
            latest_first_person_trial_text(),
            gr.update(),
            "Active first-person recording is still running.",
        )
    duration = max(1, float(duration_sec or 10))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(FIRST_PERSON_TRIAL_RECORDER_SCRIPT),
                "--record-seconds",
                str(duration),
                "--topic",
                image_topic or "/nico/vision/left",
                "--trial-id",
                trial_id or "",
                "--action-type",
                action_type or "point",
                "--target-object",
                target_object or "red object",
                "--x",
                str(x_m if x_m is not None else 0.20),
                "--y",
                str(y_m if y_m is not None else -0.08),
                "--z",
                str(z_m if z_m is not None else 0.14),
                "--fps",
                str(fps if fps is not None else 5),
            ],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=duration + 10,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else load_json_safe(latest_first_person_trial_path())
        status = (
            first_person_trial_summary(record, "Fixed first-person recording completed.")
            if completed.returncode == 0
            else output[-1200:]
        )
    except subprocess.TimeoutExpired:
        record = load_json_safe(latest_first_person_trial_path())
        status = "Fixed first-person recording timed out. Saved if frames were available."
    except Exception as exc:
        record = load_json_safe(latest_first_person_trial_path())
        status = f"Fixed first-person recording failed: {sanitize_text(exc)}"
    frames = first_person_frame_paths(record)
    return (
        status,
        latest_first_person_trial_text(),
        gr.update(maximum=max(0, len(frames) - 1), value=0),
        first_person_frame_info(record, 0) if record else "No trial loaded.",
    )


def load_latest_first_person_trial_dashboard(path_value: str):
    path = resolve_first_person_trial_path(path_value)
    record = load_json_safe(path)
    frames = first_person_frame_paths(record)
    image_value = frames[0] if frames and Path(frames[0]).exists() else None
    return (
        str(path),
        gr.update(maximum=max(0, len(frames) - 1), value=0),
        image_value,
        first_person_frame_info(record, 0) if record else f"No trial found at {path}",
    )


def render_first_person_frame_dashboard(path_value: str, frame_index: float):
    record = read_first_person_trial(path_value)
    frames = first_person_frame_paths(record)
    if not frames:
        return None, first_person_frame_info(record, 0), gr.update(value=0)
    index = max(0, min(int(frame_index or 0), len(frames) - 1))
    image_value = frames[index] if Path(frames[index]).exists() else None
    return image_value, first_person_frame_info(record, index), gr.update(value=index)


def previous_first_person_frame(path_value: str, frame_index: float):
    new_index = max(0, int(frame_index or 0) - 1)
    return render_first_person_frame_dashboard(path_value, new_index)


def next_first_person_frame(path_value: str, frame_index: float):
    record = read_first_person_trial(path_value)
    max_index = max(0, len(first_person_frame_paths(record)) - 1)
    new_index = min(max_index, int(frame_index or 0) + 1)
    return render_first_person_frame_dashboard(path_value, new_index)


def save_first_person_key_frame_dashboard(path_value: str, frame_index: float):
    path = resolve_first_person_trial_path(path_value)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(FIRST_PERSON_TRIAL_RECORDER_SCRIPT),
                "--save-key-frame",
                "--trial-record",
                str(path),
                "--frame-index",
                str(int(frame_index or 0)),
            ],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else load_json_safe(path)
        if completed.returncode != 0:
            return output[-1000:]
        return first_person_trial_summary(record)
    except subprocess.TimeoutExpired:
        return "Saving key frame timed out."
    except Exception as exc:
        return f"Saving key frame failed: {sanitize_text(exc)}"


def save_first_person_manual_label_dashboard(
    path_value: str,
    offset_direction: str,
    offset_size: str,
    contact_status: str,
    outcome: str,
    failure_reason: str,
    notes: str,
):
    path = resolve_first_person_trial_path(path_value)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(FIRST_PERSON_TRIAL_RECORDER_SCRIPT),
                "--save-manual-label",
                "--trial-record",
                str(path),
                "--offset-direction",
                offset_direction or "unknown",
                "--offset-size",
                offset_size or "unknown",
                "--contact-status",
                contact_status or "unknown",
                "--outcome",
                outcome or "unknown",
                "--failure-reason",
                failure_reason or "unknown",
                "--notes",
                notes or "",
            ],
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else load_json_safe(path)
        if completed.returncode != 0:
            return output[-1000:]
        return first_person_trial_summary(record)
    except subprocess.TimeoutExpired:
        return "Saving manual trial label timed out."
    except Exception as exc:
        return f"Saving manual trial label failed: {sanitize_text(exc)}"


def preset_debug_summary(record: Optional[Dict[str, Any]], status: str) -> str:
    if not record:
        return status
    lines = [
        f"Preset: {record.get('preset_name', 'unknown')}\n"
        f"Mode: {record.get('mode', 'unknown')}\n"
        f"Execution status: {record.get('execution_status', 'unknown')}\n"
        f"Safety confirmed: {record.get('safety_confirmed', False)}\n"
        f"Real robot motion commanded: {record.get('real_robot_motion_commanded', False)}\n"
        f"Failure reason: {record.get('failure_reason') or 'none'}"
    ]
    if record.get("joint_positions_deg"):
        lines.append(f"Joint positions deg: {record.get('joint_positions_deg')}")
    if record.get("ros_service"):
        lines.append(f"ROS service: {record.get('ros_service')}")
    if record.get("ros_topic"):
        lines.append(f"ROS topic: {record.get('ros_topic')}")
    if record.get("finger_motor_ids"):
        lines.append(f"Finger motor IDs: {record.get('finger_motor_ids')}")
    if record.get("target_raw") is not None:
        lines.append(f"XL-320 target raw: {record.get('target_raw')}")
    lines.append(f"Notes: {record.get('notes', '')}")
    return "\n".join(lines)


def run_safe_preset_dashboard(preset_name: str, mode: str, safety_confirmed: bool):
    command = [sys.executable, str(SAFE_PRESET_DEBUG_SCRIPT), "--mode", mode, "--preset", preset_name]
    if safety_confirmed:
        command.append("--safety-confirmed")
    try:
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        status = output[-1200:] if completed.returncode != 0 else "Preset debug updated."
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
    except subprocess.TimeoutExpired:
        status = "Preset debug timed out."
        record = None
    except Exception as exc:
        status = f"Preset debug failed: {sanitize_text(exc)}"
        record = None
    image_path = record.get("preview_image_path") if isinstance(record, dict) else None
    image_value = image_path if image_path and Path(image_path).exists() else None
    json_path = KINEMATIC_LOG_DIR / "latest_preset_debug.json"
    return preset_debug_summary(record, status), image_value, str(json_path)


def run_captured_direct_template_dashboard(template_name: str, safety_confirmed: bool):
    name = (template_name or "").strip()
    if not name:
        return (
            "No captured direct template selected. Save a replay frame as a template first.",
            None,
            str(KINEMATIC_LOG_DIR / "latest_preset_debug.json"),
        )
    return run_safe_preset_dashboard(name, "execute", safety_confirmed)


def run_cartesian_xyz_dashboard(x: float, y: float, z: float, orientation_mode: str, safety_confirmed: bool):
    try:
        command = [
            sys.executable,
            str(SAFE_PRESET_DEBUG_SCRIPT),
            "--mode",
            "execute",
            "--preset",
            "cartesian_xyz",
            "--cartesian",
            "--x",
            str(float(x)),
            "--y",
            str(float(y)),
            "--z",
            str(float(z)),
            "--orientation-mode",
            orientation_mode or "point",
        ]
        if safety_confirmed:
            command.append("--safety-confirmed")
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=12,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
        status = output[-1200:] if completed.returncode != 0 else "Cartesian IK command completed."
    except subprocess.TimeoutExpired:
        status = "Cartesian IK command timed out."
        record = None
    except Exception as exc:
        status = f"Cartesian IK command failed: {sanitize_text(exc)}"
        record = None
    return preset_debug_summary(record, status), None, str(KINEMATIC_LOG_DIR / "latest_preset_debug.json")


def grasp_debug_latest_json_path() -> str:
    return str(GRASP_DEBUG_LOG_DIR / "latest_grasp_plan.json")


def latest_grasp_debug_record() -> Optional[Dict[str, Any]]:
    return load_json_safe(GRASP_DEBUG_LOG_DIR / "latest_grasp_plan.json")


def detected_image_coords(record: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not record:
        return None, None
    detection = record.get("detection") or {}
    selected = detection.get("selected_target") or {}
    x_value = selected.get("bottom_x", record.get("image_x"))
    y_value = selected.get("bottom_y", record.get("image_y"))
    try:
        return float(x_value), float(y_value)
    except (TypeError, ValueError):
        return None, None


def grasp_real_coords(record: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not record:
        return None, None
    coord = record.get("coordinate_transfer") or {}
    x_value = coord.get("real_x", record.get("real_x"))
    y_value = coord.get("real_y", record.get("real_y"))
    try:
        return float(x_value), float(y_value)
    except (TypeError, ValueError):
        return None, None


def image_coords_need_detection_fallback(image_x: Any, image_y: Any) -> bool:
    try:
        x_value = float(image_x)
        y_value = float(image_y)
    except (TypeError, ValueError):
        return True
    # 0,0 is almost never a useful detected object point here; it usually means
    # the Gradio number fields were left empty.
    return abs(x_value) < 1e-9 and abs(y_value) < 1e-9


def grasp_debug_summary(record: Optional[Dict[str, Any]], fallback: str = "") -> str:
    if not record:
        return fallback or "No grasp debug plan has been generated yet."
    if record.get("summary_text"):
        return str(record.get("summary_text"))
    plan = record.get("plan") or {}
    detection = record.get("detection") or {}
    selected = detection.get("selected_target") or {}
    lines = [
        f"Stage: {record.get('stage', 'unknown')}",
        f"Status: {record.get('status', 'unknown')}",
        f"Action: {record.get('action_type', 'grasp')}",
        f"Target: {record.get('target_object', 'red object')}",
        f"Detected image coordinates: ({selected.get('bottom_x', record.get('image_x', 'n/a'))}, {selected.get('bottom_y', record.get('image_y', 'n/a'))})",
        f"Bounding box: {selected.get('bbox', 'not available')}",
        f"Real xyz: ({plan.get('real_x', record.get('real_x', 'n/a'))}, {plan.get('real_y', record.get('real_y', 'n/a'))}, {plan.get('target_z', record.get('target_z', 'n/a'))})",
        f"Selected arm: {plan.get('selected_arm', 'right')}",
        f"Workspace clamped: {plan.get('workspace_clamped', False)}",
        f"Wrist IDs 31/33 will be commanded: {plan.get('wrist_ids_31_33_will_be_commanded', False)}",
        "Planned sequence:",
    ]
    for step in plan.get("planned_steps", []):
        lines.append(
            f"- {step.get('stage')}: pose={step.get('pose')}, orientation={step.get('orientation')}, hand_action={step.get('hand_action')}"
        )
    execution = record.get("execution") or {}
    if execution:
        lines.append(
            f"Execution: {execution.get('execution_status', 'unknown')} "
            f"(motion commanded: {execution.get('real_robot_motion_commanded', False)})"
        )
        if execution.get("failure_reason"):
            lines.append(f"Execution note: {execution.get('failure_reason')}")
    lines.append(f"Log: {grasp_debug_latest_json_path()}")
    return "\n".join(lines)


def run_grasp_debug_dashboard(
    stage: str,
    target_object: str,
    image_x: Optional[float],
    image_y: Optional[float],
    real_x: Optional[float],
    real_y: Optional[float],
    target_z: float,
    safety_confirmed: bool,
    use_detection: bool = False,
    use_coordinate: bool = False,
):
    if stage != "detect" and image_coords_need_detection_fallback(image_x, image_y):
        latest_x, latest_y = detected_image_coords(latest_grasp_debug_record())
        if latest_x is not None and latest_y is not None:
            image_x, image_y = latest_x, latest_y

    command = [
        sys.executable,
        str(GRASP_DEBUG_SCRIPT),
        "--stage",
        stage,
        "--action-type",
        "grasp",
        "--target-object",
        target_object or "red object",
        "--target-z",
        str(float(target_z or 0.70)),
    ]
    if image_x is not None:
        command.extend(["--image-x", str(float(image_x))])
    if image_y is not None:
        command.extend(["--image-y", str(float(image_y))])
    if real_x is not None:
        command.extend(["--real-x", str(float(real_x))])
    if real_y is not None:
        command.extend(["--real-y", str(float(real_y))])
    if use_detection:
        command.append("--use-detection")
    if use_coordinate:
        command.append("--use-coordinate")
    if safety_confirmed:
        command.append("--confirm")
    try:
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=12,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
        status = output[-1500:] if completed.returncode != 0 else "Grasp debug updated."
    except subprocess.TimeoutExpired:
        status = "Grasp debug stage timed out."
        record = None
    except Exception as exc:
        status = f"Grasp debug failed: {sanitize_text(exc)}"
        record = None
    return grasp_debug_summary(record, status), grasp_debug_latest_json_path()


def run_grasp_debug_detect(target_object: str, image_x, image_y, real_x, real_y, target_z):
    summary, json_path = run_grasp_debug_dashboard(
        "detect", target_object, image_x, image_y, real_x, real_y, target_z, False
    )
    detected_x, detected_y = detected_image_coords(latest_grasp_debug_record())
    x_update = gr.update(value=detected_x) if detected_x is not None else gr.update()
    y_update = gr.update(value=detected_y) if detected_y is not None else gr.update()
    return summary, json_path, x_update, y_update


def run_grasp_debug_coordinate(target_object: str, image_x, image_y, real_x, real_y, target_z):
    summary, json_path = run_grasp_debug_dashboard(
        "coordinate", target_object, image_x, image_y, real_x, real_y, target_z, False
    )
    record = latest_grasp_debug_record()
    detected_x, detected_y = detected_image_coords(record)
    resolved_real_x, resolved_real_y = grasp_real_coords(record)
    return (
        summary,
        json_path,
        gr.update(value=detected_x) if detected_x is not None else gr.update(),
        gr.update(value=detected_y) if detected_y is not None else gr.update(),
        gr.update(value=resolved_real_x) if resolved_real_x is not None else gr.update(),
        gr.update(value=resolved_real_y) if resolved_real_y is not None else gr.update(),
    )


def run_grasp_debug_plan(target_object: str, image_x, image_y, real_x, real_y, target_z):
    return run_grasp_debug_dashboard(
        "plan", target_object, image_x, image_y, real_x, real_y, target_z, False
    )


def ik_trace_latest_json_path() -> str:
    return str(IK_TRACE_LOG_DIR / "latest_ik_trace.json")


def ik_trace_summary(record: Optional[Dict[str, Any]], fallback: str = "") -> str:
    if not record:
        return fallback or "No IK trace has been generated yet."
    if record.get("summary_text"):
        return str(record.get("summary_text"))
    coord = record.get("coordinate_transfer_result") or {}
    ik_response = record.get("ik_response") or {}
    filtering = record.get("command_filtering") or {}
    final_command = record.get("final_command") or {}
    lines = [
        "IK Trace Dump, No Motion",
        f"Status: {record.get('status', 'unknown')}",
        f"Action: {record.get('action_type', 'grasp')}",
        f"Target: {record.get('target_object', 'red object')}",
        f"Real target: x={coord.get('real_x')}, y={coord.get('real_y')}, z={coord.get('target_z')}",
        f"IK status: {ik_response.get('status', 'unknown')}",
        f"Dropped joints: {filtering.get('dropped_joints', [])}",
        f"Hand actions if executed: {final_command.get('hand_action_sequence', [])}",
        f"XL-320 wrist IDs 31/33 will be commanded: {final_command.get('xl320_wrist_ids_31_33_will_be_commanded')}",
        f"Diagnosis: {record.get('diagnosis', '')}",
        f"Log: {ik_trace_latest_json_path()}",
    ]
    return "\n".join(lines)


def run_ik_trace_dump_dashboard(target_object: str, image_x, image_y, real_x, real_y, target_z):
    if image_coords_need_detection_fallback(image_x, image_y):
        latest_x, latest_y = detected_image_coords(latest_grasp_debug_record())
        if latest_x is not None and latest_y is not None:
            image_x, image_y = latest_x, latest_y
    command = [
        sys.executable,
        str(IK_TRACE_DUMP_SCRIPT),
        "--action-type",
        "grasp",
        "--target-object",
        target_object or "red object",
        "--target-z",
        str(float(target_z or 0.70)),
        "--use-coordinate",
    ]
    if image_x is not None:
        command.extend(["--image-x", str(float(image_x))])
    if image_y is not None:
        command.extend(["--image-y", str(float(image_y))])
    if real_x is not None:
        command.extend(["--real-x", str(float(real_x))])
    if real_y is not None:
        command.extend(["--real-y", str(float(real_y))])
    try:
        completed = subprocess.run(
            command,
            cwd=str(API_ROOT),
            capture_output=True,
            text=True,
            timeout=12,
        )
        output = sanitize_text((completed.stdout or "") + (completed.stderr or ""))
        record = json.loads(completed.stdout or "{}") if completed.stdout else None
        status = output[-1800:] if completed.returncode != 0 else "IK trace dumped."
    except subprocess.TimeoutExpired:
        status = "IK trace dump timed out."
        record = None
    except Exception as exc:
        status = f"IK trace dump failed: {sanitize_text(exc)}"
        record = None
    return ik_trace_summary(record, status), ik_trace_latest_json_path()


def direct_control_service_available() -> bool:
    result = run_env_command("rosservice list", timeout_s=3)
    return (
        result.get("returncode") == 0
        and "/right/open_manipulator_p/goal_joint_space_path" in result.get("stdout", "")
    )


def direct_motion_bridge_ready() -> bool:
    result = run_env_command("rostopic info /nico/motion/setAngle", timeout_s=3)
    output = result.get("stdout", "")
    return result.get("returncode") == 0 and "Subscribers:" in output and "Subscribers: None" not in output


def direct_control_ready() -> bool:
    return direct_control_service_available() and direct_motion_bridge_ready()


def ik_service_available() -> bool:
    result = run_env_command("rosservice list", timeout_s=3)
    return result.get("returncode") == 0 and "/inverse_kinematics" in result.get("stdout", "")


def direct_control_status() -> str:
    ik_text = "IK available" if ik_service_available() else "IK not available"
    if direct_control_ready():
        return (
            "Direct robot control ready. "
            "Right-arm preset buttons can use /right/open_manipulator_p/goal_joint_space_path. "
            f"{ik_text}."
        )
    if direct_control_service_available() and not direct_motion_bridge_ready():
        return (
            "Joint service is available, but Motion is not ready. "
            "No subscriber on /nico/motion/setAngle, so commands will not move the physical robot yet."
        )
    if process_alive(direct_control_process):
        return "Direct robot control is starting. Wait for Motion and joint_controller_right to appear."
    ros_status = check_ros_status()
    if ros_status == "Running":
        return (
            "ROS is running, but the right-arm joint service is not available yet. "
            "Start Direct Robot Control or check the direct control log."
        )
    return "Direct robot control not running. Click Start Direct Robot Control first."


def start_direct_control(dummy_mode: bool):
    global direct_control_process, direct_control_log_handle
    if dummy_mode:
        return (
            "Dummy mode uses V-REP and will not move physical NICO. "
            "Leave Direct control dummy mode unchecked for the real robot.",
            tail_file(LOG_DIRECT_CONTROL_PATH),
        )
    if direct_control_ready():
        if not ik_service_available():
            threading.Thread(target=start_direct_ik_solver, daemon=True).start()
        return direct_control_status(), tail_file(LOG_DIRECT_CONTROL_PATH)
    if process_alive(direct_control_process):
        return direct_control_status(), tail_file(LOG_DIRECT_CONTROL_PATH)

    try:
        LOG_DIRECT_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if direct_control_log_handle:
            try:
                direct_control_log_handle.close()
            except Exception:
                pass
        direct_control_log_handle = LOG_DIRECT_CONTROL_PATH.open("w", encoding="utf-8", buffering=1)
        command = (
            "source activate.bash 2>/dev/null || true; "
            "source devel/setup.bash 2>/dev/null || true; "
            "export CUDA_VISIBLE_DEVICES=''; "
            "export PYTHONUNBUFFERED=1; "
            "roslaunch nicoros joint_controller.launch "
            "dummy:=false"
        )
        direct_control_process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(API_ROOT),
            stdout=direct_control_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        threading.Thread(target=start_direct_ik_solver, daemon=True).start()
        return (
            "Direct robot control starting. Wait 10-20 seconds, then click Refresh Control Status. IK solver is starting too.",
            tail_file(LOG_DIRECT_CONTROL_PATH),
        )
    except Exception as exc:
        return f"Direct robot control failed to start: {sanitize_text(exc)}", tail_file(LOG_DIRECT_CONTROL_PATH)


def stop_direct_control():
    global direct_control_process, direct_control_log_handle
    if not process_alive(direct_control_process):
        return "Direct robot control was not started by this dashboard.", tail_file(LOG_DIRECT_CONTROL_PATH)
    msg, ok = stop_process_group(direct_control_process, "direct robot control")
    direct_control_process = None
    if direct_control_log_handle:
        try:
            direct_control_log_handle.close()
        except Exception:
            pass
        direct_control_log_handle = None
    return ("stopped" if ok else "partial") + "\n" + msg, tail_file(LOG_DIRECT_CONTROL_PATH)


def start_direct_ik_solver():
    global direct_ik_process, direct_ik_log_handle
    if ik_service_available() or process_alive(direct_ik_process):
        return
    time.sleep(4.0)
    try:
        LOG_IK_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if direct_ik_log_handle:
            try:
                direct_ik_log_handle.close()
            except Exception:
                pass
        direct_ik_log_handle = LOG_IK_CONTROL_PATH.open("w", encoding="utf-8", buffering=1)
        command = (
            "source activate.bash 2>/dev/null || true; "
            "source devel/setup.bash 2>/dev/null || true; "
            "export CUDA_VISIBLE_DEVICES=''; "
            "export PYTHONUNBUFFERED=1; "
            "rosrun elmira ik_solver.py"
        )
        direct_ik_process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(API_ROOT),
            stdout=direct_ik_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
    except Exception:
        return


def refresh_main_status(dummy_mode: bool):
    return main_status_values(dummy_mode)


def auto_refresh_main(dummy_mode: bool):
    return (
        launch_progress_message(),
        *main_status_values(dummy_mode),
    )


def refresh_advanced_status():
    nodes_tail, sm_tail, _, _ = refresh_logs()
    return compact_evidence_summary(), nodes_tail, sm_tail


def launch_simple_handler(
    provider,
    api_key,
    saved_api_key_state,
    offset_x,
    offset_y,
    offset_z,
    dummy_mode,
    use_mllm_grounding,
    mic_device,
):
    global last_system_message
    resolved_key, saved_config, key_status = resolve_launch_api_key(
        provider, api_key, saved_api_key_state
    )
    if not resolved_key:
        last_system_message = key_status
        return (
            key_status,
            gr.update(interactive=False),
            saved_config,
            key_status,
            *main_status_values(dummy_mode),
            tail_file(LOG_NODES_PATH),
            tail_file(LOG_SM_PATH),
        )

    raw_status = launch_robot(
        provider,
        use_mllm_grounding,
        dummy_mode,
        resolved_key,
        mic_device,
        offset_x,
        offset_y,
        offset_z,
    )
    nodes_tail, sm_tail, _, _ = refresh_logs()
    launch_message = compact_launch_message(raw_status, provider, dummy_mode)
    if raw_status.startswith("launch started"):
        launch_message = (
            launch_message
            + " Starting nodes now; the state machine appears after the warm-up."
        )
    last_system_message = launch_message
    return (
        launch_message,
        gr.update(interactive=True),
        saved_config,
        key_status,
        *main_status_values(dummy_mode),
        nodes_tail,
        sm_tail,
    )


def stop_simple_handler(dummy_mode):
    global last_system_message, last_launch_started_at
    raw_status = stop_robot()
    nodes_tail, sm_tail, _, _ = refresh_logs()
    last_system_message = compact_stop_message(raw_status)
    last_launch_started_at = None
    return (
        last_system_message,
        *main_status_values(dummy_mode),
        nodes_tail,
        sm_tail,
    )


def run_probe_mode_simple(mode: str):
    result = run_probe_mode(mode, False)
    status_text = result[0]
    nodes_tail = result[7]
    sm_tail = result[8]
    return status_text, compact_evidence_summary(), nodes_tail, sm_tail


def conversation_from_logs(existing_transcript: str = "") -> str:
    log_text = tail_file(LOG_SM_PATH, lines=240)
    entries: List[Tuple[str, str]] = []
    for raw_line in log_text.splitlines():
        user_match = re.search(r"USER:\s*(.*)$", raw_line)
        if user_match:
            text = user_match.group(1).strip()
            if text:
                entries.append(("USER", text))
            continue

        robot_match = re.search(r"Text:\s*(.*)$", raw_line)
        if robot_match:
            text = robot_match.group(1).strip()
            if text:
                entries.append(("NICO", text))

    if not entries:
        return existing_transcript or ""

    lines = [f"{speaker}: {text}" for speaker, text in entries[-30:]]
    return "\n".join(lines)


def refresh_conversation(transcript: str):
    return conversation_from_logs(transcript)


def return_home_handler(transcript: str):
    return chat_interface("Return to initial pose.", transcript)


def build_dashboard():
    with gr.Blocks(
        title="ELMiRA NICO Console",
        theme=gr.themes.Soft(),
        css=CLEAN_DASHBOARD_CSS,
    ) as demo:
        saved_api_key_state = gr.State(value=dict(runtime_api_config))

        gr.Markdown(
            "<div class='main-title'>"
            "<h1>ELMiRA NICO Console</h1>"
            "<p>Talk to NICO, send a backup text command, and keep launch control visible.</p>"
            "</div>"
        )

        with gr.Row(elem_classes=["status-strip"]):
            ros_status = gr.Textbox(label="ROS Core", value=main_status_values(False)[0], interactive=False)
            pipeline_status = gr.Textbox(
                label="ELMiRA", value=main_status_values(False)[1], interactive=False
            )
            robot_mode_status = gr.Textbox(
                label="Robot", value=main_status_values(False)[2], interactive=False
            )

        with gr.Row():
            launch_btn = gr.Button("Launch ELMiRA", variant="primary")
            home_btn = gr.Button("Return Home")
            stop_btn = gr.Button("Stop System", variant="stop")
            refresh_status_btn = gr.Button("Refresh")

        launch_status = gr.Textbox(
            label="System",
            value="Ready. Launch ELMiRA before sending robot commands.",
            lines=2,
            interactive=False,
            elem_classes=["system-note"],
        )

        gr.Markdown("## Conversation")
        transcript = gr.Textbox(
            label="",
            lines=15,
            value="",
            interactive=False,
            placeholder="Your conversation with NICO will appear here.",
            elem_classes=["conversation"],
        )

        with gr.Row():
            talk_btn = gr.Button("Talk to NICO", variant="primary")
            refresh_conversation_btn = gr.Button("Refresh Conversation")
            voice_status = gr.Textbox(
                label="Voice",
                value="",
                lines=1,
                interactive=False,
                elem_classes=["voice-note"],
            )

        with gr.Row():
            message_input = gr.Textbox(
                label="Backup text command",
                placeholder="Type a backup instruction, e.g. 'Point to the red block.'",
                interactive=False,
                scale=5,
            )
            send_btn = gr.Button("Send", scale=1)

        with gr.Accordion("Advanced Settings", open=False):
            provider = gr.Dropdown(["OpenAI", "Google"], label="AI Provider", value="OpenAI")
            api_key = gr.Textbox(label="API Key", type="password", value="")
            with gr.Row():
                save_key_btn = gr.Button("Save API Key", variant="primary")
                clear_key_btn = gr.Button("Clear API Key")
            api_key_status = gr.Textbox(
                label="Saved API Key",
                value="No API key saved.",
                lines=1,
                interactive=False,
            )
            with gr.Row():
                offset_x = gr.Slider(label="X Offset", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
                offset_y = gr.Slider(label="Y Offset", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
                offset_z = gr.Slider(label="Z Offset", minimum=-0.4, maximum=0.4, value=0.03, step=0.01)
            dummy_mode = gr.Checkbox(label="Dummy mode", value=False)
            use_mllm_grounding = gr.Checkbox(label="Use MLLM grounding state", value=True)
            mic_device = gr.Textbox(label="Microphone device", value="")

        with gr.Accordion("Advanced / Demo Tools", open=False):
            gr.Markdown(
                "Neural Jacobian Field is tested here as a candidate manipulation-refinement method. "
                "This dry-run takes visual error as input and outputs a bounded correction, but it does not command robot motion."
            )
            with gr.Row():
                dry_run_btn = gr.Button("Run NJF Feasibility Test (No Motion)")
                ros_check_btn = gr.Button("Run ROS Check")
                robot_evidence_btn = gr.Button("Record Robot Evidence, No Motion")
            with gr.Row():
                service_check_btn = gr.Button("Run NJF Service Check")
                refresh_logs_btn = gr.Button("Refresh Logs")
            tool_status = gr.Textbox(label="Demo tool status", lines=2, interactive=False)
            evidence_status = gr.Textbox(
                label="NJF Feasibility Test Result",
                value=compact_evidence_summary(),
                lines=9,
                interactive=False,
            )
            with gr.Accordion("Log Tail", open=False):
                nodes_log = gr.Textbox(label="Launch nodes", lines=8, interactive=False)
                sm_log = gr.Textbox(label="State machine", lines=8, interactive=False)

        with gr.Accordion("Kinematic Preview / Action Template", open=False):
            gr.Markdown(
                "Preview a coarse right-arm action template before real execution. "
                "This is visualization only and sends no robot motion command."
            )
            kinematic_templates = kinematic_template_names()
            kinematic_template = gr.Dropdown(
                choices=kinematic_templates,
                value=kinematic_templates[0],
                label="Action template",
            )
            kinematic_target = gr.Textbox(
                label="Target object",
                value="red object",
                placeholder="Optional target object name",
            )
            with gr.Row():
                kinematic_target_x = gr.Number(label="Target x (m)", value=0.20)
                kinematic_target_y = gr.Number(label="Target y (m)", value=-0.10)
                kinematic_target_z = gr.Number(label="Target z (m)", value=0.10)
            kinematic_btn = gr.Button("Preview Action Template")
            kinematic_status = gr.Textbox(
                label="Preview result",
                value="No kinematic preview generated yet.",
                lines=10,
                interactive=False,
            )
            kinematic_image = gr.Image(label="Stick-figure preview", interactive=False)
            kinematic_json_path = gr.Textbox(label="JSON log path", interactive=False)

        with gr.Accordion("Live Motion Viewer", open=False):
            gr.Markdown(
                "Read-only view of live NICO joint states. This subscribes to joint-state topics and sends no motion command."
            )
            live_refresh_btn = gr.Button("Refresh Live Pose")
            live_status = gr.Textbox(
                label="Joint-state status",
                value="No live pose read yet.",
                lines=8,
                interactive=False,
            )
            live_image = gr.Image(label="Live stick-figure preview", interactive=False)
            with gr.Row():
                live_hand_xyz = gr.Textbox(label="Estimated right_hand_xyz", interactive=False)
                live_timestamp = gr.Textbox(label="Latest update timestamp", interactive=False)

        with gr.Accordion("Capture Natural Grasp Template", open=False):
            gr.Markdown(
                "Capture the current supervised NICO right-arm pose as a reusable grasp template. "
                "This reads joint states only and sends no robot motion command."
            )
            with gr.Row():
                natural_template_name = gr.Textbox(
                    label="Template name",
                    value="captured_natural_grasp_red_object",
                )
                natural_template_target = gr.Textbox(label="Target object", value="red object")
                natural_template_stage = gr.Dropdown(
                    ["pre_grasp", "approach", "close", "lift"],
                    label="Action stage",
                    value="pre_grasp",
                )
            natural_template_notes = gr.Textbox(label="Notes", value="", lines=2)
            natural_template_overwrite = gr.Checkbox(
                label="Allow overwrite existing template",
                value=False,
            )
            with gr.Row():
                capture_natural_template_btn = gr.Button("Capture Current Pose as Template")
                preview_natural_template_btn = gr.Button("Preview Captured Template")
                seed_natural_template_btn = gr.Button("Use Captured Template as IK Seed, Plan Only")
            natural_template_status = gr.Textbox(
                label="Capture status",
                value="No natural grasp template captured yet.",
                lines=10,
                interactive=False,
            )
            natural_template_image = gr.Image(label="Captured template preview", interactive=False)
            natural_template_json_path = gr.Textbox(
                label="Captured template JSON",
                value=latest_captured_grasp_template_path(),
                interactive=False,
            )

        with gr.Accordion("Live Recording / Replay", open=False):
            gr.Markdown(
                "Record joint states during an existing ELMiRA action, then replay the motion in the stick viewer. "
                "This is read-only and sends no robot motion command."
            )
            with gr.Row():
                recording_label = gr.Textbox(
                    label="Label",
                    value="right_arm_point_red_object_trial",
                )
                recording_action_type = gr.Dropdown(
                    ["point", "reach", "touch", "pre_grasp", "grasp_attempt"],
                    label="Action type",
                    value="point",
                )
            recording_instruction = gr.Textbox(
                label="Instruction",
                value="Point to the red object with your right hand.",
            )
            recording_target_object = gr.Textbox(label="Target object", value="red object")
            with gr.Row():
                start_recording_btn = gr.Button("Start Joint Recording")
                stop_recording_btn = gr.Button("Stop Joint Recording")
                record_10_btn = gr.Button("Record 10 Seconds")
            recording_status = gr.Textbox(
                label="Recording status",
                value="No joint recording started.",
                lines=8,
                interactive=False,
            )
            recording_path = gr.Textbox(
                label="Recording path",
                value=latest_recording_text(),
                interactive=True,
            )
            with gr.Row():
                load_recording_btn = gr.Button("Load Latest Recording")
                previous_frame_btn = gr.Button("Previous Frame")
                render_frame_btn = gr.Button("Render Selected Frame")
                next_frame_btn = gr.Button("Next Frame")
            replay_frame = gr.Slider(
                label="Replay frame",
                minimum=0,
                maximum=0,
                value=0,
                step=1,
            )
            replay_image = gr.Image(label="Replay stick-figure frame", interactive=False)
            replay_frame_info = gr.Textbox(
                label="Frame info",
                value="Load a recording, then render a frame.",
                lines=8,
                interactive=False,
            )
            with gr.Row():
                captured_template_name = gr.Textbox(
                    label="Template name",
                    value="captured_successful_point_red_object",
                )
                save_frame_template_btn = gr.Button("Save Selected Frame as Template")
            save_template_status = gr.Textbox(
                label="Template save status",
                value="No replay frame saved as template yet.",
                lines=4,
                interactive=False,
            )

        with gr.Accordion("First-Person Trial Recorder", open=False):
            gr.Markdown(
                "Record NICO camera frames during an existing ELMiRA action. "
                "This recorder is read-only and sends no robot motion command."
            )
            with gr.Row():
                first_person_image_topic = gr.Textbox(
                    label="Image topic",
                    value="/nico/vision/left",
                    placeholder="/nico/vision/left or another image topic",
                )
                first_person_trial_id = gr.Textbox(
                    label="Trial ID",
                    value="",
                    placeholder="Auto-generated if empty",
                )
            with gr.Row():
                first_person_action_type = gr.Dropdown(
                    ["point", "reach", "touch", "grasp_attempt"],
                    label="Action type",
                    value="point",
                )
                first_person_target_object = gr.Textbox(label="Target object", value="red object")
            with gr.Row():
                first_person_x = gr.Number(label="X value (m)", value=0.20)
                first_person_y = gr.Number(label="Y value (m)", value=-0.08)
                first_person_z = gr.Number(label="Z value (m)", value=0.14)
            with gr.Row():
                first_person_duration = gr.Number(label="Record duration seconds", value=10)
                first_person_fps = gr.Number(label="Frame rate (fps)", value=5)
            with gr.Row():
                list_image_topics_btn = gr.Button("List Image Topics")
                start_first_person_btn = gr.Button("Start Trial Recording")
                stop_first_person_btn = gr.Button("Stop Trial Recording")
                record_first_person_10_btn = gr.Button("Record 10 Seconds")
            first_person_status = gr.Textbox(
                label="Trial recording status",
                value="No first-person trial recorded yet.",
                lines=8,
                interactive=False,
            )
            first_person_trial_path = gr.Textbox(
                label="Trial record path",
                value=latest_first_person_trial_text(),
                interactive=True,
            )
            with gr.Row():
                load_first_person_btn = gr.Button("Load Latest Trial Frames")
                previous_first_person_btn = gr.Button("Previous Frame")
                render_first_person_btn = gr.Button("Render Selected Frame")
                next_first_person_btn = gr.Button("Next Frame")
            first_person_frame_slider = gr.Slider(
                label="Trial frame",
                minimum=0,
                maximum=0,
                value=0,
                step=1,
            )
            first_person_frame_image = gr.Image(label="Selected camera frame", interactive=False)
            first_person_frame_info_box = gr.Textbox(
                label="Frame info",
                value="Load a trial, then select a frame.",
                lines=8,
                interactive=False,
            )
            save_first_person_key_frame_btn = gr.Button("Save Selected Frame as Key Frame")
            gr.Markdown("Manual label after reviewing the selected frames.")
            with gr.Row():
                first_person_offset_direction = gr.Dropdown(
                    ["centered", "left", "right", "too_high", "too_low", "too_far", "too_close", "unknown"],
                    label="Offset direction",
                    value="unknown",
                )
                first_person_offset_size = gr.Dropdown(
                    ["none", "small", "medium", "large", "unknown"],
                    label="Offset size",
                    value="unknown",
                )
            with gr.Row():
                first_person_contact_status = gr.Dropdown(
                    ["no_contact", "near", "touch", "push", "grasped", "lifted", "unknown"],
                    label="Contact status",
                    value="unknown",
                )
                first_person_outcome = gr.Dropdown(
                    ["success", "partial", "failed", "unknown"],
                    label="Outcome",
                    value="unknown",
                )
                first_person_failure_reason = gr.Dropdown(
                    [
                        "none",
                        "y_offset",
                        "z_too_high",
                        "z_too_low",
                        "x_distance",
                        "vision",
                        "ROS",
                        "IK",
                        "calibration",
                        "hardware",
                        "unknown",
                    ],
                    label="Failure reason",
                    value="unknown",
                )
            first_person_notes = gr.Textbox(label="Notes", value="", lines=3)
            save_first_person_label_btn = gr.Button("Save Manual Trial Label")

        with gr.Accordion("Grasp Debug Mode", open=False):
            gr.Markdown(
                "Break grasp into inspectable stages. Planning stages command no robot motion. "
                "Execution buttons are right-arm/right-hand only and require the safety checkbox."
            )
            with gr.Row():
                grasp_debug_target = gr.Textbox(label="Target object", value="red object")
                grasp_debug_target_z = gr.Number(
                    label="ELMiRA target/table z (m)",
                    value=0.70,
                    precision=3,
                )
                grasp_debug_safety = gr.Checkbox(
                    label="I confirm the robot workspace is clear for this stage.",
                    value=False,
                )
            with gr.Row():
                grasp_debug_image_x = gr.Number(label="Image x", value=None, precision=4)
                grasp_debug_image_y = gr.Number(label="Image y", value=None, precision=4)
                grasp_debug_real_x = gr.Number(label="Real x (m)", value=0.20, precision=3)
                grasp_debug_real_y = gr.Number(label="Real y (m)", value=-0.08, precision=3)
            with gr.Row():
                detect_target_btn = gr.Button("1. Detect Target Only")
                coordinate_only_btn = gr.Button("2. Coordinate Transfer Only")
                plan_grasp_btn = gr.Button("3. Plan Grasp Only, No Motion")
                dump_ik_trace_btn = gr.Button("Dump IK Trace, No Motion")
            with gr.Row():
                execute_pre_grasp_only_btn = gr.Button("4. Execute Pre-Grasp Only")
                align_wrist_only_btn = gr.Button("5. Align Wrist Only")
                open_hand_only_btn = gr.Button("6a. Open Hand Only")
                close_hand_only_btn = gr.Button("6b. Close Hand Only")
            with gr.Row():
                execute_approach_only_btn = gr.Button("7. Execute Approach Only")
                full_grasp_debug_btn = gr.Button("8. Full Grasp Check")
            grasp_debug_status = gr.Textbox(
                label="Grasp debug output",
                value="No grasp debug stage has been run yet.",
                lines=16,
                interactive=False,
            )
            grasp_debug_json_path = gr.Textbox(
                label="Grasp debug JSON log",
                value=grasp_debug_latest_json_path(),
                interactive=False,
            )
            ik_trace_json_path = gr.Textbox(
                label="IK trace JSON log",
                value=ik_trace_latest_json_path(),
                interactive=False,
            )

        with gr.Accordion("Safety Debug: Right Arm Presets", open=False):
            gr.Markdown(
                "These controls may move the real robot if execution is connected. "
                "Use only with the physical robot supervised. Execution uses fixed right-arm presets through the existing nicoros joint-controller service."
            )
            with gr.Row():
                direct_dummy_mode = gr.Checkbox(
                    label="Simulation dummy mode (V-REP, leave OFF for real robot)",
                    value=False,
                )
                start_direct_control_btn = gr.Button("Start Direct Robot Control", variant="primary")
                refresh_direct_control_btn = gr.Button("Refresh Control Status")
                stop_direct_control_btn = gr.Button("Stop Direct Robot Control", variant="stop")
            direct_control_status_box = gr.Textbox(
                label="Direct control status",
                value=direct_control_status(),
                lines=3,
                interactive=False,
            )
            safety_confirmed = gr.Checkbox(label="I confirm the robot workspace is clear.", value=False)
            with gr.Row():
                preview_reset_btn = gr.Button("Preview Reset Pose")
                preview_point_btn = gr.Button("Preview Right-Arm Point Pose")
                preview_reach_btn = gr.Button("Preview Right-Arm Reach Pose")
            with gr.Row():
                preview_pre_grasp_btn = gr.Button("Preview Right-Arm Pre-Grasp Pose")
                preview_touch_forward_btn = gr.Button("Preview Touch Forward")
                preview_touch_side_btn = gr.Button("Preview Touch Side")
            with gr.Row():
                preview_close_hand_btn = gr.Button("Preview Close Hand Pose")
                preview_lift_btn = gr.Button("Preview Lift Pose")
            with gr.Row():
                execute_reset_btn = gr.Button("Execute Reset Pose")
                execute_point_btn = gr.Button("Execute Right-Arm Point Pose")
                execute_reach_btn = gr.Button("Execute Right-Arm Reach Pose")
            with gr.Row():
                execute_pre_grasp_btn = gr.Button("Execute Right-Arm Pre-Grasp Pose")
                execute_touch_forward_btn = gr.Button("Execute Touch Forward")
                execute_touch_side_btn = gr.Button("Execute Touch Side")
            with gr.Row():
                execute_lift_btn = gr.Button("Execute Lift Pose")
                execute_open_hand_btn = gr.Button("Open Right Hand")
                execute_close_hand_btn = gr.Button("Close Right Hand")
            with gr.Accordion("Experimental XYZ IK Control", open=False):
                gr.Markdown(
                    "Constrained ELMiRA IK coordinates for the right arm. "
                    "This is not the stick-preview ground coordinate system."
                )
                with gr.Row():
                    cartesian_x = gr.Slider(label="X forward (m)", minimum=0.10, maximum=0.32, value=0.20, step=0.01)
                    cartesian_y = gr.Slider(label="Y right/left (m)", minimum=-0.25, maximum=0.05, value=-0.20, step=0.01)
                    cartesian_z = gr.Slider(label="Z height (m)", minimum=0.60, maximum=0.78, value=0.70, step=0.01)
                cartesian_orientation = gr.Dropdown(
                    ["point", "reach", "touch_forward", "touch_side"],
                    label="Orientation",
                    value="touch_forward",
                )
                move_xyz_btn = gr.Button("Move Right Arm to XYZ (IK)")
            gr.Markdown("Captured templates reuse joint states recorded from successful ELMiRA motion.")
            captured_template_choices = captured_direct_template_names()
            with gr.Row():
                captured_direct_template = gr.Dropdown(
                    choices=captured_template_choices,
                    value=captured_template_choices[0] if captured_template_choices else None,
                    label="Captured direct template",
                )
                refresh_captured_template_btn = gr.Button("Refresh Captured Templates")
                execute_captured_template_btn = gr.Button("Execute Captured Template")
            captured_template_status = gr.Textbox(
                label="Captured template status",
                value=(
                    f"Captured direct templates available: {', '.join(captured_template_choices)}"
                    if captured_template_choices
                    else "No captured direct templates yet."
                ),
                lines=2,
                interactive=False,
            )
            preset_status = gr.Textbox(
                label="Preset debug status",
                value="No preset debug action yet.",
                lines=8,
                interactive=False,
            )
            preset_image = gr.Image(label="Preset preview image", interactive=False)
            preset_json_path = gr.Textbox(label="Preset debug JSON log path", interactive=False)
            with gr.Accordion("Direct Control Log", open=False):
                direct_control_log = gr.Textbox(
                    label="direct_robot_control.log",
                    value=tail_file(LOG_DIRECT_CONTROL_PATH),
                    lines=8,
                    interactive=False,
                )

        launch_btn.click(
            launch_simple_handler,
            inputs=[
                provider,
                api_key,
                saved_api_key_state,
                offset_x,
                offset_y,
                offset_z,
                dummy_mode,
                use_mllm_grounding,
                mic_device,
            ],
            outputs=[
                launch_status,
                message_input,
                saved_api_key_state,
                api_key_status,
                ros_status,
                pipeline_status,
                robot_mode_status,
                nodes_log,
                sm_log,
            ],
        )

        save_key_btn.click(
            save_api_key_handler,
            inputs=[provider, api_key],
            outputs=[saved_api_key_state, api_key_status, api_key],
        )
        clear_key_btn.click(
            clear_api_key_handler,
            inputs=provider,
            outputs=[saved_api_key_state, api_key_status, api_key],
        )

        stop_btn.click(
            stop_simple_handler,
            inputs=dummy_mode,
            outputs=[
                launch_status,
                ros_status,
                pipeline_status,
                robot_mode_status,
                nodes_log,
                sm_log,
            ],
        )

        refresh_status_btn.click(
            refresh_main_status,
            inputs=dummy_mode,
            outputs=[ros_status, pipeline_status, robot_mode_status],
        )
        try:
            status_timer = gr.Timer(3.0)
            status_timer.tick(
                auto_refresh_main,
                inputs=dummy_mode,
                outputs=[launch_status, ros_status, pipeline_status, robot_mode_status],
                show_progress="hidden",
            )
        except AttributeError:
            demo.load(
                auto_refresh_main,
                inputs=dummy_mode,
                outputs=[launch_status, ros_status, pipeline_status, robot_mode_status],
                every=3.0,
                show_progress="hidden",
            )

        send_btn.click(
            chat_interface,
            inputs=[message_input, transcript],
            outputs=[transcript, message_input],
        )
        message_input.submit(
            chat_interface,
            inputs=[message_input, transcript],
            outputs=[transcript, message_input],
        )
        home_btn.click(
            return_home_handler,
            inputs=transcript,
            outputs=[transcript, message_input],
        )
        talk_btn.click(listen_audio, outputs=voice_status)
        refresh_conversation_btn.click(
            refresh_conversation,
            inputs=transcript,
            outputs=transcript,
        )
        try:
            conversation_timer = gr.Timer(2.0)
            conversation_timer.tick(
                refresh_conversation,
                inputs=transcript,
                outputs=transcript,
                show_progress="hidden",
            )
        except AttributeError:
            demo.load(
                refresh_conversation,
                inputs=transcript,
                outputs=transcript,
                every=2.0,
                show_progress="hidden",
            )

        dry_run_btn.click(
            lambda: run_probe_mode_simple("dry-run"),
            outputs=[tool_status, evidence_status, nodes_log, sm_log],
        )
        ros_check_btn.click(
            lambda: run_probe_mode_simple("ros-check"),
            outputs=[tool_status, evidence_status, nodes_log, sm_log],
        )
        robot_evidence_btn.click(
            lambda: run_probe_mode_simple("robot-evidence"),
            outputs=[tool_status, evidence_status, nodes_log, sm_log],
        )
        service_check_btn.click(
            lambda: run_probe_mode_simple("service-check"),
            outputs=[tool_status, evidence_status, nodes_log, sm_log],
        )
        refresh_logs_btn.click(
            refresh_advanced_status,
            outputs=[evidence_status, nodes_log, sm_log],
        )
        kinematic_btn.click(
            run_kinematic_preview_dashboard,
            inputs=[
                kinematic_template,
                kinematic_target,
                kinematic_target_x,
                kinematic_target_y,
                kinematic_target_z,
            ],
            outputs=[kinematic_status, kinematic_image, kinematic_json_path],
        )
        live_refresh_btn.click(
            refresh_live_pose_dashboard,
            outputs=[live_status, live_image, live_hand_xyz, live_timestamp],
        )
        capture_natural_template_btn.click(
            run_capture_natural_grasp_template_dashboard,
            inputs=[
                natural_template_name,
                natural_template_target,
                natural_template_stage,
                natural_template_notes,
                natural_template_overwrite,
            ],
            outputs=[natural_template_status, natural_template_image, natural_template_json_path],
        )
        preview_natural_template_btn.click(
            run_preview_captured_natural_grasp_template_dashboard,
            inputs=natural_template_name,
            outputs=[natural_template_status, natural_template_image, natural_template_json_path],
        )
        seed_natural_template_btn.click(
            run_captured_template_seed_plan_dashboard,
            inputs=natural_template_name,
            outputs=[natural_template_status, natural_template_image, natural_template_json_path],
        )
        start_direct_control_btn.click(
            start_direct_control,
            inputs=direct_dummy_mode,
            outputs=[direct_control_status_box, direct_control_log],
        )
        refresh_direct_control_btn.click(
            lambda: (direct_control_status(), tail_file(LOG_DIRECT_CONTROL_PATH)),
            outputs=[direct_control_status_box, direct_control_log],
        )
        stop_direct_control_btn.click(
            stop_direct_control,
            outputs=[direct_control_status_box, direct_control_log],
        )
        start_recording_btn.click(
            start_joint_recording_dashboard,
            inputs=[
                recording_label,
                recording_instruction,
                recording_target_object,
                recording_action_type,
            ],
            outputs=[recording_status, recording_path],
        )
        stop_recording_btn.click(
            stop_joint_recording_dashboard,
            outputs=[recording_status, recording_path, replay_frame, replay_frame_info],
        )
        record_10_btn.click(
            record_10_seconds_dashboard,
            inputs=[
                recording_label,
                recording_instruction,
                recording_target_object,
                recording_action_type,
            ],
            outputs=[recording_status, recording_path, replay_frame, replay_frame_info],
        )
        load_recording_btn.click(
            load_latest_recording_dashboard,
            inputs=recording_path,
            outputs=[recording_path, replay_frame, replay_frame_info],
        )
        render_frame_btn.click(
            render_replay_frame_dashboard,
            inputs=[recording_path, replay_frame],
            outputs=[replay_image, replay_frame_info, replay_frame],
        )
        previous_frame_btn.click(
            previous_replay_frame,
            inputs=[recording_path, replay_frame],
            outputs=[replay_image, replay_frame_info, replay_frame],
        )
        next_frame_btn.click(
            next_replay_frame,
            inputs=[recording_path, replay_frame],
            outputs=[replay_image, replay_frame_info, replay_frame],
        )
        save_frame_template_btn.click(
            save_selected_frame_template_dashboard,
            inputs=[
                recording_path,
                replay_frame,
                captured_template_name,
                recording_action_type,
                recording_target_object,
            ],
            outputs=save_template_status,
        )
        list_image_topics_btn.click(
            list_first_person_image_topics_dashboard,
            outputs=first_person_status,
        )
        start_first_person_btn.click(
            start_first_person_trial_recording_dashboard,
            inputs=[
                first_person_image_topic,
                first_person_trial_id,
                first_person_action_type,
                first_person_target_object,
                first_person_x,
                first_person_y,
                first_person_z,
                first_person_fps,
            ],
            outputs=[first_person_status, first_person_trial_path],
        )
        stop_first_person_btn.click(
            stop_first_person_trial_recording_dashboard,
            outputs=[
                first_person_status,
                first_person_trial_path,
                first_person_frame_slider,
                first_person_frame_info_box,
            ],
        )
        record_first_person_10_btn.click(
            record_first_person_10_seconds_dashboard,
            inputs=[
                first_person_image_topic,
                first_person_trial_id,
                first_person_action_type,
                first_person_target_object,
                first_person_x,
                first_person_y,
                first_person_z,
                first_person_duration,
                first_person_fps,
            ],
            outputs=[
                first_person_status,
                first_person_trial_path,
                first_person_frame_slider,
                first_person_frame_info_box,
            ],
        )
        load_first_person_btn.click(
            load_latest_first_person_trial_dashboard,
            inputs=first_person_trial_path,
            outputs=[
                first_person_trial_path,
                first_person_frame_slider,
                first_person_frame_image,
                first_person_frame_info_box,
            ],
        )
        render_first_person_btn.click(
            render_first_person_frame_dashboard,
            inputs=[first_person_trial_path, first_person_frame_slider],
            outputs=[first_person_frame_image, first_person_frame_info_box, first_person_frame_slider],
        )
        previous_first_person_btn.click(
            previous_first_person_frame,
            inputs=[first_person_trial_path, first_person_frame_slider],
            outputs=[first_person_frame_image, first_person_frame_info_box, first_person_frame_slider],
        )
        next_first_person_btn.click(
            next_first_person_frame,
            inputs=[first_person_trial_path, first_person_frame_slider],
            outputs=[first_person_frame_image, first_person_frame_info_box, first_person_frame_slider],
        )
        save_first_person_key_frame_btn.click(
            save_first_person_key_frame_dashboard,
            inputs=[first_person_trial_path, first_person_frame_slider],
            outputs=first_person_status,
        )
        save_first_person_label_btn.click(
            save_first_person_manual_label_dashboard,
            inputs=[
                first_person_trial_path,
                first_person_offset_direction,
                first_person_offset_size,
                first_person_contact_status,
                first_person_outcome,
                first_person_failure_reason,
                first_person_notes,
            ],
            outputs=first_person_status,
        )
        grasp_debug_inputs = [
            grasp_debug_target,
            grasp_debug_image_x,
            grasp_debug_image_y,
            grasp_debug_real_x,
            grasp_debug_real_y,
            grasp_debug_target_z,
        ]
        detect_target_btn.click(
            run_grasp_debug_detect,
            inputs=grasp_debug_inputs,
            outputs=[
                grasp_debug_status,
                grasp_debug_json_path,
                grasp_debug_image_x,
                grasp_debug_image_y,
            ],
        )
        coordinate_only_btn.click(
            run_grasp_debug_coordinate,
            inputs=grasp_debug_inputs,
            outputs=[
                grasp_debug_status,
                grasp_debug_json_path,
                grasp_debug_image_x,
                grasp_debug_image_y,
                grasp_debug_real_x,
                grasp_debug_real_y,
            ],
        )
        plan_grasp_btn.click(
            run_grasp_debug_plan,
            inputs=grasp_debug_inputs,
            outputs=[grasp_debug_status, grasp_debug_json_path],
        )
        dump_ik_trace_btn.click(
            run_ik_trace_dump_dashboard,
            inputs=grasp_debug_inputs,
            outputs=[grasp_debug_status, ik_trace_json_path],
        )
        execute_pre_grasp_only_btn.click(
            lambda target, image_x, image_y, real_x, real_y, target_z, safety: run_grasp_debug_dashboard(
                "execute-pre-grasp",
                target,
                image_x,
                image_y,
                real_x,
                real_y,
                target_z,
                safety,
            ),
            inputs=grasp_debug_inputs + [grasp_debug_safety],
            outputs=[grasp_debug_status, grasp_debug_json_path],
        )
        align_wrist_only_btn.click(
            lambda target, image_x, image_y, real_x, real_y, target_z, safety: run_grasp_debug_dashboard(
                "align-wrist",
                target,
                image_x,
                image_y,
                real_x,
                real_y,
                target_z,
                safety,
            ),
            inputs=grasp_debug_inputs + [grasp_debug_safety],
            outputs=[grasp_debug_status, grasp_debug_json_path],
        )
        open_hand_only_btn.click(
            lambda target, image_x, image_y, real_x, real_y, target_z, safety: run_grasp_debug_dashboard(
                "open-hand",
                target,
                image_x,
                image_y,
                real_x,
                real_y,
                target_z,
                safety,
            ),
            inputs=grasp_debug_inputs + [grasp_debug_safety],
            outputs=[grasp_debug_status, grasp_debug_json_path],
        )
        close_hand_only_btn.click(
            lambda target, image_x, image_y, real_x, real_y, target_z, safety: run_grasp_debug_dashboard(
                "close-hand",
                target,
                image_x,
                image_y,
                real_x,
                real_y,
                target_z,
                safety,
            ),
            inputs=grasp_debug_inputs + [grasp_debug_safety],
            outputs=[grasp_debug_status, grasp_debug_json_path],
        )
        execute_approach_only_btn.click(
            lambda target, image_x, image_y, real_x, real_y, target_z, safety: run_grasp_debug_dashboard(
                "execute-approach",
                target,
                image_x,
                image_y,
                real_x,
                real_y,
                target_z,
                safety,
            ),
            inputs=grasp_debug_inputs + [grasp_debug_safety],
            outputs=[grasp_debug_status, grasp_debug_json_path],
        )
        full_grasp_debug_btn.click(
            lambda target, image_x, image_y, real_x, real_y, target_z, safety: run_grasp_debug_dashboard(
                "full-grasp",
                target,
                image_x,
                image_y,
                real_x,
                real_y,
                target_z,
                safety,
            ),
            inputs=grasp_debug_inputs + [grasp_debug_safety],
            outputs=[grasp_debug_status, grasp_debug_json_path],
        )
        preview_reset_btn.click(
            lambda safety: run_safe_preset_dashboard("reset_pose", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        preview_point_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_point_pose", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        preview_reach_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_reach_pose", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        preview_pre_grasp_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_pre_grasp_pose", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        preview_touch_forward_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_touch_forward", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        preview_touch_side_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_touch_side", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        preview_close_hand_btn.click(
            lambda safety: run_safe_preset_dashboard("close_hand_pose", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        preview_lift_btn.click(
            lambda safety: run_safe_preset_dashboard("lift_pose", "preview", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_reset_btn.click(
            lambda safety: run_safe_preset_dashboard("reset_pose", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_point_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_point_pose", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_reach_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_reach_pose", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_pre_grasp_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_pre_grasp_pose", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_touch_forward_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_touch_forward", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_touch_side_btn.click(
            lambda safety: run_safe_preset_dashboard("right_arm_touch_side", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_lift_btn.click(
            lambda safety: run_safe_preset_dashboard("lift_pose", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_open_hand_btn.click(
            lambda safety: run_safe_preset_dashboard("open_right_hand", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        execute_close_hand_btn.click(
            lambda safety: run_safe_preset_dashboard("close_right_hand", "execute", safety),
            inputs=safety_confirmed,
            outputs=[preset_status, preset_image, preset_json_path],
        )
        move_xyz_btn.click(
            run_cartesian_xyz_dashboard,
            inputs=[cartesian_x, cartesian_y, cartesian_z, cartesian_orientation, safety_confirmed],
            outputs=[preset_status, preset_image, preset_json_path],
        )
        refresh_captured_template_btn.click(
            refresh_captured_direct_templates,
            outputs=[captured_direct_template, captured_template_status],
        )
        execute_captured_template_btn.click(
            run_captured_direct_template_dashboard,
            inputs=[captured_direct_template, safety_confirmed],
            outputs=[preset_status, preset_image, preset_json_path],
        )
        try:
            logs_timer = gr.Timer(3.0)
            logs_timer.tick(
                refresh_advanced_status,
                outputs=[evidence_status, nodes_log, sm_log],
                show_progress="hidden",
            )
        except AttributeError:
            demo.load(
                refresh_advanced_status,
                outputs=[evidence_status, nodes_log, sm_log],
                every=3.0,
                show_progress="hidden",
            )

    return demo


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def cleanup_on_exit() -> None:
    global direct_control_process, direct_control_log_handle, direct_ik_process, direct_ik_log_handle
    global joint_record_process, joint_record_log_handle, first_person_record_process, first_person_record_log_handle
    if process_alive(direct_ik_process):
        try:
            os.killpg(os.getpgid(direct_ik_process.pid), signal.SIGINT)
            direct_ik_process.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(direct_ik_process.pid), signal.SIGKILL)
            except Exception:
                pass
        direct_ik_process = None
    if direct_ik_log_handle:
        try:
            direct_ik_log_handle.close()
        except Exception:
            pass
        direct_ik_log_handle = None
    if process_alive(direct_control_process):
        try:
            os.killpg(os.getpgid(direct_control_process.pid), signal.SIGINT)
            direct_control_process.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(direct_control_process.pid), signal.SIGKILL)
            except Exception:
                pass
        direct_control_process = None
    if direct_control_log_handle:
        try:
            direct_control_log_handle.close()
        except Exception:
            pass
        direct_control_log_handle = None
    if process_alive(joint_record_process):
        try:
            os.killpg(os.getpgid(joint_record_process.pid), signal.SIGINT)
            joint_record_process.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(joint_record_process.pid), signal.SIGKILL)
            except Exception:
                pass
        joint_record_process = None
    if joint_record_log_handle:
        try:
            joint_record_log_handle.close()
        except Exception:
            pass
        joint_record_log_handle = None
    if process_alive(first_person_record_process):
        try:
            os.killpg(os.getpgid(first_person_record_process.pid), signal.SIGINT)
            first_person_record_process.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(first_person_record_process.pid), signal.SIGKILL)
            except Exception:
                pass
        first_person_record_process = None
    if first_person_record_log_handle:
        try:
            first_person_record_log_handle.close()
        except Exception:
            pass
        first_person_record_log_handle = None
    if process_alive(launch_process_nodes) or process_alive(launch_process_sm):
        stop_robot()


def main() -> int:
    if gr is None:
        print(f"Gradio import failed: {GRADIO_IMPORT_ERROR}", flush=True)
        print(fallback_commands(), flush=True)
        return 1

    try:
        demo = build_dashboard()
    except Exception as exc:
        print(f"Dashboard build failed: {exc}", flush=True)
        print(fallback_commands(), flush=True)
        return 1

    atexit.register(cleanup_on_exit)

    for port in (7860, 7861):
        if not port_available(port):
            continue
        local_url = f"http://127.0.0.1:{port}"
        print(f"ELMiRA FYP2 Monitoring Console: {local_url}", flush=True)
        try:
            demo.queue().launch(server_name="0.0.0.0", server_port=port)
            return 0
        except Exception as exc:
            print(f"Port {port} failed: {exc}", flush=True)
            continue

    print("Could not start dashboard on 7860 or 7861.", flush=True)
    print(fallback_commands(), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
