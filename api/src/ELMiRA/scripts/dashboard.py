#!/usr/bin/env python3
import gradio as gr
import subprocess
import os
import signal
import time
import psutil
import threading

# Global state for display
displayed_cmd_nodes = "Waiting for launch..."
displayed_cmd_sm = "Waiting for launch..."

# Global Process Handles
launch_process_nodes = None
launch_process_sm = None
roscore_process = None

def launch_robot(provider, use_mllm_grounding, use_dummy, api_key, mic_device=""):
    """
    Launches:
    1. init_nodes_v2.launch (Terminal 2)
    2. state_machine.py (Terminal 3)
    """
    global launch_process_nodes, launch_process_sm, displayed_cmd_nodes, displayed_cmd_sm
    
    # ... (Checks) ...
    
    # Update Globals for Display
    displayed_cmd_nodes = cmd_nodes
    displayed_cmd_sm = cmd_sm
    
    # ... (Launch Logic) ...

def read_log_safe(path, lines=20):
    if not os.path.exists(path):
        return "Waiting for log file..."
    try:
        # Simple tail implementation
        with open(path, 'r') as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except:
        return "Error reading log."

def refresh_terminals():
    cwd = WORKSPACE_ROOT
    log_nodes = read_log_safe(os.path.join(cwd, "launch_nodes.log"))
    log_sm = read_log_safe(os.path.join(cwd, "launch_sm.log"))
    
    # Terminal 1 is static roscore
    return displayed_cmd_nodes, log_nodes, displayed_cmd_sm, log_sm



def install_requirements():
    """Installs requirements via pip."""
    try:
        # Using sys.executable to ensure we use the same python environment
        import sys
        
        # Install v2 requirements
        cmd = [sys.executable, "-m", "pip", "install", "-r", "src/ELMiRA/requirements_v2.txt", "--user"]
        yield f"Running: {' '.join(cmd)}...\n"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.getcwd())
        
        for line in process.stdout:
            yield line
            
        process.wait()
        if process.returncode == 0:
            yield "\n✅ Installation Complete!"
        else:
            yield f"\n❌ Installation Failed with code {process.returncode}"
            
    except Exception as e:
        yield f"\n❌ Error: {str(e)}"

# --- Configuration Persistence ---
import json

CONFIG_FILE = os.path.expanduser("~/.elmira_config.json")

def load_config():
    """Loads configuration from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    return {}

def save_config(provider, api_key):
    """Saves current configuration."""
    config = {
        "provider": provider,
        "api_key": api_key
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        print(f"Error saving config: {e}")

# Initial Load
loaded_config = load_config()
default_provider = loaded_config.get("provider", "OpenAI")
default_api_key = loaded_config.get("api_key", "")

# Determine Layout
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming structure: .../NICO-software/api/src/ELMiRA/scripts/dashboard.py
# Root should be .../NICO-software/api (where activate.bash is)
# SCRIPT_DIR = .../scripts
# .. = src/ELMiRA
# ../.. = src
# ../../.. = api
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

def check_ros_status():
    """Checks if roscore is running."""
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'roscore' or proc.info['name'] == 'rosmaster':
            return "🟢 ROS Core Running"
    return "🔴 ROS Core Stopped"

def toggle_roscore(start):
    global roscore_process
    if start:
        if check_ros_status() == "🟢 ROS Core Running":
            return "Already running"
        roscore_process = subprocess.Popen(['roscore'])
        time.sleep(2)
        return "Started roscore"
    else:
        # Kill roscore - a bit tricky, usually requires killing rosmaster and roscore
        subprocess.run(['pkill', 'roscore'])
        subprocess.run(['pkill', 'rosmaster'])
        return "Stopped roscore"

def launch_robot(provider, use_mllm_grounding, use_dummy, api_key, mic_device="", offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Launches:
    1. init_nodes_v2.launch (Terminal 2)
    2. state_machine.py (Terminal 3)
    """
    global launch_process_nodes, launch_process_sm, displayed_cmd_nodes, displayed_cmd_sm
    
    if launch_process_nodes is not None or launch_process_sm is not None:
        return "⚠️ Robot already running! Stop it first."
    
    if not api_key:
        return "❌ Error: API Key is required!"
        
    # Determine which key to export based on provider
    key_var = "OPENAI_API_KEY" if provider == "OpenAI" else "GOOGLE_API_KEY"
    
    # Use calculated root to be safe regardless of where dashboard is run from
    cwd = WORKSPACE_ROOT
    
    # Log files
    log_nodes_path = os.path.join(cwd, "launch_nodes.log")
    log_sm_path = os.path.join(cwd, "launch_sm.log")
    
    # --- Terminal 2 Command: Nodes ---
    cmd_nodes = (
        f"cd {cwd} && "
        f"source activate.bash && "
        f"source devel/setup.bash && "
        f"export {key_var}='{api_key}' && "
        f"export CUDA_VISIBLE_DEVICES='' && "
        f"export PYTHONUNBUFFERED=1 && "
        f"roslaunch elmira init_nodes_v2.launch "
        f"mllm_provider:={provider.lower()} "
        f"use_mllm_grounding:={'true' if use_mllm_grounding else 'false'} "
        f"dummy:={'true' if use_dummy else 'false'} "
        f"mic_device:='{mic_device}' "
        f"offset_x:={offset_x} "
        f"offset_y:={offset_y} "
        f"offset_z:={offset_z}"
    )
    
    # --- Terminal 3 Command: State Machine ---
    cmd_sm = (
        f"cd {cwd} && "
        f"source activate.bash && "
        f"source devel/setup.bash && "
        f"export PYTHONUNBUFFERED=1 && "
        f"rosrun elmira state_machine.py"
    )

    # Update Globals for Display
    displayed_cmd_nodes = cmd_nodes
    displayed_cmd_sm = cmd_sm
    
    try:
        # cleanup zombies from previous dashboard sessions
        subprocess.run(['pkill', '-f', 'roslaunch elmira'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-f', 'state_machine.py'], stderr=subprocess.DEVNULL)
        time.sleep(2) # wait for cleanup

        # Open Log Files - Line Buffered to see output immediately
        f_nodes = open(log_nodes_path, "w", buffering=1)
        f_sm = open(log_sm_path, "w", buffering=1)
        
        # Launch Nodes
        launch_process_nodes = subprocess.Popen(
            ['bash', '-c', cmd_nodes], 
            stdout=f_nodes, 
            stderr=subprocess.STDOUT, 
            text=True,
            cwd=cwd,
            preexec_fn=os.setsid
        )
        
        # Wait a bit for nodes to initialize before starting State Machine
        time.sleep(15)
        
        # Launch State Machine
        launch_process_sm = subprocess.Popen(
            ['bash', '-c', cmd_sm],
            stdout=f_sm,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            preexec_fn=os.setsid
        )
        
        return (
            f"🚀 System Launched!\n"
            f"Logs: {log_nodes_path}\n"
            f"      {log_sm_path}\n"
            f"Nodes PID: {launch_process_nodes.pid}\n"
            f"State Machine PID: {launch_process_sm.pid}"
        )
        
    except FileNotFoundError:
        return "❌ Commands not found. Check setup."
    except Exception as e:
        return f"❌ Launch Error: {str(e)}"

def stop_robot():
    global launch_process_nodes, launch_process_sm
    status = []
    
    if launch_process_sm:
        try:
            os.killpg(os.getpgid(launch_process_sm.pid), signal.SIGINT)
            launch_process_sm.wait(timeout=2)
        except:
            try:
                 os.killpg(os.getpgid(launch_process_sm.pid), signal.SIGKILL)
            except: pass
        launch_process_sm = None
        status.append("Stopped State Machine")

    if launch_process_nodes:
        try:
            os.killpg(os.getpgid(launch_process_nodes.pid), signal.SIGINT)
            launch_process_nodes.wait(timeout=5)
        except:
             try:
                 os.killpg(os.getpgid(launch_process_nodes.pid), signal.SIGKILL)
             except: pass
        launch_process_nodes = None
        status.append("Stopped Nodes")

    if monitor_process:
        try:
             monitor_process.terminate()
             monitor_process.wait(timeout=1)
        except: pass
        # Don't set to None immediately if we want to restart it? 
        # Actually start_monitor_thread creates a new one.
        # But global variable needs to be cleared or re-assigned.
        # The start_monitor_thread loop checks 'check_ros_status'.
        # If ROS is stopped, the loop keeps spinning waiting for ROS.
        # If we kill the process, the loop crashes or needs to restart the process.
        # Implemented logic: wrapper loop inside thread.
        # But wait, start_monitor_thread just runs the function.
        # monitor_conversation_loop starts one process and reads it until EOF.
        # If we kill the process, readline returns empty -> loop breaks.
        # The thread terminates.
        # Perfect.
        status.append("Stopped Monitor")

    if not status:
        return "Robot not running"
    return "🛑 " + ", ".join(status)

def get_launch_logs():
    """Generator to stream launch logs"""
    global launch_process
    if not launch_process:
        yield "Robot not running..."
        return
        
    # access stdout of the process - simple non-blocking read is hard in simple python logic without threads
    # For now, just a placeholder. Real streaming requires a thread reading stdout to a queue.
    yield "Log streaming not fully implemented in this basic script.\nCheck terminal for logs."

def chat_interface(message, history):
    # shell out to rosservice with correct env
    cmd_str = (
        f"source {os.getcwd()}/devel/setup.bash && "
        f"rosservice call /mllm_chat \"prompt: '{message}'\""
    )
    
    try:
        result = subprocess.run(
            ['bash', '-c', cmd_str], 
            capture_output=True, 
            text=True, 
            timeout=45
        )
        print(f"DEBUG: Raw rosservice output: {result.stdout}") # Log for debugging
        
        if result.returncode == 0:
            output = result.stdout.strip()
            
            # Robust Parsing Strategy: Regex Extraction
            # We look specifically for the "text": "..." pattern which contains the spoken response.
            # This avoids issues with broken JSON, python-dict string formats, or escaped characters.
            import re
            
            # Pattern: "text": "CONTENT" or 'text': 'CONTENT'
            # \1 matches the opening quote type to ensure we find the correct closing quote
            # re.DOTALL allows matching across newlines
            text_match = re.search(r'[\'"]text[\'"]\s*:\s*([\'"])(.*?)\1', output, re.DOTALL)
            
            if text_match:
                # Group 2 is the content inside the quotes
                spoken_text = text_match.group(2)
                
                # --- TTS INTEGRATION ---
                # Attempt to make the robot speak the response
                try:
                    # Construct the service call command
                    # field names: text, language, pitch, speed, blocking
                    # YAML-like syntax for rosservice call
                    tts_cmd = (
                        f"source {os.getcwd()}/devel/setup.bash && "
                        f"rosservice call /nico/text_to_speech/say \"{{text: '{spoken_text}', language: 'en', pitch: 0.25, speed: 1.0, blocking: false}}\""
                    )
                    # Run in background so UI doesn't freeze
                    subprocess.Popen(['bash', '-c', tts_cmd])
                except Exception as e:
                    print(f"TTS Trigger Error: {e}")
                # -----------------------

                return spoken_text
                
            # --- Fallback Strategy ---
            # If no "text" field found, clean up the response wrapper
            
            # If output looks like: response: "..."
            if "response:" in output:
                 val = output.split("response:")[1].strip()
                 # Strip outer quotes if they exist
                 if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                     val = val[1:-1]
                 # Unescape newlines
                 val = val.replace("\\n", "\n")
                 
                 # Try TTS fallback here too
                 try:
                    tts_cmd = (
                        f"source {os.getcwd()}/devel/setup.bash && "
                        f"rosservice call /nico/text_to_speech/say \"{{text: '{val}', language: 'en', pitch: 0.25, speed: 1.0, blocking: false}}\""
                    )
                    subprocess.Popen(['bash', '-c', tts_cmd])
                 except: pass

                 return val
                 
            # Return raw output if all else fails
            return output
                
        else:
            return f"Error: {result.stderr}"
            
    except Exception as e:
        return f"System Error: {str(e)}"

# --- ASR Functionality ---
stop_listening_flag = False

def list_audio_devices():
    """Lists available audio input devices."""
    try:
        import sounddevice
        devices = sounddevice.query_devices()
        inputs = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                inputs.append(f"{i}: {d['name']}")
        return inputs
    except ImportError:
        return ["Error: sounddevice not found"]
    except Exception as e:
        return [f"Error: {e}"]

def stop_listening():
    """Cancels the current ASR goal but expects a result."""
    global stop_listening_flag
    stop_listening_flag = True
    
    # Send cancel to action server command topic (ActionLib convention: /cancel)
    # Target: /speech_asr/cancel
    # Type: actionlib_msgs/GoalID
    # Content: empty id to cancel all
    cmd = (
        f"source {os.getcwd()}/devel/setup.bash && "
        f"rostopic pub /speech_asr/cancel actionlib_msgs/GoalID \"{{}}\" -1"
    )
    subprocess.Popen(['bash', '-c', cmd])
    return "🛑 Stop requested. Processing..."

def listen_audio():
    """Triggers ASR and streams recognized text."""
    global stop_listening_flag
    stop_listening_flag = False
    
    yield "👂 Robot is listening..."
    
    # 1. Start the goal with LIVE TEXT enabled
    cmd_pub = (
        f"source {os.getcwd()}/devel/setup.bash && "
        f"rostopic pub /speech_asr/goal elmira/PerformASRActionGoal "
        f"\"{{header: {{seq: 0, stamp: now, frame_id: ''}}, goal_id: {{stamp: now, id: ''}}, "
        f"goal: {{detect_start: true, detect_stop: true, start_timeout: 5.0, "
        f"min_duration: 1.0, max_duration: 10.0, min_period: 0.5, live_text: true}}}}\" -1"
    )
    
    # 2. Monitor FEEDBACK for partial results
    # We grep for 'cur_text'
    cmd_feedback = (
         f"source {os.getcwd()}/devel/setup.bash && "
         f"rostopic echo /speech_asr/feedback"
    )
    
    # 3. Monitor RESULT for final text
    cmd_result = (
         f"source {os.getcwd()}/devel/setup.bash && "
         f"rostopic echo /speech_asr/result"
    )
    
    try:
        # Start processes
        pub_proc = subprocess.Popen(['bash', '-c', cmd_pub])
        
        # Start monitors
        feedback_proc = subprocess.Popen(
            ['bash', '-c', cmd_feedback], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        result_proc = subprocess.Popen(
            ['bash', '-c', cmd_result], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        start_time = time.time()
        final_text = None
        
        import re
        import select
        
        # Polling loop
        while time.time() - start_time < 30: # 30s timeout to allow processing
            # If stopped, we DON'T break, we wait for result
            # But if too much time passes after stop, we force break
            
            # Check for data on feedback pipe
            reads = [feedback_proc.stdout.fileno(), result_proc.stdout.fileno()]
            ret = select.select(reads, [], [], 0.1) # 100ms timeout
            
            if feedback_proc.stdout.fileno() in ret[0]:
                line = feedback_proc.stdout.readline()
                # Parse: cur_text: "..."
                match = re.search(r'cur_text:\s*"?(.*?)"?\n', line)
                if match:
                    partial = match.group(1)
                    if partial:
                        prefix = "🛑 Processing: " if stop_listening_flag else "👂 Hearing: "
                        yield f"{prefix}{partial}..."
            
            if result_proc.stdout.fileno() in ret[0]:
                line = result_proc.stdout.readline()
                # Parse text: "..."
                match = re.search(r'text:\s*"?(.*?)"?\n', line)
                if match:
                    final_text = match.group(1)
                    yield final_text
                    break
                    
            if pub_proc.poll() is not None:
                # Pub finished, just waiting
                pass
                
        # Cleanup
        feedback_proc.terminate()
        result_proc.terminate()
        if final_text:
            return # Yielded above
            
        if not final_text:
             if stop_listening_flag:
                 yield "⚠️ Stopped manually, but no final text received."
             else:
                 yield "❌ No speech detected / Timeout."

    except Exception as e:
        yield f"Error: {e}"


# --- UI Setup ---

# Initial device listing
mic_options = list_audio_devices()
default_mic = mic_options[-1] if mic_options else None 
# Usually default is good, but user can pick specific one. 
# We default to empty string "" which means "default device", 
# but passing an index is better if selected.

def parse_mic_choice(choice):
    if not choice or "Error" in choice:
        return ""
    return choice.split(":")[0]

def launch_robot_wrapper(provider, use_mllm_grounding, use_dummy, api_key, mic_choice, offset_x, offset_y, offset_z):
    # Save config on launch
    save_config(provider, api_key)
    
    mic_idx = parse_mic_choice(mic_choice)
    return launch_robot(provider, use_mllm_grounding, use_dummy, api_key, mic_idx, offset_x, offset_y, offset_z)
    
# Update launch_robot to accept mic_device argument in call
    
# --- UI Setup ---

# --- Chat Sync Logic ---
conversation_history = []
monitor_process = None

import rospy
import std_msgs.msg

def get_chat_updates():
    return conversation_history

def chat_callback(msg):
    global conversation_history
    clean_line = msg.data.strip()
    
    # Remove quotes if present
    if len(clean_line) >= 2 and clean_line[0] == '"' and clean_line[-1] == '"':
        clean_line = clean_line[1:-1]
        
    # Parse Prefix
    if clean_line.startswith("USER:"):
        msg = clean_line[5:].strip()
        conversation_history.append([msg, None])
    elif clean_line.startswith("ROBOT:"):
        msg = clean_line[6:].strip()
        # Attach to last user msg if possible
        if conversation_history and conversation_history[-1][1] is None:
            conversation_history[-1][1] = msg
        else:
             # Or start new line if unexpected robot msg
            conversation_history.append([None, msg])
            
    # Keep history manageable (e.g., last 50)
    if len(conversation_history) > 50:
        conversation_history.pop(0)

def monitor_conversation_loop():
    global monitor_process
    
    # Wait for ROS to be ready
    while check_ros_status() != "🟢 ROS Core Running":
        time.sleep(2)
        
    try:
        # Initialize ROS node for this thread if not already initialized
        # This is safe to call multiple times, but only the first call will initialize
        # the node. Subsequent calls with the same name will be ignored.
        # Using anonymous=True to avoid name conflicts if multiple dashboard instances run.
        rospy.init_node('elmira_dashboard_monitor', anonymous=True, disable_signals=True)
        rospy.loginfo("Dashboard: Subscribing to conversation topic")
        rospy.Subscriber("/elmira/conversation", std_msgs.msg.String, chat_callback)
        # Just keep thread alive
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"Monitor error: {e}")

def start_monitor_thread():
    t = threading.Thread(target=monitor_conversation_loop, daemon=True)
    t.start()

with gr.Blocks(title="ELMiRA Ops Center", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 ELMiRA Operations Center")
    
    with gr.Tabs():
        # --- TAB 1: SETUP ---
        with gr.TabItem("🛠️ Setup"):
            gr.Markdown("### Environment Setup")
            install_btn = gr.Button("Install Requirements (v2)", variant="primary")
            config_output = gr.Textbox(label="Installation Log", lines=10, interactive=False)
            
            install_btn.click(install_requirements, outputs=config_output)
            
        # --- TAB 2: CONTROL ---
        with gr.TabItem("⚙️ Mission Control"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Configuration")
                    provider_drp = gr.Dropdown(
                        ["OpenAI", "Google"], 
                        label="AI Provider", 
                        value=default_provider
                    )
                    api_key_input = gr.Textbox(
                        label="API Key", 
                        placeholder="sk-...", 
                        type="password",
                        value=default_api_key
                    )
                    # grounding_chk = gr.Checkbox(label="Enable visual grounding", value=True)
                    grounding_chk = gr.State(value=True)
                    # dummy_chk = gr.Checkbox(label="Sim Mode / No Robot", value=True)
                    dummy_mode_state = gr.State(value=False)
                    
                    # Microphone Selection
                    # mic_drp = gr.Dropdown(
                    #     label="Microphone", 
                    #     choices=mic_options, 
                    #     value=default_mic,
                    #     interactive=True
                    # )
                    # refresh_mic_btn = gr.Button("Refresh Mics", size="sm")
                    
                    # Use default microphone (empty string)
                    mic_drp_state = gr.State(value="")

                    # def refresh_mics():
                    #      mics = list_audio_devices()
                    #      return gr.Dropdown(choices=mics, value=mics[-1] if mics else None)
                    
                    # refresh_mic_btn.click(refresh_mics, outputs=mic_drp)
                    
                    gr.Markdown("### 📐 Calibration Offsets")
                    offset_x_sld = gr.Slider(label="X Offset (Forward/Back)", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
                    offset_y_sld = gr.Slider(label="Y Offset (Left/Right)", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
                    offset_z_sld = gr.Slider(label="Z Offset (Up/Down)", minimum=-0.2, maximum=0.2, value=0.0, step=0.01)
                
                with gr.Column():
                    gr.Markdown("### Status")
                    ros_status = gr.Textbox(label="ROS Core Status", value=check_ros_status())
                    refresh_btn = gr.Button("Refresh Status")
                    
                    refresh_btn.click(check_ros_status, outputs=ros_status)
            
            # with gr.Row():
            #     start_ros_btn = gr.Button("Start ROS Core")
            #     stop_ros_btn = gr.Button("Kill ROS Core")
                
            #     start_ros_btn.click(fn=lambda: toggle_roscore(True), outputs=ros_status)
            #     stop_ros_btn.click(fn=lambda: toggle_roscore(False), outputs=ros_status)
                
            gr.Markdown("---")
            gr.Markdown("### Robot Launch")
            
            
            with gr.Row():
                launch_btn = gr.Button("🚀 Launch ELMiRA", variant="primary")
                stop_btn = gr.Button("🛑 Stop System", variant="stop")
            
            launch_log = gr.Textbox(label="Launch Status", lines=2)
            
            # Use the wrapper to handle mic strings
            launch_btn.click(
                launch_robot_wrapper, 
                inputs=[provider_drp, grounding_chk, dummy_mode_state, api_key_input, mic_drp_state, offset_x_sld, offset_y_sld, offset_z_sld], 
                outputs=launch_log
            )
            stop_btn.click(stop_robot, outputs=launch_log)
 
        # --- TAB 3: CHAT ---
        with gr.TabItem("💬 Command Center"):
            
            with gr.Row():
                listen_btn = gr.Button("🎤 Listen (ASR)", variant="secondary")
                stop_listen_btn = gr.Button("🛑 Stop Listen", variant="stop")
            
            status_box = gr.Textbox(label="ASR Status", lines=1, interactive=False)
            msg_box = gr.Textbox(placeholder="Talk to the robot via MLLM... (Requires Robot Running)")
            chatbot = gr.Chatbot(height=400)
            
            # Helper to clear textbox after send
            def user_msg(user_message, history):
                return "", history + [[user_message, None]]

            def bot_msg(history):
                user_message = history[-1][0]
                bot_response = chat_interface(user_message, history)
                history[-1][1] = bot_response
                return history

            msg_box.submit(user_msg, [msg_box, chatbot], [msg_box, chatbot], queue=False).then(
                bot_msg, chatbot, chatbot
            )
            
            # Listen button handler
            listen_btn.click(listen_audio, outputs=msg_box)
            stop_listen_btn.click(stop_listening, outputs=status_box)

        # --- TAB: TERMINALS ---
        with gr.TabItem("🖥️ Terminals"):
            gr.Markdown("### 1. ROS Core")
            gr.Code(value="roscore", language="shell", label="Terminal 1 Command")
            
            gr.Markdown("### 2. ELMiRA Nodes (Hardware/Vision)")
            term2_cmd = gr.Textbox(label="Terminal 2 Command", lines=2)
            term2_log = gr.Textbox(label="Terminal 2 Output", lines=20, interactive=False, max_lines=20)
            
            gr.Markdown("### 3. State Machine (Logic)")
            term3_cmd = gr.Textbox(label="Terminal 3 Command", lines=2)
            term3_log = gr.Textbox(label="Terminal 3 Output", lines=20, interactive=False, max_lines=20)
            
            refresh_term_btn = gr.Button("🔄 Refresh Logs")
            
            refresh_term_btn.click(
                refresh_terminals, 
                outputs=[term2_cmd, term2_log, term3_cmd, term3_log]
            )
            
            # Auto-refresh Terminal Logs (every 2s)
            try:
                term_timer = gr.Timer(2)
                term_timer.tick(refresh_terminals, outputs=[term2_cmd, term2_log, term3_cmd, term3_log], show_progress="hidden")
            except AttributeError:
                # Fallback: using demo.load with every=2
                demo.load(refresh_terminals, None, [term2_cmd, term2_log, term3_cmd, term3_log], every=2, show_progress="hidden")

            
            # Auto-refresh when launching
            launch_btn.click(
                refresh_terminals, 
                outputs=[term2_cmd, term2_log, term3_cmd, term3_log]
            )
            
            # Start Chat Monitor on Launch
            launch_btn.click(start_monitor_thread)
            
            # Auto-refresh Chat UI
            try:
                chat_timer = gr.Timer(2.0)
                chat_timer.tick(get_chat_updates, outputs=chatbot, show_progress="hidden")
            except AttributeError:
                demo.load(get_chat_updates, None, chatbot, every=2.0, show_progress="hidden")
            
if __name__ == "__main__":
    import atexit
    import sys
    import signal # Import signal module here
    
    
    # Ensure cleanup on exit
    def cleanup_on_exit():
        print("\nStopping robot processes...")
        stop_robot()
        
    atexit.register(cleanup_on_exit)
    # Handle Ctrl+C specifically to ensure atexit runs
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    
    print("Starting dashboard on http://0.0.0.0:7860")
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)


