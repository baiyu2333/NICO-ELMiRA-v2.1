# 🤖 ELMiRA v2 Robot Operation Guide

## System Requirements
- **OS**: Ubuntu 20.04 LTS
- **ROS**: Noetic
- **Python**: 3.8

---

## Step 1: Check USB Connections

### 1.1 Check Motor Controller (Dynamixel)
```bash
# List all USB serial devices
ls -la /dev/ttyACM*
ls -la /dev/ttyUSB*

# You should see something like:
# /dev/ttyACM0 -> Motor controller
```

If no devices appear, check:
- USB cable is connected
- Power is on for the robot

### 1.2 Set Permissions for Motor Controller
```bash
# Option A: Add user to dialout group (permanent, requires logout/login)
sudo adduser $USER dialout

# Option B: Set permissions manually (temporary, per session)
sudo chmod 777 /dev/ttyACM*
```

### 1.3 Check Camera Devices
```bash
# List all video devices
ls -la /dev/v4l/by-id/

# List video devices directly
ls -la /dev/video*

# Check camera details
v4l2-ctl --list-devices
```

Expected output for NICO eyes (See3CAM cameras):
```
usb-e-con_systems_See3CAM_CU135_XXXXXXXX-video-index0
```

### 1.4 Test Camera Access
```bash
# Quick camera test (requires v4l-utils)


# Or test with Python
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f'Camera works: {ret}, Frame shape: {frame.shape if ret else None}')
cap.release()
"
```

---

## Step 2: Environment Setup

### 2.1 Source the Workspace
```bash
cd ~/catkin_ws/src/NICO-software/api
source activate.bash
```

### 2.2 Set API Key (choose one provider)
```bash
# Option A: Google Gemini (recommended for multimodal)
export GOOGLE_API_KEY="your-google-api-key-here"

# Option B: OpenAI GPT-4o
export OPENAI_API_KEY="your-openai-api-key-here"
```

**To make permanent**, add to `~/.bashrc`:
```bash
echo 'export GOOGLE_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### 2.3 Install v2 Dependencies
```bash
pip install -r ~/catkin_ws/src/NICO-software/api/src/ELMiRA/requirements_v2.txt
```

### 2.4 Build Catkin Workspace
```bash
cd ~/catkin_ws/src/NICO-software/api
catkin_make
source devel/setup.bash
```

---

## Step 3: Configure Robot Motors

Since your **left arm is not functional** (but elbow and shoulder work), we need to configure the motor JSON file.

### 3.1 Check Current Motor Config
The default config is [`json/nico_humanoid_upper.json`]nico_humanoid_upper.json ). For a robot with limited left arm, edit or create a custom config:

```bash
# View current motor IDs
cat ~/catkin_ws/src/NICO-software/json/nico_humanoid_upper.json | grep -A3 '"l_'
```

### 3.2 Disable Non-Functional Motors (Optional)
If specific left arm motors don't work, you can disable them in the Motion node config. Edit the disabled motor IDs in the ROS param or JSON.

In [`api/src/nicoros/scripts/Motion.py`]Motion.py ) line 53:
```python
"disabledMotorIds": [23, 24, 26, 28, 29, 30, 32],  # Add non-working motor IDs here
```

Or set via launch argument:
```bash
roslaunch elmira init_nodes_v2.launch mllm_provider:=openai motion_disabled_motor_ids:="[23, 24, 26, 28, 29, 30, 32]"
```

### 3.3 If Plugging the New Left Hand Kills `motion-2`
If `motion-2` starts when the left hand cable is unplugged but fails when the left hand is plugged in, this is a Dynamixel bus problem, not a state machine problem. The new hand may be using a different protocol, wrong voltage, or a default/conflicting ID such as `1`, which can block replies from the main arm motors.

Use this diagnostic command to remove all hand/wrist motors from PyPot startup and test whether the arm/head bus can still initialize:
```bash
roslaunch elmira init_nodes_v2.launch \
  mllm_provider:=openai \
  motion_disabled_motor_ids:="[23, 24, 25, 26, 28, 29, 30, 32]"
```

If it still fails with the left hand plugged in, do not keep that hand on the same `/dev/ttyUSB0` bus. Put the new left hand on a separate XL-320/Protocol 2 compatible adapter and power rail, or reconfigure its IDs/protocol safely before reconnecting it to the robot.

---

## Step 4: Launch the Robot

### 4.1 Terminal 1: Start ROS Core
```bash
roscore
```

### 4.2 Terminal 2: Launch Motor Controller
```bash
cd ~/catkin_ws/src/NICO-software/api
source activate.bash
source devel/setup.bash

# Launch motion controller with your motor config
roslaunch nicoros joint_controller.launch json_path:=$(rospack find nicoros)/../../../json/nico_humanoid_upper.json
```

Watch for errors like:
- `No motor found at ID XX` - motor disconnected or broken
- `Connection refused` - USB not connected or permissions issue

### 4.3 Terminal 3: Launch ELMiRA v2 (MLLM Gateway)
```bash
cd ~/catkin_ws/src/NICO-software/api
source activate.bash
source devel/setup.bash

# With Google Gemini provider
roslaunch elmira init_nodes_v2.launch mllm_provider:=google

# OR with OpenAI provider
roslaunch elmira init_nodes_v2.launch \
  mllm_provider:=openai \
  move_wait_max_checks:=350 \
  offset_z:=-0.08 \
  grasp_contact_z_offset:=0.02
```

If the arm reaches the object but waits too long before closing the hand, reduce
`move_wait_max_checks`. At 50 Hz, `350` is about 7 seconds and `250` is about 5
seconds.

If the hand is still too high above the object, lower `offset_z` gradually. Use
`offset_z:=-0.08` first, then try `-0.12`, and only use `-0.15` if it is still
clearly high. The refined grasp closes at `table_z + grasp_contact_z_offset`;
keep `grasp_contact_z_offset` near `0.02` unless you are testing very carefully.

### 4.4 Optional Workspace Debug View
```bash
source ~/catkin_ws/src/NICO-software/api/devel/setup.bash
python3 ~/catkin_ws/src/NICO-software/api/src/ELMiRA/scripts/visualize_workspace.py

source ~/catkin_ws/src/NICO-software/api/devel/setup.bash
rqt_image_view /elmira/workspace_debug
```

---

## Step 5: Verify Everything is Running

### 5.1 Check ROS Topics
```bash
# List all topics
rostopic list

# You should see:
# /nico/vision/left         - Camera feed (current reliable default)
# /nico/motion/...          - Motor control
# /joint_states             - Joint positions
```

### 5.2 Check ROS Services
```bash
# List MLLM services
rosservice list | grep mllm

# Expected:
# /mllm_chat
# /mllm_vision  
# /mllm_detect
# /mllm_visibility
```

### 5.3 Test Camera Topic
```bash
# Check if camera is publishing
rostopic hz /nico/vision/left

# View camera image (requires image_view)
rosrun image_view image_view image:=/nico/vision/left
```

### 5.4 Test MLLM Service
```bash
# Test the MLLM gateway
rosservice call /mllm_chat "prompt: 'Hello, what can you see?'"
```

---

## Step 6: Run the State Machine

### 6.1 Launch the Full ELMiRA System
```bash
cd ~/catkin_ws/src/NICO-software/api
source activate.bash
source devel/setup.bash

# Run the state machine
rosrun elmira state_machine.py
```

### 6.2 Interact with the Robot
The robot will:
1. Listen for speech (ASR)
2. Process with MLLM (Gemini/GPT-4o)
3. Execute actions (speak, move, detect objects)

---

## Troubleshooting

### Camera Not Found
```bash
# Check USB devices
lsusb

# Check v4l devices
v4l2-ctl --list-devices

# Start the reliable left camera stream
roslaunch elmira camera.launch mode:=left
```

### Left Hand Not Moving
The current left hand is controlled as XL-320 / Dynamixel Protocol 2.0 motors on `/dev/ttyUSB0`.
Stop `Motion.py` before scanning the bus:

```bash
rosnode kill /motion
rosrun elmira xl320_scan.py --port /dev/ttyUSB0 --start-id 28 --end-id 38
```

Expected responders on the current robot are:

```text
30, 31, 33, 34, 35, 36, 37
```

The default launch mapping is:

```bash
roslaunch elmira init_nodes_v2.launch mllm_provider:=openai \
  left_xl320_wrist_z_id:=30 \
  left_xl320_wrist_x_id:=31 \
  left_xl320_finger_ids:="[33, 34, 35, 36, 37]"
```

If a replacement hand uses different IDs, update those launch args instead of editing code.

### Coordinate Not Accurate
The image-to-real coordinate model was trained for the original camera setup. The current default uses the reliable left camera, so the mapping may need retuning:

```bash
roslaunch elmira init_nodes_v2.launch mllm_provider:=openai \
  offset_x:=0.0 offset_y:=0.105 coord_scale_x:=1.0 coord_scale_y:=1.0
```

Use the `CoordTransfer` log lines to compare detected image coordinates with where the robot actually reaches, then adjust `offset_x` and `offset_y` first.

### Motor Connection Failed
```bash
# Check serial permissions
ls -la /dev/ttyUSB* /dev/ttyACM*

# Try resetting USB
sudo usbreset /dev/bus/usb/XXX/YYY  # Get XXX/YYY from lsusb

# Check if port is in use
sudo fuser /dev/ttyUSB0
```

### MLLM Gateway Error
```bash
# Check API key is set
echo $GOOGLE_API_KEY
echo $OPENAI_API_KEY

# Test provider directly
python3 -c "
import os
print('GOOGLE_API_KEY:', 'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET')
print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')
"
```

### Left Arm Not Moving
Since left arm is partially functional (elbow/shoulder work), check:
```bash
# Get current joint states
rostopic echo /joint_states -n 1

# Test individual joint
rosservice call /nico/motion/setAngle "motorName: 'l_shoulder_y'
value: 0.0
speed: 0.5"
```

---

## Quick Start Summary

```bash
# Terminal 1
roscore

# Terminal 2
cd ~/catkin_ws/src/NICO-software/api
source activate.bash && source devel/setup.bash
sudo chmod 777 /dev/ttyACM*
roslaunch nicoros joint_controller.launch

# Terminal 3
cd ~/catkin_ws/src/NICO-software/api
source activate.bash && source devel/setup.bash
export GOOGLE_API_KEY="your-key"
roslaunch elmira init_nodes_v2.launch mllm_provider:=google

# Terminal 4 (after nodes are up)
rosrun elmira state_machine.py
```








# 🤖 ELMiRA v2 Robot Operation Guide - OpenAI GPT-4o

## Pre-Flight Checklist

### 1. USB Connections Verified
```
✅ /dev/ttyUSB0 - FTDI Dynamixel Motor Controller
✅ /dev/ttyACM0 - Teensyduino (face expressions)
✅ /dev/ttyACM1 - OptoForce DAQ (force sensor)
✅ /dev/video* - Cameras
```

### 2. Working Motors (13 total)
| Part | Motors |
|------|--------|
| Head | `head_z`, `head_y` |
| Left Arm | `l_shoulder_y`, `l_shoulder_z`, `l_arm_x`, `l_elbow_y` |
| Right Arm | `r_shoulder_y`, `r_shoulder_z`, `r_arm_x`, `r_elbow_y` |
| Right Hand | `r_wrist_z`, `r_wrist_x`, `r_indexfingers_x` |

---

## Step 1: Set Up Environment

### Terminal 1: Start ROS Core
```bash
roscore
```

### Terminal 2: Set API Key & Source Workspace
```bash
cd ~/catkin_ws/src/NICO-software/api
source activate.bash
source devel/setup.bash

# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Verify it's set
echo $OPENAI_API_KEY
```

---

## Step 2: Launch the Robot

### Terminal 2 (same terminal): Launch ELMiRA v2
```bash
roslaunch elmira init_nodes_v2.launch mllm_provider:=openai
```

**Expected output:**
```
... loading Motion.py
... Connecting to /dev/ttyUSB0
✅ Robot initialized with 13 motors
... Camera node started
... MLLM Gateway started (provider: openai)
... Speech ASR started
```

---

## Step 3: Verify Nodes Are Running

### Terminal 3: Check ROS Status
```bash
cd ~/catkin_ws/src/NICO-software/api
source activate.bash
source devel/setup.bash

# Check all nodes
rosnode list
```

**Expected nodes:**
```
/rosout
/motion
/joint_controller_left
/joint_controller_right
/joint_controller_head
/camera_node (or similar)
/speech_asr
/mllm_gateway
/coordinate_transfer
/ik_solver
/text_to_speech
```

### Check Services
```bash
rosservice list | grep -E "mllm|llm|motion"
```

**Expected services:**
```
/mllm_chat
/mllm_vision
/nico/motion/getAngle
/nico/motion/setAngle
```

---

## Step 4: Test Components Individually

### Test 1: Motor Control
```bash
# Read a motor position
rosservice call /nico/motion/getAngle "joint: 'head_z'"

# Move head slightly (be careful!)
rosservice call /nico/motion/setAngle "joint: 'head_z'
angle: 10.0
speed: 0.3"
```

### Test 2: Camera
```bash
# Check camera topic
rostopic hz /nico/vision/left

# View camera (if you have display)
rosrun image_view image_view image:=/nico/vision/left
```

### Test 3: MLLM Gateway (GPT-4o)
```bash
# Simple text chat
rosservice call /mllm_chat "prompt: 'Hello, what can you do?'"
```

---

## Step 5: Run the State Machine

### Terminal 4: Start Interactive Session
```bash
cd ~/catkin_ws/src/NICO-software/api
source activate.bash
source devel/setup.bash

# Run the state machine
rosrun elmira state_machine.py
```

**The robot will now:**
1. 🎤 Listen for your voice commands
2. 🧠 Process with GPT-4o
3. 🤖 Execute actions (speak, move, detect objects)
4. 🔁 Loop back to listening

---

## Step 6: Interact with the Robot

### Voice Commands to Try:
- "Hello NICO, how are you?"
- "What objects can you see on the table?"
- "Can you touch the red ball?"
- "Push the cup to the left"
- "Wave your right arm"
- "Look down at the table"
- "Goodbye" (to quit)

---

## Troubleshooting

### Motor Connection Failed
```bash
# Check USB permissions
sudo chmod 666 /dev/ttyUSB0

# Verify config
cat ~/catkin_ws/src/NICO-software/json/nico_humanoid_upper_fixed.json | grep port
# Should show: "port": "/dev/ttyUSB0"
```

### MLLM Gateway Not Responding
```bash
# Check API key
echo $OPENAI_API_KEY

# Check node logs
rosnode info /mllm_gateway

# Test directly
python3 -c "
import openai
client = openai.OpenAI()
r = client.chat.completions.create(model='gpt-4o', messages=[{'role':'user','content':'Hi'}])
print(r.choices[0].message.content)
"
```

### Camera Not Publishing
```bash
# List video devices
ls -la /dev/video*

# Check camera node
rosnode info /camera_node
rostopic echo /nico/vision/left --noarr -n1
```

### Speech ASR Not Hearing
```bash
# Check microphone
arecord -l

# Test ASR node
rostopic echo /speech_text
```

---

## Quick Command Reference

| Action | Command |
|--------|---------|
| Start ROS | `roscore` |
| Launch robot | `roslaunch elmira init_nodes_v2.launch mllm_provider:=openai move_wait_max_checks:=350 offset_z:=-0.08 grasp_contact_z_offset:=0.02` |
| Run state machine | `rosrun elmira state_machine.py` |
| Check nodes | `rosnode list` |
| Check topics | `rostopic list` |
| View camera | `rosrun image_view image_view image:=/nico/vision/left` |
| Stop all | `Ctrl+C` in each terminal |

---

## Ready to Test?

Open **4 terminals** and run:

```bash
# Terminal 1
roscore

# Terminal 2
cd ~/catkin_ws/src/NICO-software/api && source activate.bash && source devel/setup.bash
export OPENAI_API_KEY="sk-your-key-here"
roslaunch elmira init_nodes_v2.launch mllm_provider:=openai move_wait_max_checks:=350 offset_z:=-0.08 grasp_contact_z_offset:=0.02

# Terminal 3 (after Terminal 2 is fully loaded)
cd ~/catkin_ws/src/NICO-software/api && source activate.bash && source devel/setup.bash
rosrun elmira state_machine.py

# Terminal 4 (for monitoring/debugging)
cd ~/catkin_ws/src/NICO-software/api && source activate.bash && source devel/setup.bash
rostopic echo /speech_text


```
## Peview OWLv2 Object detector

```bash
rostopic list | grep -E "result|debug|owlv2|detect"
```
You can view the OWLv2 bounding box visualization using `image_view`:

```bash
rosrun image_view image_view image:=/owlv2_server/result_image
```

This will open a window showing the camera image with bounding boxes drawn around detected objects.

**Note:** The debug image is only published when a detection request is made (when you ask the robot to interact with an object). So:

1. First run the image viewer:
   ```bash
   rosrun image_view image_view image:=/owlv2_server/result_image
   ```

2. Then ask the robot to do something like "push the orange ball" - the detection will run and you'll see the bounding boxes appear in the viewer.

Alternatively, you can also use `rqt_image_view` which has a dropdown to select topics:
```bash
rqt_image_view
```

Then select `/owlv2_server/result_image` from the dropdown menu.
