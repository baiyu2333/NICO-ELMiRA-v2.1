# FYP2 Kinematic Action Preview Plan

## Purpose

The Kinematic Action Preview module is a lightweight planning and visualization tool for NICO right-arm actions. It shows a simplified stick-figure upper-body model before real execution, logs the preview, and supports Week 10-12 experiments around pointing, reaching, touching, and simplified grasp attempts.

It is not a physics simulator and it does not move the robot by itself.

## Measured Setup Values

Current coarse measurements:

- Table surface height from ground: `0.08 m`
- NICO right hand / arm height from ground: `0.23-0.25 m`
- Default right hand height from ground: `0.24 m`
- Red object distance from NICO: `0.20 m`
- Small object center height on the table: about `0.10 m`
- Minimum safe preview z: `0.085 m`

These are approximate measured values for coarse preview only. They are not final calibration.

Coordinate convention:

- `x`: forward from NICO toward the object, meters
- `y`: left/right from NICO, meters; negative y is robot right side
- `z`: upward from ground, meters

Default target:

```text
target_object = red object
target_x_m = 0.20
target_y_m = -0.10
target_z_m = 0.10
```

The dashboard keeps target `y` configurable because the red object may not be centered.

## Static Preview

Static preview uses predefined right-arm action templates. It performs forward-kinematics-style estimation:

```text
joint angles -> estimated hand XYZ
```

Run from `api`:

```bash
python3 src/ELMiRA/scripts/kinematic_preview/preview_action.py --template right_arm_pre_grasp
python3 src/ELMiRA/scripts/kinematic_preview/preview_action.py --template right_arm_point_red_object
python3 src/ELMiRA/scripts/kinematic_preview/preview_action.py --template right_arm_grasp_attempt
```

Run with measured target values:

```bash
python3 src/ELMiRA/scripts/kinematic_preview/preview_action.py --template right_arm_pre_grasp --target-object "red object" --target-x 0.20 --target-y 0.00 --target-z 0.10
```

List templates:

```bash
python3 src/ELMiRA/scripts/kinematic_preview/preview_action.py --list
```

Generated files:

```text
api/logs/kinematic_preview/latest_preview.png
api/logs/kinematic_preview/latest_preview.json
api/logs/kinematic_preview/preview_trials.csv
api/logs/kinematic_preview/latest_preview_summary.md
```

## Live Preview

Live preview is optional and read-only. It subscribes to joint-state topics if ROS is available, maps relevant right-arm joints to simplified preview angles, and renders a stick-figure pose.

Run:

```bash
python3 src/ELMiRA/scripts/kinematic_preview/live_joint_state_reader.py --once
```

If ROS joint states are unavailable, it reports `unavailable` or `timeout` and exits cleanly. It never commands robot motion.

## Live Recording And Replay

Joint recording captures `/joint_states` during a real action triggered through the existing ELMiRA pipeline. The recorder is read-only: it does not publish motor commands, create trajectories, or move NICO.

Record a fixed 10-second trial from `api`:

```bash
python3 src/ELMiRA/scripts/kinematic_preview/joint_trajectory_recorder.py --record-seconds 10 --label point_red_object_trial --instruction "Point to the red object with your right hand." --target-object "red object" --action-type point
```

Replay a selected frame:

```bash
python3 src/ELMiRA/scripts/kinematic_preview/replay_joint_trajectory.py --recording logs/kinematic_preview/latest_recording.json --frame 0
python3 src/ELMiRA/scripts/kinematic_preview/replay_joint_trajectory.py --recording logs/kinematic_preview/latest_recording.json --frame 10
```

Save a useful replay frame as a captured template:

```bash
python3 src/ELMiRA/scripts/kinematic_preview/save_replay_frame_as_template.py --recording logs/kinematic_preview/latest_recording.json --frame 10 --template-name captured_successful_point_red_object --action-type point --target-object "red object"
```

Generated files:

```text
api/logs/kinematic_preview/recordings/<recording_id>.json
api/logs/kinematic_preview/recordings/<recording_id>.csv
api/logs/kinematic_preview/latest_recording.json
api/logs/kinematic_preview/latest_recorded_trial_summary.json
api/logs/kinematic_preview/recorded_motion_trials.csv
api/logs/kinematic_preview/replay/latest_replay_frame.png
api/src/ELMiRA/scripts/kinematic_preview/templates/captured_right_arm_templates.json
```

## Action Templates

Templates are stored in:

```text
api/src/ELMiRA/scripts/kinematic_preview/templates/right_arm_templates.json
```

Current right-arm templates:

- `right_arm_reset_pose`
- `right_arm_point_red_object`
- `right_arm_reach_forward`
- `right_arm_pre_grasp`
- `right_arm_grasp_attempt`
- `right_arm_close_hand`
- `right_arm_lift_pose`

Each template includes:

- `name`
- `description`
- `purpose`
- `action_type`
- `arm_used`
- `joint_angles_deg`
- `hand_state`
- `target_object`
- `target_object_xyz`
- `safety_note`
- `real_robot_motion_commanded: false`

Template values are demonstration values only. Do not send them directly to real motors.

## FK And IK

Forward kinematics:

```text
joint angles -> estimated hand XYZ
```

Inverse kinematics:

```text
target XYZ -> joint angles
```

This version prioritizes templates and FK-style preview. Full IK is a future extension. `gaikpy` or a URDF-based IK tool may be used later, but it is not required here.

## VLM-Based Grasping Connection

Planned pipeline:

```text
User instruction
-> VLM understands task
-> vision/object grounding estimates target object position
-> action template selects point / reach / pre-grasp / close-hand / lift
-> simplified kinematic preview visualizes right-arm pose
-> real NICO executes only through existing safe ELMiRA pipeline
-> visual error is measured after coarse movement
-> NJF-style bounded correction can later refine residual error
-> result is logged for evaluation
```

NJF is a future refinement direction here. This module does not train NJF and does not let NJF control the real robot.

## Dashboard Use

The clean dashboard keeps these tools collapsed by default:

```text
Kinematic Preview / Action Template
Live Motion Viewer
Live Recording / Replay
Safety Debug: Right Arm Presets
```

The main screen remains focused on:

- Launch ELMiRA
- Talk to NICO / text interaction
- Stop System
- compact status
- conversation area
- FYP2 NJF result card

## Safety Debug Buttons

The Safety Debug section includes preview preset buttons and optional execution buttons.

Current execution status:

```text
Execution unavailable: no safe wrapper connected yet.
```

No raw motor command publishing is implemented. Execution buttons require the safety checkbox and still return unavailable until a verified safe ELMiRA/NICO wrapper is connected.

Safety rules:

- No raw ROS motor commands from preview scripts.
- No arbitrary joint-angle UI.
- No left-arm or left-gripper control.
- No real robot movement unless routed through an existing verified safe wrapper.
- All preview outputs include `real_robot_motion_commanded: false`.
- Every preset attempt is logged.

Preset debug logs:

```text
api/logs/kinematic_preview/latest_preset_debug.json
api/logs/kinematic_preview/preset_debug_trials.csv
```

## Why Only Right Arm

The current hardware setup has unreliable left gripper behavior. Week 10-12 preview and debug controls therefore focus on the right arm/right hand only. The left arm may appear in the stick figure as a passive visual reference, but it is not used for action execution.

## Limitations

- No physics simulation.
- No contact simulation.
- No collision checking.
- No object-grasp stability estimate.
- No validated real grasp execution yet.
- No full NJF training yet.
- Stick-model geometry is approximate and not a full NICO URDF.
- Live joint mapping is approximate and may need calibration against real joint names/angles.

## Week 10-12 Usage Plan

Week 10:

- Use static previews to document planned right-arm action templates.
- Log point, reach, pre-grasp, grasp-attempt preview, close-hand preview, and lift preview.
- Verify live joint-state reading when ROS is available.

Week 11:

- Connect object detection output to target-object metadata.
- Compare preview hand pose against real NICO movement from the existing ELMiRA pipeline.
- Add trial logs for visual error after coarse movement.

Week 12:

- Add small bounded NJF-style correction experiments.
- Compare MLLM-only action selection, template preview, and preview plus refinement.
- Report limitations clearly and avoid claiming full grasping unless physically demonstrated.
