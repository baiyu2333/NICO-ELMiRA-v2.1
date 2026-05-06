# Neural Jacobian Field Integration Notes

Repository: https://github.com/sizhe-li/neural-jacobian-field

## What This Repo Provides

Neural Jacobian Fields (NJF) learn a visual action model from camera observations
and robot joint/action changes. The upstream repo is not a drop-in ROS IK solver.
For NICO, it should be treated as a learned controller research path that can
eventually sit beside or replace parts of the current EvoIK + visual refinement
pipeline after we collect NICO-specific training data.

## Current Compatibility

- Upstream recommends Python 3.10.8, conda, PyTorch, Nerfstudio, TinyCudaNN, and
  a CUDA GPU for practical training/inference.
- This robot workspace currently runs ROS Noetic on Python 3.8, so NJF should be
  installed in a separate environment, not inside the ROS Python environment.
- The upstream pretrained checkpoints are for their robots/datasets, not NICO.
  They can demonstrate the method, but they will not directly control NICO's left
  hand or arm.

## Safe First Step: Record NICO Data

I added a passive ROS recorder:

```bash
roslaunch elmira njf_data_recorder.launch output_dir:=/tmp/nico_njf_dataset image_topic:=/nico/vision/right sample_rate_hz:=2.0
```

Run it while ELMiRA or a manual test script moves the robot. It writes:

```text
/tmp/nico_njf_dataset/
  transforms.json
  images/
    00000_00000.png
    00000_00001.png
    ...
```

The recorder stores:

- Camera frames from `/nico/vision/right`.
- Left arm joint states from `/left/open_manipulator_p/joint_states`.
- Left XL-320 hand goal positions observed on `/nico/motion/xl320_cmd`.
- A `transforms.json` layout compatible with the NJF data parser style.

## Current Control Step: NJF-Inspired Visual Jacobian Refinement

The current project now includes a lightweight ROS-native refinement module:

```text
api/src/ELMiRA/scripts/njf_predict_action_server.py
api/src/ELMiRA/scripts/njf_visual_error_estimator.py
api/src/ELMiRA/srv/NJFPredictAction.srv
api/src/ELMiRA/launch/njf_grasp_refinement.launch
```

This is **not** a full reproduction of the Neural Jacobian Field paper. It does
not load PyTorch, Nerfstudio, TinyCudaNN, or upstream NJF checkpoints. It is an
NJF-inspired placeholder that exposes the right service boundary:

```text
visual/metric hand-object error + planning group + optional joint state
  -> safe small x/y correction + optional pseudo joint delta
```

Internally, the server currently uses a bounded proportional visual-Jacobian
controller. The implementation clamps every correction before it reaches the IK
planner:

- `max_correction_m`: default `0.025` m per NJF refinement call.
- `max_joint_delta`: default `0.08` rad for pseudo joint-delta output.
- `target_error_threshold`: default `0.005` m below which correction is zero.
- Workspace clamping via the existing ELMiRA workspace limits.

Pure `njf` mode currently gets its visual error from
`njf_visual_error_estimator.py`, a simple OpenCV color-marker detector. The
service internals are intended to be replaced later by a trained NICO NJF model
without changing the SMACH grasp pipeline.

## Refinement Modes

`grasp_refinement.py` supports:

- `mllm`: existing behavior. Calls `mllm_refine_grasp` and uses MLLM-estimated
  `x_offset_meters` / `y_offset_meters`.
- `njf`: requires valid visual error from the color-marker estimator, then calls
  `njf_predict_action`. If either marker is missing or too small, it returns zero
  correction safely.
- `hybrid`: calls MLLM first for a coarse correction, then feeds that correction
  into `njf_predict_action` for a bounded micro-correction. By default only
  `25%` of the MLLM offset is used as the NJF micro-error input
  (`hybrid_njf_error_scale:=0.25`) to avoid amplifying a bad visual estimate. If
  NJF is unavailable, the MLLM correction is still used.

Default mode remains `mllm` for safety.

## Color-Marker Visual Error Estimator

`njf_visual_error_estimator.py` subscribes to a camera image, detects two colored
markers, and writes normalized image error params:

```text
/elmira/njf_image_error_u = object_center_u - hand_center_u
/elmira/njf_image_error_v = object_center_v - hand_center_v
/elmira/njf_visual_error_valid
/elmira/njf_hand_visible
/elmira/njf_object_visible
/elmira/njf_visual_error_stamp
```

The `u` and `v` errors are normalized by half the image width/height so values
are roughly in `[-1, 1]`. If either marker is not visible or the blob area is
below `min_area`, `njf_visual_error_valid` is set to `false` and all errors are
set to zero. Pure `njf` mode also rejects stale visual errors older than
`/elmira/njf_visual_error_max_age` seconds, default `1.0`.

Default marker colors:

- Object marker: green tape/sticker, HSV `[40, 80, 50]` to `[85, 255, 255]`.
- Hand marker: blue tape/sticker, HSV `[100, 80, 50]` to `[130, 255, 255]`.

Put a clearly visible colored marker on the target object and another marker on
the visible side of the NICO gripper/hand. Avoid colors already common in the
scene. Red markers can work, but red wraps around the HSV hue boundary, so blue
is the safer default unless you tune the thresholds carefully.

Debug output:

```bash
rqt_image_view /elmira/njf_visual_error/debug_image
rostopic echo /elmira/njf_visual_error/status
```

Tune HSV thresholds by launching with custom values:

```bash
roslaunch elmira njf_grasp_refinement.launch \
  refinement_mode:=njf \
  image_topic:=/nico/vision/right \
  object_hsv_lower:="[40, 80, 50]" \
  object_hsv_upper:="[85, 255, 255]" \
  hand_hsv_lower:="[100, 80, 50]" \
  hand_hsv_upper:="[130, 255, 255]" \
  min_area:=80
```

The estimator does **not** output precise metric errors by default. It sets
`/elmira/njf_x_error_m` and `/elmira/njf_y_error_m` to `0.0`. If you provide
`meters_per_norm_error`, it will approximate:

```text
x_error_m = image_error_v * meters_per_norm_error
y_error_m = image_error_u * meters_per_norm_error
```

This scale is camera/setup dependent. Keep it small and treat it as rough visual
servoing, not calibrated 3D geometry.

## Build After Adding the Service

Because `NJFPredictAction.srv` is a new ROS service, rebuild the catkin workspace
before running the new node:

```bash
cd ~/catkin_ws/src/NICO-software/api
catkin_make
source devel/setup.bash
```

## Run and Test

Start the normal ELMiRA stack with the default safe MLLM refinement:

```bash
roslaunch elmira init_nodes_v2.launch mllm_provider:=openai refinement_mode:=mllm
```

Start with the NJF-inspired service available but use hybrid refinement:

```bash
roslaunch elmira init_nodes_v2.launch mllm_provider:=openai refinement_mode:=hybrid
```

Start the full stack with pure marker-based NJF refinement:

```bash
roslaunch elmira init_nodes_v2.launch \
  mllm_provider:=openai \
  refinement_mode:=njf \
  enable_njf_visual_error_estimator:=true
```

Run only the NJF-inspired refinement server for isolated testing:

```bash
roslaunch elmira njf_grasp_refinement.launch refinement_mode:=njf max_correction_m:=0.02
```

Manual service test:

```bash
rosservice call /njf_predict_action "planning_group: 'l_arm'
x_error_m: 0.02
y_error_m: -0.01
image_error_u: 0.0
image_error_v: 0.0
joint_names: []
joint_positions: []
prefer_joint_delta: false"
```

Expected output is a small, clamped correction, not a direct large robot command.

## Important Calibration Gap

The generated `transforms.json` uses identity camera extrinsics and approximate
intrinsics unless you pass calibrated values through ROS params. That is enough
for data plumbing tests, but not enough for high-quality 3D NJF training.

For serious NJF training, collect:

- Calibrated right and left eye intrinsics.
- Camera-to-world or camera-to-robot transforms for each eye.
- Synchronized image streams from both eyes, ideally with depth or masks if
  available.
- Many small randomized action sequences with recorded joint positions before
  and after movement.

## Suggested Integration Path

1. Keep the existing ELMiRA `inverse_kinematics` service as the production path.
2. Use `njf_data_recorder.py` to collect NICO-specific visual/action data.
3. Use `njf_predict_action_server.py` as the current safe visual-Jacobian
   placeholder for small correction experiments.
4. Use `njf_visual_error_estimator.py` for early marker-based visual-error
   experiments while collecting data.
5. Train a real NJF model outside ROS in a separate Python 3.10/CUDA environment.
6. Replace the placeholder internals with a trained NICO checkpoint that proposes
   joint deltas or Cartesian corrections through the same ROS service.
7. Gate NJF output through the existing workspace limits and motor filters before
   sending commands to `/nico/motion/setAngle` or `/nico/motion/xl320_cmd`.

## Environment Notes

Do not run the upstream `install.sh` in `~/.NICO-python3`; it can break ROS
package compatibility by upgrading Python ML dependencies. Use a separate conda
or uv environment.
