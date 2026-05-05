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
3. Train NJF outside ROS in a separate Python 3.10/CUDA environment.
4. Add a new ROS service, for example `njf_predict_action`, that loads a trained
   NICO checkpoint and proposes joint deltas.
5. Gate NJF output through the existing workspace limits and motor filters before
   sending commands to `/nico/motion/setAngle` or `/nico/motion/xl320_cmd`.

## Environment Notes

Do not run the upstream `install.sh` in `~/.NICO-python3`; it can break ROS
package compatibility by upgrading Python ML dependencies. Use a separate conda
or uv environment.
