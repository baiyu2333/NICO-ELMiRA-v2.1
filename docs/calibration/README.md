# Print and Capture the NICO Calibration Board

Print `nico_checkerboard_A4_9x6_25mm.pdf`, not a screenshot of it.
No robot movement or ELMiRA launch is needed for this procedure.

## 1. Print

- A4 paper, landscape, **Actual size / 100%**. Turn off Fit, Shrink and borderless enlargement.
- Use black-and-white printing on matte paper. Keep the white border.
- Mount the paper flat on a rigid board. Avoid folds, bends, glossy lamination and tape over the grid.
- The PDF has **10 columns x 7 rows of squares**, which means **9 x 6 inner corners** for OpenCV.
- Nominal square size: **25 mm**. Nominal full grid: **250 x 175 mm**.

## 2. Measure the Printed Copy

1. Measure between the end ticks of the **100 mm ruler** at the bottom.
2. Measure the full grid horizontally and vertically, preferably with a rigid ruler.
3. Calculate horizontal square size = measured grid width / 10.
4. Calculate vertical square size = measured grid height / 7.
5. Check several individual squares as well. If horizontal/vertical sizes differ
   beyond measurement uncertainty, the sheet is warped or printing is distorted:
   reprint instead of averaging a visibly non-square grid.

Send me the ruler length, grid width, grid height and actual square size in mm.
We will use the measured value divided by 1,000 for `--square-m`.
**25 mm is a print specification, not a verified measurement.** No calibration
will be marked verified just because the PDF was generated.

## 3. Capture with NICO's Camera Only

Keep NICO stationary and supported. Move **the board**, not the robot head.
Close Cheese and any other camera viewer first. Do not launch ELMiRA for this
capture: its startup can move the robot. The command below never opens the
motor bus or imports ROS motion code.

Both See3CAM cameras produced native 1280x720 frames on 2026-10-13. This command
selects serial **22275000**, independently of its still-unverified left/right
eye label. Do not mix images from the other camera into this calibration.

```bash
cd ~/catkin_ws/src/NICO-software/api
python3 src/ELMiRA/scripts/active_vision_cli.py calibration-capture \
  --device /dev/v4l/by-id/usb-e-con_systems_See3CAM_CU135_22275000-video-index0 \
  --output logs/active_vision/calibration_22275000 \
  --count 25 --columns 9 --rows 6
```

At each prompt, reposition the board, hold it still, then press **Enter**.
Type **q** then Enter to finish early. Use a new output directory if repeating.
The tool reports `board detected` or `board NOT detected`; it cannot verify your
physical square measurements. Native distorted PNGs are saved without resizing.
It times out if the camera stops delivering, rather than waiting indefinitely.

Collect different views, not 25 identical pictures:

- Start with the whole grid visible near the image centre and reasonably large.
- Move it toward left, right, upper and lower parts of the usable image circle.
- Include moderate tilts about both axes and several distances.
- Keep every inner corner visible and sharp, without fingers covering the grid.
- Hold still during each capture. Avoid blur, glare and extreme edge stretching.
- Do not change camera zoom, resolution, lens focus or mounting during a session.

View the first saved `train/frame_000.png` in the file browser to check framing.
`review/` contains separate corner overlays when detection succeeds; **do not use
overlays for fitting**. You can leave an image viewer open; do not reopen a live
camera application during capture.

Outputs in the selected directory:

- `train/`: normally 20 original PNGs plus per-image JSON.
- `holdout/`: every fifth view, normally 5 originals for independent checks.
- `review/`: detected-corner overlays only.
- `capture_session.json`: camera identity/settings, host timestamps, detection
  counts and missing square measurement. No invented head angles or camera poses.

Automatic exposure changes are recorded per image; format/identity/zoom/focus
changes stop the session. No camera settings are changed by this tool. A count
of saved files is not a count of usable views: get at least 10 distinct detected
training views and 3 distinct detected holdout views, preferably the full 20/5.
If more are needed, use a new session directory and retain its metadata.

## 4. Fit Only After Measuring

After you provide measurements, use the existing `calibrate` and
`calibration-check` commands described in [the integration guide](../active_single_rgb_nico.md).
We will use `--columns 9 --rows 6` and the measured `--square-m`.
The fisheye images suggest testing `equidistant`, but holdout reprojection,
rectification and independent scale checks still need review.

Moving the board while NICO stays still is appropriate for **intrinsic camera
calibration**. It does **not** measure NICO head kinematics or give a fixed-world
camera trajectory. Later, for multiview object reconstruction, the board/world
reference and object must stay fixed while the verified camera viewpoint changes.
The printed TOP LEFT text helps a human check board orientation; it is not a
machine-detected fiducial that resolves every planar-pose ambiguity.

Technical reference: [OpenCV camera calibration](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html).
