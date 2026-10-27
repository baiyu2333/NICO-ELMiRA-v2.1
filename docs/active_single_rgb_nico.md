# Single-RGB Active Vision on NICO

For the current collaborator-facing summary, start with
[Vision research overview](vision_research_overview.md). This document retains
chronological experiments, including superseded candidates and unsuccessful
tests. Local `api/logs/` references and illustrated reports are not bundled in
GitHub. Running the adapted professor branches requires the separately supplied
[external dependency](../api/src/ELMiRA/scripts/active_vision/vendor/README.md).

Integration date: 2026-10-13. This module is separate from ELMiRA's manipulation
pipeline. The capture/reconstruction/active-loop tools do not change grasp
parameters, run the state machine, or command NICO. A separate supervised
head-only diagnostic was tested on 2026-10-15; its original camera execution
entry point remains disabled. A separate fixed-profile camera diagnostic was
supervised on 2026-10-19 as recorded below. After earlier guarded stops, a separately authorized raised-head
5/10/20-degree staged yaw-and-return check completed at 17:13 local; see the
chronological evidence below. No depth camera, CUDA, ROS 2, or new model
training is required to run the vision tools.

## Latest Hardware Check: 2026-10-19

The physical left-eye camera, serial 22275000, still captures 1280x720 UYVY with
the original optical-control readback. After the user raised the upright board,
all 54 inner corners were detected in two stationary images. Candidate rectified
PnP RMS was 0.64..0.67 px. This passes the existing 1 px image-fit gate; it does not
verify full intrinsics or metric accuracy. The legacy corner detector missed
these views, so `corners()` now falls back to OpenCV's sector-based detector.
The fixed-board pose helper can also align a reversed corner order to the
reference image. Symmetry and physical board identity still require review.

The first head read received no response. After the user confirmed motor power,
a fresh diagnostic returned 38 clean READ replies from IDs 19/20. Positions were
stable at yaw 2119/pitch 2128 (approximately +6.29/+7.08 motor degrees), both torques
off, 11.8 V, 30 C. This differs from the October 15 raised-head pose; that old anchor
was not reused. The user then confirmed current head/shoulder and cable clearance.

`active_vision.measured_viewpoint_test` uses this hash-pinned head reference and
the new board session. Default mode saves a plan without opening hardware:

```bash
cd ~/catkin_ws/src/NICO-software/api
PYTHONPATH=src/ELMiRA/scripts python3 -m active_vision.measured_viewpoint_test
```

The supervised execution on October 19 used `--execute --workspace-clear
--full-sweep-clear`. Its only motion writes were ID19 RAM goals/torque; speed2,
cap150 and the six-count/five-stable-sample criterion stayed unchanged. Two
fresh stationary images were checked before any motion. Each fixed5/10/20-degree
stage obtains another image. A camera rejection cancels larger stages and
returns through completed waypoints; a joint/health fault stops and disables yaw.
No pitch, arm, hand, PID, EEPROM or state-machine command is part of this test.

Actual October 19 result:

| Image | Candidate rectified reprojection RMS | Result |
| --- | --- | --- |
| Before/repeat | 0.654/0.656 px | Passed image-fit gate |
| About 5-degree stage | 0.955 px | Passed image-fit gate |
| About 10-degree stage | 1.473 px | Rejected; 20-degree stage cancelled |
| Returned | 0.657 px | Passed image-fit gate |

All 54 corners remain detectable even in the rejected image. Completed outward
encoder changes were about 4.57 and 9.58 degrees relative to the preflight origin.
The return error was 3 counts (0.264 degrees). Independent 38-reply READ-only checks
confirmed both torques off; speed 2/cap 150 were deliberately left in RAM rather
than restoring the power-on maximum settings. The motor/return sequence worked;
the overall test is **partial camera validation**, not a full 20-degree pass.

The accepted 5-degree image estimates 14.4 mm camera-centre displacement and 2.56
degrees viewing-ray change using the unverified candidate intrinsics. This meets
only the software geometric gate; no usable metric 3D baseline is certified.
The returned camera estimate differs from baseline by about 0.60 mm, also a
candidate repeatability figure rather than accuracy. Offline pose rotation and
encoder changes disagree: about 3.11 vs 4.48 degrees at the first image, and
6.11 vs 9.49 degrees at the rejected image. Check camera-model coverage, board
flatness and the physical encoder-to-camera relationship before attributing
this to a single cause or accepting calibrated extrinsics.

Evidence: `api/logs/active_vision/measured_viewpoint_20261019T063159_481647Z/`
contains original images, overlays, per-view metadata, `viewpoint_report.json`
and `test_assessment.json`. The latter links the independent final head read.
The original calibration candidate and historical failures were preserved.
Compile checks, 27 head tests, 6 measured-viewpoint tests and 36 vision tests passed;
the shared pose helper also passed regression checks on both saved real-board
images. No reconstruction or real-object recognition result was generated.

Next improve or restrict the validated camera region, check board flatness,
and evaluate independent holdout viewpoints and metric scale. Keep accepted
and rejected observations distinct; do not lower the 1 px gate to accept this run.

### Offline Calibration Follow-Up

An experimental refit used the original 16 training images plus the new baseline
and 5-degree images. It excluded the near-duplicate stationary/returned images
and reserved the 10-degree image for development validation. The fit detected
all 18 selected training views, with training RMS 0.202 px. All seven old held-out
images still pass, with RMS 0.149..0.214 px.

Comparing the same detected corners in the **same original distorted pixel
coordinates**, the 10-degree residual falls from 1.319 to 0.265 px. This supports
limited coverage in the original fit as a contributor. The earlier live rejection
was 1.473 in rectified pixels; do not compare that number directly to raw pixels
or relabel the original hardware test. The 10-degree image had already been
inspected during debugging, so it is a development holdout, not a fresh blind test.

The estimated 5-degree baseline also changes from about 14.1 to 8.5 mm in this
same-pixel comparison; the new candidate estimates about 18.4 mm at 10 degrees.
Thus neither a low residual nor the old baseline estimate establishes accurate
3D. The new 5-degree estimate is below the existing 10 mm geometric gate.
Candidate rotation changes improve toward encoder changes, but their physical
relationship and independent metric accuracy remain unverified.

Files under `api/logs/active_vision/calibration_update_20261019T063835_042995Z/`:
`candidate_18views.json`, `holdout_report.json`, `assessment.json` and
`same_pixel_comparison.json`. The candidate remains `verified:false` and was not
deployed to the robot, ROS adapter or fixed hardware diagnostic. The original
candidate is unchanged. Next use fresh unseen board views and independent ruler
measurements before another reconstruction-oriented motion test.

### First Real-Object Multiview Result

On October 19, the user placed a stationary white dice beside the upright board.
The supervised camera diagnostic used the explicitly selected 18-view candidate
and collected start/approximately 5/10/20-degree RGB images, then returned and
disabled yaw. A separate 38-reply diagnostic confirmed both head torques off.
All 54 corners were detected per view; rectified pose-fit RMS was about
0.17/0.23/0.28/0.32 px. No arm or hand was commanded.

`active_vision.sparse_preview` is an explicitly experimental offline exporter.
It accepts candidate intrinsics/poses without changing production verification
gates, uses original-pixel mask proposals and AKAZE or ORB correspondences,
and exports coloured PLY/NPZ/PNG with all verification flags false. Mutual
matching, epipolar, depth, ray-angle, reprojection and all-view silhouette
checks filter the sparse points. Initial ORB trials had large outliers; the
selected AKAZE result contains 167 points after 1 mm voxel merging.

The same preview also executes the professor's supplied CPU visual-hull branch
through `GuardedGeometry`. With sparse-estimated volume bounds, it produced
6,288 surface voxel samples from three accepted views. The short-baseline
5-degree view is excluded from carving. The hull is elongated and reaches the
estimated volume boundary: this is partial exploratory reconstruction, not
validated full-object geometry. Colours project the first view for display.

Data and selected outputs:
`api/logs/active_vision/dice_multiview_20261019T065922_054813Z/`.
`manifest_v2.json` references the corrected polygon-guided GrabCut proposals;
`reconstruction_delivery/` contains the packaged sparse and hull previews. The original
capture is `measured_viewpoint_20261019T065432_836036Z/`. A README records the
actual result, provenance, missing data and reproduction requirements.

Shareable offline package:
`api/logs/active_vision/exports/nico_dice_multiview_20261019.zip`.
It includes four unchanged original RGB images, the selected masks and overlays,
candidate calibration, per-view pose/head evidence, selected results and the
minimal offline geometry code. The ZIP was extracted under `/tmp` and replayed
without ROS or hardware access: 167 sparse points and 6,288 hull surface voxels
were reproduced. ZIP CRC and packaged SHA-256 checks passed, including the
original image and mask checksums. The validation record is
`api/logs/active_vision/exports/nico_dice_multiview_20261019_replay_validation.json`.

```bash
cd ~/catkin_ws/src/NICO-software/api
PYTHONPATH=src/ELMiRA/scripts python3 -m active_vision.sparse_preview \
  --manifest logs/active_vision/dice_multiview_20261019T065922_054813Z/manifest_v2.json \
  --output logs/active_vision/dice_multiview_20261019T065922_054813Z/new_replay \
  --detector akaze
```

Three synthetic exporter tests check known plane depth and files, insufficient
baseline rejection, and duplicate/moving-input rejection. Capture tests also
verify the explicit experimental-calibration selection. Metric object dimensions,
ground-truth masks, hidden-surface recovery, learned recognition and autonomous
next-view performance remain unverified. This is a real RGB multiview result,
with experimental status retained, rather than another synthetic demonstration.

### Rigid Wood-Block Follow-Up

The user replaced the cloth dice with an assembled coloured wood block reported
as approximately 36 x 36 x 36 mm (not independently measured ground truth).
After explicit sweep clearance, four supervised head-yaw views were captured in
`measured_viewpoint_20261019T072952_298270Z/`. Every view detected all 54 board
corners; candidate pose-fit RMS was 0.165/0.229/0.271/0.319 px. The head returned;
independent READ-only diagnostics confirmed both head torques off and no motor
error replies. No arms or hands were commanded.

`wood_multiview_20261019T073155_422464Z/reconstruction_delivery/` contains 47 merged
sparse points and 3,739 experimental hull surface voxels. The hull remains
elongated and touches its estimated bounds: rigid-object substitution did not
establish accurate cuboid recovery. Initial-to-10/20-degree pairs had 42/34 mutual
descriptor matches but zero candidate epipolar inliers, so both correspondences
and object-region camera geometry need review. Do not attribute the previous
distortion solely to cloth deformation or rescale output to force 36 mm.

The raw board-axis sparse spans are 75.3/54.0/57.5 mm, not independently measured
object dimensions or a valid accuracy score. Masks, metric scale and per-image
poses remain unverified. The dataset README records these limitations. A portable
package is saved as `api/logs/active_vision/exports/nico_wood_multiview_20261019.zip`.

### Shape-Quality Failure Diagnosis

The wood result must not be presented as successful cube recovery. An offline
audit of the original output found 938 of 3,739 exported surface samples outside
at least one original object mask, even with a 2 px native-pixel margin. The
preview now filters hull exports against every native mask, preserving the
original inputs/results and production verification gates. The separate
`reconstruction_quality_checked/` output contains 2,801 remaining hull samples and
is explicitly marked `unreliable_clipped_volume`; it still does not recover the
cube. CLI output now states that object shape is not validated.

`active_vision.reconstruction_diagnostics` runs without hardware. For the saved
wood candidate poses, maximum object-ray direction change is about 8.9 degrees
using the unverified sparse-median object centre, not the 20-degree head sweep.
A known 36 mm SYNTHETIC cube with perfect pinhole silhouettes at those same poses
still produces a clipped elongated intersection in the test volume. Six virtual
axial directions instead yield spans about 38.6 mm (raster/grid error remains).
This isolates a viewpoint-observability limitation; the synthetic cube is NOT a
fitted real object or proof of real metric accuracy. It does not resolve the
separate real matching, mask, calibration and timing uncertainties.

The comparison/report are in
`wood_multiview_20261019T073155_422464Z/quality_diagnosis/`. Four exporter tests and
two diagnostic-control tests pass. No hardware was accessed for these tests.

A proposed next collection method is a rigid object-plus-checkerboard assembly:
keep NICO stationary and move the entire assembly manually to obtain genuinely
different sides. The object MUST remain fixed relative to the checkerboard,
which then defines the reconstruction reference frame. This is not the earlier
fixed-room-board motor sweep. Log that the board moves in the room and measure
every per-image camera pose in the common rig frame. Rotating only the object
while retaining old camera poses is invalid. Begin with a few distinct views
and review geometry before another full collection; this rig method has not
yet been physically tested. It does not by itself verify calibration or recover
surfaces that remain unseen.

### Opposite-Yaw Check and Fisheye Inverse Failure

After separate user clearance, `--opposite-yaw` performed fixed -5/-10/-20-degree
stages at the unchanged low speed/torque, using the independently measured
2124/2128 head anchor. Actual capture feedback was -4.75/-9.67/-19.60 degrees;
all54 board corners and the target appearance remained visible. The head
returned and an independent READ-only diagnostic confirmed both torques off.
No30-degree, pitch, arm or hand movement was attempted. Original capture:
`measured_viewpoint_20261019T080153_514941Z/`.

The old four-coefficient candidate has a radial derivative fold at80.31 degrees,
normalized distorted radius1.288. In the opposite20-degree target grid,32/49
samples failed numerical inverse/roundtrip checks (median21.48px, max803.58px).
`Calibration.pixel_domain_report` now exposes this, and experimental feature
unprojection/supervised target collection reject invalid inverse domains. A
numerical roundtrip pass is not a physical calibration verification.

An OFFLINE lower-order candidate fixes k3/k4 to zero using the same18 training
images. Training RMS changes0.20163 ->0.20375px; seven old holdouts and four new
opposite images pass native-board pixel checks. No radial fold remains. This
candidate is still unverified and NOT deployed or selected for motor execution.
Candidate/comparison: `fisheye_domain_comparison_20261019T080716_898987Z/`.

Experimental pose fitting can now use native distorted corners directly. Hull
preview rectification rotates a virtual pinhole camera toward each object mask,
updates T_wc axes accordingly, and keeps the physical optical centre unchanged.
This avoids losing a peripheral fisheye target to a single global pinhole crop;
it does not generate extra baseline. Four synthetic domain/native-PnP/virtual-
projection tests cover these transformations.

The original return appearance check searched near the outward position. Its
failure is preserved in the raw evidence; baseline-window offline reassessment
passes. Source now resets that window and reports unsuccessful return-camera
checks as partial rather than full success.

`wood_bilateral_20261019T081301_283472Z/` combines seven unchanged RGB views and
unreviewed mask proposals. With poses recomputed in a common native-board frame
using the SAME lower-order candidate, one-sided vs bilateral object-ray spans
are8.90 vs15.64 degrees and baselines41.75 vs79.48mm. The result is108 sparse
points and2,656 hull samples, still clipped and NOT successful cube recovery.
Greater view range helped coverage but did not by itself solve shape accuracy.

### Native Object Landmark Audit

The bilateral dataset now contains
`corner_audit_20261019T082704_195309Z/corner_audit_report.md` and annotated images.
This is entirely offline: no ROS, serial, camera or motor access. Three
assistant-selected colour junctions are refined locally and tracked by image
appearance in all seven views; they are not independent human ground truth.
All21 observations pass the appearance checks (maximum forward/backward
closure0.061px). The widest pair, indices3/6, triangulates points with
native-pixel point-only refinement; five other images stay out of the fit.

The15 held-out observations have median3.425px and maximum3.790px residuals,
all above the diagnostic2px threshold. The same views' native board-pose fit
RMS values are0.239..0.489px. This discrepancy cannot be dismissed as sparse
descriptor matching alone, but it does not uniquely identify calibration,
pose, annotation, board-flatness or physical scene-motion error. No cube-size
constraint was used and no edge-length accuracy is reported.

An image-coverage audit of972 training corners finds six target-mask regions
entirely outside their convex hull; positive20 has59.5% sampled pixels inside.
This reveals a spatial validation gap, not proof that every outside ray is
wrong or every inside ray accurate. The next proposed collection is a small
securely supported checkerboard batch covering the target's left/lower image
region, with NICO still, followed by held-out checks before further reconstruction.
Do not extend a platform beyond the table edge or increase the head sweep to
compensate for uncertain object-region geometry. Three new synthetic audit
controls bring the offline regression suite to98 passing tests.

## Hardware Checks: 2026-10-15

- Complementary eye-cover checks identify serial `22275000` as the physical
  **left** eye and `270B500C` as right. These differ from legacy `NICO_EYES`
  labels. Existing ELMiRA topic assignments were not changed; select the camera
  by serial and separately verify its ROS/optical-frame binding.
- The measured 9x6 / 25 mm board yielded 23 accepted images (16 fit, 7 holdout).
  All seven held-out pixel checks passed. Central rectification and a coarse
  independently measured distance were checked. Outer-image coverage and full
  metric accuracy remain unverified; the candidate is not deployed to ELMiRA.
- Head IDs 19/20 return stable feedback. They were not at the assumed
  `(head_z=0, head_y=-0.5)` pose. Earlier readings were approximately
  `(-0.02225, -1.00577)` radians; both head torques were confirmed off.
- With user-confirmed clearance, one yaw-only test requested 2.99 degrees at
  1.368 degrees/s and a temporary 14.66% torque cap. The encoder changed only
  0.62 degrees before the six-second settling timeout. No stronger retry or
  arm/hand/pitch command was sent, and no settled after-view was captured.
- Initial cleanup read yaw torque off, but a subsequent check found it on after
  a goal/settings restoration write. A yaw-only disable with no later goal
  writes was followed by repeated off checks and an independent final read.
  Cleanup now performs no register writes after disable. Camera-viewpoint
  execution remains blocked, not claimed safe based on offline tests alone.
- All 34 existing offline tests and 17 head-diagnostic tests pass. The new
  tests cover the packet allowlist, startup preload, feedback failures, timeout,
  cleanup, unexpected torque-on and the execution lock. Read-only mode is:

  ```bash
  cd ~/catkin_ws/src/NICO-software/api
  PYTHONPATH=src/ELMiRA/scripts python3 -m active_vision.head_viewpoint_test
  python3 -m unittest discover -s src/ELMiRA/tests -p test_head_viewpoint_test.py -v
  ```

Evidence: `api/logs/active_vision/head_viewpoint_review_20261015.json`, including
links to original images, test feedback and the final torque-off report.
Follow-up investigation: `api/logs/active_vision/head_root_cause_review_20261015.json`.
It recorded 38 clean read replies with no motor-reported error bits, both torques
off, 11.8 V and 36..37 C. Both head PID gains read P32/I0/D0, matching the
[documented MX-64 defaults](https://emanual.robotis.com/docs/en/dxl/mx/mx-64/#pid-gains-26-27-28).
Offline packet reconstruction confirmed correct ID19/address encoding. The
earlier encoder trace plateaued for about 4.64 seconds, leaving a 2.37-degree
error. The low test speed/torque and mechanical resistance are plausible factors;
the old trace lacks current/load evidence to distinguish them. The exact source
of post-cleanup torque-on is not established. A low-torque test failure is not
proof of a defective motor.

Read-only transport now blocks writes by default, and future motion diagnostics
log current/load plus continuous goal/speed/limit readback and reject overwritten
commands. Camera-viewpoint execution is still locked; PID and EEPROM limits were
not changed. Read-only
extended diagnostics, with no camera or motor movement:

```bash
cd ~/catkin_ws/src/NICO-software/api
PYTHONPATH=src/ELMiRA/scripts python3 -m active_vision.head_diagnostics
```

### Supervised Controller Follow-Up, 16:41 Local

The user reported being able to turn the head left/right manually while power
was on, and subsequently confirmed a clear workspace for one supervised check.
Readbacks showed torque off, which distinguishes power from active holding.
`--controller-check --workspace-clear` is an explicit diagnostic mode, not an
unlocked camera or active-vision workflow. It keeps the original raw speed2,
torque cap150 and fixed34-count step, with no stronger retry or pitch/arm commands.

The check stopped **before the yaw step**: writing the current-position goal
made torque read enabled before the explicit enable call. The guard immediately
disabled yaw. Five delayed checks and a separate raw-serial diagnostic confirmed
both torques off. No later goal/settings write was made. Yaw speed2/limit150 are
left in RAM deliberately, rather than restoring the previous maximum settings.
This reproduces the observed torque transition but does not identify its
firmware/intermediary mechanism or prove the earlier motion stall's cause.

Final independent feedback was yaw2050/pitch1405, approximately +0.22/-56.48
motor degrees. The user's low-head photo suggests checking clearance before yaw,
but does not prove contact. Raising pitch is not yet tested and would not fix
the preload assumption. No pitch motion was attempted. Current/load/phase and
cleanup telemetry and 17 offline head tests are retained; hardware tracking and
useful camera baseline are still unverified.

Subsequent physical inspection by the user confirmed the head is touching the
shoulder and preventing a turn. Treat that as a mechanical obstruction, not a
reason to increase yaw torque. Establish head/shoulder clearance and support
against pitch dropping before any further powered test. No pitch command was
sent in this follow-up. Contact does not explain the separate torque-on-preload
observation, and no hardware defect or safe general motion range is established.

Evidence: `api/logs/active_vision/head_controller_review_20261015.json` links both
the original controller trace and independent post-stop register readback.

### Raised-Head Yaw Check, 17:01 Local

After the user raised the head clear of the shoulder with power off and restored
power, independent readings gave yaw 2082..2083 / pitch 2243..2244. The controller
diagnostic now requires `--raised-head-clear` in addition to its explicit execution
and workspace flags. It accepts only a four-count neighbourhood of that reviewed
pose, checks the reference report hash, and rejects the old shoulder-contact
posture. This is not a general head workspace limit or an automatic recenter.

For this reviewed setup only, the first write is the measured current yaw goal
with the existing raw speed 2 / torque 150 cap. Goal-triggered activation is accepted only
after settings readback, near-zero setup movement and pitch-drift checks; five
settled current-position samples are required before sending the small step.
The generic unreviewed preload guard and camera-workflow lock remain in place.
No pitch, arm, hand, PID, EEPROM or ID changes were made.

The user explicitly requested one yaw test. Encoder feedback moved from 2082 to 2112
against target 2116: **2.637 degrees achieved of 2.989 requested**. Pitch varied by
only one count. Goal/speed/limit readbacks matched throughout. The final four-count
error (0.352 degrees) exceeded the unchanged two-count settling tolerance, so the
six-second guard stopped the test without a stronger retry or return command.
Five cleanup samples plus a separate 38-reply raw diagnostic confirmed both head
torques off. The head remains about 2.64 degrees from its test-start yaw.

This is encoder-confirmed movement, **not** a complete out-and-return pass. The
user could not tell whether the small movement was visually smooth, so physical
observation remains unverified. Current and speed registers stayed at their idle
encodings while position changed; do not infer zero physical effort or torque
saturation from these readings. The raised camera view crops the board: no useful
optical-centre baseline or new 3D result was established. Do not rerun blindly; the
final yaw is outside the original narrow starting anchor.

Validation: 21 offline head tests, 34 active-vision tests and compile checks passed. Evidence:
`api/logs/active_vision/head_raised_yaw_review_20261015.json`, including the original
feedback trace and independent post-test torque-off report. No thresholds were
relaxed to relabel this recorded timeout as a success.

The yaw test does not verify a useful optical-centre baseline for 3D. Do not
enable normal ELMiRA startup or assume a head command is a camera pose to bypass
the pending hardware review. Keep the fixed board reference unchanged for a
future measured-viewpoint test.

### Supervised Staged Yaw Check, 17:13 Local

The user requested a larger visible test and explicitly confirmed full-range
clearance, loose cables and readiness at the motor power switch. Fresh read-only
feedback was yaw2112 / pitch2243, both torques off. The separately gated
`--yaw-20deg-test --full-sweep-clear` controller mode uses only fixed relative
offsets 57,114,227 counts (approximately 5,10,20 degrees), then returns through
114,57,0. It retains raw speed2 / torque cap150; no pitch, arm or hand command
is permitted. This is a one-direction excursion and return, not a +/-20-degree
sweep or a general approved workspace.

Before execution, a new observation-test settling criterion was documented:
within six counts (0.527 degrees) and five consecutive readings stable within
two counts. Per-segment travel bounds and timeouts stop later stages on failure.
The earlier three-degree timeout retains its original two-count criterion and
is not retroactively relabelled a pass. Neither tolerance is camera-pose accuracy.

All six stages completed. The outward settled changes were 4.57, 9.58 and
19.43 degrees; the largest measured excursion was 19.52 degrees. The return
ended three counts (0.264 degrees) from the starting yaw. Pitch varied only one
count; voltage stayed 11.8V and yaw temperature 37C. Goal/speed/cap readbacks
matched the requested settings during the motion. The user reported:
"Moved smoothly; no contact or unusual noise."

The final register write disabled yaw. Five delayed checks and an independent
38-reply read-only diagnostic confirmed both head torques off. Independent yaw
readings were 2114..2115, pitch2243. Speed2 / cap150 remain in RAM deliberately;
no old maximum settings were restored. This confirms this supervised raised-head
motion, not the safety of arbitrary poses or a diagnosis of every earlier issue.

Validation: compile checks, 27 offline head tests and 34 active-vision tests passed.
Evidence: `api/logs/active_vision/head_staged_yaw_review_20261015.json`, linking
the original trace and independent torque-off readback. No camera frames were
captured during this controller-only test. Useful optical-centre displacement,
calibrated camera poses and reconstruction remain unverified; camera-workflow
execution stays locked. Next obtain a fully visible, rigidly fixed board at the
raised posture, then measure actual camera poses in a separately supervised test.

## What Is Implemented

| Area | Implementation | Validation boundary |
| --- | --- | --- |
| Object masks | GrabCut with padded target box, foreground/background scribbles, overlays and pixel IoU/Dice evaluation | Proposals need review; box initialization is not segmentation ground truth |
| Calibration | OpenCV pinhole/fisheye fitting, held-out reprojection checks, resolution/device identity checks, rectification | 23-view candidate; sampled central-region checks passed, full metric calibration unverified |
| Camera poses | Exact-image-time ROS1 TF, or fixed metric chessboard PnP; saves `T_wc`, timestamps and head feedback | Requires verified camera extrinsics or fixed board coordinates |
| Reconstruction | Professor's visual-hull code with observed-voxel support, displacement gates; ORB + checked triangulation | Coarse hull/sparse points, not precise dense reconstruction |
| Recognition | Professor's pretrained RGB branch, real-object enrollment/prototypes, rejection threshold calibration and independent test sessions | Synthetic checkpoint is not a real-object classifier; real training/test data still needed |
| Active loop | Measured-catalog bootstrap, professor's CPU Gaussian-proxy next-view scorer when geometry/recognition are ready, continued settling checks | No automatic motor execution; physical displacement and ranking quality not verified |

The original eleven vendored Python files remain unchanged. Provenance and hashes:
`api/src/ELMiRA/scripts/active_vision/vendor/PROVENANCE.txt`. The checkpoint is
`api/model_checkpoints/active_vision_v8/single_rgb_model.pt`; it is loaded with
`weights_only=True` on CPU. Confirm the professor's redistribution permission
before uploading the vendored source/checkpoint. No GitHub upload was performed.

The next-view selector bootstraps from measured geometric coverage. After two
accepted views and a real-object recognition evaluation are available, it uses
the professor's `BatchedSE3NBVPlanner` and CPU `SE3ObjectGaussianMap`. Proxy depth
comes only from multi-view-supported RGB hull voxels, not a depth sensor. The map
is rebuilt after carving; invisible/already-observed candidates are rejected.
The proxy is not a precision surface, and prototype entropy is not validated
real-world conformal coverage. The RGB encoder never receives invented zero
geometry in its 3D branch. Ranking quality still needs physical evaluation.

## Initial Integration Results (2026-10-13)

- All 34 offline unit tests passed. The suite covers segmentation, pinhole/fisheye rectification,
  calibration fitting with synthetic corner sets, PnP, triangulation, visibility-
  supported hulls, observation provenance, active-loop settling/timeouts, ROS image
  decoding and recognition dataset isolation. Run the command below for the current count.
- An isolated ROS1 master with **synthetic** Image/Mask/JointState/CameraInfo/TF
  messages captured two views, saved synchronized head feedback and metric poses,
  produced a coarse two-view hull, and rejected a stale image. No robot nodes ran.
  Evidence: `api/logs/active_vision/ros1_smoke_report.json`.
- Real NICO trial `T20260602_081617`, frames `030..054`: 25 masks generated using
  the manually selected padded box and foreground scribble below. First/middle/
  last overlays were inspected. The initial box-only attempt collapsed; a tiny
  foreground scribble also failed. These failure cases are now rejected.
- Corrected proposals contain approximately 1,283..1,454 foreground pixels.
  Some silhouette/shadow boundaries still need manual correction. No real-image
  IoU/Dice is claimed without independent pixel annotations.
- All 25 corrected masks/images ran through the supplied RGB encoder: finite
  `25 x 64` embeddings. This proves execution, not object-recognition accuracy.
- The initial sandbox device check was incomplete. A host-level check found two
  See3CAM_CU135 cameras (serials 270B500C and 22275000) and the FTDI `/dev/ttyUSB0`
  adapter. Both cameras delivered 1280x720 UYVY frames. A new camera-only capture
  helper also passed a real-camera smoke test without changing settings.
- A current die image produced a visually plausible pixel-mask proposal and a
  finite `1 x 64` feature. This used a manually supplied box/foreground point,
  not automatic object recognition or independently measured mask accuracy.
- Real ROS master on port 11311 refused the connection. A serial head-read
  attempt failed during DTR setup; a second transport leaving modem lines alone
  stalled and was stopped. Head feedback remains unverified, not evidence of a
  failed motor. All diagnostic camera/serial handles were released.
- Existing stored calibration is for different camera serials at 1920x1080;
  it is not accepted for the connected 1280x720 cameras. User has no verified
  board ready yet. A printable A4 board and capture guide are now provided below.
- The original next-view scorer ran with synthetic metric geometry and the
  supplied RGB checkpoint. Scores were finite; no motion was commanded.
  Evidence: `api/logs/active_vision/nbv_synthetic_smoke.json`. This is software
  validation, not real-world viewpoint/recognition performance.
- The old sequence has no synchronized exposure/ROS timestamps, measured camera
  poses or verified session calibration. It is deliberately **not** reconstructed.
  Operator-entered trial XYZ and `head_y=-0.5` are not camera extrinsics.

## Run Offline

Run from the repository's `api` directory. Existing Python 3.8 with OpenCV 4.3,
NumPy 1.21.6 and PyTorch 2.4.1 worked in this environment. `unittest` is sufficient.

```bash
cd ~/catkin_ws/src/NICO-software/api
python3 -m unittest discover -s src/ELMiRA/tests -p test_active_vision.py -v
python3 -m compileall -q src/ELMiRA/scripts/active_vision src/ELMiRA/scripts/active_vision_cli.py
python3 src/ELMiRA/scripts/active_vision_cli.py --help
python3 src/ELMiRA/scripts/active_vision_cli.py ros-check
```

Optional middleware test (starts and stops its own private localhost ROS master,
not ELMiRA; publishes only synthetic test data):

```bash
python3 src/ELMiRA/tests/ros1_active_vision_smoke.py
```

Reproduce masks in a **new** output directory (existing directories are not
overwritten). Original JPEGs and trial records are untouched:

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py segment \
  --images logs/first_person_trials/T20260602_081617/frames \
  --first 30 --count 25 --box 374 222 54 61 \
  --foreground-point 401 253 --foreground-radius 14 \
  --output logs/active_vision/masks_new

python3 src/ELMiRA/scripts/active_vision_cli.py features \
  --manifest logs/active_vision/nico_25_review_proposals/masks.json \
  --checkpoint model_checkpoints/active_vision_v8/single_rgb_model.pt \
  --output logs/active_vision/features_new
```

For a difficult frame, create an aligned one-channel PNG of GrabCut scribbles:
0=definite background, 1=definite foreground, 2=probable background,
3=probable foreground. Use `segment --count 1 --scribbles FILE.png` with that
frame's box. Include several object surfaces, exclude shadows/hand/table, and
inspect the overlay. A binary foreground mask is 0/255 and is a different format.
External reviewed segmentation masks can be supplied directly in manifests or
on the ROS mask topic; no particular external segmentation model is required.

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py mask-evaluate \
  --prediction /path/predicted_mask.png --ground-truth /path/human_pixel_mask.png
```

The outputs already created are under `api/logs/active_vision/`:
`nico_25_review_proposals/` (use these proposals), `nico_25_review_features/`, and
`ros1_smoke_report.json`. Earlier `nico_25_seeded_masks/`, `nico_mask_surface_test/`
and `nico_25_features/` are failed-mask experiments, not accepted segmentation.

## Collect Verified Calibration

Start with [Print and Capture the NICO Calibration Board](calibration/README.md).
The generated vector PDF is `docs/calibration/nico_checkerboard_A4_9x6_25mm.pdf`.
It has 9x6 inner corners and nominal 25 mm squares; measure the printed copy.
The new `calibration-capture` CLI operates directly on one camera with no ROS
or motor access and saves original images, device settings and host timestamps.

1. Record camera serial/device, selected eye, optical frame, resolution, zoom,
   focus and exposure settings. Do not identify a camera only by image size.
2. Capture at least 10 distinct views of a measured chessboard, spanning the
   usable image area and multiple tilts/distances. Reserve at least 3 different
   views for holdout checks. Do not use repeated copies of the same image.
3. Choose `equidistant` for a verified fisheye model or `plumb_bob` for pinhole
   distortion. The existing `json/nico_vision_calibration_params.json` is an
   unverified reference for this recording, not a drop-in calibration.

Example board dimensions below are **examples**, replace them with measured
inner-corner counts and square size:

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py calibrate \
  --images /path/calibration_train --columns 9 --rows 6 --square-m 0.020 \
  --camera-id ACTUAL_SERIAL --optical-frame ACTUAL_OPTICAL_FRAME \
  --model equidistant --output /path/calibration_candidate.json
python3 src/ELMiRA/scripts/active_vision_cli.py calibration-check \
  --calibration /path/calibration_candidate.json --images /path/calibration_holdout \
  --output /path/calibration_holdout_report.json
```

Both commands leave `verified=false`. Review per-view errors, corner coverage,
held-out error, straight-line rectification, actual board dimensions and an
independent measured distance. Only then set `verified=true` and a nonempty
`verification_evidence` describing who/when/how it was checked. A low fit error
alone does not establish correct metric scale. Images and masks are rectified
together; original distorted pixels must not be projected with rectified K.
See [OpenCV calibration](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)
and [fisheye model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html).

## Measured Camera Poses and ROS1 Capture

`T_wc` maps OpenCV optical coordinates (+X right, +Y down, +Z forward) into a
fixed world frame, translations in metres. Reconstruction never uses ELMiRA's
learned image-to-arm coordinate service or recorded grasp XYZ as a camera pose.

Two pose sources are supported:

- Verified TF chain from world to the actual eye's **optical** frame, queried at
  each image timestamp. Measure/validate camera-to-head extrinsics and head
  kinematics, including axis signs, offsets and rotation order. Do not invent
  a translation from head angles. Verify the chain against a fixed measured board.
- A rigidly fixed, metric chessboard visible in each RGB image. Set
  `pose_source=fixed_metric_chessboard_pnp`, `board_columns`, `board_rows`,
  `square_m`, and `board_fixed_and_order_verified=true` in the local config.
  The board defines the world frame. Verify corner origin/order: a symmetric
  board can flip its detected orientation. Reject inconsistent/occluded views.

The current NICO publisher provides a software image timestamp but no `frame_id`.
The adapter therefore rejects metric processing unless its physical camera/frame
binding has been checked and documented using `image_frame_binding_verified`
and `image_frame_binding_evidence`. Do not merely toggle these to bypass errors.
Software publish times are not verified exposure times: measure latency and use
stationary/settled captures before trusting moving-camera synchronization.

Use the example config at
`src/ELMiRA/scripts/active_vision/config/ros1_capture.example.json` as a local
configuration. Replace identities/frame names, set verified calibration and pose
evidence when available. Without calibration the node still records raw RGB and
available head feedback for later calibration; it does not output metric geometry.

```bash
source activate.bash
source devel/setup.bash
python3 src/ELMiRA/scripts/active_vision_cli.py ros-node --config /path/local_capture.json
```

From another terminal in the sourced ROS1 environment:

```bash
rosservice call /acf_single_rgb/capture "{}"
```

This saves a PNG, per-image JSON and `session.json` under a new
`api/logs/active_vision/session_*` directory. Verified calibration is snapshotted
inside the session. Original image header, head joint sample, image/pose times,
pose provenance, missing-data reasons and image/mask hashes are recorded.
`Capture success` means a raw frame was saved; check its `missing` and `geometry`
fields before treating it as a valid multiview observation. Masks must have the
same image timestamp and original frame ID, not just similar arrival time.

For geometry, set `object_static=true` only for a stationary rigid object.
Measure `volume_center_m` and `half_extent_m` in the **same world frame** and
document `volume_verified`/`volume_evidence`. Hand/object movement invalidates
rigid-object reconstruction; exclude those frames. Reviewed masks require
`mask_verified` and `mask_evidence`, or review saved per-image masks individually.

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py reconstruct \
  --manifest /path/session.json --output /path/reconstruction
```

The NPZ separates `visual_hull_xyz_m` from `triangulated_xyz_m`; it also contains
rectified K, accepted T_wc, volume centre and scale. JSON explains acceptance and
rejection. A single view, repeated viewpoint or pure rotation cannot pass as 3D.
The 10 mm baseline / 1 degree view-angle thresholds are conservative software
quality gates, **not** a claim that NICO's head produces these displacements or
that such baselines guarantee accurate shape. Validate against measured objects.

## Real Recognition and Active Views

Create a labelled JSON dataset with `samples`. Each entry needs `image`, `mask`,
`mask_reviewed:true`, `label`, `session_id`, and `split` (`enroll`, `calibration`,
`test`). Image/mask paths are relative to the manifest. Use at least two real
object classes, plus unknown objects in calibration and test. Different frames
from one video must stay in the same split. Duplicate images are rejected.

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py recognition-evaluate \
  --manifest /path/real_object_dataset.json \
  --checkpoint model_checkpoints/active_vision_v8/single_rgb_model.pt \
  --output /path/real_object_evaluation
```

This enrolls real-object prototypes, chooses a similarity rejection threshold on
calibration sessions only and reports accuracy on separate test sessions. To
enable ROS inference, set `checkpoint` and `recognition_evaluation` in the local
config. Similarity is not a probability. No real accuracy is currently available.

For the active loop, collect a verified `candidates` catalog. Each entry requires
`name`, measured `T_wc`, `world_frame`, `optical_frame`, `head_positions_rad`
containing only `head_z` and `head_y`, `verified:true`, and `evidence`. All poses
must refer to the same camera/world calibration. Define **supervised workspace-
safe** `head_limits` with `verified`/`evidence`; do not copy motor hard limits.

Current configuration references: head IDs 19/20; broad motor JSON limits are
yaw -179..180 degrees and pitch -89..79 degrees. These are NOT approved scan
limits. The FYP operating reference remains yaw 0, pitch -0.5 radians. The
example permits no sweep and is unverified. `/NICOL/joint_states` reports these
positions in radians. Verify both physically before extending the range.

Observe using `capture` -> inspect the logged candidate -> supervisor performs
an approved head movement using the existing controller -> adapter waits for
fresh measured joints to settle -> `capture` a new image. It compares measured
camera displacement, not command success. Requests expire after 20 seconds;
`rosservice call /acf_single_rgb/reset_loop "{}"` resets only the software loop.
The adapter does not call `/NICOL/head/goal_joint_space_path`. That existing
service returns after publishing angles and ignores path_time; this alone is
not evidence that a calibrated, collision-free viewpoint was reached.

The loop rechecks head feedback after settling. Further movement cancels readiness;
stale images cannot satisfy a pending request. Observations must agree with the
measured catalog within the current software tolerance (10 mm translation,
3 degrees rotation); these gates are not physical calibration accuracy claims.
Geometry rejection never marks a viewpoint completed. Candidates pointing behind
the camera/outside its image and low-visibility rendered proxies are not requested.
Without an evaluated real-object memory, bootstrap remains geometric and is
labelled as such; no synthetic-label classifier is presented as object recognition.

If yaw/pitch mostly rotate the optical centre, stop the reconstruction test.
Collect a measured camera translation using a supervised rig/repositioning
method or revise the experiment with the professor. Do not fabricate baseline.

## Remaining NICO Validation

1. Verify camera identity, focus/zoom, measured intrinsics and holdout images.
2. Human-correct object masks and collect independent pixel-mask ground truth.
3. Measure camera extrinsics/fixed-board poses and image/feedback latency.
4. Verify real head feedback, axis directions, conservative limits and useful
   optical-centre translation while the target remains static and visible.
5. Collect calibrated multiview sessions; measure reconstruction scale/error
   against known dimensions. Test pose uncertainty and weak texture/occlusion.
6. Collect separate labelled object enrollment/calibration/test sessions and
   unknown-object examples. Report real confusion/rejection results.
7. Validate supervised observe/move/observe before considering any separate,
   explicitly safety-gated automatic head executor. None is enabled here.

## Real Object Recognition Diagnostics: 2026-10-21

All data and results for this experiment are under
`api/logs/active_vision/real_object_recognition_20261021/`:

- `manifest.json`: sample roles, capture sessions and collection progress.
- `sessions/`: unchanged native images, timestamps, assistant-assisted masks,
  mask review images and `capture_session.json` per placement.
- `enrollment_checks/`: the original 30-image RGB feature/enrollment check.
- `threshold_calibration/`: the preserved professor-RGB failure diagnosis.
- `offline_diagnostics/`: ROI normalization, reproducible calibration-only
  comparisons and diagnostic control memories. Nothing is deployed to ROS.
- `method_comparison.csv` and `experiment_index.json`: readable result table
  and machine-readable artifact index.

There are 30 enrollment images (can, bottle and green apple model; two static
placements per label) and 20 calibration images (one new placement each of
can, bottle, apple and a corn model). Each placement has five correlated
frames: these are **not 50 independent recognition trials**. Corn is labelled
`unknown` for rejection-threshold calibration, never enrolled as a known class.
An unseen transport model is reserved for future unknown-object testing.
No independent test images have been collected at this checkpoint.

The professor's RGB visual branch does run, but its frontend converts BGR to
grayscale before semilog-polar/spectral processing. The original weights are
unchanged. Re-encoding an identical saved input is repeatable, but some bursts
and newly positioned objects give inconsistent nearest-class matches. Cropping
each masked object onto a padded 192x192 canvas did not improve aggregate
calibration performance. Replacing class-mean prototypes with nearest stored
exemplars also failed to improve it. These observations do not isolate one
cause: synthetic-to-real transfer, sampling/appearance sensitivity and mask
limitations remain plausible contributors. This is not proof of reliable real
recognition or of a defect in the complete professor 3D pipeline.

An explicitly separate `color-shape` control uses OpenCV masked HSV histograms
(18x4x4 bins, square-root normalization) and five silhouette measurements
(relative bounding-box width/height, extent, circularity and solidity).
Normalized histogram and shape blocks have squared weights 0.85 and 0.15.
It has no trained network and is **not the professor's learned model**.
It cannot distinguish objects with the same color distribution and silhouette.

| Method | Correct Calibration Frames / 20 | Independent Test |
| --- | --- | --- |
| Professor RGB branch, original input and class means | 10/20 | Not collected |
| Professor RGB branch, object-normalized ROI and class means | 10/20 | Not collected |
| Native/ROI nearest-exemplar alternatives, each | 5/20, rejecting every frame | Not collected |
| OpenCV color + silhouette diagnostic control | 20/20 | Not collected |

These are development-set fits, not final accuracy or 20 independent trials.
The color/shape control confirms that a simple appearance-based separation is
possible in these assisted-mask examples. It does not establish semantic object
recognition, the professor model's success, autonomous detection, novel-object
generalization or deployment readiness. Transparency leaves transmitted scene
inside bottle masks; minor boundary/shadow uncertainty remains in other masks.
No independent pixel-mask ground truth has been collected.

Reproduce calibration only, from `api`. Use new output directories; previous
calibrations are not overwritten:

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py recognition-calibrate \
  --manifest logs/active_vision/real_object_recognition_20261021/manifest.json \
  --checkpoint model_checkpoints/active_vision_v8/single_rgb_model.pt \
  --output logs/active_vision/real_object_recognition_20261021/offline_diagnostics/professor_rgb_repeat

python3 src/ELMiRA/scripts/active_vision_cli.py recognition-calibrate \
  --manifest logs/active_vision/real_object_recognition_20261021/manifest.json \
  --encoder color-shape \
  --output logs/active_vision/real_object_recognition_20261021/offline_diagnostics/color_shape_repeat
```

`recognition-calibrate` enrolls only `enroll` rows, fits thresholds only on
`calibration` rows and never decodes or embeds test images. It writes a
`calibration_report.json`, `calibration_memory.npz` and source-manifest snapshot;
it does not produce an independent-test report. The snapshot preserves original
relative paths for provenance; run commands against the dataset's main manifest.

The default encoder remains `professor-rgb`. `recognition-evaluate` additionally
accepts `--encoder color-shape` for explicitly labelled offline control results.
The ROS adapter rejects control-baseline reports instead of silently loading
them as a learned professor model. It does not activate calibration-only files.

Before independent tests, freeze both methods, their memories, margin 0.045
and fitted similarity thresholds (native RGB 0.98; color/shape 0.87). Record new
positions/orientations of the three known physical objects and a distinct,
previously unseen transport model. Keep complete capture groups out of both
enrollment and threshold fitting; do not reuse corn as the unseen test object.
Compare the frozen methods on exactly the same reviewed masks. Report both
frame counts and placement counts; never treat one short burst as five
independent repetitions. Any future refinement after inspecting test results
requires another fresh test set. No head, arm or hand motion is needed.

## Optional DINOv3 RGB Comparison

The offline CLI now accepts `--encoder dinov3`. This is a **separate frozen RGB
backbone**, not a modification of the professor's pretrained weights and not
an already validated improvement. The default remains `professor-rgb`.
The implementation targets the official
[DINOv3 ViT-S/16 snapshot](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
Access to those weights requires accepting Meta's conditions with an authorized
account. No weights, tokens or licenses are bypassed by this integration.

The current NICO environment has Python 3.8.10 / Transformers 4.34.1, with no
cached DINOv3 weights or HF authentication found during the 2026-10-21 check.
That environment cannot run this encoder. Native Transformers DINOv3 support
starts at 4.56.0, as documented in the
[official repository](https://github.com/facebookresearch/dinov3).
**Do not upgrade the ROS environment in place.** The following is a proposed
isolated setup, not a claim that pretrained inference has already been tested.
Use a computer with Python 3.10/3.11 available, or install that interpreter
separately with the system administrator. No robot is needed.

From the repository root:

```bash
python3.10 -m venv .venv-dinov3
source .venv-dinov3/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r api/src/ELMiRA/scripts/active_vision/requirements_dinov3_offline.txt
```

After accepting the model's access conditions in your browser, log in privately
using the HF CLI. Never put a token in a command, manifest or experiment log.
This is the **explicit download step**; the NICO encoder itself never downloads
models or executes remote model code.

```bash
hf auth login
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir api/model_checkpoints/dinov3_vits16 \
  --include config.json preprocessor_config.json model.safetensors
```

An authorized snapshot copied from another computer is also supported. It must
contain all three files above. A `.pth` torch-hub checkpoint alone is not the HF
snapshot expected by this adapter. Do not rename a pickle file to safetensors.
The loader requires the ViT-S/16 configuration, native local model code and
complete matching safetensors weights; missing/unexpected weights are rejected
instead of using randomly initialized layers.

Preflight works even in the existing Python 3.8 environment and prints the
specific blockers. `ready_for_load` means only that files/package versions pass
preflight, **not** successful model inference. An unavailable preflight exits
with code 2, with no robot action.

```bash
cd api
python src/ELMiRA/scripts/active_vision_cli.py dinov3-check \
  --model-dir model_checkpoints/dinov3_vits16

python src/ELMiRA/scripts/active_vision_cli.py recognition-calibrate \
  --manifest logs/active_vision/real_object_recognition_20261021/manifest.json \
  --encoder dinov3 --model-dir model_checkpoints/dinov3_vits16 \
  --output logs/active_vision/real_object_recognition_20261021/offline_diagnostics/dinov3_calibration_01
```

Choose a **new** output directory for every calibration. This reuses the saved
30 enrollment + 20 calibration frames, requires no camera/ROS connection and
does not read test images. It never changes the original images or masks.

### Input And Evidence

- Use the same reviewed object masks as the other methods. Mask accuracy and
  autonomous segmentation are not improved or proven by adding an encoder.
- The adapter removes masked-out background, preserves the object's RGB colors
  and aspect ratio, and centers its tight bounding box on a black square with
  padding factor 1.25. This object-ROI step is our preprocessing, not a claimed
  official segmentation method.
- The snapshot's official image processor performs subsequent resizing,
  rescaling and normalization. The professor's grayscale/log-polar/FFT frontend
  is bypassed for DINOv3 only.
- Extract the 384-dimensional last-layer CLS token, then L2-normalize. The
  backbone is CPU float32, evaluation mode, frozen, with inference mode enabled.
  Enrollment fits labelled prototypes; calibration selects a rejection threshold.
  It does not train DINOv3 or make the features into a zero-shot named classifier.
- A successful `calibration_report.json` records feature mode, individual model/
  processor/config checksums, encoder implementation checksum, package versions,
  predictions and `held_out_test_accuracy: not evaluated`. This branch is
  offline-only; the existing ROS adapter does not activate its memory.

Software tests use explicitly synthetic tensors/mocked HF loading to check RGB
ordering, masking, token selection, normalization, local-only safetensors loading,
failure handling and provenance. They are **not pretrained DINOv3 accuracy tests**.
Real DINOv3 feature extraction and the 50-image comparison remain unverified
until the authorized snapshot and a compatible isolated runtime are available.

The 2026-10-21 integration checks passed all 126 offline tests, including 16
DINOv3 plumbing tests. All 50 real development images passed RGB object-input
preparation; a ten-placement contact sheet is saved for inspection. The previous
408 experiment files were preserved. Professor-RGB and color/shape regression
predictions, thresholds and memory hashes match their original reports exactly.
These regression scores remain 10/20 and 20/20 **calibration frames**, respectively.

New checks, input review, regression results and the exact blockers are saved in
`api/logs/active_vision/real_object_recognition_20261021/offline_diagnostics/dinov3_integration_20261021/`.
Start with `integration_report.json`, `runtime_preflight.json` and
`rgb_input_review.png`. No DINOv3 accuracy result or trained DINOv3 memory has
been generated. The original `frozen_comparison.json` remains unchanged and
contains only the two previously tested methods.

### Next Experiment

1. Obtain the authorized snapshot and pass real local inference on the saved
   development images. Preserve failures as well as successes.
2. Freeze its preprocessing, weights, prototypes, rejection threshold and source
   hashes before collecting the independent test placements. Add a new freeze
   record; do not rewrite the original two-method `frozen_comparison.json`.
3. Evaluate all frozen methods on identical new masks of can, bottle and apple,
   with a different never-enrolled transport model as unknown. Corn stays in
   calibration. Report per-class/unknown rejection and placement-level counts.
4. Only consider combining dense DINO features with trustworthy measured 3D
   geometry after the RGB comparison. Existing point clouds contain RGB samples,
   but today's recognition pipeline does not use them. Colored experimental
   reconstruction is not verified metric shape, and DINOv3 does not fix camera
   calibration, insufficient baseline or inaccurate poses automatically.

No head/arm/hand motion, ROS publisher, motor service, NJF training or live
recognition deployment is part of this offline integration.
