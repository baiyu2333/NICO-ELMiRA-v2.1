# NICO Single-RGB Vision: Collaborator Overview

Status snapshot: **2026-10-27**. This is a research progress summary, not a claim
of completed autonomous perception or reliable grasping.

## Research Question

Can limited, measured multiview geometry help recognize real tabletop objects
more reliably than image features alone on NICO? Geometry is an auxiliary cue;
recovering a complete high-quality mesh is not the main objective.

The integration adapts the professor-supplied
`Active_Conformal_Foveated_Vision_Single_RGB_v8` approach to NICO's ROS 1 system.
It is distinct from ELMiRA's VLM/API-based task-understanding and arm-control
pipeline. A recognition result does not automatically command a grasp.

## Hardware and Observations

- Ubuntu 20.04 / ROS 1 Noetic; offline replay does not require ROS or a robot.
- Main current source: one NICO See3CAM CU135 RGB camera, serial 22275000,
  native 1280x720 images. Different cameras must not share intrinsics blindly.
- Logitech BRIO was also investigated separately; no verified depth stream is
  used by this pipeline. The experiments discussed here are single-RGB.
- Original images, reviewed foreground masks, camera settings, and candidate
  per-frame poses are recorded. Camera positions/rotations are estimated from
  a stationary checkerboard, not invented from commanded head angles.
- The board has 9x6 inner corners with a user-measured 25 mm square size.
  Calibration image fit and independent metric accuracy are different checks.
- Supervised head sweeps provide limited viewpoint displacement. A 20-degree
  head rotation alone does not establish an adequate triangulation baseline.

## What Enters the Recognition Model?

1. A manually assisted foreground mask isolates the object in the camera image.
   GrabCut/scribble proposals are visually reviewed; ground-truth mask quality
   has not been established for all samples.
2. The supplied v8 visual frontend transforms the masked image into grayscale
   log-polar and spectral inputs. Thus, the name "RGB branch" refers to the
   image source; it is not an unmodified full-colour semantic image encoder.
3. Matched foreground observations from different measured views produce sparse
   XYZ points after geometry checks. Each fusion prefix uses only views already
   observed, not future views from the sequence.
4. The supplied neural point branch receives XYZ, not RGB point colours. Coloured
   PLY visualizations do not imply that colour texture is used in this branch.
5. Image and geometry embeddings support prototype matching and rejection of
   unknown objects. No new full neural training is claimed by this adaptation.

| Comparison | Recorded representation |
| --- | --- |
| Single view | First-view visual embedding, 64 dimensions |
| Multiview images | Normalized mean of visual embeddings, 64 dimensions |
| Image + geometry | Normalized 64D neural fusion plus weighted 37D analytic signature, 101 dimensions |

The fusion comparison includes that analytic signature, so it is not an
isolated point-branch-only ablation. At least ten finite, non-degenerate XYZ
points are required by the fusion gate; this is an availability condition,
not a certificate of shape correctness. Missing geometry is reported, not
replaced with invented points. Experimental visual-hull outputs can be shown
separately but clipped hulls are not substituted as neural fusion input.

An optional local DINOv3 encoder is provided as a separate comparison path.
Its interface/software tests do not establish a real-world DINOv3 accuracy
result, and it is not required to reproduce the supplied v8 baseline.

## Current Three-Object Pilot

Known labels: **apple model, beverage can, bottle**. The same known physical
instances are reused at different placements; this is not category-level
generalization to unseen instances.

| Split | Sequences | Purpose |
| --- | ---: | --- |
| Enrollment | 3 | One reference sequence for each known object |
| Calibration/development | 4 | Three known objects and one unknown blue toy; fit rejection thresholds |
| Independent test | 0 | Still to be collected; no final test accuracy available |

One physical placement/view sequence is one evaluation sample. Adjacent images
inside it are not independent trials. Each current sequence contains three
views. Unknown test data must use a different physical object from the blue
toy used for threshold calibration.

Thresholds and prototype files are frozen before independent testing. The
software checks image/capture-group overlap, source hashes, and unchanged
development data; test data must not update prototypes or rejection thresholds.

On the **same four development sequences used for calibration**, each mode
currently has 2/4 correct decisions: the apple is accepted and the blue toy is
rejected. Both image-only modes reject the can and bottle; the image+geometry
mode accepts the can as a bottle and rejects the bottle. These are development
observations, **not 50% held-out accuracy** and not evidence of a geometry gain.

Source of these counts: the local
`api/logs/active_vision/real_object_registration_20261026/current_three_object_comparison_v1/progress.json`
and `calibrated_v1/sequence_calibration_report.json`. Raw session evidence is
not bundled publicly. These results must not be mixed with historical AP2
physical grasp success rates, which measure a different task.

## What Is and Is Not Verified

| Area | Current evidence / limitation |
| --- | --- |
| Software integration | Offline model/geometry replay, ROS 1 adapter, and synthetic/mock regression tests exist |
| Real camera capture | Native images and supervised head-viewpoint sequences were recorded |
| Calibration and pose | Candidate parameters and checkerboard PnP; complete metric accuracy remains unverified |
| Reconstruction | Real sparse coloured point clouds / exploratory hulls exist; incomplete surfaces and outliers remain |
| Recognition | Real-object enrollment and development calibration completed; independent test pending |
| Observe-move-observe | Supervised fixed-profile captures and guarded software workflow, not validated autonomous next-best-view selection |
| Manipulation integration | No automatic execution from this recognition experiment; no new grasp success claim |

Weak results may involve feature separation, masks/background leakage,
transparent or low-texture objects, limited baseline, and candidate calibration.
The available evidence does not isolate one root cause or establish that the
professor's original algorithm is intrinsically faulty.

## Next Controlled Test

Collect four new sequences: one fresh placement of each known object and one
different unknown physical object. Compare all three feature modes on the
same captures with frozen parameters. Report correct known acceptance,
wrong-class acceptance, false rejection, unknown rejection, and unavailable
geometry, including both full-set and common-available comparisons.

This would still be a four-sequence pilot. More independent repetitions and
different physical instances are required before general performance claims.
Targeted independent calibration checks can accompany the work; repeatedly
stacking an unstable board platform is not required or appropriate.

## Code Reading Order

1. [Recognition workflow and commands](real_object_recognition_workflow.md).
2. [Offline RGB/geometry replay](../api/src/ELMiRA/scripts/active_vision/fusion_replay.py).
3. [Sequence registration, frozen calibration, and evaluation](../api/src/ELMiRA/scripts/active_vision/sequence_recognition.py).
4. [Calibration](../api/src/ELMiRA/scripts/active_vision/calibration.py),
   [masks](../api/src/ELMiRA/scripts/active_vision/masks.py), and
   [sparse geometry](../api/src/ELMiRA/scripts/active_vision/sparse_preview.py).
5. [Read-only ROS 1 adapter](../api/src/ELMiRA/scripts/active_vision/ros1.py) and
   [no-hardware tests](../api/src/ELMiRA/tests/).
6. [Chronological integration evidence](active_single_rgb_nico.md).

Historical diagnostic scripts contain session-specific anchors and paths.
They are evidence/research utilities, not portable commands to move another
robot. Do not bypass their guards or run them on connected hardware as a test.

## Publication Validation

On 2026-10-27, local unittest discovery ran **147 tests, all passing**, with the
authorized professor dependency installed. The suite uses synthetic data and
mocked hardware; it did not start ROS or command the robot. This verifies
software contracts, not physical recognition accuracy, calibration accuracy,
or a complete fresh-clone installation without the external dependency.

## Reproduction and Sharing Boundary

This publication includes our adaptation source, tests, printable calibration
board, and explanatory documentation. The original professor-supplied package
and weights require authorized local installation; see the
[dependency instructions](../api/src/ELMiRA/scripts/active_vision/vendor/README.md).
Several modules and the unified CLI import that package at startup. Without it,
a fresh GitHub clone is useful for code review but is not a complete runnable
neural vision installation.

Raw recordings and illustrated reports can contain identifiable people, and
are retained locally rather than uploaded automatically. Original ZIPs, model
weights, runtime logs, and credentials are also excluded. Reference paths in
the historical documentation identify local evidence, not public downloads.

Questions for a vision collaborator: how best to validate the camera model in
the working image region; whether mask quality or representation is the main
bottleneck; which semantic RGB baseline is appropriate; and how to design a
fair geometry ablation under NICO's limited viewpoint displacement.
