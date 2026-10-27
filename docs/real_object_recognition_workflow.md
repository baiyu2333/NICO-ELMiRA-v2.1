# Real-Object Registration and Paired Recognition Testing

The objective is recognition, with coarse 3D geometry as an auxiliary feature.
This workflow never opens a camera or motor interface and never starts ELMiRA.
It consumes features from the existing `fusion-replay` command.

Start with the [collaborator overview](vision_research_overview.md) for current
scope and results. The adapted professor branches require an authorized local
[external dependency](../api/src/ELMiRA/scripts/active_vision/vendor/README.md).
The `api/logs/` paths below identify local evidence, not publicly bundled data.

## Historical Five-Class Enrollment on 26 October 2026

- Archived NICO dice and wooden-block sequences have been registered under the
  real identity labels `dice` and `wood_blocks`, not the synthetic label `cube`.
- The three galleries are separate: `rgb_single` (64D), `rgb_multiview` (64D),
  and `rgb_geometry` (101D). Neural weights have not been trained or changed.
- These archived registration sequences are enrollment only. Their registration
  is not an independent recognition test, and this five-class archive retains
  uncalibrated default thresholds. The separate three-class comparison below
  has since completed development calibration with frozen thresholds.
- A new, stationary beverage-can sequence has also been registered. Its three
  real viewpoints have encoder offsets of about 0, 9.5 and 18.8 degrees. Camera
  poses come from the checkerboard in each image, not from commanded angles.
- The can sequence produced 59 and 118 candidate sparse points at its two
  fusion steps. These are partial, unverified geometry, not a complete model.
  A transparent bottle sequence has now also been recorded and registered as
  an explicitly high-risk experimental example. Its two fusion steps produced
  53 and 142 candidate points. The full cloud has a large extent and may contain
  transmitted-background matches; it is not reliable bottle-surface geometry.
  No points were removed to produce a more favourable result.
- The apple-model sequence is also registered. It produced 10 and 30 candidate
  sparse points. Few features and possible outliers remain; it is not a complete
  or validated apple shape. All three new objects have enrollment data, not
  independent recognition results.
- Existing reports and historical recognition results have not been replaced.

Existing registration report:

```text
api/logs/active_vision/real_object_registration_20261026/registered_v5/registration_report.json
```

### Active Three-Object Comparison

The prospective comparison uses only `beverage_can`, `bottle`, and
`apple_model`, with the same class subset in all three modes. The five-class
archive above, including the old dice and wooden blocks, is retained separately
and is not the denominator for this new experiment.

```text
api/logs/active_vision/real_object_registration_20261026/current_three_object_comparison_v1/dataset.json
api/logs/active_vision/real_object_registration_20261026/current_three_object_comparison_v1/registered_v1/registration_report.json
api/logs/active_vision/real_object_registration_20261026/current_three_object_comparison_v1/progress.json
api/logs/active_vision/real_object_registration_20261026/current_three_object_comparison_v1/sample_inventory.csv
api/logs/active_vision/real_object_registration_20261026/current_three_object_comparison_v1/calibrated_v1/sequence_calibration_report.json
api/logs/active_vision/real_object_registration_20261026/current_three_object_comparison_v1/calibrated_v1/calibration_decision_diagnostics.csv
api/logs/active_vision/real_object_sequences_20261026/blue_toy_unknown_calibration_01/sequence_result.json
api/logs/active_vision/real_object_sequences_20261026/apple_model_calibration_01/sequence_result.json
api/logs/active_vision/real_object_sequences_20261026/beverage_can_calibration_01/sequence_result.json
api/logs/active_vision/real_object_sequences_20261026/bottle_calibration_01/sequence_result.json
```

The user-confirmed blue toy is an **unknown calibration** object, not a fourth
known class and not a held-out test result. A new apple placement has now been
collected for calibration too. Its two-view prefix had only 2 points and could
not be fused; the three-view prefix reached 11 candidate points and passed the
unchanged minimum-point gate. The unavailable prefix is retained. Nearby hand
and clothing pixels required assistant-refined background scribbles; no mask
ground truth, validated geometry or recognition accuracy is claimed.

The new can calibration placement has also been collected. Its two fusion
prefixes produced 70 and 183 candidate points. Padded, assistant-proposed
silhouette envelopes exclude adjacent person/table pixels; mask and metric
accuracy remain unverified. The bottle calibration capture is also complete:
59 and 140 candidate points, with the transparent-background limitation retained.

All four calibration sequences are collected. Rejection thresholds were fitted
using only enrollment/development data and frozen in `calibrated_v1`: single RGB
0.98, multi-view RGB 0.92, RGB/geometry 0.52; margin remains 0.045 for all modes.
Known-class prototype arrays and network weights are unchanged. Each mode gets
2/4 on these same calibration examples: apple accepted and blue toy rejected;
RGB controls reject can and bottle, while fusion mislabels can as bottle and
rejects bottle. These are development results, not held-out recognition rates
or evidence that geometry improves recognition. Eleven software-contract tests
passed; they are not physical recognition trials.

No independent test sequences are collected yet. Next collect a separate test
placement per known class plus a different physical unknown object, with these
parameters kept fixed. A one-placement-per-class test
is only a small pilot, not a stable large-sample accuracy estimate. Neural
weights and known-class prototype files remain unchanged by calibration data
collection.

The user authorized automatic continuation of the established, supervised
same-direction head-yaw capture profile without per-run approval prompts:
unchanged low speed/torque cap, at most about 20 degrees, fixed pitch, reverse
return, and feedback/view checks before movement. This does not authorize arm
or hand motion, larger ranges, higher limits, or continuation after anomalies.
The bottle calibration used that standing instruction. Its feedback confirmed
return; no separate visual-motion confirmation was requested or invented.

Current can evidence:

```text
api/logs/active_vision/real_object_sequences_20261026/beverage_can_enroll_01/manifest_sweep_v2.json
api/logs/active_vision/real_object_sequences_20261026/beverage_can_enroll_01/fusion_replay_v2/fusion_replay_report.json
api/logs/active_vision/real_object_sequences_20261026/beverage_can_enroll_01/evidence/yaw_capture_review.json
api/logs/active_vision/real_object_sequences_20261026/beverage_can_enroll_01/evidence/sweep_mask_review.png
```

The raw controller reported a post-release goal-readback mismatch. The separate
read-only check confirmed return within about 0.35 degrees of the command
origin, yaw torque off, and unchanged pitch hold. The operator confirmed smooth
return without contact or unusual noise. The original stopped controller log
is retained, not rewritten as an uninterrupted successful run. Use the current
manifest above: the older frozen input retains stale single-view metadata.
Source images and previous registration outputs are preserved.

Current bottle evidence:

```text
api/logs/active_vision/real_object_sequences_20261026/bottle_enroll_01/manifest_sweep.json
api/logs/active_vision/real_object_sequences_20261026/bottle_enroll_01/fusion_replay_v1/fusion_replay_report.json
api/logs/active_vision/real_object_sequences_20261026/bottle_enroll_01/evidence/transparent_geometry_diagnostic.json
api/logs/active_vision/real_object_sequences_20261026/bottle_enroll_01/sequence_result.json
```

The bottle sweep completed and returned normally. Its session-specific state
check accepts a controller goal reset to actual position only after verified
return and yaw torque-off; active-motion goal, speed and torque-cap checks were
not relaxed. The operator confirmed smooth motion without contact or unusual
noise. Pitch hold was preserved. Transparent geometry remains a quality warning
for the later paired comparison, not an established recognition improvement.

Current apple evidence:

```text
api/logs/active_vision/real_object_sequences_20261026/apple_model_enroll_01/manifest_sweep.json
api/logs/active_vision/real_object_sequences_20261026/apple_model_enroll_01/fusion_replay_v1/fusion_replay_report.json
api/logs/active_vision/real_object_sequences_20261026/apple_model_enroll_01/evidence/yaw_capture_review.json
api/logs/active_vision/real_object_sequences_20261026/apple_model_enroll_01/sequence_result.json
api/logs/active_vision/real_object_registration_20261026/today_object_enrollment_summary.csv
```

The apple raw controller log retains its stopped status: after verified return
and yaw torque-off, sequential goal and position readbacks differed by one
encoder count (about 0.09 degrees). A subsequent independent read-only snapshot
confirmed stable return, yaw torque off, unchanged pitch hold and no motor
error flags; the operator confirmed smooth return without contact or unusual
noise. No movement retry or stronger settings were used. The separate review
records this distinction rather than rewriting the original controller result.

The next stage is independent recognition evaluation, not more shape-polishing.
Use fresh placements, separate rejection-threshold calibration from held-out
testing, and keep any selected class subset identical across all three modes.
Registration photographs must not count as test successes.

## Recording Requirements

Use the same camera and preprocessing for all conditions. A posed sequence needs
the original images, target masks, same-camera intrinsics, and recorded per-frame
camera poses in a common frame. Keep the object fixed within each sequence.
Repeated still frames or head commands without measured camera displacement are
not additional geometric views. Image-only photographs remain usable for RGB
work, but cannot be given invented poses or depth to make a fusion sequence.

The current camera can be reused. This workflow does not require another camera,
a depth sensor, a dense mesh, or another calibration-board stacking exercise.
Candidate geometry stays explicitly experimental; it is not a motion target.
Any later physical viewpoint change needs separate connection/feedback checks
and safety confirmation. These commands do not perform that movement.

## 1. Extract One Recorded Sequence

From `api`, use the previously implemented command:

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py fusion-replay \
  --manifest logs/your_capture/manifest.json \
  --checkpoint model_checkpoints/active_vision_v8/single_rgb_model.pt \
  --experimental-unverified \
  --output logs/your_capture/fusion_replay_v1
```

The output includes `fusion_replay_report.json` and `recognition_features.npz`.
The optional synthetic primitive memory is not needed for extraction or real
object registration. Prefix geometry never uses later photographs.

## 2. Add the Human Label

No hand-written checksums are necessary:

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py sequence-add \
  --manifest logs/real_objects/dataset.json \
  --replay-report logs/your_capture/fusion_replay_v1/fusion_replay_report.json \
  --label beverage_can \
  --split enroll \
  --capture-group can_enroll_01 \
  --object-instance can_physical_01
```

Use `bottle` and `apple_model` for the other prepared objects. The label is a
human annotation, not an automatic inference from the old primitive prediction.
`capture-group` identifies one physical placement/capture sequence. Do not change
the group name merely to reuse the same photographs in another split.

## 3. Register All Three Galleries

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py sequence-enroll \
  --manifest logs/real_objects/dataset.json \
  --output logs/real_objects/registered_v1
```

At least two known real-object labels are needed. Registration requires features
for all three conditions; an enrollment without usable geometry must be completed
or reported as pending. Existing outputs are never overwritten.

For the existing old-data demonstration, substitute:

```text
--manifest logs/active_vision/real_object_registration_20261026/enrollment_manifest.json
--output logs/active_vision/real_object_registration_20261026/registered_NEW_RUN
```

Do not report the enrolled sequences as test successes.

## 4. Calibrate Rejection Using Independent Data

Add separate sequences with `--split calibration`, including every known class
and at least one genuinely unknown physical object. Then run:

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py sequence-calibrate \
  --manifest logs/real_objects/dataset.json \
  --output logs/real_objects/calibrated_v1
```

This reuses the existing development-only threshold search independently for
each gallery, with a fixed margin of 0.045. It does not open test replay/feature
files. `sequence_calibration_report.json` freezes the memories and thresholds.
Without calibration data, the command refuses to invent a calibrated result.

## 5. Evaluate Fresh Test Placements

Add independent `--split test` sequences for all known classes and an unknown
object different from the calibration unknown. Keep the development rows intact.

```bash
python3 src/ELMiRA/scripts/active_vision_cli.py sequence-evaluate \
  --manifest logs/real_objects/dataset.json \
  --calibration-dir logs/real_objects/calibrated_v1 \
  --output logs/real_objects/test_v1
```

Outputs are `sequence_comparison.json` and `sequence_predictions.csv`.

- One sequence, not one adjacent frame, is the unit of analysis.
- Report known-object correctness, wrong-class acceptance, false rejection,
  unknown false acceptance and geometry availability.
- Missing geometry stays `unavailable` and counts in the full-set denominator.
  Also report the common available subset, never just the favourable subset.
- Compare geometry against multi-view RGB as well as single-view RGB, so extra
  images alone are not mistaken for a geometric improvement.
- Cached classification timing is not full reconstruction/inference latency.
- No memory learning, threshold refit or file update occurs during evaluation.
- Small same-instance tests are not evidence of broad semantic generalization.

## Software Checks

```bash
python3 -m py_compile src/ELMiRA/scripts/active_vision/sequence_recognition.py
python3 -m py_compile src/ELMiRA/scripts/active_vision/cli.py
python3 -m unittest discover -s src/ELMiRA/tests -p 'test_sequence_recognition.py' -v
```

These tests use synthetic feature fixtures. They validate software contracts,
split protection and error handling, not physical recognition accuracy.
