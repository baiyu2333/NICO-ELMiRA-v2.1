# AP2 Viva Experiment Summary

## Objective

The AP2 experiment evaluates whether ELMiRA can move from a VLM-grounded voice command to a controlled right-arm NICO manipulation attempt, while producing measurable evidence for perception, coordinate transfer, planning, wrist/hand control, safety, and trial outcome.

## Experiment Design

The controlled setup uses the right arm only, a table-top target object, first-person camera recording, manual key-frame annotation, staged grasp debug controls, and plan-only IK/NJF diagnostics. The main task is a grasp attempt on a table-top object. Point/touch trials and debug-only tests are retained as supporting evidence but are not counted as final grasp success unless labelled as grasp trials.

## Current Results

Historical all-log labelled grasp rate: `2/12 = 16.67%`. This includes early Z, wrist, distance, and hand-closing tuning failures, so it is debugging evidence and not the final system performance.

Post-calibration controlled mini evaluation:

|object|trials|success|partial|failed|success_rate|main_failure_reason|
|---|---:|---:|---:|---:|---:|---|
|dice|16|6|3|7|37.5%|residual object-centering / offset before hand closing|

Interpretation: representative successful grasp/lift was achieved, but stable repeated grasping is still under evaluation. The `37.5%` result is a post-calibration mini-evaluation result, not a final large-sample success rate.

## Object Selection and Variation

The dice/cube object is used as the baseline because it is rigid and cube-shaped. The red fish / red object was tested exploratorily but was not suitable for the current grasp setup because it is soft, low-profile, and irregular.

Formal object variation success rates are not claimed yet. Controlled object variation across dice/cube, soft low-profile objects, and cylindrical/other shapes is to be collected.

## Capacity Test

Maximum lifting capacity was not formally tested. Capacity remains to be collected with measured object weights and repeated trials.

Planned method: measure object weight, run 3 or 5 trials per weight, and define success as object enclosed and lifted/held for 2-3 seconds. Maximum reliable capacity should be the highest weight with at least 2/3 or 3/5 successful trials.

## Unsupported Command Handling

The unsafe command `throw the object` was not physically tested. It is treated as an unsupported/unsafe command handling plan. Expected behaviour: safe rejection or no-motion response, with `real_robot_motion_commanded=false` in the log.

## IK Seed and Debug Evidence

The IK seed guard is documented as:

`q_final = clip(q_IK, q_seed - delta, q_seed + delta)`

The latest IK trace and captured template logs show the use of captured natural-grasp posture evidence for plan-only debugging. The system still needs further refinement because the current target can remain wrist/TCP-based rather than fully fingertip/palm-contact based.

## Collaboration Status

A formal collaborator letter has been provided. Collaboration is not marked as missing.

## Limitations

- The historical `2/12 = 16.67%` all-log rate includes early calibration failures and is not final performance.
- The post-calibration mini evaluation is limited to 16 dice trials.
- Stable repeated grasping remains under evaluation.
- Object variation success rates are to be collected.
- Capacity testing is to be collected.
- Unsupported command testing is to be collected as a no-motion safety test.
- XL-320 wrist/finger values are last-commanded logs, not sensor feedback.
- Full NJF training/deployment is not claimed; NJF evidence is currently dry-run/service-boundary level.

## Next Work

Run repeated clean post-calibration trials, complete object variation tests, collect formal capacity data, add unsupported-command no-motion evidence, and continue improving object-centering before hand closing using captured templates and bounded IK/NJF-style correction.
