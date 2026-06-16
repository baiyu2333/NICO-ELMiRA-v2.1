# AP2 Viva Experiment Summary

## Objective
The AP2 experiment evaluates whether ELMiRA can move from a chat/VLM instruction to a controlled right-arm NICO manipulation attempt, while producing measurable evidence for perception, coordinate transfer, planning, wrist/hand control, and trial outcome.

## Experiment Design
The controlled setup uses the right arm only, a table-top target object, first-person camera recording, manual key-frame annotation, staged grasp debug controls, and plan-only IK/NJF diagnostics. The main task is a grasp attempt on a table-top object; point trials are retained as supporting interaction evidence but are not counted in grasp success rate.

## Current Results From Logs
- First-person trial rows in current CSV: 27.
- Grasp-attempt rows: 24.
- Valid labelled grasp trials: 12.
- Success / partial / failed: 2 / 0 / 10.
- Success rate over valid labelled grasp trials: 16.67%.
- Unlabelled grasp rows excluded from success rate: 12.

## Success Rate By Object
|target_object|valid_trials|success|partial|failed|success_rate_percent|main_failure_reason|
|---|---|---|---|---|---|---|
|dice|12|2|0|10|16.67|unknown|

## Evidence for Error Handling
The logs include no-motion NJF dry-run evidence, passive robot/ROS checks, staged grasp debug logs, IK trace logs, and first-person frame labels. Unsafe commands should be rejected or converted into safe no-motion responses; no direct throw command should execute.

## IK Seed Evidence
The latest IK trace uses captured seed template `natural_red_grasp_current`. The seed guard is documented as `q_final = clip(q_IK, q_seed - delta, q_seed + delta)`. Latest trace reports dropped joints `['r_wrist_x', 'r_wrist_z']` and XL-320 wrist IDs 31/33 planned through the wrist hand stage.

## Limitations
- Object weight/capacity data is not available yet.
- Several grasp rows are still `unknown` and require manual frame review.
- XL-320 wrist/finger values are last-commanded logs, not sensor feedback.
- Current IK target remains wrist/TCP-based, not fully fingertip/palm-contact based.
- Full NJF training/deployment is not claimed; NJF evidence is currently dry-run/service-boundary level.

## Next Work
Complete object variation and capacity testing, add unsupported-command negative tests, review unknown trials, collect more labelled trials by object, and improve contact-target planning using captured templates plus bounded IK/NJF-style correction.
