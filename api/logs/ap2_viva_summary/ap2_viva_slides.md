# AP2 Viva Slide Outline

## Slide 1 - AP2 Objective

- Goal: evaluate ELMiRA from VLM-grounded voice command to controlled right-arm NICO manipulation.
- Evidence: first-person camera trials, staged grasp debug logs, IK trace logs, captured template logs, and no-motion NJF diagnostics.
- Scope: representative grasp/lift achieved; stable repeated grasping still under evaluation.

## Slide 2 - Experiment Design

- Robot: NICO right arm / right hand only.
- Setup: table-top object, first-person camera recording, manual key-frame annotation.
- Pipeline: perception -> coordinate transfer -> planned pose -> IK/captured seed -> wrist/hand control -> trial label.
- Metrics: success, partial, failed, offset direction/size, contact status, failure reason.

## Slide 3 - Historical Tuning Evidence

- Historical all-log labelled grasp rate: `2/12 = 16.67%`.
- This includes early Z, wrist, distance, table-collision, and hand-closing tuning failures.
- Use this as debugging evidence, not final performance.

## Slide 4 - Post-Calibration Mini Evaluation

|object|trials|success|partial|failed|success rate|
|---|---:|---:|---:|---:|---:|
|dice|16|6|3|7|37.5%|

- Main failure reason: residual object-centering / offset before hand closing.
- Interpretation: representative successful grasp/lift achieved, but this is not a final large-sample rate.

## Slide 5 - Object Choice

- Dice/cube selected as baseline because it is rigid and cube-shaped.
- Red fish / red object was tested exploratorily.
- It was not suitable for the current grasp setup because it is soft, low-profile, and irregular.
- Full object variation success rates are to be collected.

## Slide 6 - Capacity and Safety Tests

- Capacity test: not formally tested; to be collected.
- Planned capacity method: measure weight, run 3 or 5 trials per weight, success means object enclosed and lifted/held for 2-3 seconds.
- Unsupported command test: `throw the object` not physically tested.
- Expected unsupported-command behaviour: safe rejection / no-motion response, `real_robot_motion_commanded=false`.

## Slide 7 - IK Seed and Refinement Evidence

- IK seed guard formula: `q_final = clip(q_IK, q_seed - delta, q_seed + delta)`.
- Captured natural-grasp template provides plan/debug evidence.
- Remaining issue: grasp can still fail from residual object-centering / offset before hand closing.
- NJF remains a future bounded correction direction; no full NJF deployment is claimed.

## Slide 8 - AP2 Conclusion

- AP2 delivered measurable manipulation evidence and debugging workflow.
- Post-calibration dice mini evaluation: `6/16 = 37.5%`.
- Representative grasp/lift achieved.
- Stable repeated grasping, object variation, capacity, and unsupported-command tests remain to be collected.
- Formal collaborator letter has been provided.
