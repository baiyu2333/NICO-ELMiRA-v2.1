# IK Seed Guard Formula Evidence

## Formula
`q_final = clip(q_IK, q_seed - delta, q_seed + delta)`

This means IK can still solve toward the requested target pose, but each guarded joint is bounded around the captured natural-grasp seed. The purpose is to stop IK from drifting back to an unnatural arm shape when a better captured posture is available.

## Captured Seed Template
- Template name: `natural_red_grasp_current`
- Captured template used as seed in latest IK trace: `True`
- Max delta: `0.45 rad from /elmira/captured_grasp_seed_max_delta_rad if current param is active`
- IK status: `success` / service status `success`

## Seed Joint Values
- `r_shoulder_z` = 0.2907718533822553
- `r_shoulder_y` = 0.10663961729685355
- `r_arm_x` = 0.43493604959698695
- `r_elbow_y` = -1.7484708446479194
- `r_wrist_z` = -1.39
- `r_wrist_x` = 0.0

## Planned Target Poses Sent/Prepared Before IK
- pre_grasp: position={'x': 0.1217746102809906, 'y': -0.09386234730482101, 'z': 0.79}, orientation=None, target_basis=right_tcp / wrist-based IK target, not fingertip contact target
- approach: position={'x': 0.1417746102809906, 'y': -0.09386234730482101, 'z': 0.76}, orientation=None, target_basis=right_tcp / wrist-based IK target, not fingertip contact target
- touch: position={'x': 0.1567746102809906, 'y': -0.09386234730482101, 'z': 0.625}, orientation=None, target_basis=right_tcp / wrist-based IK target, not fingertip contact target
- lift_retreat: position={'x': 0.1567746102809906, 'y': -0.09386234730482101, 'z': 0.82}, orientation=None, target_basis=right_tcp / wrist-based IK target, not fingertip contact target

## Example Guard Interpretation
- For `r_shoulder_z`, guard interval is approximately `0.2907718533822553 - delta` to `0.2907718533822553 + delta`; with delta `0.45 rad from /elmira/captured_grasp_seed_max_delta_rad if current param is active`.
- For `r_shoulder_y`, guard interval is approximately `0.10663961729685355 - delta` to `0.10663961729685355 + delta`; with delta `0.45 rad from /elmira/captured_grasp_seed_max_delta_rad if current param is active`.
- For `r_arm_x`, guard interval is approximately `0.43493604959698695 - delta` to `0.43493604959698695 + delta`; with delta `0.45 rad from /elmira/captured_grasp_seed_max_delta_rad if current param is active`.

## IK and Filtering Evidence
- IK returned `r_wrist_z`: `True`
- IK returned `r_wrist_x`: `True`
- Dropped joints after filtering: `['r_wrist_x', 'r_wrist_z']`
- XL-320 wrist IDs 31/33 will be commanded by hand stage: `True`

## Important Limitation
The trace indicates that the arm target is still wrist/TCP-based rather than a fully fingertip/palm-contact target. The XL-320 wrist/finger values are last-commanded values, not sensor feedback.
