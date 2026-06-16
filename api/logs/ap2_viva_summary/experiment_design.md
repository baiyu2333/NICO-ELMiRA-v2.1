# Controlled Grasp Experiment Design

## Command Phrase
Primary command family: `try to grasp the dice/object on the table` or equivalent dashboard/chat instruction. The current logs record `grasp_attempt` and `point` action types.

## Object Setup
The current logged object set includes: dice, red object. Object weight is not available in the current logs. Object shape is marked as `cube` only where the target name itself indicates dice/cube; otherwise it is `not available`.

## Robot Setup
- Right arm only for manipulation trials.
- Left arm/gripper is not used for execution.
- First-person camera records action evidence from `/nico/vision/left`.
- XL-320 wrist/finger commands are logged as last-commanded values, not sensor feedback.

## Metrics
- `contact_status`: no_contact / near / touch / grasped / lifted / unknown.
- `outcome`: success / partial / failed / unknown.
- `offset_direction`: centered / left / right / too_high / too_low / too_far / too_close / unknown.
- `offset_size`: none / small / medium / large / unknown.
- `failure_reason`: none / y_offset / z_too_high / z_too_low / x_distance / vision / ROS / IK / calibration / hardware / unknown.
- `valid_trial`: true only when `action_type=grasp_attempt` and outcome is explicitly success, partial, or failed.

## Success Definitions
- Success: object is grasped or lifted according to manual first-person annotation.
- Partial: hand reaches/touches/partially encloses the object but does not complete the desired grasp/lift.
- Failed: object is not contacted or the robot misses/collides/slips according to the manual label.
- Unknown: recorded but not yet manually labelled; excluded from success-rate calculation.

## Current Logged Results
|metric|value|
|---|---|
|first-person rows in current CSV|27|
|grasp_attempt rows|24|
|valid labelled grasp trials|12|
|success / partial / failed|2 / 0 / 10|
|success rate over valid labelled grasp trials|16.67%|
|unlabelled grasp rows|12|

See `grasp_success_summary.csv` and `success_rate_by_object.csv` for machine-readable summaries.
