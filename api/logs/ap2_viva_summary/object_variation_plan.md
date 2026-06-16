# Object Variation Plan

Purpose: test whether ELMiRA/NICO grasp performance depends on object shape, height, material, and contact geometry.

## Controlled Objects

| Object | Expected Shape | Reason for Test | Data Status |
|---|---|---|---|
| dice / cube | cube | Baseline rigid object with graspable edges. Current logs contain dice/cube evidence. | partially collected |
| red fish / soft low-profile object | soft low-profile object | Tests low-height and deformable object difficulty. | to be collected |
| cylinder or other object | cylinder / alternative shape | Tests curved surface and rolling/slipping behaviour. | to be collected |

## Metrics

For each object, collect the same command phrase and target setup, then label:

- `success`: object is enclosed and lifted/held briefly.
- `partial`: hand reaches/touches/encloses object but lift or hold is unstable.
- `failed`: no contact, wrong offset, collision, slip, hardware issue, or unsupported execution.
- `main_failure_reason`: y_offset, z_too_high, z_too_low, x_distance, vision, ROS, IK, calibration, hardware, weak_grip, slip, unknown.

## Trial Protocol

Use 3 or 5 trials per object after final calibration. Keep table height, robot position, camera view, target coordinates, and command phrase fixed where possible. Do not combine early tuning trials with post-calibration object variation results.
