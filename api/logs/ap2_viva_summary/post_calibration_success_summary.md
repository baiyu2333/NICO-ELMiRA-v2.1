# Post-Calibration Success Summary

This file separates historical tuning failures from post-calibration evidence. It is generated from existing logs only; no robot motion was commanded.

## Historical All-Log Rate

Across all valid labelled grasp-attempt rows currently available, the historical all-log result is `2/12 = 16.67%` success, with `0` partial and `10` failed trials.

This must not be presented as the final system success rate because it mixes early tuning failures with later calibrated trials.

## Why Early Failures Matter

The early failures are still useful AP2 evidence. They show the debugging process for Z height, table collision, hand weakness, wrist/hand closure behaviour, object distance, and repeatability. These trials justify why the system added staged grasp debugging, first-person trial recording, captured-template IK seeding, and XL-320 hand tuning.

## Post-Calibration Interpretation

The logs show improved post-calibration evidence after the hand-closing fix and later demo-phase tuning. However, the final performance still needs clean repeated post-calibration trials under a controlled protocol before claiming a final success rate.

Recommended statement: the current evidence includes representative successful post-calibration dice/cube grasp/lift examples, but final success rate is `to be collected` using repeated trials after the latest calibration.

## Representative Successful Evidence

- Trial: `T20260612_101148`
- Timestamp: `2026-06-12T10:11:48Z`
- Target object: `dice`
- Contact status: `grasped`
- Outcome: `success`
- Key frame: `306`
- Notes: nico successfully grabed the object | Change: Grasp tuning used right_grasp_x_bias=-0.055 m, right_grasp_y_bias=+0.027 m, grasp_contact_z_offset=+0.005 m, grasp_hover_z_offset=+0.060 m, captured_seed=True (natural_red_grasp_current), seed_guard=True (+/-0.45 rad), contact_seed_guard_shoulder_y=+/-0.85 rad, contact_seed_guard_elbow_y=+/-0.75 rad, final_contact_nudge_shoulder_y=-0.300 rad, final_contact_nudge_elbow_y=+0.000 rad, right_hand_close_deg=90.0, right_hand_close_deg_by_id={'34': 150.0, '35': 150.0, '36': 150.0, '37': 70.0}, right_xl320_position_max_raw=1500, right_xl320_command_repeats=4, right_xl320_command_interval_sec=0.04, captured_stage=False.

## Next Required Trial Set

Run 3 or 5 repeated trials using the final calibrated settings only, with the same object position, command phrase, object, and manual labels. Report this as the final post-calibration success rate.
