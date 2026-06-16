# Viva Panel Answer Summary

## What are your AP2 objectives?

My AP2 objectives are to demonstrate a controlled NICO right-arm manipulation pipeline, record measurable trial evidence, separate perception/coordinate/IK/hand-control debugging stages, and prepare a safe refinement path using captured templates and NJF-style bounded corrections.

## How is the experiment designed?

The controlled experiment uses a fixed table setup, a target object such as dice/cube, the command to grasp or touch the object, first-person video recording, key-frame selection, manual outcome labels, and staged grasp-debug logs. Success, partial, and failure are labelled explicitly rather than assumed.

## What is the current result?

The full historical log contains early failures from Z, wrist, hand-closing, table-collision, and offset tuning. Later logs show representative successful dice/cube grasp/lift evidence after calibration, but repeated clean post-calibration trials are still required.

## What is the success rate?

The historical all-log rate is `2/12 = 16.67%`, but this is not the final system success rate because it includes calibration failures. The final post-calibration success rate is `to be collected` from repeated trials using only the latest calibrated settings.

## How do different objects affect performance?

The cube/dice object is the current baseline. Soft low-profile objects such as the red fish and cylindrical objects still need controlled testing because shape, height, friction, and contact geometry affect grasp reliability.

## What is NICO's maximum lifting capacity?

Maximum lifting capacity is `to be collected`. The correct method is to measure object weight and run 3 or 5 trials per weight, defining reliable capacity as the highest weight with at least 2/3 or 3/5 successful lifts held for 2-3 seconds.

## How do you handle unsupported commands?

Unsafe or unsupported commands such as `throw the object` should be rejected or converted to a no-motion response. The system should log the unsupported command with `real_robot_motion_commanded=false`.

## What is the IK seed formula?

The guard formula is `q_final = clip(q_IK, q_seed - delta, q_seed + delta)`. The captured seed template is `natural_red_grasp_current`. Seed joint evidence: r_shoulder_z=0.2907718533822553, r_shoulder_y=0.10663961729685355, r_arm_x=0.43493604959698695, r_elbow_y=-1.7484708446479194, r_wrist_z=-1.39, r_wrist_x=0.0. Max delta evidence: +/-0.45 rad in logged tuning notes.

## How do you handle errors and testing?

ROS/camera/IK unavailable cases are logged as unavailable/partial rather than crashes. Execution tools are safety-gated. The first-person recorder stores video frames and manual labels. Isolated Open Hand Only and Close Hand Only tests are used to debug XL-320 hand behaviour. Video evidence is kept as backup for viva explanation.
