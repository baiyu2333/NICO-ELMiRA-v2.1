# Error Handling Summary

## ROS, Camera, and IK Availability
- ROS check log status: `success`.
- Robot evidence log status: `success`.
- IK trace status: `success`.
- IK available in latest trace: `True`.
- Camera evidence topics in robot-evidence log: `['/nico/vision/left']`.

## No-Motion Tools
- NJF dry-run status: `success`.
- NJF physical motion executed: `False`.
- IK trace real robot motion commanded: `False`.
- First-person recorder command motion flag is recorded per trial as `real_robot_motion_commanded_by_recorder=false`.

## Safety-Gated Execution
Grasp debug logs separate execution stages such as wrist alignment, open hand, close hand, pre-grasp, and approach. Safety-gated commands are logged in `grasp_debug_trials.csv`. The latest grasp debug stage is `close-hand` with status `executed`.

## Isolated Hand Tests
Open Hand Only / Close Hand Only tests are represented through the grasp-debug stage logs and XL-320 last-command logs. Latest XL-320 tracked motor IDs: `[31, 33, 34, 35, 36, 37]`.

## First-Person Recorder Labels
Manual labels are stored in the first-person trial CSV and trial JSON records: key frame, offset direction, offset size, contact status, outcome, failure reason, and notes.

## Video Backup Strategy
First-person frame folders and contact sheets provide video/frame evidence even when automatic perception or IK debugging is inconclusive. Unknown outcomes remain labelled `unknown` until manually reviewed.
