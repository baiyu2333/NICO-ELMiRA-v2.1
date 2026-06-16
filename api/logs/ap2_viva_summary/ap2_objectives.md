# AP2 Objectives and Evidence Map

This summary is generated from existing ELMiRA/NICO logs only. No robot motion was commanded during generation.

## Objective O1: Demonstrate a controlled right-arm manipulation pipeline
Evidence: first-person trial logs, grasp debug logs, captured natural-grasp template, and IK trace logs.

## Objective O2: Measure grasp/touch behaviour using explicit trial labels
Evidence: `grasp_success_summary.csv` contains trial ID, target object, target coordinates, key frame, offset label, contact status, outcome, and failure reason.

## Objective O3: Separate perception, coordinate transfer, planning, wrist/hand control, and execution debugging
Evidence: `grasp_debug_trials.csv`, `latest_grasp_plan.json`, and `latest_ik_trace.json` record staged debug outputs: detect, coordinate transfer, plan-only, wrist alignment, hand open/close, pre-grasp, and approach.

## Objective O4: Provide safety and error-handling evidence
Evidence: no-motion NJF logs, passive robot evidence logs, safety-gated grasp debug execution, and first-person labels showing unknown/failed cases instead of forced success.

## Objective O5: Explain the IK seed guard and captured-template method
Evidence: latest IK trace reports captured template seed usage: `natural_red_grasp_current`, seed joint names, seed joint positions, returned IK joints, dropped joints, and final command notes.

## Source Logs
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/first_person_trials/first_person_trials.csv`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/grasp_debug/grasp_debug_trials.csv`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/grasp_debug/latest_grasp_plan.json`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/ik_trace/latest_ik_trace.json`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/grasp_templates/latest_captured_template.json`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/grasp_templates/captured_grasp_templates.csv`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/grasp_templates/latest_xl320_command.json`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/fyp2_monitoring/latest_njf_probe.json`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/fyp2_monitoring/latest_ros_check.json`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/fyp2_monitoring/latest_robot_evidence.json`: available
- `/home/baiyuzhe/catkin_ws/src/NICO-software/api/logs/fyp2_monitoring/latest_njf_service_check.json`: available
