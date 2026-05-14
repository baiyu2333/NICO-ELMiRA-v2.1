# FYP2 Clean Dashboard Demo

## Purpose

The clean dashboard is a stable AP2 monitoring surface for a 3-minute system demo. It keeps the FYP1 operational story visible through launch, stop, chat, ASR controls, and log monitoring, then adds the FYP2 manipulation-evaluation evidence path through an NJF dry-run probe and no-motion ROS/robot checks.

It does not replace the existing ELMiRA pipeline, train Neural Jacobian Fields, or add new robot trajectories.

## What Was Removed From The Main Demo UI

- Install requirements button.
- Setup and terminal-command tabs.
- Raw launch command display.
- Debug/prompt-heavy output.
- Any UI path that exposes API keys.

## Essential Functions Kept

- Launch ELMiRA.
- Stop System.
- Refresh Status.
- Text chat through the existing `/mllm_chat` pipeline.
- Listen and Stop Listen controls, with graceful ASR fallback.
- Tail display for `launch_nodes.log` and `launch_sm.log`.

## Commands

Run from the API workspace:

```bash
cd ~/catkin_ws/src/NICO-software/api
```

Compile checks:

```bash
python3 -m py_compile src/ELMiRA/scripts/dashboard_clean.py
python3 -m py_compile src/ELMiRA/scripts/fyp2_njf_demo_probe.py
python3 -m py_compile src/ELMiRA/scripts/utils/fyp2_trial_logger.py
```

Generate NJF dry-run logs:

```bash
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode dry-run
```

Start clean dashboard:

```bash
python3 src/ELMiRA/scripts/dashboard_clean.py
```

Expected dashboard URL:

```text
http://127.0.0.1:7860
```

If `7860` is busy, the dashboard automatically tries:

```text
http://127.0.0.1:7861
```

ROS check:

```bash
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode ros-check
```

Robot evidence, no motion:

```bash
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode robot-evidence
```

Summary:

```bash
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode summary
```

Optional service check:

```bash
python3 src/ELMiRA/scripts/fyp2_njf_demo_probe.py --mode service-check
```

`NJFPredictAction.srv` was valid during implementation. Run `catkin_make` and `source devel/setup.bash` only if the service has not previously been built locally or if the service file is changed later.

## Log Files

All monitoring logs are written under:

```text
~/catkin_ws/src/NICO-software/api/logs/fyp2_monitoring/
```

Files:

- `latest_njf_probe.json`: dry-run result.
- `latest_njf_service_check.json`: optional NJF service check.
- `latest_ros_check.json`: ROS availability check.
- `latest_robot_evidence.json`: passive robot/ROS evidence check.
- `latest_njf_probe.csv`: append-only CSV for all modes.
- `monitoring_summary.md`: aggregate summary of latest available evidence.

Robot evidence and service checks do not overwrite `latest_njf_probe.json`, so the dry-run result stays visible for the demo.

## 3-Minute Click Sequence And Expected Output

`0:00-0:15` show physical NICO setup.

Expected output: physical robot is visible in the recording. In the dashboard checklist, tick `Physical NICO shown in recording` after opening the dashboard.

`0:15-0:35` open clean dashboard.

Expected output: title `ELMiRA FYP2 Monitoring Console`; status bar shows ROS Core, ELMiRA Pipeline, Robot Mode, NJF Probe, Safety; Demo Mode text separates Real Robot Mode from NJF Probe Mode.

`0:35-0:55` click `Launch ELMiRA`.

Expected output: Launch / Stop Status shows `launch started`, selected provider, dummy mode, nodes log path, state machine log path, and `API key: [MASKED]`. Raw commands and secret values are not displayed. Log tails begin updating after refresh.

`0:55-1:20` optional safe robot action.

If the robot is stable, type one of these into the chat input and click `Send`:

```text
Point to the red block.
```

or:

```text
Show the red block with the right hand.
```

Expected output: transcript shows `USER:` followed by the command and `ROBOT:` followed by the existing ELMiRA response. If the chat/action service is unavailable or times out, the dashboard shows `Chat/action service unavailable; use NJF dry-run and robot evidence fallback.`

Fallback if motion is unreliable: click `Record Robot Evidence, No Motion`.

Expected output: Latest Probe Action shows `robot-evidence: completed` or a clear `unavailable`, `partial`, or `timeout` status. No robot motion is commanded.

`1:20-1:50` click `Run NJF Dry-run Probe`.

Expected output: Latest Probe Action shows `dry-run: completed`. The FYP2 card shows status `success`, implementation level `NJF-inspired feasibility probe / ROS service boundary`, target object `red block`, fixed visual error, bounded correction, metrics, success criteria, limitation, and next action.

`1:50-2:20` show JSON/CSV/Markdown logs.

Expected output: `latest_njf_probe.json` displays the dry-run record, the CSV path points to `api/logs/fyp2_monitoring/latest_njf_probe.csv`, and `monitoring_summary.md` shows sections for FYP2 Monitoring Update, What the NJF Probe Tests, Evaluation Design, AI Training and Testing Status, Current Limitations, and Next Actions for Week 10-12.

`2:20-2:40` click `Run ROS Check` or `Record Robot Evidence, No Motion`.

Expected output: the button returns within a few seconds with `success`, `partial`, `unavailable`, or `timeout`. `monitoring_summary.md` aggregates the latest dry-run plus latest ROS/robot evidence. NJF probe commands do not move the robot.

`2:40-3:00` click `Stop System`.

Expected output: Launch / Stop Status starts with `stopped`, `partial`, or `failed`. It reports only tracked dashboard-launched process groups, so the dashboard itself stays running. End by showing the Week 10-12 next action in the monitoring summary.

## Recording Modes

Mode A, real robot available:

```text
Launch ELMiRA -> optional safe point/show command -> NJF dry-run -> logs -> Stop System
```

Mode B, robot or ROS unstable:

```text
show physical robot -> clean dashboard -> NJF dry-run -> robot evidence no motion -> logs -> stop/summary
```

## Evaluation Metrics And Success Criteria

Metrics:

- probe_status
- log_generation_success
- ros_availability
- njf_service_availability
- physical_robot_evidence
- physical_motion_executed_false_for_probe
- future_controlled_trial_success_rate

Success criteria:

- dry_run_generates_json_csv_markdown
- dashboard_displays_latest_probe_result
- ros_check_does_not_crash
- robot_evidence_check_commands_no_motion
- service_check_reports_available_or_unavailable_clearly
- physical_trials_scheduled_for_week_10_12

## AI Training And Testing Status

VLM/LLM modules are used as pretrained reasoning components. No full NICO-specific Neural Jacobian Field training is claimed at AP2 monitoring stage.

Current testing focuses on deterministic NJF-inspired dry-run, ROS service-boundary validation, structured logs, and preparation for future NICO image-action data collection.

## What Is Not Claimed

- Full trained Neural Jacobian Field deployment on NICO.
- Full physical grasping success unless physically demonstrated.
- NJF-controlled robot motion.
- New arbitrary robot trajectories.
- External NJF repository installation as a blocking requirement.

## Week 10-12 Plan

- Record controlled NICO image-action data.
- Run physical manipulation trials after monitoring feedback.
- Compare MLLM-only, NJF-inspired, and hybrid refinement modes.
- Report success rate, failure modes, ROS/service availability, and whether physical motion remained within safe bounds.
