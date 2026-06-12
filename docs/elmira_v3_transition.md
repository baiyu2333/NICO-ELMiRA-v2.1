# ELMiRA v3 Transition

## Version Boundary

`elmira-v2.1-final` marks the AP2 dashboard baseline: one-click launch/stop, chat-style monitoring, NJF dry-run evidence, and AP2 recording support.

`elmira-v3` is the Week 10-12 manipulation-development line. It keeps the dashboard but shifts the project focus toward right-arm reaching, touching, grasp debugging, first-person trial recording, captured grasp templates, RH7D/XL-320 hand tuning, and future NJF-style refinement.

## What Was Not Renamed

The ROS package is still named `elmira`. This is intentional.

Do not rename these without a full ROS migration:

```bash
roslaunch elmira init_nodes_v2.launch
rosrun elmira state_machine.py
```

Keeping the package name stable avoids breaking catkin builds, launch files, service imports, and existing dashboard controls.

## v3 Scope

ELMiRA v3 includes:

- Clean NICO dashboard as the main control surface.
- Right-hand grasp debug mode.
- Kinematic action preview.
- First-person trial recorder.
- Captured natural grasp templates.
- IK trace dumps.
- RH7D/XL-320 right-hand tuning.
- Structured logs for grasp trials and parameter changes.

ELMiRA v3 does not claim:

- Full physics simulation.
- Guaranteed grasp success.
- Full trained Neural Jacobian Field deployment.
- Direct arbitrary robot trajectory generation.

## Current Branch Plan

Use:

```bash
git checkout elmira-v3
```

for Week 10-12 grasping and manipulation development.

Keep:

```bash
elmira-v2.1-final
```

as the AP2 dashboard reference point.
