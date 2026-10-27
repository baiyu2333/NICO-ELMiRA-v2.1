# ELMiRA v3 on NICO

**FYP: Embodying Visual Language Models in Robot Action on the NICO humanoid robot.**

This research repository extends the NICO software and ELMiRA interaction
pipeline with a robot dashboard, right-arm manipulation debugging, captured
grasp templates, and structured trial recording. A separate experimental
single-RGB vision integration investigates multiview reconstruction and
object recognition; it is not a validated autonomous grasping system.

Current development branch: **[`elmira-v3`](https://github.com/baiyu2333/NICO-ELMiRA-v2.1/tree/elmira-v3)**.
The repository URL retains the older `NICO-ELMiRA-v2.1` name. The ROS package
remains `elmira`, and `init_nodes_v2.launch` keeps its existing filename for
compatibility. See the [v3 transition notes](docs/elmira_v3_transition.md).

## Start Here

| Topic | Entry point |
| --- | --- |
| For computer-vision collaborators | [Vision research overview and current evidence](docs/vision_research_overview.md) |
| Single-RGB integration history | [Integration and validation notes](docs/active_single_rgb_nico.md) |
| Recognition experiment and data splits | [Real-object recognition workflow](docs/real_object_recognition_workflow.md) |
| Camera calibration board and capture | [Calibration guide](docs/calibration/README.md) |
| Version boundary and v3 scope | [ELMiRA v3 transition](docs/elmira_v3_transition.md) |
| Dashboard implementation | [`dashboard_clean.py`](api/src/ELMiRA/scripts/dashboard_clean.py) |
| ROS nodes and hardware defaults | [`init_nodes_v2.launch`](api/src/ELMiRA/launch/init_nodes_v2.launch) |
| Manipulation state machine | [`state_machine.py`](api/src/ELMiRA/scripts/state_machine.py) |
| Kinematic preview and recording | [Kinematic preview guide](docs/fyp2_kinematic_preview_plan.md) |
| Historical AP2 evaluation and limitations | [AP2 viva summary](api/logs/ap2_viva_summary/ap2_viva_summary.md) |

The [AP2 dashboard demo guide](docs/fyp2_clean_dashboard_demo.md) describes an
earlier monitoring-stage interface. Some labels and screen layouts have since
changed; it is not a specification of every current dashboard control.

## Platform and Communication

| Layer | Current project setup |
| --- | --- |
| Operating system | Ubuntu 20.04 LTS; the current development VM is 20.04.6 |
| Middleware | ROS 1 Noetic, Catkin, Python 3 |
| Robot | NICO upper body; manipulation development focuses on the physical right arm and hand |
| Motor connection | FTDI USB-to-serial adapter; addressed motor-bus commands, not USB image messages |
| Head and arm motors | Dynamixel MX-64 through the Protocol 1.0/PyPot path; head diagnostics use 1 Mbps |
| Current physical right hand | Separate XL-320-compatible Protocol 2.0 command path for wrist IDs 31/33 and finger IDs 34-37 |
| Cameras | NICO See3CAM CU135 USB cameras through UVC/V4L2; current experiments mainly use camera serial 22275000 at 1280x720 |
| Application communication | ROS topics, services, and actions between speech, vision, planning, and motion nodes |
| AI services | Configurable OpenAI or Google provider for the existing ELMiRA multimodal gateway |

The hands have been physically swapped. Legacy joint names, older documentation,
and default disabled-motor lists do not by themselves identify the currently
connected physical hand. Verify the active configuration and motor identities
before hardware execution. Dynamixel Protocol 2.0 is not ROS 2.

## Architecture and Scope

The existing robot interaction pipeline is:

```text
Voice/text instruction -> VLM reasoning and object grounding
    -> image-to-robot coordinate transfer -> action planning / EvoIK
    -> existing ROS motion and hand-control interfaces -> trial evidence
```

The v3 tools support isolated grasp-stage tests, no-motion IK trace dumps,
natural-pose capture, template-seeded IK, joint-state recording/replay, and
first-person camera trial recording with manual outcome labels. Simplified
forward-kinematics previews are planning aids, not physics/contact simulation.
Captured seeds and seed guards do not provide collision avoidance or guarantee
a successful grasp. Pose capture, preview, and replay visualization must be
distinguished from explicitly selected real execution controls.

The separate experimental vision workflow is:

```text
Single-RGB observations + object masks + per-frame camera poses
    -> multiview geometry -> image/geometry features
    -> prototype recognition and known/unknown evaluation
```

Supervised head-viewpoint collection and offline RGB/geometry replay have been
tested locally. Masks currently include manual assistance, and calibration,
camera poses, reconstructed shape, and recognition performance retain explicit
verification limits. This does not establish precise metric 3D, automatic
segmentation, a depth-camera pipeline, or autonomous learned viewpoint selection.
The vision experiment is separate from the real-arm grasp execution path.

**Published scope:** the NICO adaptation code, vision tests, calibration board,
and research/workflow documentation are included. The professor-supplied
`acf3d` implementation and checkpoint are deliberately excluded pending
redistribution permission. Follow the [external dependency instructions](api/src/ELMiRA/scripts/active_vision/vendor/README.md)
before running dependent tools. A fresh clone is not a self-contained neural
vision installation: even the current unified vision CLI imports this external
package. Raw camera recordings, generated reports containing people, private
runtime logs, and model weights are not published. Documentation references to
those local evidence paths describe the experiments; they are not download links.

For visual recognition, this release supports a controlled comparison of
single-view image features, multiview image features, and image-plus-geometry
features. As of 2026-10-27, the current three-object pilot has three enrollment
sequences and four calibration sequences, but no independent test sequences.
Calibration results must not be presented as final recognition performance.
See the [vision overview](docs/vision_research_overview.md) for the exact inputs,
current results, limitations, and collaboration questions.

## Open the Dashboard

On an already configured local workspace, from the repository root:

```bash
cd api
source activate.bash
source devel/setup.bash
python3 src/ELMiRA/scripts/dashboard_clean.py
```

Open `http://127.0.0.1:7860`; if occupied, the dashboard tries port `7861`.
Opening the dashboard alone does not start the robot pipeline. Launch and
execution controls can affect real hardware and require physical supervision.
Do not start an unfamiliar configuration on a connected robot unattended.

This is not a clean-machine installation procedure: local dependencies,
model files, built ROS messages/services, and hardware configuration may be
required. Never commit API keys or publish raw commands/logs containing secrets.

## Evidence and Limitations

- Representative physical grasp/lift trials exist, but stable repeated grasping
  and generalization must be evaluated with clearly defined repeated trials.
- Historical tuning failures, post-calibration trials, and new recognition
  tests must not be combined into an unsupported final performance claim.
- Recognition enrollment, threshold calibration, and independent testing are
  separate stages; calibration scores are not held-out accuracy.
- NJF support is an NJF-inspired feasibility probe / service boundary, not a
  fully trained and validated NICO-specific Neural Jacobian Field deployment.
- No full physics simulator, validated contact model, certified collision
  checking, or verified force-feedback grasp controller is claimed.
- Before sharing raw recordings, review visible people, credentials, supplied
  third-party code permissions, and model redistribution requirements.

## Upstream NICO Software

This project builds on software for NICO (Neuro-Inspired COmpanion). The
upstream robot platform and its academic attribution are retained below.

[![](https://www2.informatik.uni-hamburg.de/wtm/pictures/nico_tomato.jpg)](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/neurobotics/nico.html)

## Academic Citation

If you use the NICO robot, this software or parts of it or the NICO CAD files for your research, please cite the paper 

NICO - Neuro-Inspired COmpanion: A Developmental Humanoid Robot Platform for Multimodal Interaction  
Matthias Kerzel, Erik Strahl, Sven Magg, Nicolás Navarro-Guerrero, Stefan Heinrich, Stefan Wermter
Proceedings of the IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN), pages 113-120, 2017.

> BibTeX Style Citation

```
@InProceedings\{KSMNHW17,
  author       = "Kerzel, Matthias and Strahl, Erik and Magg, Sven and Navarro-Guerrero, Nicol\'as and Heinrich, Stefan and Wermter, Stefan",
  title        = "NICO - Neuro-Inspired COmpanion: A Developmental Humanoid Robot Platform for Multimodal Interaction",
  booktitle    = "Proceedings of the IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN)",
  pages        = "113--120",
  month        = "Aug",
  year         = "2017",
  url          = "https://www2.informatik.uni-hamburg.de/wtm/publications/2017/KSMNHW17/"
}
```
## Documentation

Find all information to run a simulated NICO in the wiki at https://github.com/knowledgetechnologyuhh/NICO-software/wiki

Find much more information and how to build your own NICO at http://nico.knowledge-technology.info
