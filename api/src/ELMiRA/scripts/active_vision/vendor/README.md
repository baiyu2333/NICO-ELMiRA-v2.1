# External Professor-Supplied Dependency

The NICO adaptation imports the professor's `acf3d` package from this directory.
Its original source and checkpoint were supplied privately in
`Active_Conformal_Foveated_Vision_Single_RGB_v8(1).zip`. No separate redistribution
license was found in that archive, so neither the implementation nor weights
are included in this GitHub publication. No claim of an open-source license
for the supplied package is made.

## Authorized Local Setup

Obtain the original archive and permission to use it from the author. From its
`acf_lifelong3d_v8/acf3d/` directory, place the following original files into
`api/src/ELMiRA/scripts/active_vision/vendor/acf3d/`:

```text
__init__.py
frontend.py
frontend_v8.py
model.py
synthetic.py
pose_se3.py
rgb_geometry_v8.py
memory.py
planner_v6.py
se3_gaussian_world.py
gaussian_world.py
```

[PROVENANCE.txt](PROVENANCE.txt) records the original archive hash and the
hash of each imported file. The NICO adaptations live outside these files.
Do not substitute random model weights or an unrelated package with the same
name. The neural encoder requires the matching authorized `single_rgb_model.pt`
checkpoint; commands take its location explicitly with `--checkpoint`.

The directory and local checkpoint folder are git-ignored to prevent accidental
publication. This does not remove an authorized local installation.

## What Runs Without This Dependency?

- The pre-existing ELMiRA dashboard/manipulation stack is separate from this
  vision package and retains its own dependencies.
- Standalone calibration, mask, and helper modules can be inspected without the
  supplied neural implementation, subject to their NumPy/OpenCV dependencies.
- The current unified `active_vision_cli.py`, geometry wrappers, recognition
  workflow, and several tests import `acf3d` and require it even for some
  non-neural commands. Without installation, an import error is expected.
- Neural inference additionally needs PyTorch and the matching checkpoint.
- Optional DINOv3 comparison needs its separately authorized local model and
  compatible dependencies; nothing downloads automatically.

After installing the authorized dependency in the configured environment,
the offline tests can be run from the repository root:

```bash
PYTHONPATH=api/src/ELMiRA/scripts python3 -m unittest discover \
  -s api/src/ELMiRA/tests -p 'test_*.py'
```

These tests use synthetic inputs and mocked hardware where applicable. They do
not certify real-object recognition accuracy or authorize physical execution.
Do not run hardware diagnostic entry points as installation smoke tests.
