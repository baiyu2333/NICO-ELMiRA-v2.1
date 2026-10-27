#!/usr/bin/env python3
"""Replay the original v8 CPU graph, never a camera or robot execution test."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, default=serializable,
                               allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    inventory = json.loads((audit / "source_inventory.json").read_text())
    for row in inventory["files"]:
        if digest(audit / row["file"]) != row["sha256"]:
            raise ValueError("Original source/asset hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(audit / "original_source"))
    from acf3d.memory import LifelongPrototypeMemory
    from acf3d.model import DualStreamConformal3DNet
    from acf3d.pipeline_v8 import SingleRGBActiveRecognizer
    from acf3d.planner_v6 import BatchedSE3NBVPlanner
    from acf3d.pose_se3 import candidate_pose_lattice, synthetic_view_pose
    from acf3d.rgb_geometry_v8 import RGBPoseGeometry, mask_from_rgb
    from acf3d.synthetic import Camera, SyntheticRGBDWorld

    torch.set_num_threads(1)
    torch.manual_seed(8)
    np.random.seed(8)
    checkpoint = audit / "supplied_assets/single_rgb_model.pt"
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model = DualStreamConformal3DNet(num_classes=len(saved["base_classes"]))
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    initial_weights = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    memory = LifelongPrototypeMemory.load(audit / "supplied_assets/lifelong_memory_rgb.npz")
    initial_memory_metadata = {
        "similarity_threshold": memory.similarity_threshold,
        "margin_threshold": memory.margin_threshold,
        "class_thresholds": dict(memory.class_thresholds),
    }
    calls = Counter()
    hooks = [module.register_forward_hook(
        lambda _module, _inputs, _output, key=name: calls.update([key]))
        for name, module in (("visual", model.visual), ("points", model.points),
                             ("fusion", model.fuse))]
    world = SyntheticRGBDWorld(Camera(), 8)
    recognizer = SingleRGBActiveRecognizer(
        model, memory, BatchedSE3NBVPlanner(), world.cam, device="cpu", seed=8,
        gaussian_backend="cpu", geometry_kwargs={"half_extent": 1.05,
                                                  "resolution": 30, "seed": 8})
    start = time.monotonic()
    cases = []

    def candidates(current, visited):
        return candidate_pose_lattice(
            current, yaw_values=(-75, -50, -25, 0, 25, 50, 75),
            pitch_values=(7, 12, 18), lateral_values=(0.,))

    def run(label, yaw, phase, expected):
        frame_count = 0

        def observe(pose):
            nonlocal frame_count
            frame_count += 1
            rgb, _unused_sensor_depth, _unused_sensor_mask = world.render_pose(
                label, pose, noise=1.4)
            return rgb, mask_from_rgb(rgb)

        summary, trace, _world_map = recognizer.recognize(
            observe, synthetic_view_pose(yaw, 12.), candidates, max_views=4)
        folder = args.output / (label + "_" + phase)
        folder.mkdir()
        rows = []
        for row in trace:
            index = row["step"]
            cv2.imwrite(str(folder / ("rgb_%02d.png" % index)), row["rgb"])
            cv2.imwrite(str(folder / ("mask_%02d.png" % index)), row["mask"] * 255)
            np.save(str(folder / ("model_proxy_depth_%02d.npy" % index)), row["proxy_depth"])
            rows.append({
                "step": index, "pose": row["pose"],
                "prediction": row["prediction"], "similarity": row["similarity"],
                "margin": row["margin"], "gate": row["gate"],
                "visual_input_shape": list(row["obs"]["visual"].shape),
                "point_input_shape": list(row["obs"]["points"].shape),
                "nonzero_point_input_values": int(np.count_nonzero(row["obs"]["points"])),
                "world_update": row["world_update"],
                "gaussian_map": row["gaussian_map"], "plan": row.get("plan", []),
            })
        result = {key: value for key, value in summary.items() if key != "fused_embedding"}
        result.update(true_synthetic_label=label, phase=phase, expected=expected,
                      expected_prediction_matches=summary["prediction"] == expected,
                      fused_memory_vector_dimension=int(summary["fused_embedding"].size),
                      actual_observe_callback_calls=frame_count,
                      nbv_candidate_score_rows=sum(len(r["plan"]) for r in rows),
                      depth_sensor_frames_consumed=0, real_robot_motion_commanded=False,
                      input_kind="synthetic_RGB_with_known_virtual_camera_poses")
        np.save(str(folder / "fused_embedding.npy"), summary["fused_embedding"])
        write_json(folder / "trace.json", {"summary": result, "steps": rows})
        cases.append(result)
        print(json.dumps({"label": label, "phase": phase, "prediction": result["prediction"],
                          "views": result["views_used"], "nbv_score_rows": result["nbv_candidate_score_rows"]}),
              flush=True)
        return result

    try:
        for label, yaw in (("cube", -63), ("sphere", 51), ("cylinder", -57)):
            run(label, yaw, "known", label)
        before = run("cone", 58, "before_enrollment", "unknown")
        enrolled = []
        for index, yaw in enumerate((-60, -30, 0, 30, 60)):
            geometry = RGBPoseGeometry(world.cam, half_extent=1.05, resolution=26,
                                       seed=900 + index)
            pose = synthetic_view_pose(yaw, 12.)
            rgb, _unused_depth, _unused_mask = world.render_pose("cone", pose, noise=.5)
            mask = mask_from_rgb(rgb)
            geometry.update(rgb, mask, pose)
            vector, _observation, _parts = recognizer.embed_actual(
                rgb, mask, geometry.object_points(512))
            memory.learn("cone", vector)
            enrolled.append(vector)
        prototype = memory.prototype("cone")
        similarities = [float(v @ prototype) for v in enrolled]
        memory.class_thresholds["cone"] = float(max(.74, min(.97, np.percentile(similarities, 5) - .04)))
        memory.save(args.output / "synthetic_memory_after_enrollment.npz")
        after = run("cone", -67, "after_enrollment", "cone")
        for row in inventory["files"]:
            assert digest(audit / row["file"]) == row["sha256"]
        assert not model.training
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "original_professor_v8_full_CPU_synthetic_function_replay_only",
            "cases": cases, "network_branch_forward_calls": dict(calls),
            "software_path_completed": all(calls[k] > 0 for k in ("visual", "points", "fusion")),
            "all_expected_synthetic_predictions_matched": all(c["expected_prediction_matches"] for c in cases),
            "nbv_scoring_exercised": any(c["nbv_candidate_score_rows"] for c in cases),
            "new_category_enrollment_exercised": "cone" in memory.samples,
            "lifelong_before": before["prediction"], "lifelong_after": after["prediction"],
            "initial_memory_metadata": initial_memory_metadata,
            "neural_training_performed": False, "depth_sensor_frames_consumed": 0,
            "original_source_and_assets_unchanged": True,
            "network_weights_unchanged": initial_weights == hashlib.sha256(b"".join(
                value.detach().cpu().numpy().tobytes()
                for value in model.state_dict().values())).hexdigest(),
            "archive_sha256": inventory["archive_sha256"],
            "checkpoint_sha256": digest(checkpoint), "elapsed_seconds": time.monotonic() - start,
            "hardware_accessed": False, "real_robot_motion_commanded": False,
            "real_object_accuracy": "not measured by this synthetic replay",
            "real_camera_pose_calibration": "not validated by this synthetic replay",
            "NICO_ROS1_full_pipeline_deployed": False,
        }
        report["contract_checks"] = {
            "all_network_branches_exercised": report["software_path_completed"],
            "all_five_expected_synthetic_outputs": report["all_expected_synthetic_predictions_matched"],
            "next_view_scoring_exercised": report["nbv_scoring_exercised"],
            "new_category_registered_and_recognized": before["prediction"] == "unknown"
                and after["prediction"] == "cone",
            "full_neural_and_signature_vector": all(
                c["fused_memory_vector_dimension"] == 101 for c in cases),
            "original_source_and_assets_unchanged": report["original_source_and_assets_unchanged"],
            "network_weights_unchanged": report["network_weights_unchanged"],
        }
        report["status"] = "passed" if all(report["contract_checks"].values()) else "failed"
        write_json(args.output / "replay_report.json", report)
        print(args.output / "replay_report.json", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
