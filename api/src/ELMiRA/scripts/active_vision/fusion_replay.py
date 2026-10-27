"""Offline RGB/geometry fusion on recorded, posed single-camera observations.

No camera, ROS, serial or motion interfaces are imported. Experimental inputs
require explicit opt-in; they are never promoted to verified geometry. This
adapter reuses the professor's frontend/model and NICO's guarded reconstruction,
not the autonomous next-view or motor-execution paths.
"""
from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np

from .calibration import Calibration
from .common import positive, read_json, sha256, transform, write_json
from .masks import validate_mask
from .recognition import RGBEncoder
from .session import validate_observation
from .sparse_preview import reconstruct_preview
from .vendor.acf3d.frontend_v8 import prepare_rgb_observation
from .vendor.acf3d.memory import LifelongPrototypeMemory


def unit_vector(vector):
    vector = np.asarray(vector, np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
        raise ValueError("Embedding must be a finite, nonzero vector")
    return vector / np.linalg.norm(vector)


def validate_points(points):
    points = np.asarray(points, np.float32)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 10
            or not np.isfinite(points).all()
            or np.linalg.norm(np.ptp(points, axis=0)) < 1e-6):
        raise ValueError("Fusion needs at least ten finite, non-degenerate reconstructed XYZ points")
    return points


class RGBGeometryEncoder(RGBEncoder):
    """Exact v8 neural/signature fusion formula, with matched masked RGB input."""

    feature_mode = "professor_v8_neural_rgb_geometry_plus_signature"

    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.calls = Counter()
        self.hooks = [module.register_forward_hook(
            lambda _module, _inputs, _outputs, key=name: self.calls.update([key]))
            for name, module in (("visual", self.model.visual),
                                 ("points", self.model.points), ("fusion", self.model.fuse))]

    def encode_fused(self, bgr, mask, points):
        points = validate_points(points)
        if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Fusion requires a readable uint8 BGR image")
        mask = validate_mask(mask, bgr.shape)
        isolated = cv2.bitwise_and(bgr, bgr, mask=mask)
        observation = prepare_rgb_observation(isolated, mask, points,
                                              np.random.default_rng(0))
        with self.torch.no_grad():
            visual = self.torch.from_numpy(observation["visual"][None])
            tokens = self.torch.from_numpy(observation["points"][None])
            _, embedding, parts = self.model(visual, tokens, return_parts=True)
        neural = embedding[0].numpy().astype(np.float32)
        # Matches SingleRGBActiveRecognizer.embed_actual in the supplied v8.
        fused = unit_vector(np.concatenate([neural, .75 * observation["signature"]]))
        if fused.shape != (101,):
            raise ValueError("Unexpected professor full-fusion vector dimension")
        return fused, {
            "visual_input_shape": list(observation["visual"].shape),
            "point_input_shape": list(observation["points"].shape),
            "nonzero_point_input_values": int(np.count_nonzero(observation["points"])),
            "signature_dimension": int(observation["signature"].size),
            "gate": parts["gate"][0].numpy().tolist(),
            "visual_embedding": parts["visual"][0].numpy(),
            "point_embedding": parts["points"][0].numpy(),
        }

    def close(self):
        for hook in self.hooks:
            hook.remove()


def load_sequence(manifest_path, allow_experimental=False):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("object_static") is not True or manifest.get("units") != "metres":
        raise ValueError("Need a stationary object and explicit metre convention")
    if not manifest.get("world_frame") or not manifest.get("calibration"):
        raise ValueError("Need a common world frame and same-camera calibration; image-only photos cannot supply poses")
    rows = manifest.get("observations", [])
    if not 2 <= len(rows) <= 12:
        raise ValueError("Need 2..12 recorded images, masks and per-frame camera poses")
    calibration_path = (path.parent / manifest["calibration"]).resolve()
    calibration = Calibration.load(calibration_path, require_verified=not allow_experimental)
    records, digests, previous = [], set(), None
    for row in rows:
        image_path = (path.parent / row["image"]).resolve()
        mask_path = (path.parent / row["mask"]).resolve()
        digest = sha256(image_path)
        mask_digest = sha256(mask_path)
        if digest in digests:
            raise ValueError("Duplicate source image does not supply a new camera view")
        digests.add(digest)
        for key, actual in (("image_sha256", digest), ("mask_sha256", mask_digest)):
            if not row.get(key) or row[key] != actual:
                raise ValueError("Missing or mismatched source checksum: " + key)
        if row.get("camera_id") != calibration.record["camera_id"]:
            raise ValueError("Do not mix cameras or reuse BRIO calibration for NICO images")
        if row.get("world_frame", manifest["world_frame"]) != manifest["world_frame"]:
            raise ValueError("World frame changes within recording")
        if row.get("units", "metres") != "metres":
            raise ValueError("Mixed pose units")
        if row.get("pose_source") not in ("fixed_metric_chessboard_pnp", "fixed_board_native_corner_pnp",
                                          "tf_calibrated", "measured_external"):
            raise ValueError("Need image-derived or measured poses, not commanded head angles")
        stamp = positive(row["timestamp_sec"], "image timestamp")
        if previous is not None and stamp <= previous:
            raise ValueError("Source images must be in capture order")
        if not allow_experimental:
            validate_observation({**manifest, **row}, calibration, previous)
        elif row.get("pose_source") in ("fixed_metric_chessboard_pnp", "fixed_board_native_corner_pnp"):
            error = float(row["pose_reprojection_rms_px"])
            if not np.isfinite(error) or not 0 <= error <= 1.0:
                raise ValueError("Candidate board pose fails the existing 1 px reprojection gate")
        previous = stamp
        pose = transform(row["T_wc"])
        image = cv2.imread(str(image_path))
        if image is None or (image.shape[1], image.shape[0]) != calibration.size:
            raise ValueError("Source image missing or resolution differs from calibration")
        mask = validate_mask(cv2.imread(str(mask_path), 0), image.shape)
        # A virtual pinhole rotation preserves the physical optical centre and
        # keeps the peripheral NICO target visible without inventing parallax.
        rectified, rectified_mask, _, _ = calibration.rectify_object(image, mask, pose)
        records.append({"image": str(image_path), "mask": str(mask_path),
                        "image_sha256": digest, "mask_sha256": mask_digest,
                        "observation": row, "rectified": rectified,
                        "rectified_mask": validate_mask(rectified_mask, rectified.shape)})
    return path, manifest, calibration_path, calibration, records


def load_fused_memory(path):
    path = Path(path).resolve()
    memory = LifelongPrototypeMemory.load(path)
    if not memory.samples:
        raise ValueError("Full-fusion memory is empty")
    for exemplars in memory.samples.values():
        for exemplar in exemplars:
            if np.asarray(exemplar).shape != (101,) or not np.isfinite(exemplar).all():
                raise ValueError("Full fusion needs a separate 101D memory, not the existing 64D RGB memory")
    return memory


def predict(memory, embedding):
    if memory is None:
        return {"prediction": None, "reason": "No matching full-fusion class memory supplied"}
    label, similarity, margin, scores, entropy = memory.predict(embedding)
    return {"prediction": label, "similarity": float(similarity), "margin": float(margin),
            "scores": scores, "entropy": float(entropy), "probability_claimed": False}


def replay(manifest_path, checkpoint, output, memory_path=None, allow_experimental=False):
    started = time.monotonic()
    path, manifest, calibration_path, calibration, records = load_sequence(
        manifest_path, allow_experimental)
    memory = load_fused_memory(memory_path) if memory_path else None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint).resolve()
    source_hashes = {path: sha256(path), calibration_path: sha256(calibration_path),
                     checkpoint: sha256(checkpoint)}
    for row in records:
        source_hashes[Path(row["image"])] = row["image_sha256"]
        source_hashes[Path(row["mask"])] = row["mask_sha256"]
    if memory_path:
        for source in (Path(memory_path).resolve(), Path(memory_path).resolve().with_suffix(".json")):
            source_hashes[source] = sha256(source)
    encoder = RGBGeometryEncoder(checkpoint)
    initial_weights = {name: value.detach().clone() for name, value in encoder.model.state_dict().items()}
    visual_vectors, fused_vectors, trace = [], [], []
    for index, record in enumerate(records):
        step = {"step": index + 1, "source_image": record["image"],
                "source_mask": record["mask"], "source_image_sha256": record["image_sha256"],
                "source_mask_sha256": record["mask_sha256"],
                "geometry_views_available_at_step": index + 1,
                "future_images_used": False, "real_robot_motion_commanded": False}
        try:
            visual = encoder.encode(record["rectified"], record["rectified_mask"])
            visual_vectors.append(visual)
            step["rgb_embedding_dimension"] = int(visual.size)
            if index == 0:
                step.update(status="rgb_only_waiting_for_second_view", point_count=0)
                continue
            prefix = {**manifest, "calibration": str(calibration_path), "observations": []}
            for entry in records[:index + 1]:
                prefix["observations"].append({**entry["observation"],
                                               "image": entry["image"], "mask": entry["mask"]})
            prefix_path = output / ("prefix_%02d.json" % (index + 1))
            write_json(prefix_path, prefix)
            reconstruction_dir = output / ("geometry_prefix_%02d" % (index + 1))
            reconstruction = reconstruct_preview(prefix_path, reconstruction_dir, "akaze")
            step.update(geometry_status=reconstruction["status"],
                        reconstruction_report=str(reconstruction_dir / "sparse_preview_report.json"),
                        point_count=reconstruction["point_count"],
                        hull_quality=reconstruction["experimental_visual_hull"].get("quality_status"),
                        point_source="prefix-only silhouette-checked sparse RGB triangulation; clipped hull not used as neural input")
            with np.load(str(reconstruction_dir / "experimental_object_cloud.npz"), allow_pickle=False) as cloud:
                points = validate_points(cloud["xyz"])
            fused, parts = encoder.encode_fused(record["rectified"], record["rectified_mask"], points)
            np.savez_compressed(str(output / ("features_step_%02d.npz" % (index + 1))),
                                rgb_visual=visual, reconstructed_xyz=points,
                                full_fused=fused, point_embedding=parts.pop("point_embedding"),
                                matched_fusion_visual_embedding=parts.pop("visual_embedding"))
            fused_vectors.append(fused)
            step.update(status="rgb_geometry_fused", full_embedding_dimension=int(fused.size),
                        **parts, **predict(memory, unit_vector(np.mean(fused_vectors, axis=0))))
        except (ValueError, KeyError, OSError, RuntimeError, cv2.error) as error:
            step.update(status="geometry_or_encoding_unavailable", reason=str(error))
        finally:
            trace.append(step)
            write_json(output / "trace.json", {"steps": trace, "real_robot_motion_commanded": False})
    try:
        assert all(sha256(source) == digest for source, digest in source_hashes.items())
        unchanged_weights = all(encoder.torch.equal(value, initial_weights[name])
                                for name, value in encoder.model.state_dict().items())
        arrays = {}
        if visual_vectors:
            arrays.update(rgb_by_view=np.stack(visual_vectors),
                          rgb_single_view=visual_vectors[0],
                          rgb_multiview_mean=unit_vector(np.mean(visual_vectors, axis=0)))
        if fused_vectors:
            arrays.update(full_fused_by_available_step=np.stack(fused_vectors),
                          full_fused_mean=unit_vector(np.mean(fused_vectors, axis=0)))
        np.savez_compressed(str(output / "recognition_features.npz"), **arrays)
        report = {
            "status": "experimental_real_image_fusion_completed" if fused_vectors else "no_usable_multiview_fusion",
            "scope": "offline integration replay; not a controlled accuracy or improvement benchmark",
            "target_object": manifest.get("target_object", "not available"),
            "manifest": str(path), "manifest_sha256": source_hashes[path],
            "calibration": str(calibration_path), "calibration_sha256": source_hashes[calibration_path],
            "camera_id": calibration.record["camera_id"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": source_hashes[checkpoint],
            "source_view_count": len(records), "rgb_encoded_view_count": len(visual_vectors),
            "fused_step_count": len(fused_vectors), "network_branch_forward_calls": dict(encoder.calls),
            "rgb_embedding_dimension": 64, "full_fused_embedding_dimension": 101,
            "full_embedding_formula": "normalize(concat(neural_RGB_XYZ_fusion_64D, 0.75 * analytic_signature_37D))",
            "geometry_source": "new prefix-only sparse reconstruction from original RGB/masks and recorded candidate poses",
            "geometry_uses_point_colours": False,
            "visual_preprocessing": "same masked-image v8 grayscale log-polar and spectrum frontend for RGB and fusion",
            "memory": str(Path(memory_path).resolve()) if memory_path else None,
            "memory_labels": list(memory.samples) if memory else [],
            "memory_scope": "Use supplied memory provenance. Synthetic primitive memory does not establish real-object category accuracy.",
            "memory_updated": False, "thresholds_tuned_on_replay": False,
            "final_fused_prediction": predict(memory, unit_vector(np.mean(fused_vectors, axis=0))) if fused_vectors else None,
            "recognition_accuracy_percent": None, "recognition_improvement_demonstrated": False,
            "rgb_single_and_multiview_control_features_saved": bool(visual_vectors),
            "controlled_comparison_status": "requires matched real-object enrollment and independent calibration/test sessions for each feature mode",
            "experimental_unverified_inputs_explicitly_allowed": bool(allow_experimental),
            "input_calibration_verified": calibration.record.get("verified") is True,
            "all_input_poses_verified": all(r["observation"].get("pose_verified") is True for r in records),
            "all_input_masks_reviewed": all(r["observation"].get("mask_reviewed") is True for r in records),
            "metric_accuracy_verified": False, "complete_object_shape_verified": False,
            "sources_unchanged": True, "network_weights_unchanged": unchanged_weights,
            "neural_training_performed": False, "future_images_used_in_prefix_geometry": False,
            "camera_opened": False, "serial_port_opened": False, "depth_sensor_frames_consumed": 0,
            "real_robot_motion_commanded": False, "robot_runtime_modified": False,
            "autonomous_observe_move_observe_tested": False,
            "elapsed_seconds": time.monotonic() - started,
            "steps": trace,
            "limitations": [
                "Candidate calibration/poses and assistant-proposed masks remain unverified even if the code completes.",
                "Sparse visible-surface geometry is incomplete; invalid/clipped hull samples are not substituted for real point input.",
                "One-frame RGB and multi-frame RGB controls are saved, but no matched labeled real-object memory exists yet for a fair fusion accuracy comparison.",
                "Ordinary RGB-only datasets without measured poses cannot be promoted to this geometry path.",
            ],
        }
        write_json(output / "fusion_replay_report.json", report)
        return report
    finally:
        encoder.close()
