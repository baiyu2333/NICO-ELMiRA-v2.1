"""Real-object prototype registration and paired sequence-level comparison.

Consumes frozen offline fusion-replay features, not camera/ROS/motor interfaces.
Enrollment, rejection-threshold calibration and held-out evaluation are separate
commands. A missing geometry result is never replaced with a made-up vector.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .common import read_json, sha256, write_json
from .fusion_replay import unit_vector
from .recognition import _fit_memory
from .vendor.acf3d.memory import LifelongPrototypeMemory


MODES = {
    "rgb_single": ("rgb_single_view", 64),
    "rgb_multiview": ("rgb_multiview_mean", 64),
    "rgb_geometry": ("full_fused_mean", 101),
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def dataset_rows(path, manifest=None):
    """Validate split metadata without opening any replay/feature/image files."""
    path = Path(path).resolve()
    manifest = read_json(path) if manifest is None else manifest
    rows = manifest.get("samples", [])
    if not rows:
        raise ValueError("Sequence dataset is empty")
    samples, groups, image_splits, unknown_instances, known_instances = set(), set(), {}, {}, set()
    normalized = []
    instance_labels = {}
    for source in rows:
        row = dict(source)
        for key in ("sample_id", "capture_group_id", "object_instance_id", "label"):
            if not isinstance(row.get(key), str) or not _IDENTIFIER.fullmatch(row[key]):
                raise ValueError("Need a nonempty ASCII identifier for " + key)
        if row.get("split") not in ("enroll", "calibration", "test"):
            raise ValueError("Use enroll/calibration/test splits")
        if row["sample_id"] in samples or row["capture_group_id"] in groups:
            raise ValueError("Duplicate sample/capture group; adjacent frames are not independent trials")
        samples.add(row["sample_id"])
        groups.add(row["capture_group_id"])
        if row["label"] == "unknown":
            if row["split"] == "enroll":
                raise ValueError("Unknown objects must not enter the known-class gallery")
            unknown_instances.setdefault(row["split"], set()).add(row["object_instance_id"])
        else:
            known_instances.add(row["object_instance_id"])
        previous_label = instance_labels.setdefault(row["object_instance_id"], row["label"])
        if previous_label != row["label"]:
            raise ValueError("Physical object instance has conflicting labels")
        hashes = row.get("source_image_sha256s")
        if not isinstance(hashes, list) or len(hashes) < 2 or len(set(hashes)) != len(hashes):
            raise ValueError("Need distinct source image checksums for each recorded sequence")
        for digest in hashes:
            if not isinstance(digest, str) or not _SHA.fullmatch(digest):
                raise ValueError("Invalid source image SHA-256")
            if digest in image_splits:
                raise ValueError("Source image reused across sequences/splits; cannot count as an independent trial")
            image_splits[digest] = row["split"]
        for key in ("replay_report_sha256", "feature_file_sha256"):
            if not isinstance(row.get(key), str) or not _SHA.fullmatch(row[key]):
                raise ValueError("Need a frozen checksum for " + key)
        for key in ("replay_report", "feature_file"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("Missing recorded-feature path: " + key)
            row[key] = str((path.parent / row[key]).resolve())
        normalized.append(row)
    if unknown_instances.get("calibration", set()) & unknown_instances.get("test", set()):
        raise ValueError("Unknown physical objects must differ between calibration and held-out testing")
    unknown_all = set().union(*unknown_instances.values())
    if known_instances & unknown_all:
        raise ValueError("An unknown object instance also appears as a known class")
    return path, manifest, normalized


def _source_signature(rows):
    keys = ("sample_id", "capture_group_id", "object_instance_id", "label", "split",
            "source_image_sha256s", "replay_report", "replay_report_sha256",
            "feature_file", "feature_file_sha256")
    return _signature([{key: row[key] for key in keys}
                       for row in sorted(rows, key=lambda value: value["sample_id"])])


def add_sequence(manifest_path, replay_path, label, split, capture_group, object_instance,
                 sample_id=None):
    """Add human labels/provenance without requiring hand-written checksums."""
    path, replay_path = Path(manifest_path).resolve(), Path(replay_path).resolve()
    previous_hash = sha256(path) if path.exists() else None
    manifest = read_json(path) if previous_hash else {"dataset_id": path.stem, "samples": []}
    report = read_json(replay_path)
    feature_path = replay_path.parent / "recognition_features.npz"
    row = {"sample_id": sample_id or capture_group, "capture_group_id": capture_group,
           "object_instance_id": object_instance, "label": label, "split": split,
           "replay_report": str(replay_path), "replay_report_sha256": sha256(replay_path),
           "feature_file": str(feature_path), "feature_file_sha256": sha256(feature_path),
           "source_image_sha256s": [step["source_image_sha256"] for step in report["steps"]],
           "label_source": "User-supplied physical object identity; not the previous model prediction."}
    candidate = {**manifest, "samples": [*manifest["samples"], row]}
    _, _, normalized = dataset_rows(path, candidate)
    load_features(normalized[-1])
    if (sha256(path) if path.exists() else None) != previous_hash:
        raise ValueError("Dataset changed concurrently; no sample added")
    write_json(path, candidate)
    return {"stage": "labelled_sequence_added_not_evaluated", "dataset": str(path),
            "sample_id": row["sample_id"], "label": label, "split": split,
            "sample_count": len(candidate["samples"]), "neural_training_performed": False,
            "held_out_accuracy_percent": None, "real_robot_motion_commanded": False}


def load_features(row):
    """Cross-check frozen features against their input replay provenance."""
    report_path, feature_path = Path(row["replay_report"]), Path(row["feature_file"])
    if feature_path != report_path.parent / "recognition_features.npz":
        raise ValueError("Feature bundle must belong to the same replay directory")
    if sha256(report_path) != row["replay_report_sha256"] or sha256(feature_path) != row["feature_file_sha256"]:
        raise ValueError("Replay report or feature file changed after dataset registration")
    report = read_json(report_path)
    if report.get("real_robot_motion_commanded") is not False or not report.get("sources_unchanged"):
        raise ValueError("Need a read-only replay with unchanged source evidence")
    if report.get("network_weights_unchanged") is not True or report.get("future_images_used_in_prefix_geometry") is not False:
        raise ValueError("Replay lacks unchanged-model/prefix-only geometry evidence")
    steps = report.get("steps", [])
    actual = [step["source_image_sha256"] for step in steps]
    if actual != row["source_image_sha256s"] or len(steps) != report.get("source_view_count"):
        raise ValueError("Dataset source-image inventory differs from replay")
    if report.get("rgb_encoded_view_count") != len(steps):
        raise ValueError("RGB control is missing source views; paired comparison would be inconsistent")
    for step in steps:
        if sha256(step["source_image"]) != step["source_image_sha256"]:
            raise ValueError("Original source image changed")
        if sha256(step["source_mask"]) != step["source_mask_sha256"]:
            raise ValueError("Original source mask changed")
    for key, expected in (("manifest", "manifest_sha256"), ("calibration", "calibration_sha256"),
                          ("checkpoint", "checkpoint_sha256")):
        if sha256(report[key]) != report[expected]:
            raise ValueError("Replay source changed: " + key)
    features = {}
    with np.load(str(feature_path), allow_pickle=False) as data:
        for mode, (key, dimension) in MODES.items():
            if key not in data:
                features[mode] = None
                continue
            vector = np.asarray(data[key], np.float32)
            if vector.shape != (dimension,):
                raise ValueError("Wrong feature dimension for " + mode)
            features[mode] = unit_vector(vector)
    if (features["rgb_geometry"] is not None) != (report.get("fused_step_count", 0) > 0):
        raise ValueError("Fusion availability in feature bundle disagrees with replay trace")
    identity = {key: report[key] for key in ("checkpoint_sha256", "camera_id",
                                            "visual_preprocessing", "full_embedding_formula")}
    return features, identity, report


def _selected_features(rows, splits):
    selected = [row for row in rows if row["split"] in splits]
    if not selected:
        raise ValueError("No samples in requested split")
    records, identity = [], None
    for row in selected:
        features, this_identity, report = load_features(row)
        if identity is not None and identity != this_identity:
            raise ValueError("Mixed cameras/checkpoints/preprocessing; not a matched comparison")
        identity = this_identity
        records.append((row, features, report))
    return records, identity


def _gallery_labels(records):
    labels = {row["label"] for row, _, _ in records if row["split"] == "enroll"}
    if "unknown" in labels or len(labels) < 2:
        raise ValueError("Need at least two real-object known classes for a comparison gallery")
    for row, features, _ in records:
        if row["split"] == "enroll" and any(features[mode] is None for mode in MODES):
            raise ValueError("Enrollment sequence needs all three feature modes: " + row["sample_id"])
    return labels


def _save_memories(output, memories):
    result = {}
    for mode, memory in memories.items():
        path = output / mode / "memory.npz"
        memory.save(path)
        result[mode] = {"path": str(path), "sha256": sha256(path),
                        "metadata_sha256": sha256(path.with_suffix(".json")),
                        "dimension": MODES[mode][1],
                        "similarity_threshold": memory.similarity_threshold,
                        "margin_threshold": memory.margin_threshold}
    return result


def _base_report(path, rows, identity):
    return {"dataset": str(path), "dataset_sha256": sha256(path),
            "selected_sample_ids": [row["sample_id"] for row in rows],
            "selected_sources_signature": _source_signature(rows), "encoder_identity": identity,
            "unit_of_analysis": "one recorded capture/placement sequence, not one frame",
            "object_labels": "human-supplied real-object labels, not automatic shape-name conversion",
            "neural_training_performed": False, "camera_opened": False,
            "serial_port_opened": False, "real_robot_motion_commanded": False,
            "model_deployed": False, "metric_accuracy_verified": False,
            "broad_category_generalization_demonstrated": False}


def enroll(manifest_path, output):
    path, _, rows = dataset_rows(manifest_path)
    records, identity = _selected_features(rows, {"enroll"})
    labels = _gallery_labels(records)
    memories = {mode: LifelongPrototypeMemory(margin_threshold=.045) for mode in MODES}
    for row, features, _ in records:
        for mode, memory in memories.items():
            memory.learn(row["label"], features[mode])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = [row for row, _, _ in records]
    report = {**_base_report(path, selected, identity), "stage": "registered_uncalibrated",
              "registered_labels": sorted(labels), "enrollment_sequences": len(records),
              "sequences_per_label": dict(Counter(row["label"] for row in selected)),
              "memories": _save_memories(output, memories), "thresholds_calibrated": False,
              "held_out_accuracy_percent": None,
              "next_step": "Collect independent calibration sequences for every known class and unknown objects; do not test on enrollment photographs."}
    write_json(output / "registration_report.json", report)
    return report


def calibrate(manifest_path, output):
    path, _, rows = dataset_rows(manifest_path)
    records, identity = _selected_features(rows, {"enroll", "calibration"})
    labels = _gallery_labels(records)
    memories, calibration_predictions = {}, {}
    for mode in MODES:
        available = [(row, features[mode]) for row, features, _ in records if features[mode] is not None]
        memory, _ = _fit_memory(available)
        memories[mode] = memory
        calibration_predictions[mode] = [
            {"sample_id": row["sample_id"], "label": row["label"],
             "predicted": memory.predict(features[mode])[0] if features[mode] is not None else "unavailable"}
            for row, features, _ in records if row["split"] == "calibration"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = [row for row, _, _ in records]
    report = {**_base_report(path, selected, identity), "stage": "calibration_only_not_held_out_test",
              "registered_labels": sorted(labels), "development_sources_signature": _source_signature(selected),
              "memories": _save_memories(output, memories), "thresholds_calibrated": True,
              "threshold_selection": "Same existing global similarity-threshold search per mode; fixed margin 0.045; development data only.",
              "test_replay_or_feature_files_opened": False,
              "calibration_predictions": calibration_predictions, "held_out_accuracy_percent": None}
    write_json(output / "sequence_calibration_report.json", report)
    return report


def _metrics(predictions):
    if not predictions:
        return {"sequences": 0, "correct": 0, "accuracy_percent": None}
    known = [row for row in predictions if row["label"] != "unknown"]
    unknown = [row for row in predictions if row["label"] == "unknown"]
    correct = sum(row["label"] == row["predicted"] for row in predictions)
    known_correct = sum(row["label"] == row["predicted"] for row in known)
    unknown_rejected = sum(row["predicted"] == "unknown" for row in unknown)
    return {"sequences": len(predictions), "correct": correct,
            "accuracy_percent": 100. * correct / len(predictions),
            "known_sequences": len(known), "unknown_sequences": len(unknown),
            "known_correct": known_correct,
            "known_accuracy_percent": 100. * known_correct / len(known) if known else None,
            "unknown_correct_rejections": unknown_rejected,
            "unknown_correct_rejection_percent": 100. * unknown_rejected / len(unknown) if unknown else None,
            "known_false_rejections": sum(row["predicted"] == "unknown" for row in known),
            "known_wrong_class_acceptances": sum(row["predicted"] not in (row["label"], "unknown", "unavailable") for row in known),
            "unknown_false_acceptances": sum(row["predicted"] not in ("unknown", "unavailable") for row in unknown),
            "unavailable_sequences": sum(row["predicted"] == "unavailable" for row in predictions),
            "unavailable_included_in_denominator": True,
            "per_label": {label: {"sequences": sum(row["label"] == label for row in predictions),
                                   "correct": sum(row["label"] == label and row["predicted"] == label for row in predictions)}
                          for label in sorted({row["label"] for row in predictions})}}


def evaluate(manifest_path, calibration_dir, output):
    path, _, rows = dataset_rows(manifest_path)
    development = [row for row in rows if row["split"] != "test"]
    calibration_path = Path(calibration_dir).resolve() / "sequence_calibration_report.json"
    calibration = read_json(calibration_path)
    if calibration.get("stage") != "calibration_only_not_held_out_test" or not calibration.get("thresholds_calibrated"):
        raise ValueError("Need frozen calibrated memories, not an uncalibrated registration gallery")
    if _source_signature(development) != calibration["development_sources_signature"]:
        raise ValueError("Development data changed since calibration; do not refit during held-out evaluation")
    records, identity = _selected_features(rows, {"test"})
    if identity != calibration["encoder_identity"]:
        raise ValueError("Test encoder/camera differs from frozen calibration")
    labels = set(calibration["registered_labels"])
    if {row["label"] for row, _, _ in records} != labels | {"unknown"}:
        raise ValueError("Held-out tests must include every registered class and unknown objects")
    memories = {}
    for mode in MODES:
        details = calibration["memories"][mode]
        memory_path = Path(details["path"])
        if sha256(memory_path) != details["sha256"] or sha256(memory_path.with_suffix(".json")) != details["metadata_sha256"]:
            raise ValueError("Frozen memory or rejection thresholds were modified")
        memory = LifelongPrototypeMemory.load(memory_path)
        if set(memory.samples) != labels or any(np.asarray(value).shape != (MODES[mode][1],)
                                               for values in memory.samples.values() for value in values):
            raise ValueError("Memory label/dimension mismatch for " + mode)
        memories[mode] = memory
    predictions = {mode: [] for mode in MODES}
    for row, features, _ in records:
        for mode, memory in memories.items():
            started = time.perf_counter()
            if features[mode] is None:
                predicted, similarity, margin = "unavailable", None, None
            else:
                predicted, similarity, margin, _, _ = memory.predict(features[mode])
            predictions[mode].append({"sample_id": row["sample_id"], "capture_group_id": row["capture_group_id"],
                                      "object_instance_id": row["object_instance_id"], "label": row["label"],
                                      "predicted": predicted, "similarity": similarity, "margin": margin,
                                      "cached_classification_ms": (time.perf_counter() - started) * 1000})
    paired_ids = {row["sample_id"] for row, features, _ in records if all(features[mode] is not None for mode in MODES)}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = [row for row, _, _ in records]
    overall_metrics = {mode: _metrics(value) for mode, value in predictions.items()}
    paired_metrics = {mode: _metrics([row for row in value if row["sample_id"] in paired_ids])
                      for mode, value in predictions.items()}
    paired_delta = None
    if paired_ids:
        paired_delta = (paired_metrics["rgb_geometry"]["accuracy_percent"]
                        - paired_metrics["rgb_multiview"]["accuracy_percent"])
    report = {**_base_report(path, selected, identity), "stage": "frozen_memory_held_out_sequence_comparison",
              "calibration_report": str(calibration_path), "calibration_report_sha256": sha256(calibration_path),
              "thresholds_refitted": False, "memories_updated": False,
              "overall_metrics": overall_metrics,
              "paired_available_sequence_ids": sorted(paired_ids),
              "paired_available_metrics": paired_metrics,
              "observed_geometry_minus_multiview_rgb_percent_points": overall_metrics["rgb_geometry"]["accuracy_percent"] - overall_metrics["rgb_multiview"]["accuracy_percent"],
              "paired_available_geometry_minus_multiview_rgb_percent_points": paired_delta,
              "observed_delta_scope": "Only these held-out sequences; availability and small-sample limitations must accompany the result.",
              "predictions": predictions, "general_recognition_improvement_demonstrated": False,
              "latency_scope": "Cached prototype classification only, not image encoding/reconstruction or end-to-end latency.",
              "limitations": ["Small same-instance tests do not establish broad semantic category recognition.",
                              "Candidate geometry and mask proposals remain unverified; no new physical accuracy claim.",
                              "Report full-set availability and errors as well as the paired available subset; never silently drop geometry failures."]}
    for details in calibration["memories"].values():
        if sha256(details["path"]) != details["sha256"] or sha256(Path(details["path"]).with_suffix(".json")) != details["metadata_sha256"]:
            raise ValueError("Frozen memories changed during evaluation")
    write_json(output / "sequence_comparison.json", report)
    fields = ("mode", "sample_id", "capture_group_id", "object_instance_id", "label", "predicted",
              "similarity", "margin", "cached_classification_ms")
    with (output / "sequence_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for mode, values in predictions.items():
            for value in values:
                writer.writerow({"mode": mode, **value})
    return report
