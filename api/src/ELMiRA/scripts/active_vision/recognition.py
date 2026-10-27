"""Reuse the professor's RGB encoder with labelled real-object prototypes.

The supplied synthetic classifier is NOT a real-object recognition benchmark.
Enrollment/calibration/test sessions must be disjoint, including unknown objects.
"""
from collections import Counter
from pathlib import Path
import shutil

import cv2
import numpy as np

from .common import read_json, sha256, write_json
from .masks import validate_mask
from .vendor.acf3d.frontend_v8 import prepare_rgb_observation
from .vendor.acf3d.memory import LifelongPrototypeMemory


class RGBEncoder:
    feature_mode = "masked_rgb_visual_branch_only"

    def __init__(self, checkpoint):
        import torch
        from .vendor.acf3d.model import DualStreamConformal3DNet
        self.torch = torch
        torch.set_num_threads(1)
        checkpoint = Path(checkpoint)
        # Never fall back to unrestricted pickle loading of a supplied checkpoint.
        saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.model = DualStreamConformal3DNet(num_classes=len(saved["base_classes"]))
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()
        self.checkpoint_sha256 = sha256(checkpoint)

    def encode(self, bgr, mask):
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("RGB feature input must be a readable uint8 BGR image")
        mask = validate_mask(mask, bgr.shape)
        # Remove background before reusing the BGR frontend. No fictitious 3D
        # point tokens are fed to the point branch to manufacture confidence.
        isolated = cv2.bitwise_and(bgr, bgr, mask=mask)
        observation = prepare_rgb_observation(isolated, mask, np.empty((0, 3), np.float32))
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(observation["visual"]).unsqueeze(0)
            feature = self.model.vproj(self.model.visual(tensor))[0].numpy()
        feature = feature / max(np.linalg.norm(feature), 1e-8)
        if not np.isfinite(feature).all():
            raise ValueError("Non-finite RGB feature")
        return feature.astype(np.float32)


class ColorShapeEncoder:
    """Explicit untrained RGB control, NOT the professor's learned encoder."""

    feature_mode = "opencv_hsv_silhouette_baseline_v1"

    def __init__(self):
        self.checkpoint_sha256 = sha256(Path(__file__))

    def encode(self, bgr, mask):
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR image")
        mask = validate_mask(mask, bgr.shape)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [18, 4, 4],
                            [0, 180, 0, 256, 0, 256]).ravel()
        hist = np.sqrt(hist / max(float(hist.sum()), 1))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        shape = np.array([w / max(w, h), h / max(w, h), area / (w*h),
                          min(1, 4*np.pi*area / max(perimeter**2, 1e-8)),
                          area / max(hull_area, 1e-8)], np.float32)
        shape /= max(float(np.linalg.norm(shape)), 1e-8)
        feature = np.concatenate([np.sqrt(.85)*hist, np.sqrt(.15)*shape])
        return (feature / max(float(np.linalg.norm(feature)), 1e-8)).astype(np.float32)


def _dataset_rows(manifest_path, include_test=True):
    manifest_path = Path(manifest_path)
    rows = read_json(manifest_path)["samples"]
    if not rows:
        raise ValueError("Empty recognition dataset")
    sessions, hashes = {}, set()
    unknown_objects = {split: set() for split in ("calibration", "test")}
    selected = []
    for row in rows:
        split, session = row["split"], row["session_id"]
        if split not in ("enroll", "calibration", "test") or not session:
            raise ValueError("Each sample needs enroll/calibration/test and a real session_id")
        if session in sessions and sessions[session] != split:
            raise ValueError("Session leakage across data splits")
        sessions[session] = split
        if row.get("exclude_from_enrollment") and split == "enroll":
            raise ValueError("Calibration/test-only image cannot enter enrollment")
        if row.get("exclude_from_final_test") and split == "test":
            raise ValueError("Development image cannot enter held-out test")
        if row["label"] == "unknown" and split in unknown_objects and row.get("object_identity"):
            unknown_objects[split].add(row["object_identity"])
        if not include_test and split == "test":
            continue
        image = manifest_path.parent / row["image"]
        if row.get("mask_reviewed") is not True:
            raise ValueError("Review real-object masks before enrollment or evaluation")
        digest = sha256(image)
        if row.get("sha256") and row["sha256"] != digest:
            raise ValueError("Image hash differs from captured original")
        if digest in hashes:
            raise ValueError("Duplicate image content in recognition dataset")
        hashes.add(digest)
        selected.append(row)
    if unknown_objects["calibration"] & unknown_objects["test"]:
        raise ValueError("Unknown calibration/test object identity leakage")
    return selected


def _encode_rows(rows, manifest_path, encoder):
    base = Path(manifest_path).parent
    features = []
    for row in rows:
        image = cv2.imread(str(base / row["image"]))
        if image is None:
            raise ValueError("Cannot read recognition image")
        mask = validate_mask(cv2.imread(str(base / row["mask"]), 0), image.shape)
        feature = np.asarray(encoder.encode(image, mask), np.float32)
        if feature.ndim != 1 or not np.isfinite(feature).all() or np.linalg.norm(feature) < 1e-8:
            raise ValueError("Invalid recognition embedding")
        features.append((row, feature))
    return features


def _fit_memory(features):
    memory = LifelongPrototypeMemory(margin_threshold=.045)
    for row, feature in features:
        if row["split"] == "enroll":
            if row["label"] == "unknown":
                raise ValueError("Do not enroll unknown objects as a known class")
            memory.learn(row["label"], feature)
    if len(memory.samples) < 2:
        raise ValueError("Need >=2 labelled real-object classes; one dice sequence is not a recognition test")
    calibration = [(r, f) for r, f in features if r["split"] == "calibration"]
    if not any(r["label"] == "unknown" for r, _ in calibration):
        raise ValueError("Calibration split needs held-out unknown objects")
    if not set(memory.samples).issubset({r["label"] for r, _ in calibration}):
        raise ValueError("Calibration split needs every enrolled class")
    # Select a global rejection threshold on calibration data only.
    best = (-1, 1.01)
    for threshold in np.linspace(-1, 1.01, 202):
        memory.similarity_threshold = float(threshold)
        accuracy = sum(memory.predict(f)[0] == r["label"] for r, f in calibration)/len(calibration)
        if (accuracy, threshold) > best:
            best = (accuracy, float(threshold))
    memory.similarity_threshold = best[1]
    return memory, calibration


def _encoder_metadata(encoder):
    result = {"encoder_sha256": encoder.checkpoint_sha256,
              "feature_mode": getattr(encoder, "feature_mode", "masked_rgb_visual_branch_only"),
              "recognition_code_sha256": sha256(Path(__file__)),
              "real_robot_motion_commanded": False}
    if hasattr(encoder, "provenance"):
        result["encoder_details"] = encoder.provenance
    return result


def calibrate_dataset(manifest_path, encoder, output):
    """Freeze development parameters without decoding/encoding any test images."""
    rows = _dataset_rows(manifest_path, include_test=False)
    features = _encode_rows(rows, manifest_path, encoder)
    memory, calibration = _fit_memory(features)
    predictions = [{"image": row["image"], "session_id": row["session_id"],
                    "label": row["label"], "predicted": memory.predict(feature)[0]}
                   for row, feature in calibration]
    per_class = {}
    for label in sorted({p["label"] for p in predictions}):
        subset = [p for p in predictions if p["label"] == label]
        per_class[label] = {"frames": len(subset),
                            "correct": sum(p["predicted"] == label for p in subset),
                            "predicted_counts": dict(Counter(p["predicted"] for p in subset))}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "source_manifest.json")
    memory.save(output / "calibration_memory.npz")
    report = {**_encoder_metadata(encoder), "stage": "calibration_only_not_held_out_test",
              "calibration_count": len(calibration),
              "calibration_correct": sum(p["label"] == p["predicted"] for p in predictions),
              "per_class_calibration": per_class, "predictions": predictions,
              "calibrated_similarity_threshold": memory.similarity_threshold,
              "margin_threshold": memory.margin_threshold,
              "enrolled_labels": sorted(memory.samples), "encoder_training_performed": False,
              "dataset_sha256": sha256(output / "source_manifest.json"),
              "memory_sha256": sha256(output / "calibration_memory.npz"),
              "held_out_test_accuracy": "not evaluated", "model_deployed": False,
              "limitation": "Development-set fit only; correlated frames are not independent trials. "
                            "Manually assisted masks do not demonstrate autonomous detection."}
    write_json(output / "calibration_report.json", report)
    return report


def evaluate_dataset(manifest_path, encoder, output):
    rows = _dataset_rows(manifest_path)
    development = _encode_rows([r for r in rows if r["split"] != "test"], manifest_path, encoder)
    memory, _ = _fit_memory(development)
    features = _encode_rows([r for r in rows if r["split"] == "test"], manifest_path, encoder)
    tests = [(r, f) for r, f in features if r["split"] == "test"]
    if not tests:
        raise ValueError("A separate test session is required")
    if not (set(memory.samples) | {"unknown"}).issubset({r["label"] for r, _ in tests}):
        raise ValueError("Test sessions need every enrolled class and unknown objects")
    predictions = [{"image": r["image"], "session_id": r["session_id"], "label": r["label"],
                    "predicted": memory.predict(f)[0]} for r, f in tests]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    memory.save(output / "real_object_memory.npz")
    report = {"test_count": len(tests), "correct": sum(r["label"] == r["predicted"] for r in predictions),
              "test_accuracy": sum(r["label"] == r["predicted"] for r in predictions)/len(tests),
              "calibrated_similarity_threshold": memory.similarity_threshold, "predictions": predictions,
              **_encoder_metadata(encoder),
              "session_splits": {r["session_id"]: r["split"] for r in rows},
              "dataset_sha256": sha256(manifest_path), "memory_sha256": sha256(output / "real_object_memory.npz"),
              "margin_threshold": memory.margin_threshold}
    report["per_class"] = {}
    for label in sorted({r["label"] for r in predictions}):
        subset = [r for r in predictions if r["label"] == label]
        report["per_class"][label] = {"count": len(subset),
                                      "correct": sum(r["predicted"] == label for r in subset)}
    write_json(output / "recognition_evaluation.json", report)
    return report
