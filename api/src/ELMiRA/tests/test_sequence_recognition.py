"""Offline split-integrity and frozen-memory tests; no robot/camera interfaces."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from active_vision.common import read_json, sha256, write_json
from active_vision.fusion_replay import unit_vector
from active_vision.sequence_recognition import (MODES, add_sequence, calibrate, dataset_rows, enroll,
                                               evaluate, load_features)
from active_vision.vendor.acf3d.memory import LifelongPrototypeMemory


class SequenceRecognitionTests(unittest.TestCase):
    def sample(self, root, index, label, split, geometry=True):
        folder = root / ("capture_%02d" % index)
        folder.mkdir()
        checkpoint = root / "checkpoint.bin"
        if not checkpoint.exists():
            checkpoint.write_bytes(b"Synthetic test fixture: no neural model loaded")
        write_json(folder / "source_manifest.json", {"synthetic_fixture_id": index})
        write_json(folder / "calibration.json", {"synthetic_fixture": True})
        steps = []
        rng = np.random.default_rng(index)
        for view in range(3):
            image = folder / ("image_%d.png" % view)
            mask = folder / ("mask_%d.png" % view)
            cv2.imwrite(str(image), rng.integers(0, 255, (20, 20, 3), np.uint8))
            value = np.zeros((20, 20), np.uint8)
            value[4:16, 4:16] = 255
            cv2.imwrite(str(mask), value)
            steps.append({"source_image": str(image), "source_image_sha256": sha256(image),
                          "source_mask": str(mask), "source_mask_sha256": sha256(mask)})
        arrays = {}
        coordinate = {"can": 0, "apple": 1, "unknown": 2}[label]
        for mode, (key, dimension) in MODES.items():
            if mode == "rgb_geometry" and not geometry:
                continue
            vector = np.zeros(dimension, np.float32)
            vector[coordinate] = 1
            vector[3] = .01 if split == "enroll" else .025
            arrays[key] = unit_vector(vector)
        feature = folder / "recognition_features.npz"
        np.savez_compressed(str(feature), **arrays)
        report = {"real_robot_motion_commanded": False, "sources_unchanged": True,
                  "network_weights_unchanged": True, "future_images_used_in_prefix_geometry": False,
                  "steps": steps, "source_view_count": 3, "rgb_encoded_view_count": 3,
                  "fused_step_count": 2 if geometry else 0,
                  "manifest": str(folder / "source_manifest.json"),
                  "manifest_sha256": sha256(folder / "source_manifest.json"),
                  "calibration": str(folder / "calibration.json"),
                  "calibration_sha256": sha256(folder / "calibration.json"),
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "camera_id": "SYNTHETIC_TEST_CAMERA", "visual_preprocessing": "synthetic fixture",
                  "full_embedding_formula": "synthetic fixture matching dimensions only"}
        report_path = folder / "fusion_replay_report.json"
        write_json(report_path, report)
        instance = label + "_instance_01" if label != "unknown" else split + "_unknown_01"
        return {"sample_id": "sample_%02d" % index, "capture_group_id": "group_%02d" % index,
                "object_instance_id": instance, "label": label, "split": split,
                "replay_report": str(report_path), "replay_report_sha256": sha256(report_path),
                "feature_file": str(feature), "feature_file_sha256": sha256(feature),
                "source_image_sha256s": [step["source_image_sha256"] for step in steps]}

    def fixture(self, root):
        rows = []
        for split in ("enroll", "calibration", "test"):
            for label in (("can", "apple") if split == "enroll" else ("can", "apple", "unknown")):
                rows.append(self.sample(root, len(rows), label, split))
        manifest = root / "dataset.json"
        write_json(manifest, {"dataset_id": "synthetic_unit_test_only", "samples": rows})
        return manifest, rows

    def test_registration_creates_separate_dimensions_and_no_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            result = enroll(manifest, root / "registration")
            self.assertEqual(result["stage"], "registered_uncalibrated")
            self.assertEqual(result["registered_labels"], ["apple", "can"])
            self.assertIsNone(result["held_out_accuracy_percent"])
            self.assertFalse(result["thresholds_calibrated"])
            self.assertFalse(result["real_robot_motion_commanded"])
            for mode, (_, dimension) in MODES.items():
                memory = LifelongPrototypeMemory.load(Path(result["memories"][mode]["path"]))
                self.assertEqual(memory.prototype("can").shape, (dimension,))
            with self.assertRaises(FileExistsError):
                enroll(manifest, root / "registration")

    def test_add_sequence_generates_checksums_and_does_not_overwrite_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.sample(root, 0, "can", "enroll")
            path = root / "new_dataset.json"
            result = add_sequence(path, row["replay_report"], "can", "enroll", "can_enroll_01", "can_physical_01")
            self.assertEqual(result["sample_count"], 1)
            self.assertIsNone(result["held_out_accuracy_percent"])
            saved = read_json(path)["samples"][0]
            self.assertEqual(saved["feature_file_sha256"], sha256(saved["feature_file"]))
            self.assertEqual(saved["source_image_sha256s"], row["source_image_sha256s"])
            before = sha256(path)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                add_sequence(path, row["replay_report"], "can", "enroll", "can_enroll_01", "can_physical_01")
            self.assertEqual(sha256(path), before)
            with self.assertRaisesRegex(ValueError, "Source image reused"):
                add_sequence(path, row["replay_report"], "can", "test", "renamed_test", "can_physical_01")
            self.assertEqual(sha256(path), before)

    def test_calibration_never_opens_test_feature_or_replay_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.fixture(root)
            for row in rows:
                if row["split"] == "test":
                    row["replay_report"] = str(root / "NOT_PRESENT" / row["sample_id"] / "report.json")
                    row["feature_file"] = str(root / "NOT_PRESENT" / row["sample_id"] / "recognition_features.npz")
            write_json(manifest, {"samples": rows})
            result = calibrate(manifest, root / "calibration")
            self.assertTrue(result["thresholds_calibrated"])
            self.assertFalse(result["test_replay_or_feature_files_opened"])
            self.assertEqual(len(result["selected_sample_ids"]), 5)
            self.assertIsNone(result["held_out_accuracy_percent"])

    def test_evaluation_uses_frozen_memories_without_fit_or_enrollment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            calibration = calibrate(manifest, root / "calibration")
            with patch("active_vision.sequence_recognition._fit_memory", side_effect=AssertionError("No test refit")), \
                 patch.object(LifelongPrototypeMemory, "learn", side_effect=AssertionError("No test learning")), \
                 patch.object(LifelongPrototypeMemory, "save", side_effect=AssertionError("No test memory update")), \
                 patch("subprocess.run", side_effect=AssertionError("No hardware calls")):
                result = evaluate(manifest, root / "calibration", root / "evaluation")
            self.assertFalse(result["thresholds_refitted"])
            self.assertFalse(result["memories_updated"])
            self.assertEqual(len(result["paired_available_sequence_ids"]), 3)
            for mode, details in calibration["memories"].items():
                self.assertEqual(sha256(details["path"]), details["sha256"])
                self.assertEqual(result["overall_metrics"][mode]["sequences"], 3)
                self.assertEqual(result["overall_metrics"][mode]["correct"], 3)
            self.assertFalse(result["general_recognition_improvement_demonstrated"])

    def test_cross_split_image_reuse_and_relabelled_duplicate_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.fixture(root)
            rows[-1]["source_image_sha256s"][0] = rows[0]["source_image_sha256s"][0]
            rows[-1]["capture_group_id"] = "renamed_but_same_image"
            write_json(manifest, {"samples": rows})
            with self.assertRaisesRegex(ValueError, "Source image reused"):
                dataset_rows(manifest)

    def test_duplicate_capture_groups_and_unknown_identity_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.fixture(root)
            rows[-1]["capture_group_id"] = rows[0]["capture_group_id"]
            write_json(manifest, {"samples": rows})
            with self.assertRaisesRegex(ValueError, "capture group"):
                dataset_rows(manifest)
            rows[-1]["capture_group_id"] = "fresh_group"
            unknown_cal = next(row for row in rows if row["label"] == "unknown" and row["split"] == "calibration")
            rows[-1]["object_instance_id"] = unknown_cal["object_instance_id"]
            write_json(manifest, {"samples": rows})
            with self.assertRaisesRegex(ValueError, "Unknown physical objects must differ"):
                dataset_rows(manifest)

    def test_geometry_unavailable_is_counted_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.fixture(root)
            missing = self.sample(root, 20, "unknown", "test", geometry=False)
            rows[-1] = missing
            write_json(manifest, {"samples": rows})
            calibrate(manifest, root / "calibration")
            result = evaluate(manifest, root / "calibration", root / "evaluation")
            metrics = result["overall_metrics"]["rgb_geometry"]
            self.assertEqual(metrics["sequences"], 3)
            self.assertEqual(metrics["unavailable_sequences"], 1)
            self.assertAlmostEqual(metrics["accuracy_percent"], 200 / 3)
            self.assertEqual(len(result["paired_available_sequence_ids"]), 2)
            for mode in MODES:
                self.assertEqual(result["paired_available_metrics"][mode]["sequences"], 2)

    def test_changed_frozen_thresholds_and_development_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.fixture(root)
            report = calibrate(manifest, root / "calibration")
            memory_meta = Path(report["memories"]["rgb_geometry"]["path"]).with_suffix(".json")
            data = read_json(memory_meta)
            data["similarity_threshold"] = -1
            write_json(memory_meta, data)
            with self.assertRaisesRegex(ValueError, "modified"):
                evaluate(manifest, root / "calibration", root / "evaluation")
            rows[0]["capture_group_id"] = "changed_development_group"
            write_json(manifest, {"samples": rows})
            with self.assertRaisesRegex(ValueError, "Development data changed"):
                evaluate(manifest, root / "calibration", root / "evaluation")

    def test_missing_independent_calibration_is_not_a_fake_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.fixture(root)
            write_json(manifest, {"samples": rows[:2]})
            with self.assertRaisesRegex(ValueError, "Calibration split"):
                calibrate(manifest, root / "calibration")
            self.assertFalse((root / "calibration").exists())
            register = enroll(manifest, root / "registration")
            self.assertEqual(register["stage"], "registered_uncalibrated")
            with self.assertRaises(FileNotFoundError):
                evaluate(manifest, root / "registration", root / "evaluation")

    def test_mixed_camera_identity_and_wrong_vector_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = self.fixture(root)
            path = Path(rows[1]["replay_report"])
            report = read_json(path)
            report["camera_id"] = "WRONG_CAMERA"
            write_json(path, report)
            rows[1]["replay_report_sha256"] = sha256(path)
            write_json(manifest, {"samples": rows})
            with self.assertRaisesRegex(ValueError, "Mixed cameras"):
                enroll(manifest, root / "registration")
            feature = Path(rows[0]["feature_file"])
            with np.load(str(feature)) as data:
                arrays = {key: data[key].copy() for key in data.files}
            arrays["full_fused_mean"] = np.ones(64, np.float32)
            np.savez_compressed(str(feature), **arrays)
            rows[0]["feature_file_sha256"] = sha256(feature)
            with self.assertRaisesRegex(ValueError, "dimension"):
                load_features(rows[0])

    def test_original_source_and_feature_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.fixture(root)
            report = read_json(rows[0]["replay_report"])
            Path(report["steps"][0]["source_image"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Original source image changed"):
                load_features(rows[0])
            Path(rows[1]["feature_file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after dataset"):
                load_features(rows[1])


if __name__ == "__main__":
    unittest.main()
