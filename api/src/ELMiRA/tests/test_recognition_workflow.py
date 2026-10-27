"""Synthetic offline plumbing checks, not real-object accuracy evidence."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from active_vision.cli import build_parser
from active_vision.common import read_json, write_json
from active_vision.recognition import ColorShapeEncoder, calibrate_dataset, evaluate_dataset
from active_vision.vendor.acf3d.memory import LifelongPrototypeMemory


class RecognitionWorkflowTests(unittest.TestCase):
    def make_dataset(self, root):
        rows = []
        for split in ("enroll", "calibration", "test"):
            for index, label in enumerate(("one", "two", "unknown")):
                if split == "enroll" and label == "unknown":
                    continue
                name = split + "_" + label
                image = np.zeros((40, 60, 3), np.uint8)
                image[10:30, 20:40, index] = 200
                image[0, 0] = len(rows) + 1
                mask = np.zeros(image.shape[:2], np.uint8)
                mask[10:30, 20:40] = 255
                cv2.imwrite(str(root / (name + ".png")), image)
                cv2.imwrite(str(root / (name + "_mask.png")), mask)
                rows.append({"image": name + ".png", "mask": name + "_mask.png",
                             "split": split, "session_id": split, "label": label,
                             "object_identity": split + "_novel" if label == "unknown" else label,
                             "mask_reviewed": True})
        write_json(root / "manifest.json", {"samples": rows})
        return rows

    def test_descriptor_ignores_background_and_integer_translation(self):
        encoder = ColorShapeEncoder()
        image = np.full((80, 100, 3), 30, np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[15:35, 25:45] = 255
        image[mask > 0] = (10, 200, 30)
        expected = encoder.encode(image, mask)
        image[mask == 0] = (255, 255, 0)
        np.testing.assert_array_equal(encoder.encode(image, mask), expected)
        moved = np.zeros_like(image)
        moved_mask = np.zeros_like(mask)
        moved_mask[45:65, 55:75] = 255
        moved[moved_mask > 0] = (10, 200, 30)
        np.testing.assert_array_equal(encoder.encode(moved, moved_mask), expected)
        self.assertEqual(expected.shape, (293,))
        self.assertAlmostEqual(float(np.linalg.norm(expected)), 1, places=6)

    def test_uniform_scaling_is_approximately_invariant(self):
        encoder = ColorShapeEncoder()
        features = []
        for size in (10, 20, 40):
            image = np.zeros((100, 100, 3), np.uint8)
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[10:10+size, 10:10+size] = 255
            image[mask > 0] = (10, 200, 30)
            features.append(encoder.encode(image, mask))
        self.assertGreater(float(features[0] @ features[-1]), .999)

    def test_bad_image_and_mask_rejected(self):
        encoder = ColorShapeEncoder()
        for image in (None, np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), float)):
            with self.assertRaises(ValueError):
                encoder.encode(image, np.zeros((10, 10), np.uint8))
        with self.assertRaises(ValueError):
            encoder.encode(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8))

    def test_calibration_does_not_open_or_encode_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_dataset(root)
            for row in rows:
                if row["split"] == "test":
                    (root / row["image"]).unlink()
                    (root / row["mask"]).unlink()
            encoder = ColorShapeEncoder()
            with patch.object(encoder, "encode", wraps=encoder.encode) as encoded:
                report = calibrate_dataset(root / "manifest.json", encoder, root / "fit")
            self.assertEqual(encoded.call_count, 5)
            self.assertEqual(report["calibration_correct"], 3)
            self.assertEqual(report["held_out_test_accuracy"], "not evaluated")
            self.assertFalse(report["real_robot_motion_commanded"])
            self.assertFalse(report["model_deployed"])
            self.assertFalse(report["encoder_training_performed"])
            self.assertEqual(report["enrolled_labels"], ["one", "two"])
            memory = LifelongPrototypeMemory.load(root / "fit/calibration_memory.npz")
            self.assertNotIn("unknown", memory.samples)
            self.assertTrue((root / "fit/source_manifest.json").is_file())
            self.assertFalse((root / "fit/recognition_evaluation.json").exists())

    def test_held_out_plumbing_is_explicitly_diagnostic_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            report = evaluate_dataset(root / "manifest.json", ColorShapeEncoder(), root / "test")
            self.assertEqual(report["correct"], 3)
            self.assertEqual(report["feature_mode"], "opencv_hsv_silhouette_baseline_v1")

    def test_identical_color_and_silhouette_cannot_reveal_identity(self):
        encoder = ColorShapeEncoder()
        image = np.zeros((40, 60, 3), np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[10:30, 20:40] = 255
        image[mask > 0] = (10, 200, 30)
        feature = encoder.encode(image, mask)
        memory = LifelongPrototypeMemory()
        memory.learn("object_a", feature)
        memory.learn("object_b", feature)
        self.assertEqual(memory.predict(feature)[0], "unknown")

    def test_unknown_identity_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_dataset(root)
            for row in rows:
                if row["label"] == "unknown":
                    row["object_identity"] = "same_physical_object"
            write_json(root / "manifest.json", {"samples": rows})
            with self.assertRaisesRegex(ValueError, "identity leakage"):
                calibrate_dataset(root / "manifest.json", ColorShapeEncoder(), root / "fit")

    def test_protected_development_rows_cannot_become_enrollment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_dataset(root)
            rows[0]["exclude_from_enrollment"] = True
            write_json(root / "manifest.json", {"samples": rows})
            with self.assertRaisesRegex(ValueError, "cannot enter enrollment"):
                calibrate_dataset(root / "manifest.json", ColorShapeEncoder(), root / "fit")

    def test_protected_development_rows_cannot_become_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_dataset(root)
            rows[-1]["exclude_from_final_test"] = True
            write_json(root / "manifest.json", {"samples": rows})
            with self.assertRaisesRegex(ValueError, "cannot enter held-out test"):
                evaluate_dataset(root / "manifest.json", ColorShapeEncoder(), root / "test")

    def test_changed_original_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_dataset(root)
            rows[0]["sha256"] = "wrong"
            write_json(root / "manifest.json", {"samples": rows})
            with self.assertRaisesRegex(ValueError, "hash differs"):
                calibrate_dataset(root / "manifest.json", ColorShapeEncoder(), root / "fit")

    def test_previous_calibration_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            calibrate_dataset(root / "manifest.json", ColorShapeEncoder(), root / "fit")
            before = (root / "fit/calibration_report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                calibrate_dataset(root / "manifest.json", ColorShapeEncoder(), root / "fit")
            self.assertEqual((root / "fit/calibration_report.json").read_bytes(), before)

    def test_cli_defaults_to_professor_not_control(self):
        parser = build_parser()
        args = parser.parse_args(["recognition-calibrate", "--manifest", "data.json", "--output", "new"])
        self.assertEqual(args.encoder, "professor-rgb")
        args = parser.parse_args(["recognition-calibrate", "--manifest", "data.json", "--output", "new",
                                  "--encoder", "color-shape"])
        self.assertIsNone(args.checkpoint)


if __name__ == "__main__":
    unittest.main()
