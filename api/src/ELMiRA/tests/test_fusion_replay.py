"""No-hardware tests for recorded RGB/geometry recognition integration."""
from collections import Counter
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from active_vision.common import read_json, sha256, write_json
from active_vision.fusion_replay import (RGBGeometryEncoder, load_fused_memory,
                                        load_sequence, replay, unit_vector, validate_points)
from active_vision.vendor.acf3d.frontend_v8 import prepare_rgb_observation
from active_vision.vendor.acf3d.memory import LifelongPrototypeMemory


class FakeEncoder:
    def __init__(self, _checkpoint):
        self.model = SimpleNamespace(state_dict=lambda: {})
        self.torch = SimpleNamespace(equal=np.array_equal)
        self.calls = Counter()

    def encode(self, _image, _mask):
        self.calls["visual"] += 1
        return unit_vector(np.arange(1, 65, dtype=np.float32))

    def encode_fused(self, _image, _mask, points):
        validate_points(points)
        self.calls.update(["visual", "points", "fusion"])
        return unit_vector(np.arange(1, 102, dtype=np.float32)), {
            "point_input_shape": [256, 3], "signature_dimension": 37,
            "nonzero_point_input_values": 768, "visual_input_shape": [2, 64, 64],
            "gate": [.5, .5], "visual_embedding": np.ones(64, np.float32),
            "point_embedding": np.ones(64, np.float32)}

    def close(self):
        pass


class FusionReplayTests(unittest.TestCase):
    def fixture(self, root, count=3):
        calibration = {
            "K": [[120, 0, 80], [0, 120, 60], [0, 0, 1]], "D": [0] * 5,
            "width": 160, "height": 120, "distortion_model": "plumb_bob",
            "camera_id": "SYNTHETIC", "optical_frame": "test_optical", "verified": False}
        write_json(root / "calibration.json", calibration)
        mask = np.zeros((120, 160), np.uint8)
        mask[40:80, 60:100] = 255
        cv2.imwrite(str(root / "mask.png"), mask)
        manifest = {"object_static": True, "units": "metres", "world_frame": "fixed_test",
                    "calibration": "calibration.json", "target_object": "test object",
                    "observations": []}
        for index in range(count):
            image = np.full((120, 160, 3), 20, np.uint8)
            image[40:80, 60:100] = (20, 80 + index * 20, 180)
            path = root / ("image_%d.png" % index)
            cv2.imwrite(str(path), image)
            pose = np.eye(4)
            pose[0, 3] = index * .03
            manifest["observations"].append({
                "image": path.name, "mask": "mask.png", "image_sha256": sha256(path),
                "mask_sha256": sha256(root / "mask.png"), "camera_id": "SYNTHETIC",
                "timestamp_sec": 100. + index, "pose_verified": False,
                "mask_reviewed": False, "pose_source": "fixed_board_native_corner_pnp",
                "pose_reprojection_rms_px": .1, "T_wc": pose.tolist()})
        write_json(root / "manifest.json", manifest)
        (root / "checkpoint.bin").write_bytes(b"mocked encoder; not a model checkpoint")
        return manifest

    def test_unverified_inputs_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "unverified"):
                load_sequence(root / "manifest.json")
            _, _, _, calibration, rows = load_sequence(root / "manifest.json", True)
            self.assertEqual(len(rows), 3)
            self.assertFalse(calibration.record["verified"])

    def test_missing_poses_cannot_be_recovered_from_ordinary_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "ordinary.json", {"samples": [{"image": "photo.png"}]})
            with self.assertRaises(ValueError):
                load_sequence(root / "ordinary.json", True)
            manifest = self.fixture(root)
            del manifest["observations"][0]["T_wc"]
            write_json(root / "manifest.json", manifest)
            with self.assertRaises(KeyError):
                load_sequence(root / "manifest.json", True)

    def test_checksum_and_camera_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest["observations"][0]["image_sha256"] = "wrong"
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_sequence(root / "manifest.json", True)
            manifest = self.fixture(root)
            manifest["observations"][0]["camera_id"] = "OTHER CAMERA"
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "mix cameras"):
                load_sequence(root / "manifest.json", True)

    def test_commanded_angles_and_bad_candidate_pnp_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest["observations"][0]["pose_source"] = "commanded_head_angles"
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "not commanded"):
                load_sequence(root / "manifest.json", True)
            for error in (float("nan"), -1., 1.1):
                manifest = self.fixture(root)
                manifest["observations"][0]["pose_reprojection_rms_px"] = error
                if np.isfinite(error):
                    write_json(root / "manifest.json", manifest)
                    with self.assertRaisesRegex(ValueError, "reprojection gate"):
                        load_sequence(root / "manifest.json", True)
                else:
                    # The JSON writer already refuses nonfinite provenance.
                    with self.assertRaises(ValueError):
                        write_json(root / "manifest.json", manifest)

    def test_duplicates_out_of_order_and_moving_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest["observations"][1] = dict(manifest["observations"][0])
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_sequence(root / "manifest.json", True)
            manifest = self.fixture(root)
            manifest["observations"][1]["timestamp_sec"] = 99
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "capture order"):
                load_sequence(root / "manifest.json", True)
            manifest["object_static"] = False
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "stationary"):
                load_sequence(root / "manifest.json", True)

    def test_empty_or_degenerate_points_are_not_fake_geometry(self):
        for points in (np.empty((0, 3)), np.ones((10, 3)), np.ones((9, 3)),
                       np.full((10, 3), np.nan), np.ones((10, 2))):
            with self.assertRaises(ValueError):
                validate_points(points)
        points = np.random.default_rng(0).normal(size=(12, 3))
        self.assertEqual(validate_points(points).shape, (12, 3))

    def test_rgb_only_memory_is_not_accepted_for_full_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb_memory.npz"
            memory = LifelongPrototypeMemory()
            memory.learn("item", np.ones(64, np.float32))
            memory.save(path)
            before = sha256(path)
            with self.assertRaisesRegex(ValueError, "101D"):
                load_fused_memory(path)
            self.assertEqual(sha256(path), before)

    def test_replay_uses_only_prefixes_and_never_claims_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            before = sha256(root / "manifest.json")
            prefix_lengths = []

            def reconstruct(prefix_path, output, _detector):
                prefix = read_json(prefix_path)
                prefix_lengths.append(len(prefix["observations"]))
                expected = manifest["observations"][:len(prefix["observations"])]
                self.assertEqual([Path(r["image"]).name for r in prefix["observations"]],
                                 [r["image"] for r in expected])
                output.mkdir()
                points = np.random.default_rng(3).normal(size=(20, 3)).astype(np.float32)
                np.savez_compressed(str(output / "experimental_object_cloud.npz"), xyz=points)
                return {"status": "experimental_sparse_cloud", "point_count": 20,
                        "experimental_visual_hull": {"quality_status": "unreliable_clipped_volume"}}

            with patch("active_vision.fusion_replay.RGBGeometryEncoder", FakeEncoder), \
                 patch("active_vision.fusion_replay.reconstruct_preview", side_effect=reconstruct), \
                 patch("subprocess.run", side_effect=AssertionError("No subprocess/hardware calls")):
                result = replay(root / "manifest.json", root / "checkpoint.bin", root / "out",
                                allow_experimental=True)
            self.assertEqual(prefix_lengths, [2, 3])
            self.assertEqual(result["fused_step_count"], 2)
            self.assertEqual(result["steps"][0]["point_count"], 0)
            self.assertEqual(result["steps"][0]["status"], "rgb_only_waiting_for_second_view")
            self.assertFalse(result["future_images_used_in_prefix_geometry"])
            self.assertFalse(result["real_robot_motion_commanded"])
            self.assertFalse(result["recognition_improvement_demonstrated"])
            self.assertIsNone(result["recognition_accuracy_percent"])
            self.assertFalse(result["input_calibration_verified"])
            self.assertFalse(result["metric_accuracy_verified"])
            self.assertEqual(sha256(root / "manifest.json"), before)
            with np.load(str(root / "out/recognition_features.npz")) as data:
                self.assertEqual(data["rgb_by_view"].shape, (3, 64))
                self.assertEqual(data["full_fused_mean"].shape, (101,))

    def test_insufficient_prefix_clouds_return_partial_evidence_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)

            def insufficient(_prefix, output, _detector):
                output.mkdir()
                np.savez_compressed(str(output / "experimental_object_cloud.npz"), xyz=np.empty((0, 3)))
                return {"status": "insufficient_points", "point_count": 0,
                        "experimental_visual_hull": {"status": "insufficient_sparse_seed"}}

            with patch("active_vision.fusion_replay.RGBGeometryEncoder", FakeEncoder), \
                 patch("active_vision.fusion_replay.reconstruct_preview", side_effect=insufficient):
                result = replay(root / "manifest.json", root / "checkpoint.bin", root / "out",
                                allow_experimental=True)
            self.assertEqual(result["status"], "no_usable_multiview_fusion")
            self.assertEqual(result["fused_step_count"], 0)
            self.assertEqual(result["rgb_encoded_view_count"], 3)
            with self.assertRaises(FileExistsError):
                replay(root / "manifest.json", root / "checkpoint.bin", root / "out",
                       allow_experimental=True)

    def test_real_checkpoint_matches_original_v8_fusion_formula(self):
        api = Path(__file__).resolve().parents[3]
        checkpoint = api / "model_checkpoints/active_vision_v8/single_rgb_model.pt"
        if not checkpoint.is_file():
            self.skipTest("Locally supplied professor checkpoint not installed")
        encoder = RGBGeometryEncoder(checkpoint)
        try:
            rng = np.random.default_rng(9)
            image = rng.integers(0, 255, (96, 128, 3), np.uint8)
            mask = np.zeros((96, 128), np.uint8)
            mask[20:75, 30:100] = 255
            points = rng.normal(size=(50, 3)).astype(np.float32)
            fused, parts = encoder.encode_fused(image, mask, points)
            observation = prepare_rgb_observation(cv2.bitwise_and(image, image, mask=mask),
                                                  mask, points, np.random.default_rng(0))
            with encoder.torch.no_grad():
                _, neural, _ = encoder.model(encoder.torch.from_numpy(observation["visual"][None]),
                                             encoder.torch.from_numpy(observation["points"][None]),
                                             return_parts=True)
            expected = unit_vector(np.concatenate([neural[0].numpy(), .75 * observation["signature"]]))
            np.testing.assert_allclose(fused, expected, atol=1e-6)
            np.testing.assert_allclose(encoder.encode(image, mask), parts["visual_embedding"], atol=1e-6)
            self.assertEqual(fused.shape, (101,))
            self.assertGreater(parts["nonzero_point_input_values"], 0)
            self.assertFalse(encoder.model.training)
        finally:
            encoder.close()


if __name__ == "__main__":
    unittest.main()
