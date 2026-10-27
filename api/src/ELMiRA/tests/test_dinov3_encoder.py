"""Offline plumbing with synthetic tensors/mocked HF loader, NOT DINO accuracy."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from active_vision.cli import build_parser, main
from active_vision.common import read_json, write_json
from active_vision.dinov3_encoder import (
    DINOv3Encoder, FEATURE_MODE, _load_backend, _snapshot_files,
    prepare_object_rgb, runtime_status,
)
from active_vision.recognition import _encoder_metadata


class StubModel(torch.nn.Module):
    """Deterministic fake output. Never passed off as a pretrained backbone."""

    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(model_type="dinov3_vit", hidden_size=384)

    def forward(self, **inputs):
        self.inference_mode_used = torch.is_inference_mode_enabled()
        result = torch.full((1, 3, 384), -100.)
        result[0, 0] = torch.arange(1, 385)
        return SimpleNamespace(last_hidden_state=result)


class DINOv3EncoderTests(unittest.TestCase):
    def snapshot(self, root):
        write_json(root / "config.json", {
            "model_type": "dinov3_vit", "hidden_size": 384, "patch_size": 16,
            "num_hidden_layers": 12, "num_attention_heads": 6})
        write_json(root / "preprocessor_config.json", {
            "image_processor_type": "DINOv3ViTImageProcessorFast"})
        # A real safetensors container, but fake data; the HF loader is mocked.
        from safetensors.numpy import save_file
        save_file({"stub": np.ones(1, np.float32)}, str(root / "model.safetensors"))

    def image(self):
        bgr = np.full((40, 60, 3), 7, np.uint8)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        mask[10:30, 20:30] = 255
        bgr[mask > 0] = (10, 20, 230)
        return bgr, mask

    def test_rgb_keeps_color_aspect_ratio_and_black_padding(self):
        bgr, mask = self.image()
        before = bgr.copy()
        rgb = prepare_object_rgb(bgr, mask)
        self.assertEqual(rgb.shape, (25, 25, 3))
        np.testing.assert_array_equal(rgb[2:22, 7:17], np.tile([230, 20, 10], (20, 10, 1)))
        self.assertEqual(np.count_nonzero(np.any(rgb != 0, axis=2)), 200)
        np.testing.assert_array_equal(bgr, before)

    def test_roi_translation_and_background_do_not_change_input(self):
        bgr, mask = self.image()
        expected = prepare_object_rgb(bgr, mask)
        moved = np.full_like(bgr, 255)
        moved_mask = np.zeros_like(mask)
        moved_mask[15:35, 40:50] = 255
        moved[moved_mask > 0] = (10, 20, 230)
        np.testing.assert_array_equal(prepare_object_rgb(moved, moved_mask), expected)

    def test_bad_image_mask_and_padding_are_rejected(self):
        bgr, mask = self.image()
        for bad in (None, bgr[:, :, 0], bgr.astype(float)):
            with self.assertRaises(ValueError):
                prepare_object_rgb(bad, mask)
        for bad in (np.zeros_like(mask), np.ones((2, 2), np.uint8), mask/2):
            with self.assertRaises(ValueError):
                prepare_object_rgb(bgr, bad)
        for value in (0, float("nan"), float("inf"), 3):
            with self.assertRaises(ValueError):
                prepare_object_rgb(bgr, mask, value)

    def test_missing_snapshot_never_calls_backend(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "active_vision.dinov3_encoder._load_backend") as backend:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                DINOv3Encoder(directory)
            backend.assert_not_called()

    def test_wrong_backbone_or_custom_code_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snapshot(root)
            config = read_json(root / "config.json")
            for change in ({"model_type": "dinov2"}, {"hidden_size": 768},
                           {"auto_map": {"AutoModel": "custom.Model"}}):
                write_json(root / "config.json", {**config, **change})
                with self.assertRaises(ValueError):
                    _snapshot_files(root)

    def test_pickle_only_weights_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snapshot(root)
            (root / "model.safetensors").rename(root / "pytorch_model.bin")
            with self.assertRaisesRegex(ValueError, "model.safetensors"):
                _snapshot_files(root)

    def test_preflight_reports_old_runtime_without_loading_or_network(self):
        versions = {"torch": "2.4.1", "torchvision": "0.19.1", "transformers": "4.34.1",
                    "safetensors": "0.5.3"}
        with patch("active_vision.dinov3_encoder.metadata.version", side_effect=versions.__getitem__), \
                patch("active_vision.dinov3_encoder.sys.version_info", (3, 8, 10)):
            result = runtime_status()
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(any("Python" in reason for reason in result["blockers"]))
        self.assertTrue(any("transformers" in reason for reason in result["blockers"]))
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["automatic_downloads"])
        self.assertFalse(result["real_robot_motion_commanded"])

    def test_loader_is_local_safetensors_only_and_freezes_model(self):
        model, processor = StubModel(), Mock()
        auto_model = SimpleNamespace(from_pretrained=Mock(return_value=(model, {})))
        auto_processor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        module = SimpleNamespace(AutoModel=auto_model, AutoImageProcessor=auto_processor)
        with patch.dict(sys.modules, transformers=module), \
                patch("active_vision.dinov3_encoder.runtime_status", return_value={"blockers": []}):
            _, actual_processor, actual_model = _load_backend(Path("/local/snapshot"))
        self.assertIs(actual_processor, processor)
        self.assertIs(actual_model, model)
        self.assertFalse(model.training)
        self.assertFalse(model.value.requires_grad)
        self.assertEqual(model.value.device.type, "cpu")
        kwargs = auto_model.from_pretrained.call_args[1]
        self.assertTrue(kwargs["local_files_only"])
        self.assertTrue(kwargs["use_safetensors"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertTrue(auto_processor.from_pretrained.call_args[1]["local_files_only"])

    def test_incomplete_weight_loading_never_uses_random_parameters(self):
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            module = SimpleNamespace(
                AutoImageProcessor=SimpleNamespace(from_pretrained=Mock()),
                AutoModel=SimpleNamespace(from_pretrained=Mock(return_value=(StubModel(), {key: ["bad"]}))))
            with patch.dict(sys.modules, transformers=module), \
                    patch("active_vision.dinov3_encoder.runtime_status", return_value={"blockers": []}):
                with self.assertRaisesRegex(RuntimeError, "refusing partial/random"):
                    _load_backend(Path("/local/snapshot"))

    def test_cls_is_normalized_without_patch_or_register_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snapshot(root)
            processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})
            model = StubModel()
            with patch("active_vision.dinov3_encoder._load_backend", return_value=(torch, processor, model)):
                encoder = DINOv3Encoder(root)
            bgr, mask = self.image()
            feature = encoder.encode(bgr, mask)
            expected = np.arange(1, 385, dtype=np.float32)
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(feature, expected, atol=1e-7)
            self.assertEqual(feature.dtype, np.float32)
            self.assertTrue(model.inference_mode_used)
            np.testing.assert_array_equal(processor.call_args[1]["images"], prepare_object_rgb(bgr, mask))
            details = _encoder_metadata(encoder)
            self.assertEqual(details["feature_mode"], FEATURE_MODE)
            self.assertEqual(len(details["encoder_details"]["snapshot_files_sha256"]), 3)
            self.assertFalse(details["encoder_details"]["geometry_features_used"])

    def test_bad_outputs_do_not_become_embeddings(self):
        encoder = object.__new__(DINOv3Encoder)
        encoder.torch = torch
        encoder.processor = Mock(return_value={})
        for value in (None, torch.ones(384), torch.ones(1, 2, 64), torch.ones(1, 0, 384),
                      torch.zeros(1, 2, 384), torch.full((1, 2, 384), float("nan"))):
            encoder.model = Mock(return_value=SimpleNamespace(last_hidden_state=value))
            with self.assertRaises(ValueError):
                encoder.encode(*self.image())

    def test_processor_config_is_part_of_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snapshot(root)
            with patch("active_vision.dinov3_encoder._load_backend", return_value=(torch, Mock(), StubModel())):
                before = DINOv3Encoder(root).checkpoint_sha256
                write_json(root / "preprocessor_config.json", {"size": {"height": 256, "width": 256}})
                after = DINOv3Encoder(root).checkpoint_sha256
            self.assertNotEqual(before, after)

    def test_cli_preflight_writes_honest_unavailable_report(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            path = Path(directory) / "status.json"
            code = main(["dinov3-check", "--model-dir", directory, "--output", str(path)])
            result = json.loads(output.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(read_json(path), result)
            self.assertEqual(result["recognition_accuracy"], "not evaluated")

    def test_cli_requires_explicit_local_snapshot_and_keeps_old_default(self):
        for command in ("features", "recognition-calibrate", "recognition-evaluate"):
            base = [command, "--manifest", "unused.json", "--output", "unused"]
            self.assertEqual(build_parser().parse_args(base).encoder, "professor-rgb")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(base + ["--encoder", "dinov3"])
            self.assertEqual(code, 2)
            self.assertIn("requires --model-dir", output.getvalue())

    def test_mocked_dinov3_calibration_uses_development_only_and_saves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.snapshot(root / "snapshot")
            rows = []
            for split, labels in (("enroll", ("one", "two")),
                                  ("calibration", ("one", "two", "unknown"))):
                for label in labels:
                    name = split + "_" + label
                    bgr, mask = self.image()
                    bgr[0, 0] = len(rows) + 1
                    cv2.imwrite(str(root / (name + ".png")), bgr)
                    cv2.imwrite(str(root / (name + "_mask.png")), mask)
                    rows.append({"image": name + ".png", "mask": name + "_mask.png",
                                 "split": split, "session_id": name, "label": label,
                                 "mask_reviewed": True})
            rows.append({"image": "missing_test.png", "mask": "missing_mask.png",
                         "split": "test", "session_id": "held_out", "label": "one"})
            write_json(root / "manifest.json", {"samples": rows})
            processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})
            with patch("active_vision.dinov3_encoder._load_backend",
                       return_value=(torch, processor, StubModel())), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(["recognition-calibrate", "--encoder", "dinov3",
                             "--model-dir", str(root / "snapshot"),
                             "--manifest", str(root / "manifest.json"),
                             "--output", str(root / "fit")])
            self.assertEqual(code, 0)
            self.assertEqual(processor.call_count, 5)
            report = read_json(root / "fit/calibration_report.json")
            self.assertEqual(report["feature_mode"], FEATURE_MODE)
            self.assertEqual(report["encoder_details"]["feature_dimension"], 384)
            self.assertEqual(report["held_out_test_accuracy"], "not evaluated")
            self.assertFalse(report["encoder_training_performed"])
            self.assertFalse(report["model_deployed"])
            self.assertFalse(report["real_robot_motion_commanded"])

    def test_default_cli_does_not_import_optional_torch_transformers_or_dinov3(self):
        check = ("import sys; sys.path.insert(0, " + repr(str(SCRIPTS)) + "); "
                 "from active_vision.cli import build_parser; build_parser(); "
                 "assert 'active_vision.dinov3_encoder' not in sys.modules; "
                 "assert 'transformers' not in sys.modules; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-c", check], check=True, timeout=15,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
