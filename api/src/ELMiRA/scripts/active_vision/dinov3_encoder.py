"""Optional, offline DINOv3 ViT-S/16 features. No ROS or automatic downloads.

Only an explicitly supplied local Hugging Face safetensors snapshot is loaded.
RGB object cutouts bypass the professor's grayscale/polar/spectral frontend.
"""
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from .common import read_json, sha256
from .masks import validate_mask


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
FEATURE_MODE = "dinov3_vits16_masked_rgb_cls_v1"
SNAPSHOT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def _snapshot_files(model_dir):
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("DINOv3 requires --model-dir pointing to a local downloaded HF snapshot")
    missing = [name for name in SNAPSHOT_FILES
               if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    if missing:
        raise ValueError("Incomplete local DINOv3 snapshot; missing: " + ", ".join(missing))
    config = read_json(directory / "config.json")
    processor = read_json(directory / "preprocessor_config.json")
    if not isinstance(config, dict) or not isinstance(processor, dict):
        raise ValueError("DINOv3 config and processor settings must be JSON objects")
    if config.get("auto_map") or processor.get("auto_map"):
        raise ValueError("Remote/custom model code is not allowed for this offline encoder")
    expected = {"model_type": "dinov3_vit", "hidden_size": 384, "patch_size": 16,
                "num_hidden_layers": 12, "num_attention_heads": 6}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected the official DINOv3 ViT-S/16 snapshot, not another backbone")
    return directory


def runtime_status(model_dir=None):
    """Metadata-only preflight; does not import/load models, access HF or ROS."""
    from packaging.version import InvalidVersion, Version

    versions, blockers = {}, []
    if sys.version_info < (3, 9):
        blockers.append("Python >=3.9 required; use a separate Python 3.10/3.11 environment")
    for name, minimum in (("torch", "2.2"), ("transformers", "4.56.0"),
                          ("torchvision", None), ("safetensors", None)):
        try:
            versions[name] = metadata.version(name)
            if minimum and Version(versions[name]) < Version(minimum):
                blockers.append(name + " >=" + minimum + " required")
        except (metadata.PackageNotFoundError, InvalidVersion):
            versions[name] = "unavailable"
            blockers.append(name + " is unavailable")
    snapshot_status = "not supplied"
    if model_dir:
        try:
            _snapshot_files(model_dir)
            snapshot_status = "files present; loading/weights not yet verified"
        except (ValueError, OSError) as exc:
            snapshot_status = "unavailable"
            blockers.append(str(exc))
    else:
        blockers.append("Supply an authorized local DINOv3 HF snapshot with --model-dir")
    return {"status": "unavailable" if blockers else "ready_for_load",
            "check_type": "metadata_only_not_model_inference",
            "requested_model": MODEL_ID, "python": sys.version.split()[0],
            "packages": versions, "snapshot_status": snapshot_status, "blockers": blockers,
            "model_loaded": False, "recognition_accuracy": "not evaluated",
            "automatic_downloads": False, "real_robot_motion_commanded": False}


def prepare_object_rgb(bgr, mask, padding=1.25):
    """Masked RGB, tight bbox, centered square padding; preserve aspect ratio.

    This is our object-ROI preparation, not an autonomous segmenter. The local
    official image processor subsequently handles resize/rescale/normalization.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("DINOv3 expects a readable uint8 BGR image")
    if not math.isfinite(padding) or not 1 <= padding <= 2:
        raise ValueError("Object padding must be finite and between 1 and 2")
    mask = validate_mask(mask, bgr.shape)
    x, y, width, height = cv2.boundingRect(mask)
    cutout = cv2.bitwise_and(bgr, bgr, mask=mask)[y:y+height, x:x+width]
    side = int(math.ceil(max(width, height) * padding))
    canvas = np.zeros((side, side, 3), np.uint8)
    top, left = (side-height)//2, (side-width)//2
    canvas[top:top+height, left:left+width] = cutout[:, :, ::-1]
    return canvas


def _load_backend(directory):
    status = runtime_status(directory)
    if status["blockers"]:
        raise RuntimeError("DINOv3 unavailable: " + "; ".join(status["blockers"]))
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        # Native implementation only: no custom code, pickle weights or network.
        processor = AutoImageProcessor.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
        model, loading = AutoModel.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, output_loading_info=True)
        if any(loading.get(key) for key in
               ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("DINOv3 weights do not completely match the model; refusing partial/random weights")
        if model.config.model_type != "dinov3_vit" or model.config.hidden_size != 384:
            raise ValueError("Loaded model is not DINOv3 ViT-S/16")
        torch.set_num_threads(1)
        model.to(device="cpu", dtype=torch.float32)
        model.eval()
        model.requires_grad_(False)
        return torch, processor, model
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Cannot load the local DINOv3 snapshot: " + str(exc)) from exc


class DINOv3Encoder:
    """Frozen 384D CLS features for the existing labelled prototype workflow."""

    feature_mode = FEATURE_MODE

    def __init__(self, model_dir):
        directory = _snapshot_files(model_dir)
        hashes = {name: sha256(directory / name) for name in SNAPSHOT_FILES}
        self.checkpoint_sha256 = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("ascii")).hexdigest()
        self.torch, self.processor, self.model = _load_backend(directory)
        self.provenance = {
            "requested_model": MODEL_ID, "snapshot_files_sha256": hashes,
            "encoder_code_sha256": sha256(Path(__file__)),
            "preprocessing": "reviewed mask -> black RGB cutout -> square bbox padding 1.25 -> local official processor",
            "feature_pooling": "last_hidden_state CLS token, L2 normalized",
            "feature_dimension": 384, "device": "cpu", "dtype": "float32",
            "packages": runtime_status(directory)["packages"],
            "encoder_frozen": True, "automatic_downloads": False,
            "geometry_features_used": False, "deployment": "offline_only"}

    def encode(self, bgr, mask):
        rgb = prepare_object_rgb(bgr, mask)
        inputs = self.processor(images=rgb, return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model(**inputs)
            hidden = getattr(output, "last_hidden_state", None)
            if (hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1
                    or hidden.shape[1] < 1 or hidden.shape[2] != 384):
                raise ValueError("DINOv3 returned invalid CLS token dimensions")
            feature = hidden[0, 0].detach().float().cpu().numpy()
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(feature).all() or not math.isfinite(norm) or norm < 1e-8:
            raise ValueError("DINOv3 returned a non-finite or zero feature")
        return (feature / norm).astype(np.float32)
