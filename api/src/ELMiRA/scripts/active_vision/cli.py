"""Offline and read-only ROS1 entry points for the single-RGB integration."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, board_pose, calibrate_images, check_holdout
from .common import read_json, sha256, write_json
from .masks import mask_metrics, overlay, segment
from .recognition import RGBEncoder, ColorShapeEncoder, _encoder_metadata, calibrate_dataset, evaluate_dataset
from .session import reconstruct


def image_files(directory):
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def build_parser():
    parser = argparse.ArgumentParser(description="Single RGB vision. All commands are no-motion.")
    sub = parser.add_subparsers(dest="command", required=True)
    seg = sub.add_parser("segment", help="GrabCut pixel masks from a padded box; requires human review")
    seg.add_argument("--images", required=True)
    seg.add_argument("--output", required=True)
    seg.add_argument("--box", type=int, nargs=4, required=True, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    seg.add_argument("--first", type=int, default=0, help="Offset in sorted image files")
    seg.add_argument("--count", type=int, default=25)
    seg.add_argument("--scribbles", help="Optional 0/1/2/3 GrabCut mask for a single image")
    seg.add_argument("--foreground-point", type=int, nargs=2, help="Manual point inside target, not a ground-truth mask")
    seg.add_argument("--foreground-radius", type=int, default=2, help="Radius of definite foreground scribble in pixels")
    evaluation = sub.add_parser("mask-evaluate")
    evaluation.add_argument("--prediction", required=True)
    evaluation.add_argument("--ground-truth", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--images", required=True)
    cal.add_argument("--columns", type=int, required=True, help="Inner chessboard corners")
    cal.add_argument("--rows", type=int, required=True)
    cal.add_argument("--square-m", type=float, required=True)
    cal.add_argument("--camera-id", required=True)
    cal.add_argument("--optical-frame", required=True)
    cal.add_argument("--model", choices=["plumb_bob", "equidistant"], required=True)
    cal.add_argument("--output", required=True)
    check = sub.add_parser("calibration-check")
    check.add_argument("--calibration", required=True)
    check.add_argument("--images", required=True, help="Held-out board images, separate from calibration fitting")
    check.add_argument("--max-error-px", type=float, default=1.0)
    check.add_argument("--output", required=True)
    board = sub.add_parser("calibration-board", help="Printable A4 vector checkerboard; measure after printing")
    board.add_argument("--output", required=True)
    board.add_argument("--square-mm", type=float, default=25)
    capture = sub.add_parser("calibration-capture", help="Camera-only board images; Enter captures, q stops")
    capture.add_argument("--device", required=True, help="Stable /dev/v4l/by-id camera capture path")
    capture.add_argument("--output", required=True, help="New directory; never overwritten")
    capture.add_argument("--count", type=int, default=25)
    capture.add_argument("--columns", type=int, default=9)
    capture.add_argument("--rows", type=int, default=6)
    pose = sub.add_parser("board-pose")
    pose.add_argument("--image", required=True)
    pose.add_argument("--calibration", required=True)
    pose.add_argument("--columns", type=int, required=True)
    pose.add_argument("--rows", type=int, required=True)
    pose.add_argument("--square-m", type=float, required=True)
    pose.add_argument("--output", required=True)
    for name in ("features", "recognition-calibrate", "recognition-evaluate"):
        item = sub.add_parser(name)
        item.add_argument("--manifest", required=True)
        item.add_argument("--encoder", choices=["professor-rgb", "color-shape", "dinov3"], default="professor-rgb",
                          help="dinov3 and color-shape are separate offline comparisons, not the professor's model")
        item.add_argument("--checkpoint", help="Required for professor-rgb; unused by color-shape")
        item.add_argument("--model-dir", help="dinov3 only: authorized local HF ViT-S/16 safetensors snapshot")
        item.add_argument("--output", required=True)
    dino = sub.add_parser("dinov3-check", help="Offline dependency/snapshot preflight; no downloads or model inference")
    dino.add_argument("--model-dir", help="Local HF DINOv3 ViT-S/16 snapshot directory")
    dino.add_argument("--output", help="Optional JSON preflight report")
    geometry = sub.add_parser("reconstruct")
    geometry.add_argument("--manifest", required=True)
    geometry.add_argument("--output", required=True)
    fusion = sub.add_parser("fusion-replay", help="Offline RGB/geometry fusion from old posed recordings; no hardware")
    fusion.add_argument("--manifest", required=True)
    fusion.add_argument("--checkpoint", required=True)
    fusion.add_argument("--memory", help="Optional matching 101D full-fusion memory; never updated")
    fusion.add_argument("--output", required=True, help="New output directory; never overwritten")
    fusion.add_argument("--experimental-unverified", action="store_true",
                        help="Explicitly allow candidate calibration/poses; outputs remain unverified")
    sequence_add = sub.add_parser("sequence-add", help="Add a labelled replay to a dataset; checksums are automatic")
    sequence_add.add_argument("--manifest", required=True)
    sequence_add.add_argument("--replay-report", required=True)
    sequence_add.add_argument("--label", required=True)
    sequence_add.add_argument("--split", choices=["enroll", "calibration", "test"], required=True)
    sequence_add.add_argument("--capture-group", required=True, help="One unique physical capture/placement ID")
    sequence_add.add_argument("--object-instance", required=True, help="Physical item ID; unknown calibration/test items must differ")
    sequence_add.add_argument("--sample-id")
    for command in ("sequence-enroll", "sequence-calibrate", "sequence-evaluate"):
        sequence = sub.add_parser(command, help="No-motion real-object registration / matched sequence comparison")
        sequence.add_argument("--manifest", required=True, help="Recorded feature dataset, not a list of unposed photos")
        sequence.add_argument("--output", required=True, help="New output directory")
        if command == "sequence-evaluate":
            sequence.add_argument("--calibration-dir", required=True, help="Frozen development-only memories")
    sub.add_parser("ros-check")
    ros = sub.add_parser("ros-node")
    ros.add_argument("--config", required=True)
    ros.add_argument("--output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        encoder = None
        if args.command in ("features", "recognition-calibrate", "recognition-evaluate"):
            if args.encoder == "professor-rgb" and not args.checkpoint:
                raise ValueError("professor-rgb requires --checkpoint")
            if args.encoder == "dinov3":
                if not args.model_dir:
                    raise ValueError("dinov3 requires --model-dir; no weights are downloaded automatically")
                if args.checkpoint:
                    raise ValueError("Use --model-dir, not --checkpoint, for dinov3")
                from .dinov3_encoder import DINOv3Encoder
                encoder = DINOv3Encoder(args.model_dir)
            else:
                if args.model_dir:
                    raise ValueError("--model-dir is only used by --encoder dinov3")
                encoder = RGBEncoder(args.checkpoint) if args.encoder == "professor-rgb" else ColorShapeEncoder()
        if args.command == "segment":
            if args.first < 0 or args.count <= 0:
                raise ValueError("first >= 0 and count > 0 required")
            files = image_files(args.images)[args.first:args.first+args.count]
            if not files:
                raise ValueError("No images selected")
            if args.scribbles and len(files) != 1:
                raise ValueError("Scribbles must be supplied for one matching frame at a time")
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=False)
            rows = []
            for path in files:
                bgr = cv2.imread(str(path))
                scribbles = cv2.imread(args.scribbles, 0) if args.scribbles else None
                if args.scribbles and scribbles is None:
                    raise ValueError("Cannot read scribbles")
                mask, details = segment(bgr, args.box, scribbles, args.foreground_point, args.foreground_radius)
                mask_path = output / (path.stem + "_mask.png")
                overlay_path = output / (path.stem + "_overlay.png")
                if not cv2.imwrite(str(mask_path), mask) or not cv2.imwrite(str(overlay_path), overlay(bgr, mask)):
                    raise OSError("Cannot save segmentation outputs")
                rows.append({"image": str(path.resolve()), "image_sha256": sha256(path),
                             "mask": mask_path.name, "overlay": overlay_path.name, **details})
            result = {"samples": rows, "status": "proposals_need_review", "real_robot_motion_commanded": False}
            write_json(output / "masks.json", result)
            result = {"frames_processed": len(rows), "manifest": str(output / "masks.json"),
                      "accuracy": "not measured; human pixel masks needed", "real_robot_motion_commanded": False}
        elif args.command == "mask-evaluate":
            prediction, truth = cv2.imread(args.prediction, 0), cv2.imread(args.ground_truth, 0)
            if truth is None:
                raise ValueError("Ground-truth mask unavailable")
            result = mask_metrics(prediction, truth)
        elif args.command == "calibrate":
            result = calibrate_images(image_files(args.images), args.columns, args.rows, args.square_m,
                                      args.camera_id, args.optical_frame, args.model)
            write_json(args.output, result)
        elif args.command == "board-pose":
            calibration = Calibration.load(args.calibration)
            bgr = calibration.rectify(cv2.imread(args.image))
            result = board_pose(bgr, calibration.rectified_K, args.columns, args.rows, args.square_m)
            result.update(pose_verified=False, pose_evidence="", real_robot_motion_commanded=False)
            write_json(args.output, result)
        elif args.command == "calibration-check":
            result = check_holdout(args.calibration, image_files(args.images), args.max_error_px)
            write_json(args.output, result)
        elif args.command == "calibration-board":
            from .calibration_board import generate_board
            result = generate_board(args.output, args.square_mm)
        elif args.command == "calibration-capture":
            from .calibration_capture import capture_board_images
            result = capture_board_images(args.device, args.output, args.count, args.columns, args.rows)
        elif args.command == "features":
            manifest = Path(args.manifest)
            records = read_json(manifest)["samples"]
            if not records:
                raise ValueError("No feature samples")
            features = []
            for row in records:
                features.append(encoder.encode(cv2.imread(str(manifest.parent / row["image"])),
                                               cv2.imread(str(manifest.parent / row["mask"]), 0)))
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(output / "rgb_features.npz"), embeddings=np.stack(features))
            result = {"frames_processed": len(features), "feature_shape": list(np.stack(features).shape),
                      **_encoder_metadata(encoder), "branch": "visual_only",
                      "input_images": [row["image"] for row in records], "finite": bool(np.isfinite(features).all()),
                      "recognition_accuracy": "unverified; no independent real-object test set",
                      "geometry_status": "not inferred from RGB features", "real_robot_motion_commanded": False}
            write_json(output / "feature_report.json", result)
        elif args.command == "recognition-calibrate":
            result = calibrate_dataset(args.manifest, encoder, args.output)
        elif args.command == "recognition-evaluate":
            result = evaluate_dataset(args.manifest, encoder, args.output)
        elif args.command == "dinov3-check":
            from .dinov3_encoder import runtime_status
            result = runtime_status(args.model_dir)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["status"] == "ready_for_load" else 2
        elif args.command == "reconstruct":
            result = reconstruct(args.manifest, args.output)
        elif args.command == "fusion-replay":
            from .fusion_replay import replay
            result = replay(args.manifest, args.checkpoint, args.output, args.memory,
                            args.experimental_unverified)
        elif args.command in ("sequence-enroll", "sequence-calibrate", "sequence-evaluate"):
            from .sequence_recognition import enroll, calibrate, evaluate
            if args.command == "sequence-enroll":
                result = enroll(args.manifest, args.output)
            elif args.command == "sequence-calibrate":
                result = calibrate(args.manifest, args.output)
            else:
                result = evaluate(args.manifest, args.calibration_dir, args.output)
        elif args.command == "sequence-add":
            from .sequence_recognition import add_sequence
            result = add_sequence(args.manifest, args.replay_report, args.label, args.split,
                                  args.capture_group, args.object_instance, args.sample_id)
        elif args.command == "ros-check":
            from .ros1 import ros_status
            result = ros_status()
        else:
            from .ros1 import run
            return run(args.config, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, subprocess.TimeoutExpired, cv2.error) as exc:
        print(json.dumps({"status": "unavailable_or_invalid_input", "reason": str(exc),
                          "real_robot_motion_commanded": False}, indent=2))
        return 2
