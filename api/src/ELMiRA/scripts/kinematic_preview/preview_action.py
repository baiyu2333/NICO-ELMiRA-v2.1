#!/usr/bin/env python3
"""CLI for offline NICO kinematic action-template previews."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from kinematic_preview.action_templates import get_template, load_templates, template_names, template_summary
from kinematic_preview.nico_stick_model import NICOStickModel
from kinematic_preview.preview_config import (
    DEFAULT_OBJECT_CENTER_Z_M,
    DEFAULT_TARGET_Y_M,
    MIN_SAFE_Z_M,
    PREVIEW_NOTE,
    RED_OBJECT_DISTANCE_FROM_NICO_M,
    measured_setup_values,
)
from kinematic_preview.render_stick_nico import render_preview
from utils.kinematic_trial_logger import build_record, find_api_root, log_preview_result, preview_paths


def resolve_target_xyz(
    template: Dict[str, object],
    target_x: Optional[float],
    target_y: Optional[float],
    target_z: Optional[float],
) -> Dict[str, float]:
    existing = template.get("target_object_xyz") or {}
    if not isinstance(existing, dict):
        existing = {}
    return {
        "x": float(target_x if target_x is not None else existing.get("x", RED_OBJECT_DISTANCE_FROM_NICO_M)),
        "y": float(target_y if target_y is not None else existing.get("y", DEFAULT_TARGET_Y_M)),
        "z": float(target_z if target_z is not None else existing.get("z", DEFAULT_OBJECT_CENTER_Z_M)),
    }


def build_preview(
    template_name: str,
    target_object: Optional[str],
    target_x: Optional[float],
    target_y: Optional[float],
    target_z: Optional[float],
):
    template = get_template(template_name)
    if target_object:
        template["target_object"] = target_object
    template["target_object_xyz"] = resolve_target_xyz(template, target_x, target_y, target_z)

    model = NICOStickModel()
    preview = model.forward_kinematics(template.get("joint_angles_deg", {}), MIN_SAFE_Z_M)
    return template, preview


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview a simplified NICO right-arm action template.")
    parser.add_argument("--template", choices=template_names(), help="Action template name")
    parser.add_argument("--target-object", default=None, help="Optional target object display name")
    parser.add_argument("--target-x", type=float, default=None, help="Optional target x in meters")
    parser.add_argument("--target-y", type=float, default=None, help="Optional target y in meters")
    parser.add_argument("--target-z", type=float, default=None, help="Optional target z in meters")
    parser.add_argument("--list", action="store_true", help="List available templates")
    parser.add_argument("--notes", default="", help="Optional notes stored in the preview log")
    args = parser.parse_args()

    if args.list:
        templates = load_templates()
        for name in sorted(templates):
            print(template_summary(get_template(name)))
        return 0
    if not args.template:
        parser.error("--template is required unless --list is used")

    template, preview = build_preview(
        args.template,
        args.target_object,
        args.target_x,
        args.target_y,
        args.target_z,
    )

    api_root = find_api_root(__file__)
    paths = preview_paths(api_root)
    image_path = render_preview(
        preview,
        paths["latest_image"],
        target_object=template.get("target_object", "target object"),
        target_object_xyz=template.get("target_object_xyz"),
        template_name=template.get("name", args.template),
    )
    setup_values = measured_setup_values()
    setup_values["preview_note"] = PREVIEW_NOTE
    record = build_record(template, preview, image_path, measured_values=setup_values, notes=args.notes)
    paths = log_preview_result(record, api_root)

    summary = {
        "selected_template": template_summary(template),
        "action_type": template.get("action_type"),
        "joint_angles_deg": template.get("joint_angles_deg", {}),
        "target_object": template.get("target_object"),
        "target_object_xyz": template.get("target_object_xyz"),
        "estimated_right_hand_xyz": preview.get("estimated_right_hand_xyz", {}),
        "estimated_right_wrist_xyz": preview.get("estimated_right_wrist_xyz", {}),
        "hand_state": template.get("hand_state", "neutral"),
        "above_table_safety_limit": preview.get("above_table_safety_limit"),
        "height_warning": preview.get("height_warning", ""),
        "saved_visualization_image_path": str(paths["latest_image"]),
        "saved_json_pose_path": str(paths["latest_json"]),
        "saved_markdown_summary_path": str(paths["latest_summary"]),
        "preview_trials_csv_path": str(paths["trials_csv"]),
        "real_robot_motion_commanded": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
