#!/usr/bin/env python3
"""Action-template loading for the offline NICO kinematic preview."""

import json
from pathlib import Path
from typing import Any, Dict, List


TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "right_arm_templates.json"
CAPTURED_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "captured_right_arm_templates.json"
)


def load_templates(path: Path = TEMPLATE_PATH) -> Dict[str, Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    templates = data.get("templates", {})
    if not isinstance(templates, dict):
        raise ValueError(f"No templates dictionary found in {path}")
    if Path(path) == TEMPLATE_PATH and CAPTURED_TEMPLATE_PATH.exists():
        with CAPTURED_TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
            captured = json.load(handle).get("templates", {})
        if isinstance(captured, dict):
            templates = {**templates, **captured}
    return templates


def template_names(path: Path = TEMPLATE_PATH) -> List[str]:
    return sorted(load_templates(path).keys())


def get_template(name: str, path: Path = TEMPLATE_PATH) -> Dict[str, Any]:
    templates = load_templates(path)
    if name not in templates:
        available = ", ".join(sorted(templates.keys()))
        raise KeyError(f"Unknown template '{name}'. Available templates: {available}")
    template = dict(templates[name])
    template.setdefault("name", name)
    template.setdefault("arm_used", "right")
    if "joint_angles_deg" not in template and "joint_angles" in template:
        template["joint_angles_deg"] = template.get("joint_angles", {})
    template.setdefault("joint_angles_deg", {})
    template.setdefault("hand_state", "neutral")
    template.setdefault("safety_note", "Preview only. No real robot motion is commanded.")
    template.setdefault("real_robot_motion_commanded", False)
    return template


def template_summary(template: Dict[str, Any]) -> str:
    angles = template.get("joint_angles_deg", template.get("joint_angles", {}))
    angle_text = ", ".join(f"{key}={value}" for key, value in sorted(angles.items()))
    return (
        f"{template.get('name', 'unknown')} | "
        f"{template.get('action_type', 'unknown')} | "
        f"hand={template.get('hand_state', 'neutral')} | "
        f"{angle_text}"
    )
