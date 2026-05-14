#!/usr/bin/env python3
"""Render a simple stick-figure NICO upper-body preview."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from kinematic_preview.preview_config import TABLE_SURFACE_Z_M
except Exception:
    from preview_config import TABLE_SURFACE_Z_M


Point3 = Tuple[float, float, float]


def _point(record: Dict[str, float]) -> Point3:
    return (float(record["x"]), float(record["y"]), float(record["z"]))


def _plot_line(ax, coordinates: Dict[str, Dict[str, float]], names, **kwargs) -> None:
    points = [_point(coordinates[name]) for name in names]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.plot(xs, ys, zs, **kwargs)


def render_preview(
    preview: Dict[str, Any],
    output_path: Path,
    target_object: str = "target object",
    target_object_xyz: Optional[Dict[str, float]] = None,
    template_name: str = "action template",
) -> Path:
    """Save a stick-figure image and return the output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coordinates = preview["coordinates"]
    right_hand = _point(coordinates["right_hand"])
    right_wrist = _point(coordinates["right_wrist"])

    fig = plt.figure(figsize=(7.0, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"NICO Kinematic Action Preview: {template_name}", pad=12)

    _plot_line(ax, coordinates, ["torso_base", "torso_top", "neck"], color="#2f4858", linewidth=4)
    _plot_line(
        ax,
        coordinates,
        ["right_shoulder", "right_elbow", "right_wrist", "right_hand"],
        color="#0077b6",
        linewidth=4,
        marker="o",
        markersize=5,
    )
    _plot_line(
        ax,
        coordinates,
        ["left_shoulder", "left_elbow", "left_wrist", "left_hand"],
        color="#8d99ae",
        linewidth=2.5,
        marker="o",
        markersize=4,
        alpha=0.75,
    )
    _plot_line(
        ax,
        coordinates,
        ["left_shoulder", "right_shoulder"],
        color="#2f4858",
        linewidth=3,
    )

    head = _point(coordinates["head_center"])
    ax.scatter([head[0]], [head[1]], [head[2]], s=420, color="#f4d35e", edgecolor="#2f4858")
    ax.scatter([right_hand[0]], [right_hand[1]], [right_hand[2]], s=90, color="#00a676", label="estimated right hand")
    ax.scatter([right_wrist[0]], [right_wrist[1]], [right_wrist[2]], s=45, color="#118ab2", label="estimated right wrist")

    if target_object_xyz:
        target = (
            float(target_object_xyz.get("x", 0.3)),
            float(target_object_xyz.get("y", -0.15)),
            float(target_object_xyz.get("z", 0.25)),
        )
        ax.scatter([target[0]], [target[1]], [target[2]], s=120, color="#d62828", marker="s", label=target_object)
        ax.text(target[0], target[1], target[2] + 0.025, target_object, color="#7f1d1d")

    table_x = [-0.06, 0.46]
    table_y = [-0.34, 0.24]
    xx = np.array([
        [table_x[0], table_x[1]],
        [table_x[0], table_x[1]],
    ])
    yy = np.array([
        [table_y[0], table_y[0]],
        [table_y[1], table_y[1]],
    ])
    zz = np.array([
        [TABLE_SURFACE_Z_M, TABLE_SURFACE_Z_M],
        [TABLE_SURFACE_Z_M, TABLE_SURFACE_Z_M],
    ])
    ax.plot_surface(xx, yy, zz, color="#d9cab3", alpha=0.28, linewidth=0)
    ax.text(0.34, 0.20, TABLE_SURFACE_Z_M + 0.01, "table z=0.08m", color="#6b5d4d")

    ax.text(right_hand[0], right_hand[1], right_hand[2] + 0.025, "right hand", color="#005f73")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left/right (m)")
    ax.set_zlabel("z up (m)")
    ax.set_xlim(-0.08, 0.48)
    ax.set_ylim(-0.38, 0.28)
    ax.set_zlim(0.0, 0.78)
    ax.view_init(elev=20, azim=-55)
    ax.legend(loc="upper left")
    ax.grid(True)

    fig.text(
        0.5,
        0.02,
        "Preview only: no ROS motion command, no physics, no contact simulation.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path
