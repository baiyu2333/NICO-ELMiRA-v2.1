"""Professor's CPU next-best-view scorer on supported, single-RGB geometry.

Rendered depth is a visual-hull proxy, NOT a depth sensor measurement. Rankings
are hypotheses about future observations, not calibrated uncertainty guarantees.
Only the supervised session loop may turn a ranking into a text view request.
"""
from dataclasses import asdict

import numpy as np

from .common import transform
from .geometry import baseline_and_angle
from .masks import validate_mask
from .vendor.acf3d.planner_v6 import BatchedSE3NBVPlanner
from .vendor.acf3d.pose_se3 import SE3Pose, pose_key
from .vendor.acf3d.rgb_geometry_v8 import _project
from .vendor.acf3d.se3_gaussian_world import SE3ObjectGaussianMap


class SupportedGaussianProxy(SE3ObjectGaussianMap):
    def __init__(self, geometry):
        spacing = 2*geometry.half_extent/(geometry.resolution-1)
        super().__init__(geometry.cam, voxel_size=spacing, base_sigma=spacing/2)
        self.target_center = geometry.center

    def novelty(self, pose):
        if not self.observed_poses:
            return 1.0
        pairs = [baseline_and_angle(p.T_wc, pose.T_wc, self.target_center) for p in self.observed_poses]
        return float(np.clip(.75*min(a for _, a in pairs)/90 + .25*min(d for d, _ in pairs)/.75, 0, 1))


def supported_proxy_depth(geometry, T_wc):
    """Project only >=2-view-supported hull voxels, with nearest-depth buffering."""
    cam = geometry.cam
    result = np.full((cam.height, cam.width), np.inf, np.float32)
    points = geometry.metric_hull()
    if len(points):
        u, v, z = _project(points, SE3Pose(transform(T_wc)), cam)
        good = (z > .1) & (u >= 0) & (v >= 0) & (u < cam.width-.5) & (v < cam.height-.5)
        ix, iy = np.rint(u[good]).astype(int), np.rint(v[good]).astype(int)
        np.minimum.at(result, (iy, ix), z[good])
    result[~np.isfinite(result)] = 0
    return result


class RGBViewScorer:
    def __init__(self, geometry, encoder, memory):
        self.geometry, self.encoder, self.memory = geometry, encoder, memory
        self.observations = []
        self.planner = BatchedSE3NBVPlanner()

    def add_observation(self, bgr, mask, T_wc):
        T = transform(T_wc)
        if not any(np.allclose(T, accepted, atol=1e-6) for accepted in self.geometry.poses):
            raise ValueError("View scorer accepts only geometrically accepted observations")
        if any(np.allclose(T, old[2], atol=1e-6) for old in self.observations):
            raise ValueError("Duplicate observation in viewpoint scorer")
        mask = validate_mask(mask, bgr.shape)
        feature = self.encoder.encode(bgr, mask)
        self.observations.append((bgr.copy(), mask.copy(), T.copy(), feature))
        self.observations = self.observations[-40:]

    def __call__(self, T_wc, candidates):
        details = {"planner": "professor_BatchedSE3NBVPlanner_CPU_RGB_proxy",
                   "real_robot_motion_commanded": False, "depth_sensor_used": False,
                   "geometry_source": "multi_view_supported_RGB_visual_hull",
                   "limitation": "Rendered proxy and prototype entropy; not measured future views or calibrated conformal coverage."}
        if self.geometry.views < 2 or len(self.observations) < 2:
            return None, {**details, "status": "bootstrap_required", "reason": "Need two accepted measured views"}
        if not len(self.geometry.metric_hull()):
            return None, {**details, "status": "insufficient_geometry", "reason": "No multi-view-supported hull"}
        # Rebuild after carving, so rejected voxels cannot persist as ghost surfaces.
        proxy = SupportedGaussianProxy(self.geometry)
        for bgr, mask, T, _ in self.observations:
            depth = supported_proxy_depth(self.geometry, T)
            proxy.update(bgr, depth, (mask > 0).astype(np.uint8), SE3Pose(T))
        fused = np.sum([row[3] for row in self.observations], axis=0)
        norm = np.linalg.norm(fused)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("Cannot normalize observed RGB features")
        fused = fused/norm
        current_uncertainty = self.memory.predict(fused)[4]
        observed = [SE3Pose(T) for T in self.geometry.poses]
        visited = {pose_key(p) for p in observed}
        eligible, rejected = [], []
        for candidate in candidates:
            pose = SE3Pose(transform(candidate["T_wc"]), name=candidate["name"])
            if pose_key(pose) in visited:
                rejected.append({"name": pose.name, "reason": "already_observed"})
                continue
            rgb, depth, mask, stats = proxy.render(pose)
            if np.count_nonzero(mask) < self.planner.min_mask_pixels:
                rejected.append({"name": pose.name, "reason": "insufficient_rendered_visibility"})
                continue
            eligible.append(pose)
        details.update(proxy=proxy.parameter_summary(), rejected_candidates=rejected)
        if not eligible:
            return None, {**details, "status": "no_visible_unvisited_candidate"}
        selected, rows = self.planner.choose(
            SE3Pose(transform(T_wc)), eligible, fused, current_uncertainty, self.memory, proxy,
            lambda rgb, _proxy_depth, mask: self.encoder.encode(rgb, (mask > 0).astype(np.uint8)*255), observed)
        scores = [asdict(row) for row in rows]
        if not scores or not all(np.isfinite(row["score"]) for row in scores):
            raise ValueError("Viewpoint scores missing or non-finite")
        candidate = next(c for c in candidates if c["name"] == selected.name)
        return candidate, {**details, "status": "ranked", "scores": scores,
                           "current_prototype_entropy": float(current_uncertainty)}
