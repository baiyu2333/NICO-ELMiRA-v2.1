"""Guarded professor RGB visual hull plus OpenCV sparse triangulation.

All geometry is in metres in a measured world frame. A rotation of the camera
axes is not a translation of the optical centre and cannot create baseline.
"""
import cv2
import numpy as np

from .common import positive, transform
from .masks import validate_mask
from .vendor.acf3d.pose_se3 import SE3Pose
from .vendor.acf3d.rgb_geometry_v8 import RGBPoseGeometry, _project
from .vendor.acf3d.synthetic import Camera


def baseline_and_angle(a, b, center):
    a, b = transform(a), transform(b)
    va, vb = np.asarray(center) - a[:3, 3], np.asarray(center) - b[:3, 3]
    if min(np.linalg.norm(va), np.linalg.norm(vb)) < 1e-8:
        raise ValueError("Object centre coincides with camera centre")
    cosine = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def triangulate_checked(K, T1, T2, pixels1, pixels2, min_angle_deg=1.0, max_error_px=2.0):
    """Known-pose epipolar geometry; reject infinity, negative depth and bad matches."""
    T1, T2 = transform(T1), transform(T2)
    if np.linalg.norm(T1[:3, 3] - T2[:3, 3]) < 1e-6:
        return np.empty((0, 3))
    p1, p2 = np.asarray(pixels1, np.float64).reshape(-1, 2), np.asarray(pixels2, np.float64).reshape(-1, 2)
    if p1.shape != p2.shape or not np.isfinite(p1).all() or not np.isfinite(p2).all():
        raise ValueError("Invalid matched pixels")
    if not len(p1):
        return np.empty((0, 3))
    C1, C2 = np.linalg.inv(T1), np.linalg.inv(T2)
    Xh = cv2.triangulatePoints(K @ C1[:3], K @ C2[:3], p1.T, p2.T)
    good = np.abs(Xh[3]) > 1e-9
    X = Xh[:3, good].T / Xh[3, good, None]
    p1, p2 = p1[good], p2[good]
    valid = np.isfinite(X).all(axis=1)
    for C, pixels in ((C1, p1), (C2, p2)):
        camera = X @ C[:3, :3].T + C[:3, 3]
        h = camera @ K.T
        projected = h[:, :2] / np.maximum(h[:, 2:], 1e-9)
        valid &= (camera[:, 2] > .001) & (np.linalg.norm(projected-pixels, axis=1) <= max_error_px)
    ray1, ray2 = X-T1[:3, 3], X-T2[:3, 3]
    cosine = np.sum(ray1*ray2, axis=1) / np.maximum(np.linalg.norm(ray1, axis=1)*np.linalg.norm(ray2, axis=1), 1e-12)
    valid &= np.degrees(np.arccos(np.clip(cosine, -1, 1))) >= min_angle_deg
    return X[valid]


class GuardedGeometry(RGBPoseGeometry):
    def __init__(self, K, size, center, half_extent, resolution=40,
                 min_baseline_m=.01, min_angle_deg=1.0):
        K = np.asarray(K, np.float64)
        center = np.asarray(center, np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Measured volume centre must have three finite coordinates")
        if not 8 <= int(resolution) <= 100:
            raise ValueError("Voxel resolution must be 8..100")
        camera = Camera(size[0], size[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        super().__init__(camera, center, positive(half_extent, "half_extent"), int(resolution))
        self.K = K
        self.min_baseline = positive(min_baseline_m, "min_baseline_m")
        self.min_angle = positive(min_angle_deg, "min_angle_deg")
        self.support = np.zeros(len(self.voxels), np.uint16)
        self.poses = []

    def _triangulate(self, rgb, mask, pose):
        if self.prev is None:
            return 0
        previous, previous_mask, previous_pose = self.prev
        baseline, angle = baseline_and_angle(previous_pose.T_wc, pose.T_wc, self.center)
        if baseline < self.min_baseline or angle < self.min_angle:
            return 0
        orb = cv2.ORB_create(nfeatures=1200, fastThreshold=8)
        a, da = orb.detectAndCompute(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), previous_mask*255)
        b, db = orb.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), mask)
        if da is None or db is None or min(len(a), len(b)) < 8:
            return 0
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .7*pair[1].distance]
        if len(good) < 8:
            return 0
        points = triangulate_checked(self.K, previous_pose.T_wc, pose.T_wc,
                                     [a[m.queryIdx].pt for m in good], [b[m.trainIdx].pt for m in good],
                                     min_angle_deg=self.min_angle)
        points = points[np.all(np.abs(points-self.center) <= self.half_extent, axis=1)]
        self.tri_points = np.concatenate([self.tri_points, points]).astype(np.float32)[-self.max_triangulated:]
        return len(points)

    def update_checked(self, bgr, mask, T_wc):
        pose = SE3Pose(transform(T_wc))
        mask = validate_mask(mask, bgr.shape)
        if bgr.shape[:2] != (self.cam.height, self.cam.width):
            raise ValueError("Rectified image size differs from geometry camera")
        if self.poses:
            pairs = [baseline_and_angle(p, pose.T_wc, self.center) for p in self.poses]
            if min(b for b, _ in pairs) < self.min_baseline or not any(b >= self.min_baseline and a >= self.min_angle for b, a in pairs):
                return {"status": "insufficient_baseline", "accepted": False,
                        "max_baseline_m": max(b for b, _ in pairs),
                        "max_view_angle_deg": max(a for _, a in pairs),
                        "reason": "No useful optical-centre displacement; rotation alone is insufficient"}
        u, v, z = _project(self.voxels, pose, self.cam)
        visible = (z > .08) & (u >= 0) & (u < self.cam.width-1) & (v >= 0) & (v < self.cam.height-1)
        self.support[visible] += 1
        stats = super().update(bgr, mask, pose)
        self.poses.append(pose.T_wc)
        hull = self.metric_hull()
        return {**stats, "accepted": True, "status": "coarse_multiview" if len(hull) else "awaiting_multiview",
                "supported_hull_points": len(hull), "depth_sensor_used": False}

    def metric_hull(self):
        # The original implementation leaves never-observed voxels occupied.
        # Do not export these as measured geometry or use them as point tokens.
        return self.voxels[self.occupied & (self.support >= 2)]

    def export(self, path):
        np.savez_compressed(str(path), visual_hull_xyz_m=self.metric_hull(),
                            triangulated_xyz_m=self.tri_points, K=self.K,
                            T_wc=np.asarray(self.poses), volume_center_m=self.center,
                            half_extent_m=self.half_extent)
