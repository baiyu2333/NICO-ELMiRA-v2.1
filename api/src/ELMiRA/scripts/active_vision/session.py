"""Read-only measured observations and supervised observe/move/observe state."""
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .common import finite, positive, read_json, sha256, transform, write_json
from .geometry import GuardedGeometry, baseline_and_angle
from .masks import validate_mask


def validate_observation(record, calibration, previous_stamp=None):
    stamp = positive(record["timestamp_sec"], "image timestamp")
    pose_stamp = positive(record["pose_timestamp_sec"], "pose timestamp")
    if abs(stamp-pose_stamp) > .05:
        raise ValueError("Pose is not synchronized with image (50 ms maximum)")
    if previous_stamp is not None and stamp <= previous_stamp:
        raise ValueError("Duplicate or out-of-order image timestamp")
    if record.get("pose_verified") is not True or not record.get("pose_evidence"):
        raise ValueError("Measured pose provenance is missing or unverified")
    if record.get("pose_source") not in ("tf_calibrated", "fixed_metric_chessboard_pnp", "measured_external"):
        raise ValueError("Unknown pose source; commanded head angles are not measured camera poses")
    if record.get("mask_reviewed") is not True or not record.get("mask_evidence"):
        raise ValueError("Pixel mask needs review/validation evidence before reconstruction")
    if record.get("object_static") is not True:
        raise ValueError("Rigid multiview reconstruction requires a stationary target")
    if record.get("camera_id") != calibration.record["camera_id"]:
        raise ValueError("Camera identity mismatch")
    if record.get("optical_frame") != calibration.record["optical_frame"]:
        raise ValueError("Optical frame mismatch")
    if not record.get("world_frame") or record.get("units") != "metres":
        raise ValueError("Need explicit metric world frame")
    return stamp, transform(record["T_wc"])


def reconstruct(manifest_path, output):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if not manifest.get("calibration"):
        raise ValueError("Verified camera calibration is missing; existing image-only trials cannot be reconstructed")
    if not manifest.get("observations"):
        raise ValueError("Need timestamped images, reviewed masks and measured per-frame camera poses")
    calibration_path = manifest_path.parent / manifest["calibration"]
    calibration = Calibration.load(calibration_path)
    if manifest.get("volume_verified") is not True or not manifest.get("volume_evidence"):
        raise ValueError("Object volume must be measured in the same world frame, not an image-to-arm guess")
    geometry = GuardedGeometry(calibration.rectified_K, calibration.size, manifest["volume_center_m"],
                               manifest["half_extent_m"], manifest.get("resolution", 40))
    previous_stamp, world, records = None, None, []
    for observation in manifest["observations"]:
        stamp, T = validate_observation(observation, calibration, previous_stamp)
        if world is not None and world != observation["world_frame"]:
            raise ValueError("World frame changed during recording")
        world, previous_stamp = observation["world_frame"], stamp
        bgr = cv2.imread(str(manifest_path.parent / observation["image"]))
        mask = cv2.imread(str(manifest_path.parent / observation["mask"]), 0)
        if bgr is None:
            raise ValueError("Unreadable session image")
        mask = validate_mask(mask, bgr.shape)
        bgr, mask = calibration.rectify(bgr, mask)
        records.append({"timestamp_sec": stamp, **geometry.update_checked(bgr, mask, T)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    geometry.export(output / "reconstruction.npz")
    hull_count = len(geometry.metric_hull())
    report = {"status": "coarse_multiview" if hull_count else "insufficient_geometry",
              "calibration_sha256": sha256(calibration_path), "world_frame": world,
              "observations": records, "accepted_views": geometry.views,
              "visual_hull_points": hull_count, "sparse_points": len(geometry.tri_points),
              "units": "metres", "depth_sensor_used": False, "real_robot_motion_commanded": False,
              "limitation": "Visual hull is an outer shape estimate, not dense surface/concavity or precision validation."}
    write_json(output / "reconstruction_report.json", report)
    return report


class ObserveMoveObserve:
    """Requests a supervised move, but never executes one or trusts a command as feedback."""
    def __init__(self, center, verified_head_limits, candidates, settling_sec=.5, timeout_sec=20,
                 K=None, image_size=None, view_scorer=None):
        self.center = np.asarray(center, np.float64)
        self.limits = verified_head_limits
        self.candidates = candidates
        self.settling_sec = positive(settling_sec, "settling_sec")
        self.timeout_sec = positive(timeout_sec, "timeout_sec")
        self.phase = "observe"
        self.pending = None
        self.visited = set()
        self.last_stamp = None
        self.stable_since = None
        self.last_feedback_stamp = None
        self.previous_head = None
        self.K = None if K is None else np.asarray(K, np.float64)
        self.image_size = image_size
        self.view_scorer = view_scorer
        if len({c["name"] for c in candidates}) != len(candidates):
            raise ValueError("Viewpoint candidate names must be unique")

    def _check_head(self, feedback, now):
        if self.limits.get("verified") is not True or not self.limits.get("evidence"):
            raise ValueError("Head motion limits are unverified; no viewpoint request allowed")
        stamp = positive(feedback["timestamp_sec"], "joint timestamp")
        if not 0 <= now-stamp <= .25:
            raise ValueError("Head feedback is stale or future-dated")
        for joint in ("head_z", "head_y"):
            value = finite(feedback["positions_rad"][joint], joint)
            low, high = map(float, self.limits[joint])
            if not np.isfinite([low, high]).all() or low > high or not low <= value <= high:
                raise ValueError("Head feedback outside verified limits: " + joint)
        return stamp

    def check_observation(self, T_wc, timestamp_sec, feedback, now):
        """Validate readiness without marking a view accepted or requesting a move."""
        now = finite(now, "now")
        stamp = positive(timestamp_sec, "image timestamp")
        T = transform(T_wc)
        feedback_stamp = self._check_head(feedback, now)
        if not 0 <= now-stamp <= .25 or abs(feedback_stamp-stamp) > .05:
            raise ValueError("Observation and head feedback must be fresh and synchronized")
        if self.pending and now-self.request_time > self.timeout_sec:
            self.phase = "timeout"
        if self.pending and (self.last_feedback_stamp is None or feedback_stamp > self.last_feedback_stamp):
            self.feedback(feedback, now)
        if self.phase == "waiting_for_move":
            if now-self.request_time > self.timeout_sec:
                self.phase = "timeout"
                raise ValueError("Viewpoint change timed out; no further request sent")
            raise ValueError("Waiting for measured settled head feedback")
        if self.phase == "timeout":
            raise ValueError("Reset session after timeout")
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError("New viewpoint needs a new image")
        if self.pending:
            if stamp <= self.settled_stamp:
                raise ValueError("Need an image captured after head settling")
            if any(abs(feedback["positions_rad"][j]-self.pending["head_positions_rad"][j]) > .02
                   for j in ("head_y", "head_z")):
                raise ValueError("Head moved away from requested viewpoint")
            distance, angle = baseline_and_angle(self.previous_pose, T, self.center)
            if distance < .01 or angle < 1.0:
                raise ValueError("Measured optical-centre displacement is insufficient for 3D")
            expected = transform(self.pending["T_wc"])
            translation_error = np.linalg.norm(expected[:3, 3]-T[:3, 3])
            rotation_error = np.degrees(np.arccos(np.clip((np.trace(expected[:3, :3].T @ T[:3, :3])-1)/2, -1, 1)))
            if translation_error > .01 or rotation_error > 3:
                raise ValueError("Measured camera pose disagrees with verified viewpoint catalog")
        return T

    def observe(self, T_wc, timestamp_sec, feedback, now):
        T = self.check_observation(T_wc, timestamp_sec, feedback, now)
        if self.pending:
            self.visited.add(self.pending["name"])
            self.pending = None
        self.last_stamp = timestamp_sec
        self.previous_pose = T
        self.phase = "observe"
        return self.propose(T, feedback, now)

    def propose(self, T, feedback, now):
        self._check_head(feedback, now)
        eligible = []
        for candidate in self.candidates:
            if candidate["name"] in self.visited or candidate.get("verified") is not True or not candidate.get("evidence"):
                continue
            joints = candidate["head_positions_rad"]
            if set(joints) != {"head_y", "head_z"}:
                continue
            if any(not self.limits[j][0] <= finite(joints[j], j) <= self.limits[j][1] for j in joints):
                continue
            baseline, angle = baseline_and_angle(T, candidate["T_wc"], self.center)
            camera_pose = transform(candidate["T_wc"])
            point = camera_pose[:3, :3].T @ (self.center-camera_pose[:3, 3])
            if point[2] <= .1:
                continue
            if self.K is not None:
                pixel = self.K @ point
                u, v = pixel[:2]/pixel[2]
                if not 0 <= u < self.image_size[0] or not 0 <= v < self.image_size[1]:
                    continue
            if baseline >= .01 and angle >= 1.0:
                # Geometric coverage heuristic, not validated information gain.
                eligible.append((angle, baseline, candidate))
        if not eligible:
            return {"status": "no_verified_useful_viewpoint", "real_robot_motion_commanded": False}
        details = {"planner": "measured_catalog_geometric_coverage"}
        selected = None
        if self.view_scorer:
            selected, details = self.view_scorer(T, [row[2] for row in eligible])
            if selected is None and details.get("status") != "bootstrap_required":
                return {**details, "real_robot_motion_commanded": False}
        if selected is None:
            _, _, selected = max(eligible, key=lambda item: item[:2])
            details.update(planner="measured_catalog_geometric_coverage")
        self.pending = selected
        self.phase = "waiting_for_move"
        self.request_time = now
        self.stable_since = None
        self.last_feedback_stamp = None
        self.previous_head = None
        return {"status": "supervised_move_requested", "candidate": selected,
                **{k: v for k, v in details.items() if k != "status"}, "real_robot_motion_commanded": False}

    def feedback(self, feedback, now):
        now = finite(now, "now")
        stamp = self._check_head(feedback, now)
        if self.phase not in ("waiting_for_move", "awaiting_new_image"):
            return self.phase
        if now-self.request_time > self.timeout_sec:
            self.phase = "timeout"
            return self.phase
        if self.last_feedback_stamp is not None and stamp <= self.last_feedback_stamp:
            raise ValueError("Repeated joint message is not settling evidence")
        values = feedback["positions_rad"]
        near = all(abs(values[j]-self.pending["head_positions_rad"][j]) <= .02 for j in ("head_z", "head_y"))
        still = False
        if self.previous_head is not None:
            dt = stamp-self.last_feedback_stamp
            still = dt <= .25 and all(abs(values[j]-self.previous_head[j])/dt <= .02 for j in values)
        self.stable_since = (self.stable_since if self.stable_since is not None else stamp) if near and still else None
        self.previous_head = dict(values)
        self.last_feedback_stamp = stamp
        if not near or not still:
            self.phase = "waiting_for_move"
        if self.phase == "waiting_for_move" and self.stable_since is not None and stamp-self.stable_since >= self.settling_sec:
            self.settled_stamp = stamp
            self.phase = "awaiting_new_image"
        return self.phase
