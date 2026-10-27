"""OpenCV single-camera calibration, rectification and measured board poses."""
from pathlib import Path

import cv2
import numpy as np

from .common import positive, read_json, sha256, transform


class Calibration:
    def __init__(self, record, require_verified=True):
        self.record = record
        self.size = (int(record["width"]), int(record["height"]))
        self.K = np.asarray(record["K"], np.float64).reshape(3, 3)
        self.D = np.asarray(record["D"], np.float64).reshape(-1)
        self.model = record["distortion_model"]
        if (min(self.size) <= 0 or not np.isfinite(self.K).all() or not np.isfinite(self.D).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1]) or abs(self.K[0, 1]) > 1e-7):
            raise ValueError("Invalid camera intrinsics")
        if self.model not in ("plumb_bob", "equidistant"):
            raise ValueError("Supported models: plumb_bob or equidistant (fisheye)")
        if len(self.D) not in ((4,) if self.model == "equidistant" else (4, 5, 8)):
            raise ValueError("Distortion coefficient count does not match camera model")
        if not record.get("camera_id") or not record.get("optical_frame"):
            raise ValueError("Camera identity and optical frame are required")
        if require_verified and (record.get("verified") is not True
                                 or not record.get("verification_evidence")):
            raise ValueError("Calibration is unverified; collect same-camera calibration and holdout evidence")
        self.rectified_K = self.K.copy()
        if self.model == "equidistant":
            self.rectified_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, self.size, np.eye(3), balance=0.0)
            self.maps = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.rectified_K, self.size, cv2.CV_32FC1)
        else:
            self.maps = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.rectified_K, self.size, cv2.CV_32FC1)

    @classmethod
    def load(cls, path, require_verified=True):
        return cls(read_json(path), require_verified)

    def rectify(self, bgr, mask=None):
        if bgr is None or (bgr.shape[1], bgr.shape[0]) != self.size:
            raise ValueError("Image resolution differs from calibration; refusing guessed scaling")
        image = cv2.remap(bgr, *self.maps, interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        if mask.shape != bgr.shape[:2]:
            raise ValueError("Mask and image are not aligned")
        return image, cv2.remap(mask, *self.maps, interpolation=cv2.INTER_NEAREST)

    def pixel_domain_report(self, pixels, max_roundtrip_px=1.0):
        """Numerical inverse check only; never certifies physical accuracy."""
        uv = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        if not len(uv) or not np.isfinite(uv).all():
            raise ValueError('Pixel-domain check requires finite, nonempty image coordinates')
        fold_angle, fold_radius = None, None
        if self.model == 'equidistant':
            normalized = cv2.fisheye.undistortPoints(uv, self.K, self.D)
            restored = cv2.fisheye.distortPoints(normalized, self.K, self.D)
            k1, k2, k3, k4 = self.D
            roots = np.roots([9*k4, 7*k3, 5*k2, 3*k1, 1])
            turns = [float(np.sqrt(r.real)) for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
            if turns:
                theta = min(turns)
                fold_angle = float(np.degrees(theta))
                fold_radius = float(theta*(1+k1*theta**2+k2*theta**4+k3*theta**6+k4*theta**8))
        else:
            normalized = cv2.undistortPoints(uv, self.K, self.D)
            xyz = np.c_[normalized.reshape(-1, 2), np.ones(len(uv))]
            restored = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.K, self.D)[0]
        error = np.linalg.norm(restored.reshape(-1, 2)-uv.reshape(-1, 2), axis=1)
        flat = uv.reshape(-1, 2)
        valid = (np.isfinite(normalized.reshape(-1, 2)).all(axis=1)
                 & np.isfinite(error) & (error <= positive(max_roundtrip_px, 'max_roundtrip_px'))
                 & (flat[:, 0] >= 0) & (flat[:, 0] < self.size[0])
                 & (flat[:, 1] >= 0) & (flat[:, 1] < self.size[1]))
        if fold_radius is not None:
            radii = np.linalg.norm((flat-self.K[:2, 2])/np.diag(self.K)[:2], axis=1)
            valid &= radii < fold_radius
        finite_errors = error[np.isfinite(error)]
        return {'sample_count': len(uv), 'valid_count': int(valid.sum()),
                'invalid_count': int((~valid).sum()), 'all_pixels_valid': bool(valid.all()),
                'invalid_indices': np.flatnonzero(~valid).tolist(),
                'max_roundtrip_error_px': float(finite_errors.max()) if len(finite_errors) else None,
                'median_roundtrip_error_px': float(np.median(finite_errors)) if len(finite_errors) else None,
                'first_radial_fold_theta_deg': fold_angle,
                'maximum_monotonic_normalized_radius': fold_radius,
                'calibration_accuracy_verified_by_check': False}

    def rectify_object(self, bgr, mask, T_wc):
        """Rotate a virtual pinhole view toward the mask; no camera displacement."""
        if bgr is None or (bgr.shape[1], bgr.shape[0]) != self.size or mask.shape != bgr.shape[:2]:
            raise ValueError('Native image/mask resolution differs from calibration')
        yy, xx = np.nonzero(mask)
        if not len(xx) or not self.pixel_domain_report(np.c_[xx, yy])['all_pixels_valid']:
            raise ValueError('Object mask leaves the calibration inverse domain')
        centroid = np.array([[[float(xx.mean()), float(yy.mean())]]])
        if self.model == 'equidistant':
            normal = cv2.fisheye.undistortPoints(centroid, self.K, self.D).reshape(2)
        else:
            normal = cv2.undistortPoints(centroid, self.K, self.D).reshape(2)
        forward = np.r_[normal, 1.]
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        old_from_virtual = np.stack((right, np.cross(forward, right), forward), axis=1)
        K = self.K.copy()
        K[:2, 2] = [(self.size[0]-1)/2., (self.size[1]-1)/2.]
        if self.model == 'equidistant':
            maps = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, old_from_virtual.T, K, self.size, cv2.CV_32FC1)
        else:
            maps = cv2.initUndistortRectifyMap(
                self.K, self.D, old_from_virtual.T, K, self.size, cv2.CV_32FC1)
        image = cv2.remap(bgr, *maps, interpolation=cv2.INTER_LINEAR)
        binary = cv2.remap(mask, *maps, interpolation=cv2.INTER_NEAREST)
        rotation = np.eye(4)
        rotation[:3, :3] = old_from_virtual
        virtual_pose = transform(transform(T_wc) @ rotation)
        return image, binary, virtual_pose, K


def board_points(columns, rows, square_m):
    if columns < 3 or rows < 3:
        raise ValueError("Board requires at least 3x3 inner corners")
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * positive(square_m, "square_m")
    return points


def corners(bgr, columns, rows):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ok, pixels = cv2.findChessboardCorners(gray, (columns, rows))
    if not ok and hasattr(cv2, "findChessboardCornersSB"):
        # The real upright NICO board is missed by the legacy detector.
        ok, pixels = cv2.findChessboardCornersSB(
            gray, (columns, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return pixels
    if not ok:
        raise ValueError("Calibration board not detected")
    return cv2.cornerSubPix(gray, pixels, (5, 5), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, .001))


def calibrate_images(paths, columns, rows, square_m, camera_id, optical_frame, model):
    world = board_points(columns, rows, square_m)
    images, objects, sources, size = [], [], [], None
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError("Unreadable calibration image: " + str(path))
        this_size = (bgr.shape[1], bgr.shape[0])
        if size and size != this_size:
            raise ValueError("Mixed calibration resolutions")
        size = this_size
        try:
            pixels = corners(bgr, columns, rows)
        except ValueError:
            continue
        images.append(pixels)
        objects.append(world.copy())
        sources.append({"path": str(Path(path).resolve()), "sha256": sha256(path)})
    if len(images) < 10 or len({s["sha256"] for s in sources}) < 10:
        raise ValueError("Need at least 10 distinct detected board views, with varied tilts and coverage")
    if model == "equidistant":
        objects = [o.reshape(1, -1, 3).astype(np.float64) for o in objects]
        images = [i.astype(np.float64) for i in images]
        rms, K, D, rotations, translations = cv2.fisheye.calibrate(
            objects, images, size, None, None,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
        project = cv2.fisheye.projectPoints
    elif model == "plumb_bob":
        rms, K, D, rotations, translations = cv2.calibrateCamera(objects, images, size, None, None)
        project = cv2.projectPoints
    else:
        raise ValueError("Unknown distortion model")
    errors = []
    for obj, pixels, r, t in zip(objects, images, rotations, translations):
        reprojection, _ = project(obj, r, t, K, D)
        errors.append(float(np.sqrt(np.mean(np.sum((reprojection.reshape(-1, 2)
                                                   - pixels.reshape(-1, 2)) ** 2, axis=1)))))
    return {"camera_id": camera_id, "optical_frame": optical_frame,
            "width": size[0], "height": size[1], "K": K.tolist(), "D": D.ravel().tolist(),
            "distortion_model": model, "rms_px": float(rms), "per_view_rms_px": errors,
            "board_columns": columns, "board_rows": rows, "square_m": square_m,
            "source_images": sources, "verified": False, "verification_evidence": "",
            "note": "Fit only. Verify held-out corners, focus/zoom, device identity and metric scale."}


def board_pose(rectified_bgr, K, columns, rows, square_m, max_error_px=1.0,
               reference_pixels=None):
    """Fixed board defines world; never use moving board as a fixed camera reference."""
    obj = board_points(columns, rows, square_m)
    pixels = corners(rectified_bgr, columns, rows)
    reversed_order = False
    if reference_pixels is not None:
        reference = np.asarray(reference_pixels, np.float64).reshape(-1, 2)
        current = pixels.reshape(-1, 2)
        if reference.shape != current.shape or not np.isfinite(reference).all():
            raise ValueError("Invalid fixed-board corner reference")
        a, b = current-current.mean(axis=0), reference-reference.mean(axis=0)
        if np.linalg.norm(a[::-1]-b) < np.linalg.norm(a-b):
            pixels = pixels[::-1].copy()
            reversed_order = True
    ok, r, t = cv2.solvePnP(obj, pixels, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or float(t[2, 0]) <= 0:
        raise ValueError("Board pose unavailable or behind camera")
    reprojection, _ = cv2.projectPoints(obj, r, t, K, np.zeros(5))
    rms = float(np.sqrt(np.mean(np.sum((pixels - reprojection) ** 2, axis=2))))
    if rms > positive(max_error_px, "max_error_px"):
        raise ValueError("Board pose reprojection error too high: %.3f px" % rms)
    T_cw = np.eye(4)
    T_cw[:3, :3] = cv2.Rodrigues(r)[0]
    T_cw[:3, 3] = t.ravel()
    return {"T_wc": transform(np.linalg.inv(T_cw)).tolist(), "reprojection_rms_px": rms,
            "corner_order_reversed_for_reference": reversed_order,
            "pose_source": "fixed_metric_chessboard_pnp",
            "warning": "Board origin/corner ordering must stay fixed; symmetric boards can flip. Verify manually."}


def native_board_pose(bgr, calibration, columns, rows, square_m, reference_pixels=None,
                      max_error_px=1.0):
    """PnP from native corners without clipping fisheye into one pinhole image."""
    if (bgr.shape[1], bgr.shape[0]) != calibration.size:
        raise ValueError('Native board image resolution differs from calibration')
    pixels = corners(bgr, columns, rows)
    reversed_order = False
    if reference_pixels is not None:
        reference = np.asarray(reference_pixels, np.float64).reshape(-1, 2)
        current = pixels.reshape(-1, 2)
        if reference.shape != current.shape or not np.isfinite(reference).all():
            raise ValueError('Invalid native fixed-board reference')
        a, b = current-current.mean(axis=0), reference-reference.mean(axis=0)
        if np.linalg.norm(a[::-1]-b) < np.linalg.norm(a-b):
            pixels = pixels[::-1].copy()
            reversed_order = True
    if not calibration.pixel_domain_report(pixels)['all_pixels_valid']:
        raise ValueError('Board corners leave the calibration inverse domain')
    objects = board_points(columns, rows, square_m)
    if calibration.model == 'equidistant':
        undistorted = cv2.fisheye.undistortPoints(pixels, calibration.K, calibration.D,
                                                P=calibration.K)
        ok, r, t = cv2.solvePnP(objects, undistorted, calibration.K, np.zeros(5))
    else:
        ok, r, t = cv2.solvePnP(objects, pixels, calibration.K, calibration.D)
    if not ok or float(t[2, 0]) <= 0:
        raise ValueError('Native board pose unavailable or behind camera')
    if calibration.model == 'equidistant':
        projected = cv2.fisheye.projectPoints(objects.reshape(1, -1, 3).astype(float),
                                             r, t, calibration.K, calibration.D)[0]
    else:
        projected = cv2.projectPoints(objects, r, t, calibration.K, calibration.D)[0]
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-pixels.reshape(-1, 2))**2, axis=1))))
    if rms > positive(max_error_px, 'max_error_px'):
        raise ValueError('Native board pose reprojection error too high: %.3f px' % rms)
    T_cw = np.eye(4)
    T_cw[:3, :3], T_cw[:3, 3] = cv2.Rodrigues(r)[0], t.ravel()
    return {'T_wc': transform(np.linalg.inv(T_cw)).tolist(), 'reprojection_rms_px': rms,
            'reprojection_space': 'native_distorted_pixels',
            'inner_corners_px': pixels.reshape(-1, 2).tolist(),
            'corner_order_reversed_for_reference': reversed_order,
            'pose_source': 'fixed_board_native_corner_pnp', 'pose_verified': False}


def check_holdout(calibration_path, paths, max_error_px=1.0):
    """Evaluate original distorted pixels, not the calibration training views."""
    calibration = Calibration.load(calibration_path, require_verified=False)
    record = calibration.record
    training = {row["sha256"] for row in record.get("source_images", [])}
    obj = board_points(record["board_columns"], record["board_rows"], record["square_m"])
    results, seen = [], set()
    limit = positive(max_error_px, "max_error_px")
    for path in paths:
        digest = sha256(path)
        if digest in training or digest in seen:
            raise ValueError("Holdout image duplicates training/another holdout image")
        seen.add(digest)
        image = cv2.imread(str(path))
        if image is None or (image.shape[1], image.shape[0]) != calibration.size:
            raise ValueError("Holdout image unavailable or wrong resolution")
        try:
            pixels = corners(image, record["board_columns"], record["board_rows"])
            if calibration.model == "equidistant":
                corrected = cv2.fisheye.undistortPoints(pixels, calibration.K, calibration.D, P=calibration.K)
                ok, r, t = cv2.solvePnP(obj, corrected, calibration.K, np.zeros(5))
                reproj, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3).astype(float), r, t, calibration.K, calibration.D)
            else:
                ok, r, t = cv2.solvePnP(obj, pixels, calibration.K, calibration.D)
                reproj, _ = cv2.projectPoints(obj, r, t, calibration.K, calibration.D)
            if not ok or t[2, 0] <= 0:
                raise ValueError("Holdout board pose failed")
            error = float(np.sqrt(np.mean(np.sum((pixels.reshape(-1, 2)-reproj.reshape(-1, 2))**2, axis=1))))
            results.append({"image": str(path), "sha256": digest, "rms_px": error, "passed": error <= limit})
        except (ValueError, cv2.error) as exc:
            results.append({"image": str(path), "sha256": digest, "passed": False, "reason": str(exc)})
    if len(results) < 3:
        raise ValueError("Need at least 3 distinct held-out calibration-board images")
    return {"calibration_sha256": sha256(calibration_path), "holdout": results,
            "threshold_px": limit, "pixel_check_passed": all(row["passed"] for row in results),
            "verified": False, "real_robot_motion_commanded": False,
            "remaining": "Still verify device identity, focus/zoom, board size, coverage and independent metric distances."}
