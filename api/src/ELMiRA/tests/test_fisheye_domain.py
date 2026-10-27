"""Pixel inverse and virtual-view tests; no camera or robot access."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from active_vision.calibration import Calibration, board_points, native_board_pose


def record(model='equidistant'):
    return {'K': [[302., 0, 631.], [0, 302., 363.], [0, 0, 1]],
            'D': [-.02014, .00793, 0, 0] if model == 'equidistant' else [0]*4,
            'width': 1280, 'height': 720, 'distortion_model': model,
            'camera_id': 'SYNTHETIC', 'optical_frame': 'synthetic', 'verified': False}


class FisheyeDomainTests(unittest.TestCase):
    def test_folded_radial_model_rejects_peripheral_pixels(self):
        r = record()
        r['D'] = [-.0013972872, -.0572188986, .0771760697, -.029714717]
        c = Calibration(r, require_verified=False)
        result = c.pixel_domain_report([[631, 363], [310, 590]])
        self.assertFalse(result['all_pixels_valid'])
        self.assertEqual(result['invalid_indices'], [1])
        self.assertAlmostEqual(result['first_radial_fold_theta_deg'], 80.31, places=1)
        self.assertFalse(result['calibration_accuracy_verified_by_check'])

    def test_lower_order_model_roundtrip_is_not_accuracy_certification(self):
        c = Calibration(record(), require_verified=False)
        result = c.pixel_domain_report([[631, 363], [310, 590], [286, 620]])
        self.assertTrue(result['all_pixels_valid'])
        self.assertLess(result['max_roundtrip_error_px'], 1e-8)
        self.assertIsNone(result['first_radial_fold_theta_deg'])
        self.assertFalse(result['calibration_accuracy_verified_by_check'])
        with self.assertRaises(ValueError):
            c.pixel_domain_report([])

    def test_native_board_pnp_keeps_common_corner_origin(self):
        c = Calibration(record(), require_verified=False)
        obj = board_points(9, 6, .025).reshape(1, -1, 3).astype(float)
        rv, tv = np.array([.1, -.2, .03]), np.array([-.08, -.04, .7])
        pixels = cv2.fisheye.projectPoints(obj, rv, tv, c.K, c.D)[0].reshape(-1, 1, 2)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = cv2.Rodrigues(rv)[0], tv
        image = np.zeros((720, 1280, 3), np.uint8)
        with patch('active_vision.calibration.corners', return_value=pixels[::-1]):
            result = native_board_pose(image, c, 9, 6, .025, reference_pixels=pixels)
        np.testing.assert_allclose(result['T_wc'], np.linalg.inv(T), atol=1e-5)
        self.assertTrue(result['corner_order_reversed_for_reference'])
        self.assertLess(result['reprojection_rms_px'], 1e-4)
        self.assertFalse(result['pose_verified'])

    def test_virtual_rectification_rotates_axes_not_optical_center(self):
        for model in ('equidistant', 'plumb_bob'):
            c = Calibration(record(model), require_verified=False)
            mask = np.zeros((720, 1280), np.uint8)
            mask[555:615, 395:450] = 255
            image = np.full((720, 1280, 3), 30, np.uint8)
            image[mask > 0] = [10, 80, 200]
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(np.array([.1, -.1, .05]))[0]
            pose[:3, 3] = [.1, .2, .3]
            _, remapped, virtual, K = c.rectify_object(image, mask, pose)
            np.testing.assert_allclose(virtual[:3, 3], pose[:3, 3], atol=1e-12)
            yy, xx = np.nonzero(mask)
            uv = np.array([[[xx.mean(), yy.mean()]]])
            if model == 'equidistant':
                ray = cv2.fisheye.undistortPoints(uv, c.K, c.D).reshape(2)
            else:
                ray = cv2.undistortPoints(uv, c.K, c.D).reshape(2)
            world = pose[:3, :3] @ np.r_[ray, 1.]*.5 + pose[:3, 3]
            camera = virtual[:3, :3].T @ (world-virtual[:3, 3])
            projected = K @ camera
            np.testing.assert_allclose(projected[:2]/projected[2], K[:2, 2], atol=1e-8)
            self.assertGreater(np.count_nonzero(remapped), 100)


if __name__ == '__main__':
    unittest.main()
