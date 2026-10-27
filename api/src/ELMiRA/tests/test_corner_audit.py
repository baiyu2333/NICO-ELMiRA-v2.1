"""Independent synthetic controls for offline, native-pixel landmark checks."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from active_vision.calibration import Calibration
from active_vision.corner_audit import triangulate_native
from active_vision.sparse_preview import native_pixels


class CornerAuditTests(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration({'width': 640, 'height': 480, 'K': [[250, 0, 320], [0, 250, 240], [0, 0, 1]],
                                'D': [-.02, .008, 0, 0], 'distortion_model': 'equidistant',
                                'camera_id': 'synthetic', 'optical_frame': 'synthetic'}, require_verified=False)
        self.poses = [np.eye(4) for _ in range(3)]
        self.poses[0][0, 3], self.poses[2][0, 3] = -.04, .04
        self.xyz = np.array([[-.02, -.015, .25], [.015, .02, .3], [.03, -.02, .22]])
        self.pixels = np.asarray([native_pixels(self.xyz, {'pose': p}, self.cal)[0] for p in self.poses])

    def test_known_geometry_predicts_held_out_native_pixels(self):
        result = triangulate_native(self.cal, self.poses, self.pixels, (0, 2))
        np.testing.assert_allclose(result['xyz_m'], self.xyz, atol=1e-9)
        self.assertLess(result['held_out_max_error_px'], 1e-6)
        self.assertEqual(result['held_out_views'], [1])
        self.assertTrue(result['positive_depth_in_all_views'])

    def test_held_out_mismatch_is_reported_not_fitted_away(self):
        self.pixels[1] += [8, 0]
        result = triangulate_native(self.cal, self.poses, self.pixels, (0, 2))
        self.assertAlmostEqual(result['held_out_median_error_px'], 8)
        self.assertEqual(result['held_out_over_2px_count'], 3)
        np.testing.assert_allclose(result['xyz_m'], self.xyz, atol=1e-9)

    def test_zero_baseline_and_missing_holdout_are_rejected(self):
        with self.assertRaises(ValueError):
            triangulate_native(self.cal, [np.eye(4)]*3, self.pixels, (0, 2))
        with self.assertRaises(ValueError):
            triangulate_native(self.cal, self.poses[:2], self.pixels[:2], (0, 1))
        with self.assertRaises(ValueError):
            triangulate_native(self.cal, self.poses, self.pixels, (0, 0))


if __name__ == '__main__':
    unittest.main()
