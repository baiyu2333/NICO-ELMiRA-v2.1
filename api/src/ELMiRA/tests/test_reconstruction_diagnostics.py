"""Viewpoint observability controls; never use hardware or real cube fitting."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from active_vision.reconstruction_diagnostics import axial_control_views, cube_observability


class ReconstructionDiagnosticsTests(unittest.TestCase):
    def test_virtual_axes_are_rigid_optical_to_world_transforms(self):
        center = np.array([.2, .1, -.1])
        poses = axial_control_views(center)
        self.assertEqual(len(poses), 6)
        for pose in poses:
            self.assertAlmostEqual(float(np.linalg.det(pose[:3, :3])), 1.)
            self.assertAlmostEqual(float(np.linalg.norm(pose[:3, 3]-center)), .25)
            camera = pose[:3, :3].T@(center-pose[:3, 3])
            np.testing.assert_allclose(camera, [0, 0, .25], atol=1e-12)

    def test_narrow_views_do_not_determine_a_complete_cube(self):
        K = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]])
        a, b = np.eye(4), np.eye(4)
        a[:3, 3], b[:3, 3] = [-.01, 0, -.25], [.01, 0, -.25]
        results, _ = cube_observability([a, b], K, (640, 480), np.zeros(3), .064)
        narrow, wide = results['same_candidate_views'], results['six_virtual_axial_views']
        self.assertTrue(narrow['touches_bounds'])
        self.assertGreater(max(narrow['span_mm']), 90.)
        self.assertFalse(wide['touches_bounds'])
        np.testing.assert_allclose(wide['span_mm'], [36]*3, atol=4.)
        self.assertTrue(wide['synthetic_only'])
        self.assertFalse(wide['robot_motion_commanded'])


if __name__ == '__main__':
    unittest.main()
