"""Synthetic reconstruction checks, without robot or camera access."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from active_vision.common import write_json
from active_vision.calibration import Calibration
from active_vision.common import read_json
from active_vision.sparse_preview import reconstruct_preview, silhouette_consistent, undistort_pixels


class SparsePreviewTests(unittest.TestCase):
    def fixture(self, root, baseline=.04):
        rng = np.random.default_rng(19)
        image = np.full((360, 480, 3), 25, np.uint8)
        texture = rng.integers(30, 245, (120, 140), dtype=np.uint8)
        image[120:240, 170:310] = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[120:240, 170:310] = 255
        shift = np.array([[1., 0, -16], [0, 1., 0]])
        image2 = cv2.warpAffine(image, shift, (480, 360))
        mask2 = cv2.warpAffine(mask, shift, (480, 360), flags=cv2.INTER_NEAREST)
        for name, value in (('first.png', image), ('second.png', image2),
                            ('mask1.png', mask), ('mask2.png', mask2)):
            cv2.imwrite(str(root/name), value)
        write_json(root/'calibration.json', {
            'K': [[240, 0, 240], [0, 240, 180], [0, 0, 1]], 'D': [0]*4,
            'width': 480, 'height': 360, 'distortion_model': 'plumb_bob',
            'camera_id': 'SYNTHETIC', 'optical_frame': 'synthetic_optical', 'verified': False})
        first, second = np.eye(4), np.eye(4)
        second[0, 3] = baseline
        manifest = {'object_static': True, 'world_frame': 'synthetic_plane',
                    'units': 'metres', 'calibration': 'calibration.json', 'observations': []}
        for image_path, mask_path, pose in (('first.png', 'mask1.png', first),
                                            ('second.png', 'mask2.png', second)):
            manifest['observations'].append({'image': image_path, 'mask': mask_path,
                'camera_id': 'SYNTHETIC', 'pose_reprojection_rms_px': .1, 'T_wc': pose.tolist()})
        write_json(root/'manifest.json', manifest)
        return manifest

    def test_known_plane_depth_and_exports_keep_unverified_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = reconstruct_preview(root/'manifest.json', root/'out', 'akaze')
            cloud = np.load(str(root/'out/experimental_object_cloud.npz'))['xyz']
            self.assertGreater(len(cloud), 20)
            self.assertAlmostEqual(float(np.median(cloud[:, 2])), .6, delta=.02)
            self.assertFalse(result['verified_for_reconstruction'])
            self.assertFalse(result['object_shape_reconstruction_validated'])
            self.assertFalse(result['real_robot_motion_commanded'])
            self.assertFalse(result['experimental_visual_hull']['volume_verified'])
            self.assertFalse(result['experimental_visual_hull']['complete_object_surface'])
            self.assertFalse(result['experimental_visual_hull']['usable_as_validated_object_shape'])
            hull = np.load(str(root/'out/experimental_visual_hull/experimental_object_cloud.npz'))['xyz']
            manifest = read_json(root/'manifest.json')
            frames = [{'pose': np.asarray(view['T_wc']),
                       'mask': cv2.imread(str(root/view['mask']), 0)}
                      for view in manifest['observations']]
            calibration = Calibration.load(root/'calibration.json', require_verified=False)
            self.assertTrue(np.all(silhouette_consistent(hull, frames, calibration)))
            self.assertTrue((root/'out/experimental_object_cloud.ply').is_file())
            self.assertTrue((root/'out/experimental_object_cloud.png').is_file())

    def test_empty_feature_coordinates_do_not_crash_opencv(self):
        class NoCalibrationAccess:
            def __getattr__(self, name):
                raise AssertionError('Empty inputs must return before using OpenCV')
        self.assertEqual(undistort_pixels([], NoCalibrationAccess()).shape, (0, 2))
        self.assertEqual(silhouette_consistent([], [{}], NoCalibrationAccess()).shape, (0,))

    def test_insufficient_baseline_reports_no_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, baseline=.005)
            result = reconstruct_preview(root/'manifest.json', root/'out', 'akaze')
            self.assertEqual(result['status'], 'insufficient_points')
            self.assertEqual(result['point_count'], 0)
            self.assertEqual(result['pairs'][0]['status'], 'baseline_below10mm')

    def test_duplicate_images_and_moving_setup_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest['observations'][1]['image'] = 'first.png'
            write_json(root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                reconstruct_preview(root/'manifest.json', root/'out')
            manifest['object_static'] = False
            write_json(root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'stationary'):
                reconstruct_preview(root/'manifest.json', root/'out')


if __name__ == '__main__':
    unittest.main()
