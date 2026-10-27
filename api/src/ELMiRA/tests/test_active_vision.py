"""Offline tests only. No ROS master, camera, depth frames or motor commands."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from active_vision.calibration import Calibration, board_points, board_pose, calibrate_images, check_holdout, corners
from active_vision.common import read_json, transform, write_json
from active_vision.geometry import GuardedGeometry, baseline_and_angle, triangulate_checked
from active_vision.masks import mask_metrics, segment, validate_mask
from active_vision.recognition import evaluate_dataset
from active_vision.ros1 import ReadOnlyNode, decode_image, ros_status
from active_vision.view_planner import RGBViewScorer, supported_proxy_depth
from active_vision.vendor.acf3d.memory import LifelongPrototypeMemory
from active_vision.session import ObserveMoveObserve, reconstruct, validate_observation


def calibration_record(model="plumb_bob"):
    return {"K": [[200, 0, 120], [0, 200, 100], [0, 0, 1]], "D": [0]*4,
            "width": 240, "height": 200, "distortion_model": model,
            "camera_id": "SYNTHETIC_TEST_CAMERA", "optical_frame": "test_optical",
            "verified": True, "verification_evidence": "synthetic known camera, not NICO calibration"}


def pose(x=0):
    result = np.eye(4)
    result[0, 3] = x
    return result


def observation(stamp=1):
    return {"timestamp_sec": stamp, "pose_timestamp_sec": stamp,
            "pose_verified": True, "pose_evidence": "synthetic fixture",
            "pose_source": "measured_external", "mask_reviewed": True,
            "mask_evidence": "analytic test silhouette", "object_static": True,
            "camera_id": "SYNTHETIC_TEST_CAMERA", "optical_frame": "test_optical",
            "world_frame": "test_world", "units": "metres", "T_wc": pose().tolist()}


def feedback(t, yaw=0):
    return {"timestamp_sec": t, "positions_rad": {"head_z": yaw, "head_y": -.5}}


def loop_fixture():
    return ObserveMoveObserve([0, 0, .5],
        {"verified": True, "evidence": "synthetic limits only", "head_z": [-.2, .2], "head_y": [-.6, -.4]},
        [{"name": "test_view", "T_wc": pose(.05).tolist(), "head_positions_rad": {"head_z": .1, "head_y": -.5},
          "verified": True, "evidence": "synthetic measured-pose catalog"}], settling_sec=.2)


class MaskTests(unittest.TestCase):
    def test_grabcut_pixels_not_box(self):
        image = np.full((120, 160, 3), 30, np.uint8)
        truth = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(image, (80, 60), 20, (10, 160, 240), -1)
        cv2.circle(truth, (80, 60), 20, 255, -1)
        mask, report = segment(image, [50, 30, 60, 60])
        self.assertGreater(mask_metrics(mask, truth)["iou"], .98)
        self.assertFalse(report["mask_reviewed"])
        self.assertLess(np.count_nonzero(mask), 60*60)

    def test_bad_masks(self):
        for mask in (None, np.zeros((10, 10), np.uint8), np.ones((10, 10), np.uint8), np.full((10, 10), 7, np.uint8)):
            with self.assertRaises(ValueError):
                validate_mask(mask, (10, 10))

    def test_fragment_rejected(self):
        image = np.full((120, 160, 3), 30, np.uint8)
        with self.assertRaisesRegex(ValueError, "fragment"):
            segment(image, [40, 20, 80, 80], foreground_point=[80, 60])


class CalibrationTests(unittest.TestCase):
    def test_unverified_and_bad_model_rejected(self):
        record = calibration_record()
        record["verified"] = False
        with self.assertRaisesRegex(ValueError, "unverified"):
            Calibration(record)
        record = calibration_record()
        record["distortion_model"] = "guessed"
        with self.assertRaises(ValueError):
            Calibration(record)

    def test_rectification_models_preserve_binary_mask(self):
        for model in ("plumb_bob", "equidistant"):
            calibration = Calibration(calibration_record(model))
            image = np.zeros((200, 240, 3), np.uint8)
            mask = np.zeros((200, 240), np.uint8)
            mask[70:90, 100:120] = 255
            rectified, result = calibration.rectify(image, mask)
            self.assertEqual(rectified.shape, image.shape)
            self.assertTrue(set(np.unique(result)).issubset({0, 255}))
            with self.assertRaises(ValueError):
                calibration.rectify(image[:100])

    def test_rigid_transform_guard(self):
        for invalid in (np.eye(3), np.zeros((4, 4)), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                transform(invalid)

    def test_board_corner_detection_on_rendered_image(self):
        image = np.full((240, 320, 3), 127, np.uint8)
        for y in range(6):
            for x in range(8):
                value = 255 if (x+y) % 2 else 0
                image[48+y*24:48+(y+1)*24, 64+x*24:64+(x+1)*24] = value
        found = corners(image, 7, 5)
        self.assertEqual(found.shape, (35, 1, 2))
        result = board_pose(image, np.array([[200, 0, 160], [0, 200, 120], [0, 0, 1.]]), 7, 5, .02)
        self.assertLess(result["reprojection_rms_px"], .01)

    def test_pnp_known_corners(self):
        K = np.array(calibration_record()["K"], float)
        points = board_points(7, 5, .02)
        r = np.array([.1, -.2, .05])
        t = np.array([-.05, -.05, .7])
        pixels, _ = cv2.projectPoints(points, r, t, K, np.zeros(5))
        with patch("active_vision.calibration.corners", return_value=pixels):
            result = board_pose(np.zeros((200, 240, 3), np.uint8), K, 7, 5, .02)
        expected = np.eye(4)
        expected[:3, :3] = cv2.Rodrigues(r)[0]
        expected[:3, 3] = t
        self.assertTrue(np.allclose(result["T_wc"], np.linalg.inv(expected), atol=1e-5))
        self.assertLess(result["reprojection_rms_px"], .001)

    def test_sector_fallback_recovers_rendered_board_when_legacy_misses(self):
        image = np.full((240, 320, 3), 127, np.uint8)
        for y in range(6):
            for x in range(8):
                image[48+y*24:48+(y+1)*24, 64+x*24:64+(x+1)*24] = (
                    255 if (x+y) % 2 else 0)
        with patch('active_vision.calibration.cv2.findChessboardCorners', return_value=(False, None)):
            result = board_pose(image, np.array([[200, 0, 160], [0, 200, 120], [0, 0, 1.]]),
                                7, 5, .02)
        self.assertLess(result['reprojection_rms_px'], .1)

    def test_reversed_corner_order_retains_fixed_board_origin(self):
        K = np.array(calibration_record()['K'], float)
        points = board_points(7, 5, .02)
        r, t = np.array([.1, -.2, .05]), np.array([-.05, -.05, .7])
        pixels = cv2.projectPoints(points, r, t, K, np.zeros(5))[0]
        with patch('active_vision.calibration.corners', return_value=pixels[::-1].copy()):
            result = board_pose(np.zeros((200, 240, 3), np.uint8), K, 7, 5, .02,
                                reference_pixels=pixels)
        expected = np.eye(4)
        expected[:3, :3], expected[:3, 3] = cv2.Rodrigues(r)[0], t
        self.assertTrue(result['corner_order_reversed_for_reference'])
        self.assertTrue(np.allclose(result['T_wc'], np.linalg.inv(expected), atol=1e-5))

    def test_calibration_fits_known_corner_sets_without_claiming_verified(self):
        K = np.array(calibration_record()["K"], float)
        points = board_points(7, 5, .02)
        rng = np.random.default_rng(9)
        pixels = []
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for index in range(12):
                path = Path(directory) / (str(index) + ".png")
                cv2.imwrite(str(path), np.full((200, 240, 3), index, np.uint8))
                files.append(path)
                r = rng.uniform(-.35, .35, 3)
                t = np.array([rng.uniform(-.1, 0), rng.uniform(-.08, 0), rng.uniform(.35, .7)])
                pixels.append(cv2.projectPoints(points, r, t, K, np.zeros(5))[0])
            with patch("active_vision.calibration.corners", side_effect=pixels):
                result = calibrate_images(files, 7, 5, .02, "synthetic", "test_optical", "plumb_bob")
        self.assertLess(result["rms_px"], .001)
        self.assertTrue(np.allclose(result["K"], K, atol=.1))
        self.assertFalse(result["verified"])

    def test_fisheye_calibration_and_holdout(self):
        K = np.array(calibration_record()["K"], float)
        D = np.array([.04, -.02, .001, -.002])
        points = board_points(7, 5, .02).reshape(1, -1, 3).astype(float)
        rng = np.random.default_rng(13)
        pixels = []
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for index in range(15):
                path = Path(directory) / (str(index)+".png")
                cv2.imwrite(str(path), np.full((200, 240, 3), index, np.uint8))
                files.append(path)
                r = rng.uniform(-.4, .4, 3)
                t = np.array([rng.uniform(-.1, 0), rng.uniform(-.08, 0), rng.uniform(.3, .6)])
                pixels.append(cv2.fisheye.projectPoints(points, r, t, K, D)[0].reshape(-1, 1, 2))
            with patch("active_vision.calibration.corners", side_effect=pixels[:12]):
                result = calibrate_images(files[:12], 7, 5, .02, "synthetic", "test_optical", "equidistant")
            self.assertLess(result["rms_px"], .001)
            self.assertTrue(np.allclose(result["K"], K, atol=.1))
            path = Path(directory) / "calibration.json"
            write_json(path, result)
            with patch("active_vision.calibration.corners", side_effect=pixels[12:]):
                holdout = check_holdout(path, files[12:])
            self.assertTrue(holdout["pixel_check_passed"])
            self.assertFalse(holdout["verified"])
            with self.assertRaisesRegex(ValueError, "duplicates"):
                check_holdout(path, files[:3])


class GeometryTests(unittest.TestCase):
    def test_triangulation_and_outlier_rejection(self):
        K = np.array(calibration_record()["K"], float)
        points = np.array([[-.02, .02, .5], [.02, -.02, .6], [.04, .03, .55]])
        pixels = []
        for T in (pose(-.05), pose(.05)):
            camera = points-T[:3, 3]
            h = camera @ K.T
            pixels.append(h[:, :2]/h[:, 2:])
        got = triangulate_checked(K, pose(-.05), pose(.05), *pixels)
        self.assertTrue(np.allclose(got, points, atol=1e-8))
        pixels[1][2, 1] += 40
        self.assertEqual(len(triangulate_checked(K, pose(-.05), pose(.05), *pixels)), 2)
        self.assertEqual(len(triangulate_checked(K, pose(), pose(), *pixels)), 0)

    def test_rotation_is_not_baseline(self):
        a, b = pose(), pose()
        b[:3, :3] = cv2.Rodrigues(np.array([0, .4, 0]))[0]
        baseline, angle = baseline_and_angle(a, b, [0, 0, .5])
        self.assertEqual(baseline, 0)
        self.assertEqual(angle, 0)

    def test_hull_no_single_view_geometry(self):
        K = np.array(calibration_record()["K"], float)
        geometry = GuardedGeometry(K, (240, 200), [0, 0, .5], .08, resolution=24)
        image = np.zeros((200, 240, 3), np.uint8)
        mask = np.zeros((200, 240), np.uint8)
        cv2.circle(mask, (140, 100), 20, 255, -1)
        self.assertTrue(geometry.update_checked(image, mask, pose(-.05))["accepted"])
        self.assertEqual(len(geometry.metric_hull()), 0)
        self.assertFalse(geometry.update_checked(image, mask, pose(-.05))["accepted"])
        mask[:] = 0
        cv2.circle(mask, (100, 100), 20, 255, -1)
        self.assertTrue(geometry.update_checked(image, mask, pose(.05))["accepted"])
        self.assertGreater(len(geometry.metric_hull()), 0)
        self.assertFalse(geometry.update_checked(image, mask, pose(-.05))["accepted"])

    def test_manifest_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "calibration.json", calibration_record())
            records = []
            for index, x in enumerate((-.05, .05)):
                image = np.zeros((200, 240, 3), np.uint8)
                mask = np.zeros((200, 240), np.uint8)
                cv2.circle(mask, (int(120-200*x/.5), 100), 20, 255, -1)
                cv2.imwrite(str(root / (str(index)+".png")), image)
                cv2.imwrite(str(root / (str(index)+"m.png")), mask)
                row = observation(index+1)
                row.update(image=str(index)+".png", mask=str(index)+"m.png", T_wc=pose(x).tolist())
                records.append(row)
            manifest = {"calibration": "calibration.json", "volume_verified": True,
                        "volume_evidence": "analytic synthetic sphere", "volume_center_m": [0, 0, .5],
                        "half_extent_m": .08, "observations": records}
            write_json(root / "session.json", manifest)
            result = reconstruct(root / "session.json", root / "out")
            self.assertEqual(result["status"], "coarse_multiview")
            self.assertEqual(result["accepted_views"], 2)
            self.assertFalse(result["depth_sensor_used"])
            self.assertTrue((root / "out/reconstruction.npz").exists())

    def test_pose_and_mask_provenance_required(self):
        calibration = Calibration(calibration_record())
        for key, value in (("pose_verified", False), ("mask_reviewed", False), ("pose_timestamp_sec", 5),
                           ("camera_id", "wrong"), ("object_static", False), ("units", "mm")):
            row = observation()
            row[key] = value
            with self.assertRaises(ValueError):
                validate_observation(row, calibration)
        with self.assertRaises(ValueError):
            validate_observation(observation(), calibration, previous_stamp=1)


class ActiveLoopTests(unittest.TestCase):
    def settled_loop(self):
        loop = loop_fixture()
        loop.observe(pose(), 1, feedback(1), 1)
        for stamp in (1.1, 1.2, 1.3, 1.4, 1.5):
            loop.feedback(feedback(stamp, .1), stamp)
        return loop

    def test_head_moves_again_after_settling(self):
        loop = self.settled_loop()
        self.assertEqual(loop.phase, 'awaiting_new_image')
        self.assertEqual(loop.feedback(feedback(1.6, 0), 1.6), 'waiting_for_move')
        with self.assertRaisesRegex(ValueError, 'settled'):
            loop.observe(pose(.05), 1.7, feedback(1.7, 0), 1.7)
        self.assertEqual(loop.visited, set())

    def test_pending_image_times_out_and_pose_mismatch_rejected(self):
        loop = self.settled_loop()
        with self.assertRaisesRegex(ValueError, 'catalog'):
            loop.observe(pose(.12), 1.6, feedback(1.6, .1), 1.6)
        self.assertEqual(loop.visited, set())
        self.assertEqual(loop.feedback(feedback(22, .1), 22), 'timeout')

    def test_target_behind_camera_or_out_of_frame_not_requested(self):
        for behind in (False, True):
            loop = loop_fixture()
            loop.K = np.array(calibration_record()['K'], float)
            loop.image_size = (240, 200)
            T = pose(2)
            if behind:
                T = pose(.05)
                T[:3, :3] = np.diag([-1, 1, -1])
            loop.candidates[0]['T_wc'] = T.tolist()
            self.assertEqual(loop.observe(pose(), 1, feedback(1), 1)['status'], 'no_verified_useful_viewpoint')

    def test_check_does_not_commit_observation(self):
        loop = self.settled_loop()
        loop.check_observation(pose(.05), 1.6, feedback(1.6, .1), 1.6)
        self.assertEqual(loop.last_stamp, 1)
        self.assertIsNotNone(loop.pending)
        self.assertEqual(loop.visited, set())

    def test_observe_supervised_move_settle_observe(self):
        loop = loop_fixture()
        request = loop.observe(pose(), 1, feedback(1), 1)
        self.assertEqual(request["status"], "supervised_move_requested")
        self.assertFalse(request["real_robot_motion_commanded"])
        with self.assertRaisesRegex(ValueError, "settled"):
            loop.observe(pose(.05), 1.05, feedback(1.05, .1), 1.05)
        for stamp in (1.1, 1.2, 1.3, 1.4, 1.5):
            loop.feedback(feedback(stamp, .1), stamp)
        self.assertEqual(loop.phase, "awaiting_new_image")
        result = loop.observe(pose(.05), 1.6, feedback(1.6, .1), 1.6)
        self.assertEqual(result["status"], "no_verified_useful_viewpoint")

    def test_unverified_limits_stale_feedback_bad_candidates(self):
        loop = loop_fixture()
        with self.assertRaises(ValueError):
            loop.observe(pose(), 1, feedback(.1), 1)
        loop.limits["verified"] = False
        with self.assertRaises(ValueError):
            loop.observe(pose(), 1, feedback(1), 1)
        loop = loop_fixture()
        loop.candidates[0]["head_positions_rad"]["head_z"] = 1
        self.assertEqual(loop.observe(pose(), 1, feedback(1), 1)["status"], "no_verified_useful_viewpoint")

    def test_timeout_and_duplicate_feedback(self):
        loop = loop_fixture()
        loop.observe(pose(), 1, feedback(1), 1)
        loop.feedback(feedback(1.1, .1), 1.1)
        with self.assertRaises(ValueError):
            loop.feedback(feedback(1.1, .1), 1.1)
        self.assertEqual(loop.feedback(feedback(22, .1), 22), "timeout")

    def test_commanded_pose_not_accepted_as_measured_displacement(self):
        loop = loop_fixture()
        loop.observe(pose(), 1, feedback(1), 1)
        for stamp in (1.1, 1.2, 1.3, 1.4, 1.5):
            loop.feedback(feedback(stamp, .1), stamp)
        with self.assertRaisesRegex(ValueError, "displacement"):
            loop.observe(pose(), 1.6, feedback(1.6, .1), 1.6)


class ROSContractTests(unittest.TestCase):
    def test_geometry_rejection_never_advances_loop(self):
        node = ReadOnlyNode.__new__(ReadOnlyNode)
        node.calibration = Calibration(calibration_record())
        node.last_stamp = None
        node.loop = Mock()
        node.geometry = Mock()
        node.geometry.update_checked.return_value = {'accepted': False}
        node.view_scorer = Mock()
        record = observation()
        record['head_feedback'] = feedback(1)
        image = np.zeros((200, 240, 3), np.uint8)
        mask = np.zeros((200, 240), np.uint8)
        mask[80:120, 100:140] = 255
        node.update_geometry(record, image, mask, 1)
        node.loop.check_observation.assert_called_once()
        node.loop.observe.assert_not_called()
        node.view_scorer.add_observation.assert_not_called()

    def test_decode_padded_rgb(self):
        msg = SimpleNamespace(encoding="rgb8", height=1, width=2, step=8,
                              data=bytes([10, 20, 30, 40, 50, 60, 0, 0]))
        self.assertEqual(decode_image(msg).tolist(), [[[30, 20, 10], [60, 50, 40]]])
        msg.step = 1
        with self.assertRaises(ValueError):
            decode_image(msg)

    def test_ros_unavailable_cleanly(self):
        with patch("active_vision.ros1.xmlrpc.client.ServerProxy", side_effect=ConnectionRefusedError("test")):
            result = ros_status()
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["real_robot_motion_commanded"])


class RecognitionTests(unittest.TestCase):
    class ToyEncoder:
        checkpoint_sha256 = "TEST_ONLY_NOT_REAL_ENCODER"

        def encode(self, image, mask):
            feature = image[mask > 0].mean(axis=0)
            return (feature/np.linalg.norm(feature)).astype(np.float32)

    def make_data(self, root):
        samples = []
        for split in ("enroll", "calibration", "test"):
            for index, label in enumerate(("one", "two", "unknown")):
                if split == "enroll" and label == "unknown":
                    continue
                name = split+"_"+label
                image = np.zeros((20, 20, 3), np.uint8)
                image[5:15, 5:15, index] = 200
                image[0, 0] = len(samples)  # Unique file content; no split duplicate.
                mask = np.zeros((20, 20), np.uint8)
                mask[5:15, 5:15] = 255
                cv2.imwrite(str(root / (name+".png")), image)
                cv2.imwrite(str(root / (name+"m.png")), mask)
                samples.append({"image": name+".png", "mask": name+"m.png", "split": split,
                                "session_id": split, "label": label, "mask_reviewed": True})
        write_json(root / "data.json", {"samples": samples})
        return samples

    def test_gallery_calibration_and_held_out_test_plumbing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_data(root)
            report = evaluate_dataset(root / "data.json", self.ToyEncoder(), root / "out")
            self.assertEqual(report["test_count"], 3)
            self.assertEqual(report["correct"], 3)
            self.assertTrue((root / "out/real_object_memory.npz").exists())

    def test_session_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.make_data(root)
            rows[-1]["session_id"] = "enroll"
            write_json(root / "data.json", {"samples": rows})
            with self.assertRaisesRegex(ValueError, "leakage"):
                evaluate_dataset(root / "data.json", self.ToyEncoder(), root / "out")


class ViewScorerTests(unittest.TestCase):
    def fixture(self):
        K = np.array(calibration_record()['K'], float)
        geometry = GuardedGeometry(K, (240, 200), [0, 0, .5], .08, resolution=24)
        memory = LifelongPrototypeMemory()
        memory.learn('synthetic_red', np.array([0., 0., 1.], np.float32))
        memory.learn('synthetic_blue', np.array([1., 0., 0.], np.float32))
        scorer = RGBViewScorer(geometry, RecognitionTests.ToyEncoder(), memory)
        for x in (-.05, .05):
            image = np.zeros((200, 240, 3), np.uint8)
            mask = np.zeros((200, 240), np.uint8)
            center = (int(120-200*x/.5), 100)
            cv2.circle(image, center, 20, (0, 0, 200), -1)
            cv2.circle(mask, center, 20, 255, -1)
            self.assertTrue(geometry.update_checked(image, mask, pose(x))['accepted'])
            scorer.add_observation(image, mask, pose(x))
        return geometry, scorer

    def test_original_professor_scorer_ranks_supported_rgb_proxy(self):
        geometry, scorer = self.fixture()
        selected, report = scorer(pose(.05), [dict(name='next', T_wc=pose(.10).tolist())])
        self.assertEqual(selected['name'], 'next')
        self.assertEqual(report['status'], 'ranked')
        self.assertTrue(np.isfinite(report['scores'][0]['score']))
        self.assertGreater(report['proxy']['gaussians'], 0)
        self.assertFalse(report['depth_sensor_used'])
        self.assertFalse(report['real_robot_motion_commanded'])

    def test_no_unknown_hull_or_invisible_candidate_ranking(self):
        geometry, scorer = self.fixture()
        selected, report = scorer(pose(.05), [dict(name='far', T_wc=pose(5).tolist())])
        self.assertIsNone(selected)
        self.assertEqual(report['status'], 'no_visible_unvisited_candidate')
        geometry.support[:] = 0
        self.assertEqual(np.count_nonzero(supported_proxy_depth(geometry, pose())), 0)
        selected, report = scorer(pose(.05), [dict(name='next', T_wc=pose(.1).tolist())])
        self.assertIsNone(selected)
        self.assertEqual(report['status'], 'insufficient_geometry')

    def test_scorer_rejects_unaccepted_and_duplicate_frames(self):
        geometry, scorer = self.fixture()
        image, mask, T, _ = scorer.observations[-1]
        for bad_pose in (pose(.4), T):
            with self.assertRaises(ValueError):
                scorer.add_observation(image, mask, bad_pose)


class BoardWorkflowTests(unittest.TestCase):
    def test_print_board_geometry_and_corner_detection(self):
        import shutil
        import subprocess
        from active_vision.calibration_board import generate_board
        if not shutil.which('pdftoppm'):
            self.skipTest('pdftoppm needed to validate rendered PDF')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'board.pdf'
            result = generate_board(path)
            self.assertEqual(result['inner_corners'], [9, 6])
            self.assertEqual(result['grid_size_mm'], [250, 175])
            self.assertFalse(result['print_verified'])
            subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', '100', str(path), str(path.with_suffix(''))],
                           check=True, capture_output=True, timeout=10)
            image = cv2.imread(str(path.with_suffix('.png')))
            pixels = corners(image, 9, 6).reshape(6, 9, 2)
            expected_spacing = 25*100/25.4
            self.assertAlmostEqual(np.linalg.norm(np.diff(pixels, axis=1), axis=2).mean(), expected_spacing, delta=.5)
            self.assertEqual(len(pixels.reshape(-1, 2)), 54)
            with self.assertRaises(ValueError):
                generate_board(path)

    def test_calibration_capture_splits_and_never_commands_motion(self):
        from active_vision.calibration_capture import capture_board_images
        info = dict(camera_id='TEST', resolved_device='TEST', width=240, height=200,
                    pixel_format='UYVY', stride=480, optical_controls={})
        image = np.full((200, 240, 3), 120, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            with patch('active_vision.calibration_capture.device_info', return_value=info), \
                 patch('active_vision.calibration_capture.snapshot', return_value=(image, info)), \
                 patch('active_vision.calibration_capture.corners', return_value=np.zeros((54, 1, 2), np.float32)), \
                 patch('builtins.print'):
                result = capture_board_images('TEST', output, count=15, prompt=lambda _: '')
            self.assertEqual(result['detected_counts'], dict(train=12, holdout=3))
            self.assertFalse(result['real_robot_motion_commanded'])
            self.assertIsNone(read_json(output/'capture_session.json')['measured_square_m'])
            self.assertEqual(len(list((output/'train').glob('*.png'))), 12)

    def test_capture_failure_saved_and_settings_change_rejected(self):
        from active_vision.calibration_capture import capture_board_images
        info = dict(camera_id='TEST', resolved_device='TEST', width=240, height=200,
                    pixel_format='UYVY', stride=480, optical_controls={})
        image = np.full((200, 240, 3), 120, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            with patch('active_vision.calibration_capture.device_info', return_value=info), \
                 patch('active_vision.calibration_capture.snapshot', return_value=(image, {**info, 'width': 640})), \
                 patch('builtins.print'):
                result = capture_board_images('TEST', output, count=15, prompt=lambda _: '')
            self.assertEqual(result['status'], 'capture_stopped')
            self.assertIn('changed', read_json(output/'capture_session.json')['reason'])
            self.assertEqual(len(list((output/'train').glob('*.png'))), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
