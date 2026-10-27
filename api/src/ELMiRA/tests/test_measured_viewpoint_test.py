"""Offline camera/transport fault tests; never open NICO hardware."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from active_vision import measured_viewpoint_test as measured
from active_vision.head_viewpoint_test import preflight, read_head, supervised_yaw
from test_head_viewpoint_test import FakeClock, FakeIO


def current_io(opposite=False):
    io = FakeIO()
    io.pos = dict(measured.OPPOSITE_HEAD_START if opposite else measured.HEAD_START)
    io.values = {i: (value, 0, 1023) for i, value in io.pos.items()}
    io.goal_enables_torque = True
    return io


class MeasuredViewpointTests(unittest.TestCase):
    def test_new_anchor_needs_clearance_and_rejects_old_pose(self):
        io = current_io()
        for kwargs in ({'reviewed_start': measured.HEAD_START},
                       {'raised_head_clear': True, 'yaw_20deg': True}):
            with self.assertRaises(ValueError):
                preflight([read_head(io)]*5, **kwargs)
        preflight([read_head(io)]*5, raised_head_clear=True, yaw_20deg=True,
                  reviewed_start=measured.HEAD_START)
        self.assertEqual(io.calls, [])
        io.pos[20] += 5
        with self.assertRaises(ValueError):
            preflight([read_head(io)]*5, raised_head_clear=True, yaw_20deg=True,
                      reviewed_start=measured.HEAD_START)

    def probe(self, io, camera, opposite=False):
        start = read_head(io)
        clock, report = FakeClock(), {}
        self.report = report
        supervised_yaw(io, start, camera, report, lambda: None, clock, clock.sleep,
                       accept_current_goal_activation=True, yaw_20deg=True,
                       reviewed_start=measured.OPPOSITE_HEAD_START if opposite else measured.HEAD_START,
                       stage_capture=camera, opposite_yaw=opposite)
        return report

    def test_opposite_sweep_only_uses_fixed_negative_steps_and_returns(self):
        io = current_io(opposite=True)

        def camera(label, feedback):
            feedback()
            return {'label': label, 'continue_outward': True}

        report = self.probe(io, camera, opposite=True)
        origin = measured.OPPOSITE_HEAD_START[19]
        self.assertEqual([c[1][19] for c in io.calls if c[0] == 'goal'],
                         [(origin+o, 2, 150) for o in (0, -57, -114, -227, -114, -57, 0)])
        self.assertTrue(report['returned_to_start'])
        self.assertEqual(io.calls[-1], ('disable', (19,)))
        self.assertEqual(io.torque, {19: 0, 20: 0})

    def test_opposite_camera_rejection_prevents_twenty_degree_stage(self):
        io = current_io(opposite=True)

        def camera(label, feedback):
            feedback()
            return {'continue_outward': label == 'yaw_-57_counts'}

        report = self.probe(io, camera, opposite=True)
        origin = measured.OPPOSITE_HEAD_START[19]
        self.assertEqual([c[1][19][0] for c in io.calls if c[0] == 'goal'],
                         [origin+o for o in (0, -57, -114, -57, 0)])
        self.assertTrue(report['further_outward_motion_cancelled_by_camera_check'])
        self.assertTrue(report['returned_to_start'])
        self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_opposite_profile_needs_review_and_checks_lower_motor_limit(self):
        io = current_io(opposite=True)
        samples = [read_head(io)]*5
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            preflight(samples, raised_head_clear=True, yaw_20deg=True, opposite_yaw=True)
        samples[0][19]['limits'][0] = measured.OPPOSITE_HEAD_START[19]-200
        with self.assertRaisesRegex(ValueError, 'limits'):
            preflight(samples, raised_head_clear=True, yaw_20deg=True,
                      reviewed_start=measured.OPPOSITE_HEAD_START, opposite_yaw=True)
        self.assertEqual(io.calls, [])

    def test_target_appearance_check_rejects_missing_target(self):
        rng = np.random.default_rng(6)
        template = rng.integers(20, 250, (66, 54, 3), dtype=np.uint8)
        image = np.full((720, 1280, 3), 50, np.uint8)
        image[555:621, 397:451] = template
        reference = {'template': template, 'last_box': (397, 555, 54, 66)}
        result = measured.target_visibility(image, reference)
        self.assertTrue(result['visible'])
        self.assertFalse(result['object_recognition_verified'])
        last = reference['last_box']
        result = measured.target_visibility(np.zeros_like(image), reference)
        self.assertFalse(result['visible'])
        self.assertEqual(reference['last_box'], last)
        reference['last_box'] = (250, 555, 54, 66)
        self.assertFalse(measured.target_visibility(image, reference)['visible'])
        reference['last_box'] = measured.OPPOSITE_TARGET_BOX
        self.assertTrue(measured.target_visibility(image, reference)['visible'])

    def test_camera_rejection_cancels_larger_steps_and_returns_low_power(self):
        io, views = current_io(), []

        def reject(label, feedback):
            feedback()
            views.append(label)
            return {'status': 'unavailable', 'continue_outward': False}

        result = self.probe(io, reject)
        origin = measured.HEAD_START[19]
        self.assertEqual(views, ['yaw_57_counts'])
        self.assertEqual([c[1][19] for c in io.calls if c[0] == 'goal'],
                         [(origin, 2, 150), (origin+57, 2, 150), (origin, 2, 150)])
        self.assertTrue(result['returned_to_start'])
        self.assertTrue(result['further_outward_motion_cancelled_by_camera_check'])
        self.assertEqual(io.calls[-1], ('disable', (19,)))
        self.assertEqual(io.torque, {19: 0, 20: 0})

    def test_feedback_fault_during_camera_capture_does_not_command_return(self):
        io = current_io()

        def fault(label, feedback):
            io.pos[20] += 7
            feedback()

        with self.assertRaises(ValueError):
            self.probe(io, fault)
        self.assertEqual(sum(c[0] == 'goal' for c in io.calls), 2)
        self.assertEqual(io.calls[-1], ('disable', (19,)))
        self.assertEqual(io.torque, {19: 0, 20: 0})

    def test_camera_capture_distinguishes_unavailable_image_from_joint_drift(self):
        io = current_io()
        start = read_head(io)
        for drift in (False, True):
            io.pos = dict(measured.HEAD_START)

            def unavailable(device):
                if drift:
                    io.pos[20] += 3
                raise ValueError('No camera frame')

            with tempfile.TemporaryDirectory() as directory, patch.object(
                    measured, 'snapshot', side_effect=unavailable):
                if drift:
                    with self.assertRaises(ValueError):
                        measured.capture_view(io, Path(directory), 'test', None, start)
                else:
                    result = measured.capture_view(io, Path(directory), 'test', None, start)
                    self.assertFalse(result['continue_outward'])
            self.assertEqual(io.calls, [])

    def test_default_plan_and_missing_confirmation_never_open_hardware(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                measured, 'LOG_ROOT', Path(directory)), patch.object(
                measured, 'strict_io') as transport, patch('builtins.print'):
            result = measured.run()
            self.assertEqual(result['status'], 'plan_only')
            result = measured.run(opposite_yaw=True)
            self.assertEqual(result['planned_outward_offsets_raw'], [-57, -114, -227])
            self.assertEqual(result['reviewed_start_raw'], measured.OPPOSITE_HEAD_START)
            self.assertFalse(result['real_robot_motion_commanded'])
            for flags in (['--execute'], ['--execute', '--workspace-clear']):
                with self.assertRaises(SystemExit):
                    measured.main(flags)
            transport.assert_not_called()

    def test_experimental_candidate_is_explicit_and_never_claims_verified(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                measured, 'LOG_ROOT', Path(directory)), patch.object(
                measured, 'strict_io') as transport, patch('builtins.print'):
            result = measured.run(experimental_calibration=True)
        self.assertEqual(result['calibration_sha256'], measured.EXPERIMENTAL_CALIBRATION_SHA)
        self.assertEqual(result['calibration'], str(measured.EXPERIMENTAL_CALIBRATION))
        self.assertFalse(result['verified_for_reconstruction'])
        self.assertFalse(result['real_robot_motion_commanded'])
        transport.assert_not_called()

    def test_runner_baseline_rejection_never_writes_and_success_closes_after_disable(self):
        for reject_baseline in (False, True):
            io, clock = current_io(), FakeClock()

            def capture(io, output, label, calibration, start, feedback=None, **kwargs):
                if feedback is not None:
                    feedback()
                T = np.eye(4)
                T[0, 3] = (io.pos[19]-measured.HEAD_START[19])*.0001
                return {'label': label, 'continue_outward': not reject_baseline,
                        'T_wc': T.tolist(), 'inner_corners_px': [[x, y] for y in range(6) for x in range(9)]}

            def probe(*args, **kwargs):
                return supervised_yaw(*args, clock=clock, sleep=clock.sleep, **kwargs)

            def digest(path):
                return {measured.HEAD_REFERENCE: measured.HEAD_REFERENCE_SHA,
                        measured.BOARD_REFERENCE: measured.BOARD_REFERENCE_SHA,
                        measured.CALIBRATION: measured.CALIBRATION_SHA}.get(path, 'test_script_hash')

            reference = {'status': 'read_only_snapshot_complete', 'motors': {
                str(i): {'present_position_raw': p, 'torque_enabled_raw': 0, 'model_number': 310}
                for i, p in measured.HEAD_START.items()}}
            with tempfile.TemporaryDirectory() as directory, patch.object(
                    measured, 'LOG_ROOT', Path(directory)), patch.object(
                    measured, 'sha256', side_effect=digest), patch.object(
                    measured, 'read_json', return_value=reference), patch.object(
                    measured, 'Calibration'), patch.object(measured, 'DEVICE') as device, patch.object(
                    measured.subprocess, 'run') as owner, patch.object(
                    measured, 'strict_io', return_value=io), patch.object(
                    measured, 'capture_view', side_effect=capture), patch.object(
                    measured, 'supervised_yaw', side_effect=probe), patch.object(
                    measured.time, 'sleep', side_effect=clock.sleep), patch.object(
                    measured.signal, 'signal'), patch.object(measured.signal, 'alarm'), patch('builtins.print'):
                device.exists.return_value = True
                owner.return_value.returncode = 1
                result = measured.run(True, True, True)
            if reject_baseline:
                self.assertEqual(result['status'], 'stopped')
                self.assertFalse(result['real_robot_motion_commanded'])
                self.assertEqual(io.calls, [('close', ())])
            else:
                self.assertEqual(result['status'], 'supervised_viewpoint_check_completed')
                self.assertEqual(len(result['stage_views']), 3)
                self.assertEqual(len(result['completed_stages']), 6)
                self.assertEqual(io.calls[-2:], [('disable', (19,)), ('close', ())])
                self.assertEqual(io.torque, {19: 0, 20: 0})
                self.assertFalse(result['verified_for_reconstruction'])


if __name__ == '__main__':
    unittest.main()
