"""No hardware, camera or ROS access: fake transport for the bounded yaw test."""
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from active_vision.head_viewpoint_test import (
    RAISED_HEAD_START, RAISED_HEAD_REFERENCE_SHA,
    SWEEP_HEAD_START, SWEEP_OFFSETS,
    SPEED_RAW, TORQUE_RAW, YAW_STEP, controller_check, main, packet_allowed, preflight,
    read_head, supervised_yaw,
)
from active_vision.head_diagnostics import BLOCKS, decode_block, exchange, request_packet
from seed_hand_bus_setup import frame


class FakeIO:
    def __init__(self, raised=False, sweep=False):
        self.pos = dict(SWEEP_HEAD_START if sweep else RAISED_HEAD_START) if (sweep or raised) else {19: 2029, 20: 1392}
        self.torque = {19: 0, 20: 0}
        self.values = {i: (value, 0, 1023) for i, value in self.pos.items()}
        self.calls = []
        self.stuck = False
        self.fail_readback = False
        self.goal_enables_torque = False

    def get_model(self, ids):
        return tuple('MX-64' for _ in ids)

    def get_present_position(self, ids):
        return tuple(self.pos[i] for i in ids)

    def get_present_position_speed_load(self, ids):
        return tuple((self.pos[i], 0, 0) for i in ids)

    def get_present_current(self, ids):
        return tuple(2048 for _ in ids)

    def is_torque_enabled(self, ids):
        return tuple(self.torque[i] for i in ids)

    def get_goal_position_speed_load(self, ids):
        return tuple((0, 0, 0) if self.fail_readback and self.values[i][1] else self.values[i] for i in ids)

    def get_angle_limit(self, ids):
        return tuple((0, 4095) if i == 19 else (1035, 2946) for i in ids)

    def get_present_voltage(self, ids):
        return tuple(118 for _ in ids)

    def get_present_temperature(self, ids):
        return tuple(35 for _ in ids)

    def get_force_control_enable(self, ids):
        return tuple(0 for _ in ids)

    def set_goal_position_speed_load(self, values):
        self.calls.append(('goal', dict(values)))
        self.values.update(values)
        if self.goal_enables_torque:
            self.torque[19] = 1
        if self.torque[19] and not self.stuck:
            self.pos[19] = values[19][0]

    def enable_torque(self, ids):
        self.calls.append(('enable', ids))
        for i in ids:
            self.torque[i] = 1

    def disable_torque(self, ids):
        self.calls.append(('disable', ids))
        for i in ids:
            self.torque[i] = 0

    def close(self):
        self.calls.append(('close', ()))


class FakeClock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, value):
        self.now += value


class HeadViewpointTests(unittest.TestCase):
    def run_probe(self, io, capture=None, reviewed=False, sweep=False):
        start = preflight([read_head(io) for _ in range(5)], raised_head_clear=reviewed, yaw_20deg=sweep)
        self.report = {}
        clock = FakeClock()
        supervised_yaw(io, start, capture or (lambda label, feedback: {'yaw': feedback()}),
                       self.report, lambda: None, clock, clock.sleep,
                       accept_current_goal_activation=reviewed, yaw_20deg=sweep)

    def test_preflight_is_read_only_and_rejects_torque_on(self):
        io = FakeIO()
        preflight([read_head(io)]*5)
        self.assertEqual(io.calls, [])
        io.torque[19] = 1
        with self.assertRaises(ValueError):
            preflight([read_head(io)]*5)

    def test_rejects_unstable_stale_goal_or_changed_start(self):
        io = FakeIO()
        samples = [read_head(io) for _ in range(5)]
        bad = deepcopy(samples)
        bad[0][19]['position'] += 4
        with self.assertRaises(ValueError):
            preflight(bad)
        io.values[19] = (2200, 0, 1023)
        with self.assertRaises(ValueError):
            preflight([read_head(io)]*5)

    def test_speed_and_current_goal_preloaded_before_enable(self):
        io = FakeIO()
        self.run_probe(io)
        self.assertEqual(io.calls[0], ('goal', {19: (2029, SPEED_RAW, TORQUE_RAW)}))
        self.assertEqual(io.calls[1], ('enable', (19,)))
        goals = [c[1][19][0] for c in io.calls if c[0] == 'goal']
        self.assertEqual(goals, [2029, 2029+YAW_STEP, 2029])
        self.assertEqual(io.torque, {19: 0, 20: 0})
        self.assertEqual(io.values[19], (2029, SPEED_RAW, TORQUE_RAW))
        self.assertEqual(io.calls[-1], ('disable', (19,)))
        self.assertTrue(self.report['returned_to_start'])

    def test_unexpected_goal_enables_torque_aborts_and_disable_is_last_write(self):
        io = FakeIO()
        io.goal_enables_torque = True
        with self.assertRaisesRegex(IOError, 'unexpectedly enabled'):
            self.run_probe(io)
        self.assertEqual(io.torque[19], 0)
        self.assertEqual(io.calls[-1], ('disable', (19,)))
        self.assertFalse(any(c[0] == 'enable' for c in io.calls))

    def test_cli_execution_is_paused_pending_hardware_review(self):
        with self.assertRaises(SystemExit):
            main(['--execute-yaw-test', '--workspace-clear'])

    def test_controller_cli_requires_confirmation_and_is_separate_from_camera_test(self):
        with patch('active_vision.head_viewpoint_test.controller_check') as check:
            with self.assertRaises(SystemExit):
                main(['--controller-check'])
            with self.assertRaises(SystemExit):
                main(['--controller-check', '--execute-yaw-test', '--workspace-clear'])
            with self.assertRaises(SystemExit):
                main(['--controller-check', '--workspace-clear'])
            with self.assertRaises(SystemExit):
                main(['--raised-head-clear'])
            check.assert_not_called()
            main(['--controller-check', '--workspace-clear', '--raised-head-clear'])
            check.assert_called_once_with(raised_head_clear=True)

    def test_controller_runner_records_and_closes_on_success_and_stall(self):
        for stuck in (False, True):
            io, clock, saved = FakeIO(raised=True), FakeClock(), []
            io.stuck = stuck

            def probe(io, start, capture, report, save, **kwargs):
                return supervised_yaw(io, start, capture, report, save, clock, clock.sleep, **kwargs)

            with patch('active_vision.head_viewpoint_test.DEVICE') as device, patch(
                    'active_vision.head_viewpoint_test.subprocess.run') as owner, patch(
                    'active_vision.head_viewpoint_test.strict_io', return_value=io) as transport, patch(
                    'active_vision.head_viewpoint_test.write_json', side_effect=lambda p, r: saved.append(deepcopy(r))), patch(
                    'active_vision.head_viewpoint_test.signal.signal'), patch(
                    'active_vision.head_viewpoint_test.signal.alarm'), patch(
                    'active_vision.head_viewpoint_test.sha256', return_value=RAISED_HEAD_REFERENCE_SHA), patch(
                    'active_vision.head_viewpoint_test.time.sleep', side_effect=clock.sleep), patch(
                    'active_vision.head_viewpoint_test.supervised_yaw', side_effect=probe), patch('builtins.print'):
                device.exists.return_value = True
                owner.return_value.returncode = 1
                result = controller_check(raised_head_clear=True)
            transport.assert_called_once_with(allow_writes=True)
            self.assertEqual(result['status'], 'stopped' if stuck else 'supervised_controller_check_completed')
            self.assertTrue(result['current_position_hold_verified'])
            self.assertEqual(len(result['cleanup_feedback']), 5)
            self.assertTrue(all(s['yaw_torque'] == s['pitch_torque'] == 0 for s in result['cleanup_feedback']))
            self.assertEqual(io.calls[-2:], [('disable', (19,)), ('close', ())])
            self.assertEqual(saved[-1]['status'], result['status'])
            self.assertEqual(sum(c[0] == 'enable' for c in io.calls), 1)
            self.assertTrue(all(c[1][19][1:] == (2, 150) for c in io.calls if c[0] == 'goal'))

    def test_raised_check_rejects_old_contact_pose_and_changed_anchor(self):
        with self.assertRaises(ValueError):
            self.run_probe(FakeIO(), reviewed=True)
        io = FakeIO(raised=True)
        io.pos[20] += 5
        with self.assertRaises(ValueError):
            self.run_probe(io, reviewed=True)
        self.assertEqual(io.calls, [])
        with self.assertRaises(ValueError):
            controller_check()

    def test_reviewed_auto_activation_is_verified_without_second_enable(self):
        io = FakeIO(raised=True)
        io.goal_enables_torque = True
        self.run_probe(io, reviewed=True)
        self.assertTrue(self.report['preload_auto_activation_observed'])
        self.assertTrue(self.report['current_position_hold_verified'])
        self.assertTrue(self.report['returned_to_start'])
        self.assertFalse(any(c[0] == 'enable' for c in io.calls))
        self.assertEqual(io.calls[-1], ('disable', (19,)))
        self.assertEqual(io.torque, {19: 0, 20: 0})
        goals = [c[1][19] for c in io.calls if c[0] == 'goal']
        self.assertEqual(goals, [(2082, 2, 150), (2116, 2, 150), (2082, 2, 150)])

    def test_reviewed_auto_activation_does_not_allow_preload_motion_or_wrong_limits(self):
        for fault in ('yaw', 'pitch', 'settings'):
            io = FakeIO(raised=True)
            io.goal_enables_torque = True
            original = io.set_goal_position_speed_load

            def faulty(values):
                original(values)
                if fault == 'yaw':
                    io.pos[19] += 5
                elif fault == 'pitch':
                    io.pos[20] += 7
                else:
                    io.values[19] = (io.pos[19], 0, 1023)

            io.set_goal_position_speed_load = faulty
            with self.assertRaises((ValueError, IOError)):
                self.run_probe(io, reviewed=True)
            self.assertEqual(sum(c[0] == 'goal' for c in io.calls), 1)
            self.assertEqual(io.calls[-1], ('disable', (19,)))
            self.assertEqual(io.torque[19], 0)

    def test_reviewed_auto_activation_stall_has_no_retry_or_stronger_command(self):
        io = FakeIO(raised=True)
        io.goal_enables_torque = True
        io.stuck = True
        with self.assertRaises(TimeoutError):
            self.run_probe(io, reviewed=True)
        self.assertEqual(sum(c[0] == 'goal' for c in io.calls), 2)
        self.assertTrue(all(c[1][19][1:] == (2, 150) for c in io.calls if c[0] == 'goal'))
        self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_wide_sweep_requires_separate_clearance_and_named_profile(self):
        with self.assertRaises(ValueError):
            controller_check(raised_head_clear=True, yaw_20deg=True)
        with patch('active_vision.head_viewpoint_test.controller_check') as check:
            flags = ['--controller-check', '--workspace-clear', '--raised-head-clear']
            for extra in (['--yaw-20deg-test'], ['--full-sweep-clear']):
                with self.assertRaises(SystemExit):
                    main(flags+extra)
            check.assert_not_called()
            main(flags+['--yaw-20deg-test', '--full-sweep-clear'])
            check.assert_called_once_with(raised_head_clear=True, yaw_20deg=True, full_sweep_clear=True)
        with self.assertRaises(ValueError):
            self.run_probe(FakeIO(raised=True), reviewed=True, sweep=True)

    def test_wide_sweep_fixed_stages_return_and_unchanged_power_limits(self):
        io = FakeIO(sweep=True)
        io.goal_enables_torque = True
        self.run_probe(io, reviewed=True, sweep=True)
        origin = SWEEP_HEAD_START[19]
        offsets = [0, *SWEEP_OFFSETS, *reversed((0,)+SWEEP_OFFSETS[:-1])]
        self.assertEqual([c[1][19][0] for c in io.calls if c[0] == 'goal'], [origin+x for x in offsets])
        self.assertTrue(all(c[1][19][1:] == (2, 150) for c in io.calls if c[0] == 'goal'))
        self.assertEqual(len(self.report['completed_stages']), 6)
        self.assertTrue(self.report['returned_to_start'])
        self.assertEqual(self.report['return_error_raw'], 0)
        self.assertEqual(io.calls[-1], ('disable', (19,)))
        self.assertEqual(io.torque, {19: 0, 20: 0})

    def test_wide_sweep_accepts_stable_four_count_error_and_reports_it(self):
        io = FakeIO(sweep=True)
        io.goal_enables_torque = True
        original = io.set_goal_position_speed_load

        def residual(values):
            before = io.pos[19]
            original(values)
            target = values[19][0]
            if target != before:
                io.pos[19] = target - (4 if target > before else -4)

        io.set_goal_position_speed_load = residual
        self.run_probe(io, reviewed=True, sweep=True)
        self.assertTrue(all(abs(s['error_raw']) == 4 for s in self.report['completed_stages']))
        self.assertTrue(all(s['allowed_error_raw'] == 6 for s in self.report['completed_stages']))
        self.assertEqual(self.report['return_error_raw'], 4)

    def test_wide_sweep_stall_does_not_advance_or_return_or_raise_torque(self):
        io = FakeIO(sweep=True)
        io.goal_enables_torque = True
        original = io.set_goal_position_speed_load

        def stall_second(values):
            if values[19][0] == SWEEP_HEAD_START[19]+SWEEP_OFFSETS[1]:
                io.stuck = True
            original(values)

        io.set_goal_position_speed_load = stall_second
        with self.assertRaises(TimeoutError):
            self.run_probe(io, reviewed=True, sweep=True)
        self.assertEqual(len(self.report['completed_stages']), 1)
        self.assertEqual(sum(c[0] == 'goal' for c in io.calls), 3)
        self.assertTrue(all(c[1][19][1:] == (2, 150) for c in io.calls if c[0] == 'goal'))
        self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_wide_sweep_rejects_skipping_a_stage_or_pitch_drift(self):
        for fault in ('overshoot', 'pitch'):
            io = FakeIO(sweep=True)
            io.goal_enables_torque = True
            original = io.set_goal_position_speed_load

            def bad(values):
                original(values)
                if values[19][0] == SWEEP_HEAD_START[19]+SWEEP_OFFSETS[0]:
                    if fault == 'overshoot':
                        io.pos[19] = SWEEP_HEAD_START[19]+150
                    else:
                        io.pos[20] += 7

            io.set_goal_position_speed_load = bad
            with self.assertRaises(ValueError):
                self.run_probe(io, reviewed=True, sweep=True)
            self.assertEqual(sum(c[0] == 'goal' for c in io.calls), 2)
            self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_wide_sweep_requires_stability_not_only_error_band(self):
        io = FakeIO(sweep=True)
        io.goal_enables_torque = True
        original = io.get_present_position_speed_load
        sign = [1]

        def oscillating(ids):
            if io.values[19][0] > SWEEP_HEAD_START[19]:
                sign[0] *= -1
                io.pos[19] = io.values[19][0]+sign[0]*4
            return original(ids)

        io.get_present_position_speed_load = oscillating
        with self.assertRaises(TimeoutError):
            self.run_probe(io, reviewed=True, sweep=True)
        self.assertEqual(sum(c[0] == 'goal' for c in io.calls), 2)
        self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_telemetry_phases_and_cleanup_no_later_write(self):
        io = FakeIO()
        self.run_probe(io)
        phases = {s['phase'] for s in self.report['feedback_samples']}
        self.assertEqual(phases, {'initial_feedback', 'hold_current_position', 'yaw_step', 'return_to_start'})
        self.assertEqual(len(self.report['cleanup_feedback']), 5)
        self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_exception_after_torque_enable_still_disables(self):
        io = FakeIO()
        enable = io.enable_torque

        def interrupted(ids):
            enable(ids)
            raise KeyboardInterrupt()

        io.enable_torque = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.run_probe(io)
        self.assertEqual(io.torque[19], 0)
        self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_readback_failure_never_enables_torque(self):
        io = FakeIO()
        io.fail_readback = True
        with self.assertRaises(IOError):
            self.run_probe(io)
        self.assertFalse(any(c[0] == 'enable' for c in io.calls))
        self.assertEqual(io.torque[19], 0)

    def test_stall_aborts_without_retry_and_releases(self):
        io = FakeIO()
        io.stuck = True
        with self.assertRaises(TimeoutError):
            self.run_probe(io)
        self.assertEqual(sum(c[0] == 'enable' for c in io.calls), 1)
        self.assertEqual(io.torque[19], 0)

    def test_capture_failure_and_pitch_drift_release_yaw(self):
        for failure in ('capture', 'pitch'):
            io = FakeIO()

            def capture(label, feedback):
                if failure == 'capture':
                    raise ValueError('No board in view')
                io.pos[20] += 10
                feedback()

            with self.assertRaises(ValueError):
                self.run_probe(io, capture)
            self.assertEqual(io.torque[19], 0)
            self.assertTrue(all(c[1] == (19,) if c[0] != 'goal' else set(c[1]) == {19}
                                for c in io.calls))

    def test_goal_overwrite_detected_and_telemetry_retained(self):
        io = FakeIO()

        def capture(label, feedback):
            io.values[19] = (2030, 0, 1023)
            feedback()

        with self.assertRaisesRegex(IOError, 'changed during test'):
            self.run_probe(io, capture)
        sample = self.report['feedback_samples'][-1]
        self.assertEqual(sample['yaw_goal_speed_limit_raw'], [2030, 0, 1023])
        self.assertIn('present_current_raw', sample)
        self.assertIn('present_load_raw', sample)
        self.assertEqual(io.torque[19], 0)
        self.assertEqual(io.calls[-1], ('disable', (19,)))

    def test_packet_allowlist_blocks_other_ids_eeprom_and_broadcast_goals(self):
        Packet = namedtuple('Packet', 'id instruction parameters')
        self.assertTrue(packet_allowed(Packet(19, 2, (36, 2))))
        self.assertTrue(packet_allowed(Packet(20, 2, (36, 2))))
        for packet in (Packet(254, 131, (24, 1, 19, 1)),
                       Packet(254, 131, (30, 6, 19, 0, 0, 2, 0, 150, 0))):
            self.assertFalse(packet_allowed(packet))
            self.assertTrue(packet_allowed(packet, allow_writes=True))
        for packet in (Packet(31, 2, (36, 2)), Packet(254, 131, (24, 1, 20, 1)),
                       Packet(254, 131, (6, 4, 19, 0, 0, 255, 15)),
                       Packet(254, 131, (24, 1, 19, 1, 20, 1)),
                       Packet(254, 3, (30, 0, 0)), Packet(19, 6, ())):
            self.assertFalse(packet_allowed(packet, allow_writes=True))


class HeadDiagnosticReadOnlyTests(unittest.TestCase):
    def test_every_request_is_a_checksummed_read_to_head_only(self):
        for motor_id in (19, 20):
            for address, count in BLOCKS:
                packet = request_packet(motor_id, address, count)
                self.assertEqual(packet[2], motor_id)
                self.assertEqual(packet[4], 2)
                self.assertEqual(packet[5:7], bytes([address, count]))
                self.assertEqual(sum(packet[2:]) & 255, 255)
        for args in ((31, 24, 5), (254, 24, 5), (19, 99, 1)):
            with self.assertRaises(ValueError):
                request_packet(*args)

    def test_little_endian_state_pid_and_current_decoding(self):
        data = list(b''.join(v.to_bytes(2, 'little') for v in (2063, 2, 150, 2036, 0, 150)))+[118, 37]
        result = decode_block(30, data)
        self.assertEqual(result['goal_position_raw'], 2063)
        self.assertEqual(result['present_position_raw'], 2036)
        self.assertEqual(result['torque_limit_raw'], 150)
        self.assertEqual(result['present_voltage_v'], 11.8)
        self.assertEqual(decode_block(24, [0, 0, 0, 0, 32])['pid_raw'], {'D': 0, 'I': 0, 'P': 32})
        self.assertEqual(decode_block(68, [0, 8, 0, 0, 0, 0])['mx64_current_a'], 0)

    def test_ambiguous_error_and_clean_replies_are_logged(self):
        class Port:
            def __init__(self, response):
                self.response, self.buffer, self.sent = response, b'', []

            def reset_input_buffer(self):
                self.buffer = b''

            @property
            def in_waiting(self):
                return len(self.buffer)

            def write(self, packet):
                self.sent.append(packet)
                self.buffer = self.response
                return len(packet)

            def read(self, count):
                data, self.buffer = self.buffer[:count], self.buffer[count:]
                return data

        clean = frame(19, 0, [0, 0, 0, 0, 32])
        for response, expect_error in ((clean, False), (clean+clean, True),
                                       (frame(19, 32, [0, 0, 0, 0, 32]), True)):
            port, report = Port(response), {'transactions': []}
            with patch('active_vision.head_diagnostics.time.sleep'), patch(
                    'active_vision.head_diagnostics.time.monotonic', side_effect=[0., .1, .3]):
                if expect_error:
                    with self.assertRaises(IOError):
                        exchange(port, 19, 24, 5, report)
                else:
                    self.assertEqual(exchange(port, 19, 24, 5, report)['torque_enabled_raw'], 0)
            self.assertEqual(report['transactions'][0]['reply_hex'], response.hex())
            self.assertTrue(all(packet[4] == 2 for packet in port.sent))


if __name__ == '__main__':
    unittest.main()
