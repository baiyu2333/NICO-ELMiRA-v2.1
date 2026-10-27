"""Supervised fixed yaw diagnostics, not an autonomous motion controller.

Default is read-only. Camera-viewpoint execution is paused after a failed
2026-10-15 test and unexpected post-cleanup torque state. A separate supervised
controller check retains the same low limits to investigate, without cameras
or automatic stronger retries. Only ID19 can
receive RAM writes; pitch and all arm/hand motors are never commanded. Reuses
the installed NICO pypot transport, without starting Motion/SMACH or its sync
threads. Candidate calibration is used for diagnostic PnP, not certified 3D.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import time

import cv2
import numpy as np

from .calibration import Calibration, board_pose, corners
from .calibration_capture import snapshot
from .common import sha256, write_json
from .geometry import baseline_and_angle

API = next(p for p in Path(__file__).resolve().parents if p.name == "api")
DEVICE = Path('/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT2XOU56-if00-port0')
CAMERA = '/dev/v4l/by-id/usb-e-con_systems_See3CAM_CU135_22275000-video-index0'
CALIBRATION = API / 'logs/active_vision/calibration_22275000_guided_20261013/calibration_candidate_23views.json'
CALIBRATION_SHA = '359a1e0622ca2cbc4f37418cf482b3001c4ae246058a9c37b0057a52f59a0099'
YAW_STEP = 34  # MX single-turn counts: approximately 2.99 degrees.
SPEED_RAW = 2  # 1.368 deg/s; zero would mean uncontrolled maximum speed.
TORQUE_RAW = 150  # Temporary yaw-only cap, 14.66%; not a measured force limit.
EXECUTION_REVIEW_REQUIRED = True
# User-cleared raised posture, independently measured after a power cycle.
# This is a narrow diagnostic anchor, not a general safe head workspace.
RAISED_HEAD_START = {19: 2082, 20: 2244}
RAISED_HEAD_REFERENCE = API / 'logs/active_vision/head_diagnostics_20261015T085432_143235Z/head_diagnostics.json'
RAISED_HEAD_REFERENCE_SHA = 'fa621488653e810d56d6e745bf276ea6d53f01df1dababa7a7acf6c2f97b22c2'
# Separately requested and cleared by the user after the three-degree test.
SWEEP_HEAD_START = {19: 2112, 20: 2243}
SWEEP_REFERENCE = API / 'logs/active_vision/head_diagnostics_20261015T090753_973464Z/head_diagnostics.json'
SWEEP_REFERENCE_SHA = 'cda1cf7c7d443f95e78656c58eeb8893740820c2bb76a11117e35646859cccb7'
SWEEP_OFFSETS = (57, 114, 227)  # About5,10,20deg; final target stays below20deg.
SWEEP_TOLERANCE = 6  # About0.53deg, plus five samples stable within two counts.


def packet_allowed(packet, allow_writes=False):
    params = list(packet.parameters)
    read = packet.instruction == 2 and packet.id in (19, 20)
    write = (packet.instruction == 131 and packet.id == 254
             and len(params) >= 3 and params[2] == 19
             and (params[0], params[1]) in ((24, 1), (30, 6))
             and len(params) == 3 + params[1])
    return read or (allow_writes and write)


def strict_io(allow_writes=False):
    from pypot.dynamixel import DxlIO

    class HeadOnlyIO(DxlIO):
        def _send_packet(self, packet, **kwargs):
            read = packet.instruction == 2 and packet.id in (19, 20)
            if not packet_allowed(packet, allow_writes=allow_writes):
                raise ValueError('Packet outside head diagnostic allowlist')
            reply = super()._send_packet(packet, **kwargs)
            if read and (reply is None or reply.id != packet.id or reply.error):
                raise IOError('Missing/mismatched/error head reply')
            return reply

    io = HeadOnlyIO(str(DEVICE), baudrate=1000000, timeout=.1,
                    use_sync_read=False, convert=False)
    io._serial.exclusive = True
    return io


def read_head(io):
    states = {}
    for motor_id in (19, 20):
        if io.get_model((motor_id,)) != ('MX-64',):
            raise ValueError('Unexpected motor model at ID%d' % motor_id)
        states[motor_id] = {
            'position': io.get_present_position((motor_id,))[0],
            'torque': io.is_torque_enabled((motor_id,))[0],
            'goal_speed_limit': list(io.get_goal_position_speed_load((motor_id,))[0]),
            'limits': list(io.get_angle_limit((motor_id,))[0]),
            'voltage': io.get_present_voltage((motor_id,))[0],
            'temperature': io.get_present_temperature((motor_id,))[0],
            # MX-64 address70 is torque-control mode, not a hand command.
            'torque_control_mode': io.get_force_control_enable((motor_id,))[0],
        }
        s = states[motor_id]
        if (not 0 <= s['position'] <= 4095 or s['torque_control_mode'] != 0
                or not 105 <= s['voltage'] <= 135 or s['temperature'] > 45
                or not s['limits'][0] < s['position'] < s['limits'][1]):
            raise ValueError('Head state outside conservative diagnostic checks')
    return states


def preflight(samples, raised_head_clear=False, yaw_20deg=False, reviewed_start=None,
              opposite_yaw=False):
    if opposite_yaw and (not yaw_20deg or reviewed_start is None):
        raise ValueError('Opposite sweep requires its separately reviewed fixed start')
    if reviewed_start is not None and (not raised_head_clear or set(reviewed_start) != {19, 20}):
        raise ValueError('A separate reviewed start requires explicit head clearance')
    if yaw_20deg and not raised_head_clear:
        raise ValueError('Staged yaw requires the reviewed raised posture')
    for motor_id in (19, 20):
        positions = [s[motor_id]['position'] for s in samples]
        if max(positions)-min(positions) > 2 or any(s[motor_id]['torque'] != 0 for s in samples):
            raise ValueError('Need stable head feedback and both torques initially off')
    start = samples[-1]
    yaw, pitch = start[19]['position'], start[20]['position']
    if raised_head_clear:
        anchor = reviewed_start if reviewed_start is not None else (
            SWEEP_HEAD_START if yaw_20deg else RAISED_HEAD_START)
        if any(abs(start[i]['position']-anchor[i]) > 4 for i in (19, 20)):
            raise ValueError('Head differs from the reviewed raised posture; do not recenter automatically')
    elif not 1990 <= yaw <= 2105 or not 1310 <= pitch <= 1460:
        raise ValueError('Head is outside the inspected starting neighbourhood; review again')
    if abs(start[19]['goal_speed_limit'][0]-yaw) > 2:
        raise ValueError('Stored yaw goal differs from feedback; review before execution')
    if start[19]['goal_speed_limit'][2] < TORQUE_RAW:
        raise ValueError('Refusing to increase an existing lower torque limit')
    offset = SWEEP_OFFSETS[-1] if yaw_20deg else YAW_STEP
    if opposite_yaw:
        offset = -offset
    if not start[19]['limits'][0]+10 <= yaw+offset <= start[19]['limits'][1]-10:
        raise ValueError('Proposed yaw violates device limits/margin')
    return start


def supervised_yaw(io, start, capture, report, save, clock=time.monotonic, sleep=time.sleep,
                   accept_current_goal_activation=False, yaw_20deg=False,
                   reviewed_start=None, stage_capture=None, opposite_yaw=False):
    """Single out-and-return test. Cleanup releases yaw only, originally off."""
    origin, pitch = start[19]['position'], start[20]['position']
    if yaw_20deg and not accept_current_goal_activation:
        raise ValueError('Staged yaw requires reviewed controller activation behaviour')
    if opposite_yaw and (not yaw_20deg or reviewed_start is None):
        raise ValueError('Opposite sweep requires the separately reviewed fixed profile')
    if accept_current_goal_activation:
        preflight([start], raised_head_clear=True, yaw_20deg=yaw_20deg,
                  reviewed_start=reviewed_start, opposite_yaw=opposite_yaw)
    if stage_capture is not None and not yaw_20deg:
        raise ValueError('Per-stage camera checks require the fixed staged diagnostic')
    offsets = SWEEP_OFFSETS if yaw_20deg else (YAW_STEP,)
    if opposite_yaw:
        offsets = tuple(-offset for offset in offsets)
    maximum_step = max(abs(offset) for offset in offsets)
    tolerance = SWEEP_TOLERANCE if yaw_20deg else 2
    report['original_yaw_goal_speed_limit_raw'] = list(start[19]['goal_speed_limit'])
    report['feedback_samples'] = []
    writes_started = False
    active_goal = None
    segment_start = origin
    phase = 'initial_feedback'

    def feedback():
        yaw_now, speed_raw, load_raw = io.get_present_position_speed_load((19,))[0]
        pitch_now = io.get_present_position((20,))[0]
        goal_settings = tuple(io.get_goal_position_speed_load((19,))[0])
        current_raw = io.get_present_current((19,))[0]
        sample = {'host_time_sec': time.time(), 'yaw_raw': yaw_now, 'pitch_raw': pitch_now,
                  'phase': phase,
                  'yaw_goal_speed_limit_raw': list(goal_settings),
                  'requested_yaw_goal_raw': active_goal,
                  'present_speed_raw': speed_raw, 'present_load_raw': load_raw,
                  'present_current_raw': current_raw, 'mx64_current_a': (current_raw-2048)*.0045,
                  'yaw_torque_enabled_raw': io.is_torque_enabled((19,))[0],
                  'present_voltage_raw': io.get_present_voltage((19,))[0],
                  'present_temperature_c': io.get_present_temperature((19,))[0]}
        report['feedback_samples'].append(sample)
        save()
        if active_goal is not None and goal_settings != (active_goal, SPEED_RAW, TORQUE_RAW):
            raise IOError('Yaw goal/speed/torque limit changed during test; possible competing writer or controller behavior')
        if not 105 <= sample['present_voltage_raw'] <= 135 or sample['present_temperature_c'] > 45:
            raise ValueError('Voltage/temperature outside diagnostic bounds')
        if abs(yaw_now-origin) > maximum_step+6 or abs(pitch_now-pitch) > 6:
            raise ValueError('Head drift/excursion exceeds the small diagnostic envelope')
        if yaw_20deg and active_goal is not None and not (
                min(segment_start, active_goal)-6 <= yaw_now <= max(segment_start, active_goal)+6):
            raise ValueError('Yaw is outside the current segment; no later stage permitted')
        if io.is_torque_enabled((20,))[0] != 0:
            raise ValueError('Pitch torque changed externally; aborting')
        return yaw_now

    def goal(value):
        nonlocal active_goal, segment_start
        if value not in (origin, *(origin+offset for offset in offsets)):
            raise ValueError('Only the initial yaw or fixed diagnostic stages are allowed')
        segment_start = report['feedback_samples'][-1]['yaw_raw']
        expected = (value, SPEED_RAW, TORQUE_RAW)
        report.setdefault('yaw_commands', []).append({'host_time_sec': time.time(),
                                                      'goal_raw': value, 'speed_raw': SPEED_RAW,
                                                      'torque_limit_raw': TORQUE_RAW})
        save()
        io.set_goal_position_speed_load({19: expected})
        if tuple(io.get_goal_position_speed_load((19,))[0]) != expected:
            raise IOError('Yaw goal/speed/torque-cap readback mismatch')
        active_goal = value

    def settle(value, allowed_error=2, timeout=6, stable=False):
        deadline, window = clock()+timeout, []
        while clock() < deadline:
            measured = feedback()
            if io.is_torque_enabled((19,))[0] != 1:
                raise IOError('Yaw torque unexpectedly disabled')
            window = (window+[measured])[-5:] if abs(measured-value) <= allowed_error else []
            if len(window) == 5 and (not stable or max(window)-min(window) <= 2):
                return
            sleep(.1)
        raise TimeoutError('Yaw did not settle; no retry or stronger command')

    def move_to(offset):
        value = origin+offset
        distance = abs(value-report['feedback_samples'][-1]['yaw_raw'])*360./4095.
        timeout = max(6., distance/(SPEED_RAW*.114*6)+4.) if yaw_20deg else 6.
        goal(value)
        settle(value, allowed_error=tolerance, timeout=timeout, stable=yaw_20deg)
        measured = report['feedback_samples'][-1]['yaw_raw']
        report.setdefault('completed_stages', []).append({
            'phase': phase, 'target_raw': value, 'actual_raw': measured,
            'error_raw': measured-value, 'allowed_error_raw': tolerance,
            'timeout_sec': timeout, 'host_time_sec': time.time()})
        save()

    try:
        if any(io.is_torque_enabled((i,))[0] != 0 for i in (19, 20)) or abs(feedback()-origin) > 2:
            raise ValueError('Starting feedback/torque changed after baseline capture')
        writes_started = True
        report['robot_register_write_attempted'] = True
        report['real_robot_motion_commanded'] = True
        save()
        phase = 'preload_current_position'
        # The observed controller can activate on a goal write. This first
        # target is the measured current position, never the outward target.
        goal(origin)
        torque_after_preload = io.is_torque_enabled((19,))[0]
        report['preload_torque_enabled_raw'] = torque_after_preload
        report['current_goal_activation_explicitly_reviewed'] = accept_current_goal_activation
        save()
        if torque_after_preload not in (0, 1):
            raise IOError('Invalid torque state after current-position goal')
        if torque_after_preload and not accept_current_goal_activation:
            raise IOError('Torque unexpectedly enabled during goal preload')
        if accept_current_goal_activation:
            if abs(feedback()-origin) > 2:
                raise ValueError('Head moved during current-position goal setup; aborting')
            report['preload_auto_activation_observed'] = bool(torque_after_preload)
        phase = 'hold_current_position'
        if not torque_after_preload:
            io.enable_torque((19,))
        settle(origin)
        report['current_position_hold_verified'] = True
        reached_offsets = []
        for offset in offsets:
            phase = 'outward_%d_counts' % offset if yaw_20deg else 'yaw_step'
            move_to(offset)
            reached_offsets.append(offset)
            if stage_capture is not None:
                view = stage_capture('yaw_%d_counts' % offset, feedback)
                report.setdefault('stage_views', []).append(view)
                save()
                if not view.get('continue_outward', False):
                    report['further_outward_motion_cancelled_by_camera_check'] = True
                    break
        report['after'] = (report['stage_views'][-1] if stage_capture is not None
                           else capture('after', feedback))
        for offset in reversed((0,)+tuple(reached_offsets[:-1])):
            phase = 'return_%d_counts' % offset if yaw_20deg else 'return_to_start'
            move_to(offset)
        report['returned_to_start'] = True
        report['return_error_raw'] = report['feedback_samples'][-1]['yaw_raw']-origin
    finally:
        if writes_started:
            try:
                io.disable_torque((19,))
                # No register writes after disabling: the first hardware test's
                # later goal/settings restoration was followed by torque-on.
                report['cleanup_feedback'] = []
                for _ in range(5):
                    sleep(.3)
                    state = {'host_time_sec': time.time(),
                             'yaw_torque': io.is_torque_enabled((19,))[0],
                             'pitch_torque': io.is_torque_enabled((20,))[0],
                             'yaw_raw': io.get_present_position((19,))[0],
                             'pitch_raw': io.get_present_position((20,))[0]}
                    report['cleanup_feedback'].append(state)
                    if state['yaw_torque'] != 0 or state['pitch_torque'] != 0:
                        raise IOError('Cannot confirm yaw torque remains off')
                report['cleanup'] = 'yaw_torque_off_verified_no_later_register_writes'
                report['original_settings_restored'] = False
            except Exception as exc:
                report['cleanup'] = 'UNVERIFIED: switch off motor power if safe; '+str(exc)
                raise
            finally:
                save()


def controller_check(raised_head_clear=False, yaw_20deg=False, full_sweep_clear=False):
    """Explicitly supervised isolation test; no camera/ROS/full-driver launch."""
    if not raised_head_clear:
        raise ValueError('Controller check requires a cleared, reviewed raised head; old low posture touched shoulder')
    if yaw_20deg and not full_sweep_clear:
        raise ValueError('Staged yaw needs separate full-sweep clearance and supervision')
    reference = SWEEP_REFERENCE if yaw_20deg else RAISED_HEAD_REFERENCE
    reference_sha = SWEEP_REFERENCE_SHA if yaw_20deg else RAISED_HEAD_REFERENCE_SHA
    anchor = SWEEP_HEAD_START if yaw_20deg else RAISED_HEAD_START
    offsets = SWEEP_OFFSETS if yaw_20deg else (YAW_STEP,)
    output = API / 'logs/active_vision' / ('head_controller_check_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    report = {'status': 'preflight', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'mode': 'supervised_yaw_20deg' if yaw_20deg else 'supervised_controller_check',
              'real_robot_motion_commanded': False,
              'robot_register_write_attempted': False, 'script_sha256': sha256(__file__),
              'yaw_only_motor_id': 19, 'pitch_commanded': False, 'arms_hands_commanded': False,
              'proposed_yaw_step_counts': offsets[-1], 'speed_raw': SPEED_RAW,
              'planned_outward_offsets_raw': list(offsets),
              'settling_tolerance_raw': SWEEP_TOLERANCE if yaw_20deg else 2,
              'settling_stability_required': yaw_20deg,
              'full_sweep_clear_confirmed': full_sweep_clear,
              'torque_limit_raw': TORQUE_RAW, 'stronger_retry_allowed': False,
              'raised_head_clear_confirmed': True,
              'reviewed_start_raw': anchor,
              'reviewed_start_report': str(reference),
              'reviewed_start_report_sha256': reference_sha,
              'verified_for_reconstruction': False,
              'safety_note': 'Physical supervision required. Named fixed stages only, unchanged low speed/torque; no calibration or general workspace validation.'}

    def save():
        write_json(output / 'head_controller_report.json', report)

    def interrupt(signum, stack):
        raise TimeoutError('Controller check interrupted or total timeout')

    io = None
    save()
    old_alarm = signal.signal(signal.SIGALRM, interrupt)
    old_term = signal.signal(signal.SIGTERM, interrupt)
    signal.alarm(80 if yaw_20deg else 30)
    try:
        if sha256(reference) != reference_sha:
            raise ValueError('Reviewed raised-head reference changed; no hardware test permitted')
        if not DEVICE.exists():
            raise ValueError('NICO FTDI adapter unavailable')
        owner = subprocess.run(['fuser', str(DEVICE.resolve())], capture_output=True, timeout=2)
        if owner.returncode != 1:
            raise ValueError('Serial port is busy or ownership cannot be checked')
        io = strict_io(allow_writes=True)
        samples = []
        for _ in range(5):
            samples.append(read_head(io))
            time.sleep(.1)
        report['preflight_samples'] = samples
        save()
        start = preflight(samples, raised_head_clear=True, yaw_20deg=yaw_20deg)
        report['start_feedback'] = start
        supervised_yaw(io, start, lambda label, feedback: {'yaw_raw': feedback()}, report, save,
                       accept_current_goal_activation=True, yaw_20deg=yaw_20deg)
        report['status'] = 'supervised_controller_check_completed'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='stopped', reason=type(exc).__name__+': '+str(exc))
    finally:
        signal.alarm(0)
        if io is not None:
            io.close()
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print('Status:', report['status'])
    print('Reason:', report.get('reason', 'none'))
    print('Motion commanded:', report['real_robot_motion_commanded'])
    print('Cleanup:', report.get('cleanup', 'no register writes attempted'))
    print('Report:', output / 'head_controller_report.json')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute-yaw-test', action='store_true')
    mode.add_argument('--controller-check', action='store_true',
                      help='Supervised fixed-step controller diagnosis only; camera workflow stays locked')
    parser.add_argument('--workspace-clear', action='store_true')
    parser.add_argument('--raised-head-clear', action='store_true',
                        help='Confirm head is clear of shoulder at the recorded raised posture')
    parser.add_argument('--yaw-20deg-test', action='store_true',
                        help='Separately authorized fixed5/10/20-degree diagnostic, not arbitrary angles')
    parser.add_argument('--full-sweep-clear', action='store_true',
                        help='Confirm full20-degree clearance, cable slack and operator at power switch')
    args = parser.parse_args(argv)
    if (args.execute_yaw_test or args.controller_check) and not args.workspace_clear:
        parser.error('Execution requires --workspace-clear and physical supervision')
    if args.execute_yaw_test and EXECUTION_REVIEW_REQUIRED:
        parser.error('Execution paused: review the 2026-10-15 settling failure and unexpected post-cleanup torque state before another hardware test. Read-only mode is available.')
    if args.raised_head_clear and not args.controller_check:
        parser.error('--raised-head-clear is only for the supervised controller check')
    if (args.yaw_20deg_test or args.full_sweep_clear) and not args.controller_check:
        parser.error('Sweep flags require --controller-check')
    if args.yaw_20deg_test != args.full_sweep_clear:
        parser.error('Staged yaw requires both --yaw-20deg-test and --full-sweep-clear')
    if args.controller_check:
        if not args.raised_head_clear:
            parser.error('Controller check requires --raised-head-clear; do not retry the shoulder-contact posture')
        if args.yaw_20deg_test:
            return controller_check(raised_head_clear=True, yaw_20deg=True, full_sweep_clear=True)
        return controller_check(raised_head_clear=True)
    output = API / 'logs/active_vision' / ('head_viewpoint_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preflight', 'real_robot_motion_commanded': False,
              'robot_register_write_attempted': False, 'camera_settings_changed': False,
              'yaw_only_motor_id': 19, 'pitch_commanded': False, 'arms_hands_commanded': False,
              'script_sha256': sha256(__file__), 'verified_for_reconstruction': False,
              'proposed_yaw_step_deg': YAW_STEP*360/4095, 'speed_deg_per_sec': SPEED_RAW*.114*6,
              'temporary_yaw_torque_limit_percent': TORQUE_RAW/1023*100,
              'safety_note': 'Small supervised diagnostic, not a verified general head workspace.'}

    def save():
        write_json(output / 'head_viewpoint_report.json', report)

    def interrupt(signum, stack):
        raise TimeoutError('Head diagnostic interrupted or total timeout')

    io = None
    save()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(55)
    try:
        if not DEVICE.exists():
            raise ValueError('NICO FTDI adapter unavailable')
        owner = subprocess.run(['fuser', str(DEVICE.resolve())], capture_output=True, timeout=2)
        if owner.returncode != 1:
            raise ValueError('Serial port is busy or ownership cannot be checked')
        if sha256(CALIBRATION) != CALIBRATION_SHA:
            raise ValueError('Calibration candidate changed; review before this diagnostic')
        calibration = Calibration.load(CALIBRATION, require_verified=False)
        io = strict_io(allow_writes=args.execute_yaw_test)
        samples = [read_head(io) for _ in range(5)]
        report['preflight_samples'] = samples
        save()
        start = preflight(samples)
        report.update(start_feedback=start, calibration_sha256=sha256(CALIBRATION),
                      proposed_target_raw=start[19]['position']+YAW_STEP,
                      camera_serial='22275000', physical_eye='left')
        reference = {}

        def capture(label, feedback):
            before = feedback()
            bgr, info = snapshot(CAMERA)
            after = feedback()
            if abs(after-before) > 2:
                raise ValueError('Head moved during camera capture')
            if (info['width'], info['height']) != calibration.size or info['optical_controls'].get('zoom_absolute') != '0':
                raise ValueError('Camera size or optical controls differ from calibration session')
            path = output / (label+'.png')
            if not cv2.imwrite(str(path), bgr):
                raise IOError('Cannot save original frame')
            rectified = calibration.rectify(bgr)
            pixels = corners(rectified, 9, 6)
            if reference and np.max(np.linalg.norm(pixels-reference['pixels'], axis=2)) > 60:
                raise ValueError('Corner ordering/view changed excessively; not comparable')
            reference.setdefault('pixels', pixels.copy())
            result = board_pose(rectified, calibration.rectified_K, 9, 6, .025)
            cv2.drawChessboardCorners(rectified, (9, 6), pixels, True)
            cv2.imwrite(str(output / (label+'_corners.png')), rectified)
            return {**info, **result, 'image': str(path), 'image_sha256': sha256(path),
                    'pose_verified': False, 'calibration_verified': False,
                    'head_image_synchronization_verified': False,
                    'yaw_before_raw': before, 'yaw_after_raw': after,
                    'board_fixed': 'user_confirmed', 'units': 'metres'}

        def baseline_feedback():
            state = read_head(io)
            if any(state[i]['torque'] or abs(state[i]['position']-start[i]['position']) > 2 for i in (19, 20)):
                raise ValueError('Head moved during stationary baseline or torque changed')
            return state[19]['position']

        report['before'] = capture('before', baseline_feedback)
        report['before_repeat'] = capture('before_repeat', baseline_feedback)
        center = np.array([.10, .0625, 0.])
        report['stationary_pose_difference_m_deg'] = baseline_and_angle(
            report['before']['T_wc'], report['before_repeat']['T_wc'], center)
        if report['stationary_pose_difference_m_deg'][0] > .003:
            raise ValueError('Stationary PnP difference exceeds diagnostic 3 mm threshold')
        report['status'] = 'plan_only_ready'
        save()
        if args.execute_yaw_test:
            supervised_yaw(io, start, capture, report, save)
            displacement, angle = baseline_and_angle(report['before_repeat']['T_wc'], report['after']['T_wc'], center)
            report.update(status='supervised_test_completed', measured_camera_translation_m=displacement,
                          viewing_ray_change_deg=angle,
                          passes_minimum_geometric_gate_only=bool(displacement >= .01 and angle >= 1.),
                          note='Candidate-calibration PnP estimate only. Low reprojection error is not verified metric accuracy; a useful reconstruction baseline is not guaranteed.')
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='stopped', reason=type(exc).__name__+': '+str(exc))
    finally:
        signal.alarm(0)
        if io is not None:
            io.close()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print('Status:', report['status'])
    print('Reason:', report.get('reason', 'none'))
    print('Motion commanded:', report['real_robot_motion_commanded'])
    print('Report:', output / 'head_viewpoint_report.json')
    return report


if __name__ == '__main__':
    main()
