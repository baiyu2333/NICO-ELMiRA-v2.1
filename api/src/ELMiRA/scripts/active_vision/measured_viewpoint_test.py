"""Fixed, supervised yaw/camera diagnostic for the reviewed 2026-10-19 setup.

Default prints a plan without opening hardware. Explicit execution reuses the
head-only transport and its tested low-speed/low-torque out-and-return routine.
Candidate checkerboard poses remain unverified for metric reconstruction.
"""
import argparse
from datetime import datetime, timezone
import signal
import subprocess
import time

import cv2
import numpy as np

from .calibration import Calibration, board_pose, corners
from .calibration_capture import snapshot
from .common import LOG_ROOT, read_json, sha256, write_json
from .geometry import baseline_and_angle
from .head_viewpoint_test import (
    CAMERA, CALIBRATION, CALIBRATION_SHA, DEVICE, SPEED_RAW, TORQUE_RAW,
    SWEEP_OFFSETS, preflight, read_head, strict_io, supervised_yaw,
)


HEAD_START = {19: 2119, 20: 2128}
HEAD_REFERENCE = LOG_ROOT / 'head_diagnostics_20261019T062313_961374Z/head_diagnostics.json'
HEAD_REFERENCE_SHA = '946e21cf8df3703c8cb717a5e38044b47018ed8e9205077102daea05d2d05bce'
BOARD_REFERENCE = LOG_ROOT / 'board_visibility_20261019T061924_299954Z/capture.json'
BOARD_REFERENCE_SHA = 'e7b9f3790bf4eff70c3aada77f696a155a5cb1a18c335ba8709269273f276349'
BOARD_CENTER = np.array([.10, .0625, 0.])
EXPERIMENTAL_CALIBRATION = LOG_ROOT / 'calibration_update_20261019T063835_042995Z/candidate_18views.json'
EXPERIMENTAL_CALIBRATION_SHA = '312e8c199f7b1736ca40652842787106f4d3b5e8066b4330e452ceaa984601b2'
OPPOSITE_HEAD_START = {19: 2124, 20: 2128}
OPPOSITE_HEAD_REFERENCE = LOG_ROOT / 'head_diagnostics_20261019T075543_630857Z/head_diagnostics.json'
OPPOSITE_HEAD_REFERENCE_SHA = '3469dc16ad4682833cc887080f03ffb98b963f98854b5a9d907f71057e24f1e9'
OPPOSITE_TARGET_IMAGE = LOG_ROOT / 'wood_multiview_20261019T073155_422464Z/images/before.png'
OPPOSITE_TARGET_IMAGE_SHA = '9a6e85294492d3fbb4414a24d4c32db5b7f3e4b0b638c8118948526f0d8a65a5'
OPPOSITE_TARGET_BOX = (397, 555, 54, 66)


def target_visibility(image, reference):
    """Conservative appearance check, not evaluated object recognition."""
    template = reference['template']
    x, y, width, height = reference['last_box']
    sx, sy = max(0, x-100), max(0, y-50)
    search = image[sy:min(image.shape[0], y+height+50),
                   sx:min(image.shape[1], x+width+100)]
    best_score, best_box = -1., None
    for scale in (.8, .9, 1., 1.1, 1.2):
        candidate = cv2.resize(template, (round(template.shape[1]*scale),
                                         round(template.shape[0]*scale)))
        if min(candidate.shape[:2]) < 4 or any(a > b for a, b in zip(candidate.shape[:2], search.shape[:2])):
            continue
        score = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
        _, peak, _, where = cv2.minMaxLoc(score)
        if peak > best_score:
            best_score = float(peak)
            best_box = (sx+where[0], sy+where[1], candidate.shape[1], candidate.shape[0])
    inside = False
    if best_box is not None:
        x, y, width, height = best_box
        patch = image[y:y+height, x:x+width]
        inside = (x >= 8 and y >= 8 and x+width < image.shape[1]-8
                  and y+height < image.shape[0]-8
                  and float(np.mean(np.max(patch, axis=2) < 5)) < .05)
    visible = bool(best_score >= .80 and inside)
    if visible:
        reference['last_box'] = best_box
    return {'visible': visible, 'appearance_ncc': best_score,
            'box_xywh': list(best_box) if best_box is not None else None,
            'method': 'Local multiscale RGB template check; score is not a probability',
            'object_recognition_verified': False}


def capture_view(io, output, label, calibration, start, feedback=None,
                 reference_pixels=None, yaw_tolerance=2, target_reference=None):
    """Camera failures cancel later stages; feedback faults abort the motion."""
    def measured_state():
        if feedback is not None:
            feedback()
        state = read_head(io)
        if state[20]['torque'] != 0 or abs(state[20]['position']-start[20]['position']) > 2:
            raise ValueError('Pitch changed during camera check')
        expected_torque = 0 if feedback is None else 1
        if state[19]['torque'] != expected_torque:
            raise ValueError('Yaw torque changed during camera check')
        if feedback is None and abs(state[19]['position']-start[19]['position']) > yaw_tolerance:
            raise ValueError('Yaw changed during stationary camera check')
        return state

    before = measured_state()
    record = {'label': label, 'status': 'unavailable', 'continue_outward': False,
              'real_robot_motion_commanded_by_camera_capture': False,
              'calibration_verified': False, 'pose_verified': False,
              'head_image_synchronization_verified': False,
              'head_before': before, 'board_fixed': 'user_confirmed',
              'physical_eye': 'left', 'camera_serial': '22275000', 'units': 'metres'}
    try:
        image, info = snapshot(CAMERA)
        record.update(info)
        path = output / (label+'.png')
        if not cv2.imwrite(str(path), image):
            raise OSError('Cannot save original image')
        record.update(image=str(path), image_sha256=sha256(path))
        if ((info['width'], info['height']) != calibration.size
                or info['camera_id'] != calibration.record['camera_id']
                or info['optical_controls'].get('zoom_absolute') != '0'):
            raise ValueError('Camera identity, size or zoom differs from calibration')
        rectified = calibration.rectify(image)
        pixels = corners(rectified, 9, 6)
        result = board_pose(rectified, calibration.rectified_K, 9, 6, .025,
                            reference_pixels=reference_pixels)
        if result['corner_order_reversed_for_reference']:
            pixels = pixels[::-1].copy()
        overlay = rectified.copy()
        cv2.drawChessboardCorners(overlay, (9, 6), pixels, True)
        if not cv2.imwrite(str(output / (label+'_corners.png')), overlay):
            raise OSError('Cannot save corner overlay')
        record.update(result, status='candidate_pose_available', continue_outward=True,
                      inner_corners_px=pixels.reshape(-1, 2).tolist(), corner_count=len(pixels))
        if target_reference is not None:
            record['target_visibility'] = target_visibility(image, target_reference)
            if not record['target_visibility']['visible']:
                record.update(status='target_visibility_unavailable', continue_outward=False,
                              reason='Target appearance/visibility check failed; cancel larger steps')
            else:
                x, y, width, height = record['target_visibility']['box_xywh']
                xx, yy = np.meshgrid(np.linspace(x+3, x+width-3, 7),
                                     np.linspace(y+3, y+height-3, 7))
                record['target_pixel_domain'] = calibration.pixel_domain_report(np.c_[xx.ravel(), yy.ravel()])
                if not record['target_pixel_domain']['all_pixels_valid']:
                    record.update(status='target_projection_unavailable', continue_outward=False,
                                  reason='Target pixels leave the calibration inverse domain; cancel larger steps')
    except (ValueError, OSError, cv2.error, subprocess.SubprocessError) as exc:
        record.update(status='unavailable', continue_outward=False,
                      reason=type(exc).__name__+': '+str(exc))
    # Keep serial/health faults outside the camera exception handler.
    after = measured_state()
    record['head_after'] = after
    if any(abs(after[i]['position']-before[i]['position']) > 2 for i in (19, 20)):
        raise ValueError('Head moved during camera acquisition')
    write_json(output / (label+'.json'), record)
    print('View:', label, record['status'], record.get('reason', ''), flush=True)
    return record


def run(execute=False, workspace_clear=False, full_sweep_clear=False,
        experimental_calibration=False, opposite_yaw=False):
    if execute and not (workspace_clear and full_sweep_clear):
        raise ValueError('Execution requires confirmed current head and full-sweep clearance')
    output = LOG_ROOT / ('measured_viewpoint_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    calibration_path = EXPERIMENTAL_CALIBRATION if experimental_calibration else CALIBRATION
    calibration_sha = EXPERIMENTAL_CALIBRATION_SHA if experimental_calibration else CALIBRATION_SHA
    head_start = OPPOSITE_HEAD_START if opposite_yaw else HEAD_START
    head_reference = OPPOSITE_HEAD_REFERENCE if opposite_yaw else HEAD_REFERENCE
    head_sha = OPPOSITE_HEAD_REFERENCE_SHA if opposite_yaw else HEAD_REFERENCE_SHA
    report = {'status': 'plan_only', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha256(__file__), 'reviewed_start_raw': head_start,
              'head_reference': str(head_reference), 'head_reference_sha256': head_sha,
              'board_reference': str(BOARD_REFERENCE), 'board_reference_sha256': BOARD_REFERENCE_SHA,
              'calibration': str(calibration_path), 'calibration_sha256': calibration_sha,
              'experimental_calibration': experimental_calibration,
              'opposite_yaw': opposite_yaw,
              'planned_outward_offsets_raw': [-v for v in SWEEP_OFFSETS] if opposite_yaw else list(SWEEP_OFFSETS),
              'speed_raw': SPEED_RAW, 'torque_limit_raw': TORQUE_RAW,
              'settling_tolerance_raw': 6, 'settling_stability_window_raw': 2,
              'workspace_clear_confirmed': workspace_clear, 'full_sweep_clear_confirmed': full_sweep_clear,
              'yaw_only_motor_id': 19, 'pitch_commanded': False, 'arms_hands_commanded': False,
              'real_robot_motion_commanded': False, 'robot_register_write_attempted': False,
              'verified_for_reconstruction': False, 'stronger_retry_allowed': False,
              'note': 'Fixed-board candidate pose comparison only. Intrinsics, metric accuracy and exposure/joint synchronization remain unverified.'}

    def save():
        write_json(output / 'viewpoint_report.json', report)

    save()
    if not execute:
        print('Plan only; no hardware opened. Report:', output / 'viewpoint_report.json')
        return report

    def interrupt(signum, stack):
        # Do not let the camera-only OSError handler consume a total timeout.
        raise RuntimeError('Supervised camera test interrupted or total timeout')

    io = None
    old_alarm = signal.signal(signal.SIGALRM, interrupt)
    old_term = signal.signal(signal.SIGTERM, interrupt)
    signal.alarm(115)
    try:
        for path, digest in ((head_reference, head_sha),
                             (BOARD_REFERENCE, BOARD_REFERENCE_SHA), (calibration_path, calibration_sha)):
            if sha256(path) != digest:
                raise ValueError('Reviewed reference changed; no execution')
        reference = read_json(head_reference)
        if reference['status'] != 'read_only_snapshot_complete':
            raise ValueError('Reviewed head reference is incomplete')
        for motor_id, position in head_start.items():
            motor = reference['motors'][str(motor_id)]
            if (motor['present_position_raw'] != position or motor['torque_enabled_raw'] != 0
                    or motor['model_number'] != 310):
                raise ValueError('Head reference does not match the named profile')
        calibration = Calibration.load(calibration_path, require_verified=False)
        target_reference = None
        if opposite_yaw:
            if sha256(OPPOSITE_TARGET_IMAGE) != OPPOSITE_TARGET_IMAGE_SHA:
                raise ValueError('Reviewed target reference changed; no execution')
            target_image = cv2.imread(str(OPPOSITE_TARGET_IMAGE))
            if target_image is None:
                raise ValueError('Target reference unavailable; no execution')
            x, y, width, height = OPPOSITE_TARGET_BOX
            target_reference = {'template': target_image[y:y+height, x:x+width].copy(),
                                'last_box': OPPOSITE_TARGET_BOX}
            report.update(target_reference=str(OPPOSITE_TARGET_IMAGE),
                          target_reference_sha256=OPPOSITE_TARGET_IMAGE_SHA)
        if not DEVICE.exists():
            raise ValueError('NICO adapter unavailable')
        owner = subprocess.run(['fuser', str(DEVICE.resolve())], capture_output=True, timeout=2)
        if owner.returncode != 1:
            raise ValueError('Serial port busy or ownership cannot be checked')
        io = strict_io(allow_writes=True)
        samples = []
        for _ in range(5):
            samples.append(read_head(io))
            time.sleep(.1)
        report['preflight_samples'] = samples
        start = preflight(samples, raised_head_clear=True, yaw_20deg=True,
                          reviewed_start=head_start, opposite_yaw=opposite_yaw)
        report['start_feedback'] = start
        report['before'] = capture_view(io, output, 'before', calibration, start,
                                      target_reference=target_reference)
        if not report['before']['continue_outward']:
            raise ValueError('Baseline camera pose unavailable; no movement permitted')
        pixels = report['before']['inner_corners_px']
        report['before_repeat'] = capture_view(io, output, 'before_repeat', calibration, start,
                                               reference_pixels=pixels, target_reference=target_reference)
        repeat = report['before_repeat']
        if not repeat['continue_outward']:
            raise ValueError('Repeated baseline unavailable; no movement permitted')
        difference, angle = baseline_and_angle(report['before']['T_wc'], repeat['T_wc'], BOARD_CENTER)
        corner_error = np.linalg.norm(np.asarray(pixels)-np.asarray(repeat['inner_corners_px']), axis=1)
        report['stationary_pose_difference_m_deg'] = [difference, angle]
        report['stationary_maximum_corner_change_px'] = float(corner_error.max())
        if difference > .003 or angle > 1 or corner_error.max() > 1:
            raise ValueError('Stationary baseline inconsistent; no movement permitted')
        report['status'] = 'supervised_motion_started'
        save()

        def capture(label, feedback):
            return capture_view(io, output, label, calibration, start,
                                feedback=feedback, reference_pixels=pixels, target_reference=target_reference)

        supervised_yaw(io, start, capture, report, save,
                       accept_current_goal_activation=True, yaw_20deg=True,
                       reviewed_start=head_start, stage_capture=capture, opposite_yaw=opposite_yaw)
        if target_reference is not None:
            # Return is a known reset to the baseline, not an adjacent image step.
            target_reference['last_box'] = OPPOSITE_TARGET_BOX
        report['returned_view'] = capture_view(io, output, 'returned', calibration, start,
                                                reference_pixels=pixels, yaw_tolerance=6,
                                                target_reference=target_reference)
        report['candidate_view_displacements'] = []
        for view in report['stage_views']:
            if not view['continue_outward']:
                continue
            distance, angle = baseline_and_angle(repeat['T_wc'], view['T_wc'], BOARD_CENTER)
            report['candidate_view_displacements'].append({
                'label': view['label'], 'camera_translation_m': distance,
                'viewing_ray_change_deg': angle,
                'passes_minimum_geometric_gate_only': bool(distance >= .01 and angle >= 1.),
                'pose_verified': False})
        if report['returned_view']['continue_outward']:
            report['return_camera_difference_m_deg'] = list(baseline_and_angle(
                repeat['T_wc'], report['returned_view']['T_wc'], BOARD_CENTER))
        if not report['returned_view']['continue_outward']:
            report['status'] = 'partial_return_camera_check'
        else:
            report['status'] = ('partial_camera_check_returned_to_start'
                                if report.get('further_outward_motion_cancelled_by_camera_check')
                                else 'supervised_viewpoint_check_completed')
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
    print('Cleanup:', report.get('cleanup', 'no register writes attempted'))
    print('Candidate camera displacements:', report.get('candidate_view_displacements', []))
    print('Report:', output / 'viewpoint_report.json')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--workspace-clear', action='store_true')
    parser.add_argument('--full-sweep-clear', action='store_true')
    parser.add_argument('--opposite-yaw', action='store_true',
                        help='Separately cleared fixed negative5/10/20-degree sweep; no arbitrary angles')
    parser.add_argument('--experimental-calibration', action='store_true',
                        help='Use the hash-pinned offline18-view candidate; results stay unverified')
    args = parser.parse_args(argv)
    if args.execute and not (args.workspace_clear and args.full_sweep_clear):
        parser.error('Execution requires both clearance flags and physical supervision')
    return run(args.execute, args.workspace_clear, args.full_sweep_clear,
               experimental_calibration=args.experimental_calibration, opposite_yaw=args.opposite_yaw)


if __name__ == '__main__':
    main()
