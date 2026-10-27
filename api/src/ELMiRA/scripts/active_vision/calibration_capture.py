"""Bounded, camera-only calibration capture using the active V4L2 format.

No ROS/serial imports. Never changes resolution, camera controls or robot pose.
The user moves the board, not NICO. Images retain native distorted pixels.
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import tempfile
import time

import cv2
import numpy as np

from .calibration import corners
from .common import sha256, write_json


def device_info(device):
    path = Path(device)
    if path.parent != Path('/dev/v4l/by-id') or not path.exists():
        raise ValueError("Select an existing /dev/v4l/by-id camera path, not a changing /dev/video number")
    owner = subprocess.run(['fuser', str(path)], capture_output=True, timeout=3)
    if owner.returncode != 1:
        raise ValueError("Camera is busy or ownership check failed; close Cheese/ELMiRA camera capture first")
    result = subprocess.run(['v4l2-ctl', '-d', str(path), '--all'], capture_output=True, text=True, timeout=3)
    if result.returncode:
        raise ValueError("Cannot read camera format: " + result.stderr)
    size = re.search(r'Width/Height\s*:\s*(\d+)/(\d+)', result.stdout)
    pixel_format = re.search(r"Pixel Format\s*:\s*'([^']+)'", result.stdout)
    stride = re.search(r'Bytes per Line\s*:\s*(\d+)', result.stdout)
    if not size or not pixel_format or not stride or pixel_format[1] not in ('UYVY', 'YUYV'):
        raise ValueError("Capture supports native UYVY/YUYV only; no camera settings were changed")
    width, height = map(int, size.groups())
    if width % 2 or min(width, height) <= 0 or int(stride[1]) < width*2:
        raise ValueError("Invalid camera dimensions/stride")
    return {'device': str(path), 'resolved_device': str(path.resolve()),
            'camera_id': path.name, 'width': width, 'height': height,
            'pixel_format': pixel_format[1], 'stride': int(stride[1]), 'v4l2_info': result.stdout,
            'optical_controls': dict(re.findall(r'(zoom_absolute|focus_absolute|focus_automatic_continuous)\s+[^\n]*?value=(-?\d+)', result.stdout))}


def snapshot(device):
    info = device_info(device)
    with tempfile.TemporaryDirectory(prefix='nico_camera_read_') as directory:
        raw_path = Path(directory) / 'frame.raw'
        start = time.time()
        command = ['v4l2-ctl', '-d', info['device'], '--stream-mmap=3', '--stream-count=1',
                   '--stream-skip=5', '--stream-no-query', '--stream-to='+str(raw_path)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=8)
        end = time.time()
        if result.returncode:
            raise ValueError("Camera capture unavailable: " + result.stderr)
        raw = np.fromfile(str(raw_path), dtype=np.uint8)
    if raw.size != info['stride']*info['height']:
        raise ValueError("Incomplete frame; not saving a calibration image")
    packed = raw.reshape(info['height'], info['stride'])[:, :info['width']*2]
    packed = packed.reshape(info['height'], info['width'], 2).copy()
    code = cv2.COLOR_YUV2BGR_UYVY if info['pixel_format'] == 'UYVY' else cv2.COLOR_YUV2BGR_YUYV
    info.update(host_capture_start_sec=start, host_capture_end_sec=end,
                timestamp_source='host operation time; not verified exposure time',
                real_robot_motion_commanded=False, camera_settings_changed=False)
    return cv2.cvtColor(packed, code), info


def capture_board_images(device, output, count=25, columns=9, rows=6, prompt=input):
    if not 15 <= count <= 100 or min(columns, rows) < 3:
        raise ValueError("Use 15..100 frames and at least 3x3 inner corners")
    first_info = device_info(device)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('train', 'holdout', 'review'):
        (output / name).mkdir()
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'camera': first_info,
              'inner_corners': [columns, rows], 'measured_square_m': None,
              'calibration_verified': False, 'real_robot_motion_commanded': False,
              'observations': [], 'status': 'collecting'}
    write_json(output / 'capture_session.json', report)
    print('Move only the board. Hold it still; Enter captures, q stops. No robot movement.')
    try:
        for index in range(count):
            answer = prompt('View %d/%d: reposition/tilt the board, then Enter: ' % (index+1, count))
            if answer.strip().lower() == 'q':
                break
            image, info = snapshot(device)
            if any(info[k] != first_info[k] for k in ('camera_id', 'resolved_device', 'width', 'height', 'pixel_format', 'stride', 'optical_controls')):
                raise ValueError('Camera identity/format/settings changed during capture; start a new session')
            split = 'holdout' if (index+1) % 5 == 0 else 'train'
            path = output / split / ('frame_%03d.png' % index)
            if not cv2.imwrite(str(path), image):
                raise OSError('Cannot save frame')
            rec = {**info, 'image': str(path.relative_to(output)), 'split': split, 'sha256': sha256(path)}
            try:
                pixels = corners(image, columns, rows)
                rec.update(board_detected=True, inner_corners_px=pixels.reshape(-1, 2).tolist())
                review = image.copy()
                cv2.drawChessboardCorners(review, (columns, rows), pixels, True)
                cv2.imwrite(str(output / 'review' / path.name), review)
            except ValueError:
                rec['board_detected'] = False
            report['observations'].append(rec)
            write_json(path.with_suffix('.json'), rec)
            write_json(output / 'capture_session.json', report)
            print('%s | board %s' % (path, 'detected' if rec['board_detected'] else 'NOT detected; improve view'))
        report['status'] = 'captured_unverified'
    except (KeyboardInterrupt, EOFError):
        report['status'] = 'stopped_by_user'
    except Exception as exc:
        report.update(status='capture_stopped', reason=str(exc))
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['detected_counts'] = {split: sum(r['split'] == split and r['board_detected'] for r in report['observations'])
                                     for split in ('train', 'holdout')}
        write_json(output / 'capture_session.json', report)
    return {'status': report['status'], 'session': str(output / 'capture_session.json'),
            'detected_counts': report['detected_counts'], 'real_robot_motion_commanded': False}
