"""Bounded protocol-1 READ-only diagnostics for NICO head IDs19/20.

No motion library, ROS, goal writes, torque changes, or retrying movement.
Raw replies and timestamps are retained so decoding can be independently checked.
"""
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import time

from seed_hand_bus_setup import frame, parse_replies
from .common import LOG_ROOT, sha256, write_json

DEVICE = Path('/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT2XOU56-if00-port0')
BLOCKS = ((0, 3), (3, 7), (11, 8), (20, 3), (24, 5), (30, 14),
          (44, 1), (46, 4), (50, 2), (68, 6))
ERROR_BITS = {1: 'input_voltage', 2: 'angle_limit', 4: 'overheating',
              8: 'range', 16: 'checksum', 32: 'overload', 64: 'instruction'}


def request_packet(motor_id, address, count):
    if motor_id not in (19, 20) or (address, count) not in BLOCKS:
        raise ValueError('Only allowlisted head register READ requests are permitted')
    return frame(motor_id, 2, [address, count])


def decode_block(address, data):
    def word(offset):
        return int.from_bytes(bytes(data[offset:offset+2]), 'little')

    if address == 0:
        return {'model_number': word(0), 'firmware': data[2]}
    if address == 3:
        return {'device_id': data[0], 'baud_raw': data[1], 'return_delay_raw': data[2],
                'cw_limit_raw': word(3), 'ccw_limit_raw': word(5)}
    if address == 11:
        return {'temperature_limit_c': data[0], 'min_voltage_v': data[1]/10.,
                'max_voltage_v': data[2]/10., 'max_torque_raw': word(3),
                'status_return_level': data[5], 'alarm_led_bits': data[6], 'shutdown_bits': data[7]}
    if address == 20:
        return {'multiturn_offset_raw': word(0), 'resolution_divider_raw': data[2]}
    if address == 24:
        return {'torque_enabled_raw': data[0], 'led_raw': data[1],
                'pid_raw': {'D': data[2], 'I': data[3], 'P': data[4]}}
    if address == 30:
        current_pos = word(6)
        return {'goal_position_raw': word(0), 'moving_speed_raw': word(2),
                'torque_limit_raw': word(4), 'present_position_raw': current_pos,
                'present_motor_degrees': current_pos*360./4095.-180.,
                'present_speed_raw': word(8), 'present_load_raw': word(10),
                'present_voltage_v': data[12]/10., 'present_temperature_c': data[13]}
    if address == 44:
        return {'registered_instruction_raw': data[0]}
    if address == 46:
        return {'moving_raw': data[0], 'eeprom_lock_raw': data[1], 'punch_raw': word(2)}
    if address == 50:
        return {'realtime_tick_raw': word(0)}
    if address == 68:
        return {'present_current_raw': word(0), 'mx64_current_a': (word(0)-2048)*.0045,
                'torque_control_mode_raw': data[2], 'goal_torque_raw': word(3),
                'goal_acceleration_raw': data[5]}
    raise ValueError('Unknown register block')


def exchange(port, motor_id, address, count, report):
    packet = request_packet(motor_id, address, count)
    row = {'motor_id': motor_id, 'address': address, 'count': count,
           'instruction': 'READ', 'request_hex': packet.hex(), 'host_start_sec': time.time()}
    report['transactions'].append(row)
    port.reset_input_buffer()
    time.sleep(.02)
    if port.in_waiting:
        raise IOError('Bus is active before request; refusing to compete with another controller')
    if port.write(packet) != len(packet):
        raise IOError('Incomplete read request')
    received = bytearray()
    end = time.monotonic()+.2
    while time.monotonic() < end:
        received.extend(port.read(min(256, max(1, port.in_waiting))))
        if len(received) > 512:
            raise IOError('Unexpected continuous bus response')
    replies, rejected = parse_replies(received)
    row.update(host_end_sec=time.time(), reply_hex=received.hex(),
               replies=replies, rejected_bytes=rejected)
    if rejected or len(replies) != 1:
        raise IOError('Missing, corrupt or ambiguous response from ID%d' % motor_id)
    reply = replies[0]
    if reply['id'] != motor_id or len(reply['data']) != count:
        raise IOError('Mismatched ID or register count')
    row['error_flags'] = [label for bit, label in ERROR_BITS.items() if reply['error'] & bit]
    row['decoded'] = decode_block(address, reply['data'])
    if reply['error']:
        raise IOError('Motor reports error bits %d: %s' % (reply['error'], row['error_flags']))
    return row['decoded']


def main():
    import serial
    output = LOG_ROOT / ('head_diagnostics_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preflight', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'device': str(DEVICE), 'protocol': 1, 'baud': 1000000,
              'motors': {}, 'transactions': [], 'repeated_state_samples': [],
              'real_robot_motion_commanded': False, 'robot_register_write_attempted': False,
              'script_sha256': sha256(__file__),
              'reference': 'https://emanual.robotis.com/docs/en/dxl/mx/mx-64/',
              'limitation': 'Point-in-time read-only evidence; present current/load with torque off cannot explain current/load during the earlier movement test.'}

    def save():
        write_json(output / 'head_diagnostics.json', report)

    def interrupted(signum, stack):
        raise TimeoutError('Read-only diagnostic interrupted or exceeded 18 seconds')

    save()
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(18)
    try:
        if not DEVICE.exists():
            raise FileNotFoundError('NICO USB adapter unavailable')
        owner = subprocess.run(['fuser', str(DEVICE.resolve())], capture_output=True, text=True, timeout=2)
        report['port_owner_check'] = {'returncode': owner.returncode, 'pids': owner.stdout.strip()}
        if owner.returncode != 1:
            raise IOError('Serial port busy or ownership check unavailable')
        with serial.Serial(str(DEVICE), baudrate=1000000, timeout=.02,
                           write_timeout=.1, exclusive=True) as port:
            for motor_id in (19, 20):
                motor = report['motors'].setdefault(str(motor_id), {})
                for address, count in BLOCKS:
                    motor.update(exchange(port, motor_id, address, count, report))
                    save()
                    if address == 0 and motor['model_number'] != 310:
                        raise ValueError('Not the expected model310; refusing other control-table reads')
            for index in range(3):
                for motor_id in (19, 20):
                    sample = {'sample': index, 'motor_id': motor_id, 'host_start_sec': time.time()}
                    for address, count in ((24, 5), (30, 14), (50, 2)):
                        sample.update(exchange(port, motor_id, address, count, report))
                    report['repeated_state_samples'].append(sample)
                    save()
        report['status'] = 'read_only_snapshot_complete'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='partial' if report['transactions'] else 'unavailable',
                      reason=type(exc).__name__+': '+str(exc))
    finally:
        signal.alarm(0)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print('Status:', report['status'])
    print('Motor settings:', report['motors'])
    print('Reason:', report.get('reason', 'none'))
    print('No motor-register writes or motion commands.')
    print('Report:', output / 'head_diagnostics.json')
    return report


if __name__ == '__main__':
    main()
