#!/usr/bin/env python3
"""Safe Dynamixel Protocol 2.0 ping scanner for the XL-320 left hand.

This script only sends PING packets. It does not enable torque or move motors.
Run it with Motion.py stopped so the serial bus is not shared.
"""

import argparse
import struct
import time

import serial


def crc16(data):
    """CRC-16 used by Dynamixel Protocol 2.0 packets."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x8005
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def build_packet(motor_id, instruction, params=b""):
    length = len(params) + 3
    packet = bytearray([0xFF, 0xFF, 0xFD, 0x00, motor_id])
    packet += struct.pack("<H", length)
    packet.append(instruction)
    packet += params
    packet += struct.pack("<H", crc16(packet))
    return bytes(packet)


def scan(port, baud, start_id, end_id, timeout):
    found = []
    with serial.Serial(port, baudrate=baud, timeout=timeout, write_timeout=0.1) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        for motor_id in range(start_id, end_id + 1):
            ser.reset_input_buffer()
            ser.write(build_packet(motor_id, 0x01))
            ser.flush()
            time.sleep(0.006)
            data = ser.read(64)
            if b"\xff\xff\xfd\x00" in data:
                found.append((motor_id, data.hex()))
    return found


def main():
    parser = argparse.ArgumentParser(description="Scan for XL-320 / Protocol 2 motors")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=0.025)
    args = parser.parse_args()

    found = scan(args.port, args.baud, args.start_id, args.end_id, args.timeout)
    print(f"Protocol 2 responders on {args.port} at {args.baud}: {len(found)}")
    for motor_id, raw in found:
        print(f"  id {motor_id}: {raw[:80]}")


if __name__ == "__main__":
    main()
