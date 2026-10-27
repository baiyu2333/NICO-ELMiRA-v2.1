#!/usr/bin/env python3
"""Repair the swapped hands' duplicate ID31 without commanding robot motion.

Default: read-only check. --apply readdresses ONLY the isolated left EROS
controller from31 to41, after backing up its identity. --verify-both checks
the repaired bus after reconnecting the right hand. Stop all serial users
first. Power off before disconnecting/reconnecting a hand.

Register3 is the persistent EROS controller ID:
https://kb.seedrobotics.com/doku.php?id=eros:mainboard_ctltable
This utility never writes torque, speed, position, PID or protection registers.
It does not enable the left arm or change ELMiRA's motion configuration.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time


OLD_ID = 31
NEW_ID = 41
LEFT_MOTORS = (23, 25, 27, 29)
RIGHT_UNIQUE_IDS = (30, 33, 34, 35, 36, 37)
LEFT_CONTROLLER_IDENTITY = (405, 38)


class SetupError(RuntimeError):
    pass


def api_root():
    for candidate in Path(__file__).resolve().parents:
        if candidate.name == "api" and (candidate / "src" / "ELMiRA").is_dir():
            return candidate
    raise SetupError("Cannot locate api root")


def frame(device_id, instruction, params):
    body = bytes([device_id, len(params) + 2, instruction]) + bytes(params)
    return b"\xff\xff" + body + bytes([(~sum(body)) & 255])


def parse_replies(raw):
    """Retain all valid replies, including duplicate-ID responders."""
    data = bytearray(raw)
    replies = []
    rejected = 0
    while data:
        start = data.find(b"\xff\xff")
        if start < 0:
            rejected += len(data)
            break
        rejected += start
        del data[:start]
        if len(data) < 4:
            rejected += len(data)
            break
        length = data[3]
        if length < 2 or length > 64:
            rejected += 1
            del data[0]
            continue
        size = length + 4
        if len(data) < size:
            rejected += len(data)
            break
        packet = bytes(data[:size])
        del data[:size]
        if sum(packet[2:]) & 255 != 255:
            rejected += size
            continue
        replies.append({"id": packet[2], "error": packet[4],
                        "data": list(packet[5:-1])})
    return replies, rejected


class Bus:
    def __init__(self, port):
        self.port = port

    def exchange(self, packet):
        # Do not treat a noisy or continuously replying bus as a free address.
        self.port.reset_input_buffer()
        time.sleep(0.03)
        if self.port.in_waiting:
            raise SetupError("Bus is not quiet; isolate the left hand and power-cycle before retrying")
        self.port.write(packet)
        self.port.flush()
        until = time.monotonic() + 0.16
        raw = bytearray()
        while time.monotonic() < until:
            raw.extend(self.port.read(min(512, max(1, self.port.in_waiting))))
            if len(raw) > 2048:
                raise SetupError("Continuous/conflicting replies; no configuration write is safe")
        return bytes(raw)

    def read(self, device_id, address, count, allow_silent=False):
        raw = self.exchange(frame(device_id, 2, [address, count]))
        if not raw and allow_silent:
            return None
        replies, rejected = parse_replies(raw)
        if rejected or len(replies) != 1:
            raise SetupError("ID%d did not give exactly one clean reply" % device_id)
        reply = replies[0]
        if reply["id"] != device_id or reply["error"] or len(reply["data"]) != count:
            raise SetupError("Unexpected/error reply at ID%d: %s" % (device_id, reply))
        return bytes(reply["data"])

    def identify(self, device_id):
        data = self.read(device_id, 0, 3, allow_silent=True)
        return None if data is None else (int.from_bytes(data[:2], "little"), data[2])

    def protocol2_silent(self, device_id):
        from xl320_scan import build_packet
        return not self.exchange(build_packet(device_id, 1))

    def change_left_controller_id(self):
        # The only WRITE this utility can send: old31, register3, new41.
        raw = self.exchange(frame(OLD_ID, 3, [3, NEW_ID]))
        replies, rejected = parse_replies(raw)
        if rejected or len(replies) > 1 or any(r["id"] not in (OLD_ID, NEW_ID) or r["error"] or r["data"] for r in replies):
            raise SetupError("ID write acknowledgement ambiguous; inspect both IDs before any retry")


def integer(bus, device_id, address, count=1):
    return int.from_bytes(bus.read(device_id, address, count), "little")


def controller_snapshot(bus, device_id):
    identities = [bus.identify(device_id) for _ in range(3)]
    if any(value != LEFT_CONTROLLER_IDENTITY for value in identities):
        raise SetupError("ID%d is not the isolated left EROS controller" % device_id)
    snapshot = {
        "bus_id": device_id,
        "stored_id": integer(bus, device_id, 3),
        "model": 405,
        "firmware": 38,
        "hardware_model": integer(bus, device_id, 60, 2),
        "serial_hex": bus.read(device_id, 64, 3).hex(),
        "attached_motor_ids": list(bus.read(device_id, 129, 10)),
        "baud_setting": integer(bus, device_id, 4),
        "external_protocol": integer(bus, device_id, 72),
    }
    if snapshot["stored_id"] != device_id or snapshot["hardware_model"] != 32:
        raise SetupError("Left controller stored ID/hardware identity mismatch")
    if snapshot["attached_motor_ids"] != [23, 25, 27, 29, 0, 0, 0, 0, 0, 0]:
        raise SetupError("Unexpected left controller attached motors")
    return snapshot


def preflight(bus):
    # Checking unique right-hand IDs first avoids querying the conflicting31.
    for _ in range(2):
        for device_id in RIGHT_UNIQUE_IDS:
            if bus.identify(device_id) is not None:
                raise SetupError("Right hand is still connected (ID%d). Power off, disconnect it, then restore power" % device_id)
    for _ in range(3):
        if bus.identify(NEW_ID) is not None or not bus.protocol2_silent(NEW_ID):
            raise SetupError("ID41 is occupied; no ID change attempted")
    snapshot = controller_snapshot(bus, OLD_ID)
    for device_id in LEFT_MOTORS:
        if bus.identify(device_id) != (406, 30):
            raise SetupError("Unexpected left actuator identity at ID%d" % device_id)
        if integer(bus, device_id, 24) != 0:
            raise SetupError("ID%d has torque enabled; stop motion before setup" % device_id)
    return snapshot


def save_record(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_repair(bus, record, path):
    record["before"] = preflight(bus)
    record["status"] = "backup_saved_before_id_write"
    save_record(record, path)  # Abort before the EEPROM write if backup fails.
    # Persist intent first: an interruption during a write leaves its outcome unknown.
    record["id_write_attempted"] = True
    record["hardware_configuration_changed"] = None
    record["status"] = "id_write_pending_verification"
    save_record(record, path)
    try:
        bus.change_left_controller_id()
        time.sleep(0.1)
        after = controller_snapshot(bus, NEW_ID)
        for field in ("serial_hex", "attached_motor_ids", "hardware_model", "firmware", "baud_setting", "external_protocol"):
            if after[field] != record["before"][field]:
                raise SetupError("Post-write identity/settings mismatch: " + field)
        if any(bus.identify(OLD_ID) is not None for _ in range(3)):
            raise SetupError("ID31 still responds with the right hand isolated")
        record.update(status="left_controller_readdress_verified", after=after,
                      hardware_configuration_changed=True)
    except Exception as exc:
        record.update(status="id_write_unverified", message=str(exc))
        raise SetupError("ID write may have taken effect. Keep the right hand disconnected; read41/31 before any retry: " + str(exc))
    finally:
        save_record(record, path)


def verify_both(bus):
    snapshot = controller_snapshot(bus, NEW_ID)
    identities = {device_id: bus.identify(device_id)
                  for device_id in LEFT_MOTORS + (30, 31, 33, 34, 35, 36, 37)}
    expected = {device_id: (406, 30) for device_id in LEFT_MOTORS}
    expected.update({device_id: (410, 53) for device_id in (31, 33, 34, 35, 36, 37)})
    expected[30] = (405, 64)
    if identities != expected:
        raise SetupError("Both-hand identities incomplete/unexpected: " + repr(identities))
    for _ in range(3):
        if bus.identify(31) != (410, 53) or bus.identify(41) != LEFT_CONTROLLER_IDENTITY:
            raise SetupError("ID31/41 verification inconsistent")
    raw = bus.read(41, 108, 20)
    return {"controller": snapshot, "identities": identities,
            "left_currents_mA": [int.from_bytes(raw[i:i+2], "little") for i in range(0, 20, 2)]}


@contextmanager
def open_bus(device, baud):
    import serial
    if not shutil.which("fuser"):
        raise SetupError("fuser is required to check serial ownership")
    ownership = subprocess.run(["fuser", device], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ownership.returncode != 1:
        raise SetupError("Serial port is in use or cannot be checked; stop ELMiRA/other serial users first")
    with serial.Serial(device, baudrate=baud, timeout=0.01,
                       write_timeout=0.1, exclusive=True) as port:
        yield Bus(port)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Permanently change isolated LEFT controller31 to41; no motion")
    mode.add_argument("--verify-both", action="store_true", help="Read-only verification after reconnecting both hands")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=1000000)
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc)
    record = {"timestamp": timestamp.isoformat(), "old_controller_id": OLD_ID,
              "new_controller_id": NEW_ID, "port": args.port,
              "mode": "apply" if args.apply else "verify_both" if args.verify_both else "check",
              "real_robot_motion_commanded": False, "id_write_attempted": False,
              "hardware_configuration_changed": False}
    path = api_root() / "logs" / "hand_diagnostics" / ("hand_bus_setup_" + timestamp.strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    status = 0
    try:
        with open_bus(args.port, args.baud) as bus:
            if args.apply:
                apply_repair(bus, record, path)
            elif args.verify_both:
                record.update(status="both_hands_unique_ids_verified", verification=verify_both(bus))
            else:
                record.update(status="isolated_left_ready_for_id_repair", before=preflight(bus))
    except Exception as exc:
        if record.get("status") != "id_write_unverified":
            record["status"] = "unavailable_or_blocked"
        record["message"] = str(exc)
        status = 2
    record["log_path"] = str(path)
    save_record(record, path)
    print(json.dumps(record, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
