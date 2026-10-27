import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "seed_hand_bus_setup.py"
SPEC = importlib.util.spec_from_file_location("seed_hand_bus_setup", PATH)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class FakeBus:
    def __init__(self, right=False, new_occupied=False, lose_after=False):
        self.right = right
        self.new_occupied = new_occupied
        self.lose_after = lose_after
        self.controller = 31
        self.writes = []
        self.identification_reads = []

    def identify(self, device_id):
        self.identification_reads.append(device_id)
        if self.lose_after and self.writes and device_id == 41:
            return None
        if device_id == self.controller:
            return (405, 38)
        if device_id == 41 and self.new_occupied:
            return (410, 53)
        if self.right and device_id == 30:
            return (405, 64)
        if self.right and device_id in (31, 33, 34, 35, 36, 37):
            return (410, 53)
        if device_id in (23, 25, 27, 29):
            return (406, 30)
        return None

    def protocol2_silent(self, device_id):
        return not self.new_occupied

    def read(self, device_id, address, count):
        if address == 24:
            return b"\x00"
        values = {3: bytes([self.controller]), 60: b"\x20\x00",
                  64: b"\x01\x02\x03", 129: bytes([23,25,27,29,0,0,0,0,0,0]),
                  4: b"\x01", 72: b"\x01", 108: bytes(20)}
        return values[address]

    def change_left_controller_id(self):
        self.writes.append((31, 3, 41))
        self.controller = 41


class SetupTests(unittest.TestCase):
    def test_duplicate_replies_are_not_hidden(self):
        first = setup.frame(31, 0, [149,1,38])
        second = setup.frame(31, 0, [154,1,53])
        replies, rejected = setup.parse_replies(first + second)
        self.assertEqual(len(replies), 2)
        self.assertEqual(rejected, 0)

    def test_bad_checksum_is_rejected(self):
        packet = bytearray(setup.frame(31, 0, [149,1,38]))
        packet[-1] ^= 1
        replies, rejected = setup.parse_replies(packet)
        self.assertEqual(replies, [])
        self.assertGreater(rejected, 0)

    def test_both_connected_blocks_before_conflicting_id_or_write(self):
        bus = FakeBus(right=True)
        with self.assertRaisesRegex(setup.SetupError, "Right hand is still connected"):
            setup.preflight(bus)
        self.assertNotIn(31, bus.identification_reads)
        self.assertEqual(bus.writes, [])

    def test_occupied_destination_blocks_write(self):
        bus = FakeBus(new_occupied=True)
        with self.assertRaisesRegex(setup.SetupError, "occupied"):
            setup.preflight(bus)
        self.assertEqual(bus.writes, [])

    def test_only_controller_id_is_changed_and_backup_exists(self):
        bus = FakeBus()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.json"
            original = bus.change_left_controller_id
            def check_backup_then_write():
                saved = json.loads(path.read_text())
                self.assertEqual(saved["before"]["serial_hex"], "010203")
                self.assertTrue(saved["id_write_attempted"])
                self.assertIsNone(saved["hardware_configuration_changed"])
                original()
            bus.change_left_controller_id = check_backup_then_write
            record = {"real_robot_motion_commanded": False}
            setup.apply_repair(bus, record, path)
            self.assertEqual(bus.writes, [(31,3,41)])
            self.assertEqual(record["status"], "left_controller_readdress_verified")
            self.assertEqual(record["after"]["stored_id"], 41)
            self.assertFalse(record["real_robot_motion_commanded"])

    def test_failed_verification_does_not_retry_or_rollback(self):
        bus = FakeBus(lose_after=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.json"
            record = {}
            with self.assertRaisesRegex(setup.SetupError, "may have taken effect"):
                setup.apply_repair(bus, record, path)
            self.assertEqual(bus.writes, [(31,3,41)])
            self.assertEqual(json.loads(path.read_text())["status"], "id_write_unverified")
            self.assertIsNone(record["hardware_configuration_changed"])

    def test_backup_failure_prevents_id_write(self):
        bus = FakeBus()
        with patch.object(setup, "save_record", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                setup.apply_repair(bus, {}, Path("unused.json"))
        self.assertEqual(bus.writes, [])

    def test_enabled_motor_torque_blocks_id_write(self):
        bus = FakeBus()
        original = bus.read
        bus.read = lambda device_id, address, count: (b"\x01" if address == 24
                                                     else original(device_id, address, count))
        with self.assertRaisesRegex(setup.SetupError, "torque enabled"):
            setup.preflight(bus)
        self.assertEqual(bus.writes, [])

    def test_write_packet_targets_only_controller_id_register(self):
        bus = setup.Bus(None)
        packets = []
        bus.exchange = lambda packet: packets.append(packet) or b""
        bus.change_left_controller_id()
        self.assertEqual(packets, [setup.frame(31, 3, [3, 41])])

    def test_duplicate_write_acknowledgements_are_rejected(self):
        bus = setup.Bus(None)
        bus.exchange = lambda packet: setup.frame(31, 0, []) * 2
        with self.assertRaisesRegex(setup.SetupError, "ambiguous"):
            bus.change_left_controller_id()

    def test_both_hands_after_repair(self):
        bus = FakeBus(right=True)
        bus.controller = 41
        data = setup.verify_both(bus)
        self.assertEqual(data["identities"][31], (410,53))
        self.assertEqual(data["controller"]["bus_id"], 41)
        self.assertEqual(bus.writes, [])


if __name__ == "__main__":
    unittest.main()
