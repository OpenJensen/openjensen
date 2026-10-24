"""Upstream control provenance, batched packets and configuration enable barriers."""

import ast
import copy
import hashlib
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_robot_lab as fixtures
from hardware import HardwareError, ServoBus


class UpstreamControlTests(unittest.TestCase):
    def test_upstream_function_bodies_match_the_pinned_provenance(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "LEROBOT-PROVENANCE.json").read_text())
        tree = ast.parse((root / "lerobot_control.py").read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for source, expected in manifest["methods_ast_sha256"].items():
            method = source.split(":")[1]
            if method in {"configure", "send_action"}:
                nodes = classes["UpstreamSOFollower"].body
            elif method == "configure_motors":
                nodes = classes["MotorBridge"].body
            else:
                nodes = tree.body
            node = copy.deepcopy(
                next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == method)
            )
            node.decorator_list = []
            if (
                isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, expected, source)
        self.assertIn("Apache License", (root / "LEROBOT-LICENSE").read_text())

    def test_group_sync_write_encodes_one_full_multi_joint_goal_packet(self):
        packets = []

        class Writer:
            def __init__(self, port, packet, address, length):
                self.parameters = []
                packets.append(self)
                self.address, self.length, self.sent, self.cleared = address, length, False, False

            def addParam(self, motor_id, value):
                self.parameters.append((motor_id, value))
                return True

            def txPacket(self):
                self.sent = True
                return 0

            def clearParam(self):
                self.cleared = True

        bus = ServoBus.__new__(ServoBus)
        bus.lock, bus.opened, bus.port, bus.packet = threading.RLock(), True, object(), object()
        bus.sdk = SimpleNamespace(GroupSyncWrite=Writer, COMM_SUCCESS=0)
        bus.write_positions({"shoulder_lift": 3000, "elbow_flex": 1024})
        packet = packets[0]
        self.assertEqual((packet.address, packet.length), (42, 2))
        self.assertEqual(packet.parameters, [(2, [184, 11]), (3, [0, 4])])
        self.assertTrue(packet.sent and packet.cleared)
        with self.assertRaises(HardwareError):
            bus.write_positions({"elbow_flex": 1000}, lambda: False)
        self.assertEqual(len(packets), 1)


class ConfigurationTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def test_stock_configuration_and_speed_cap_reset_precede_any_torque_enable(self):
        self.start()
        writes = self.buses[-1].writes
        first_enable = next(
            i for i, (name, _, value) in enumerate(writes) if name == "Torque_Enable" and value == 1
        )
        for motor_id in range(1, 7):
            for setting in [
                ("Return_Delay_Time", 0),
                ("Maximum_Acceleration", 254),
                ("Acceleration", 254),
                ("Goal_Velocity", 0),
                ("P_Coefficient", 16),
                ("I_Coefficient", 0),
                ("D_Coefficient", 32),
            ]:
                self.assertIn((setting[0], motor_id, setting[1]), writes[:first_enable])
        self.assertFalse(
            any(
                name in {"Max_Torque_Limit", "Protection_Current", "Overload_Torque"}
                and motor_id != 6
                for name, motor_id, _ in writes
            )
        )
        self.assertEqual(self.lab.status()["controller"]["target_mode"], "direct_position")

    def test_unverified_stock_configuration_never_enables_torque(self):
        original_read = fixtures.FakeBus.read
        for failed in ("Maximum_Acceleration", "Acceleration", "P_Coefficient", "Goal_Velocity"):

            def read(bus, register, motor_id):
                return -1 if register == failed else original_read(bus, register, motor_id)

            from unittest.mock import patch

            with (
                self.subTest(register=failed),
                patch.object(fixtures.FakeBus, "read", read),
                self.assertRaises(HardwareError),
            ):
                self.start()
            self.assertFalse(
                any(
                    name == "Torque_Enable" and value == 1
                    for name, _, value in self.buses[-1].writes
                )
            )
            self.assertFalse(self.buses[-1].opened)

    def test_identification_ram_cap_is_restored_to_the_saved_ceiling(self):
        factory = self.lab.bus_factory

        def limited_factory(*args):
            bus = factory(*args)
            bus.writes.extend([("Torque_Limit", 2, 100), ("Torque_Limit", 3, 100)])
            return bus

        self.lab.bus_factory = limited_factory
        self.start()
        bus = self.buses[-1]
        self.assertEqual(self.lab.controller.torque_limits["shoulder_lift"], 1000)
        self.assertIn(("Torque_Limit", 2, 1000), bus.writes)
        self.assertIn(("Torque_Limit", 3, 1000), bus.writes)
        self.assertIn(("Torque_Limit", 6, 500), bus.writes)
        self.assertNotIn(("Max_Torque_Limit", 2, 1000), bus.writes)

    def test_saved_lower_arm_torque_ceiling_is_preserved(self):
        original_read = fixtures.FakeBus.read

        def read(bus, register, motor_id):
            return (
                700
                if register == "Max_Torque_Limit" and motor_id == 2
                else original_read(bus, register, motor_id)
            )

        from unittest.mock import patch

        with patch.object(fixtures.FakeBus, "read", read):
            self.start()
        self.assertEqual(self.lab.controller.torque_limits["shoulder_lift"], 700)
        self.assertIn(("Torque_Limit", 2, 700), self.buses[-1].writes)
