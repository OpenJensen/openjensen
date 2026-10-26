"""Governed physical protocol on a fake serial bus; never hardware acceptance."""

import copy
import http.client
import json
import math
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_robot_lab as fixtures
from agent_control import AgentExecutor, Segment, digest, durable_json, strict_json
from hardware import JOINTS, HardwareError
from server import create_server


class TimedCamera:
    def __init__(self):
        self.calls = 0
        self.on_capture = None

    def capture_timed(self, index):
        self.calls += 1
        started = time.monotonic_ns()
        if self.on_capture:
            self.on_capture()
        return {
            "jpeg": b"\xff\xd8synthetic-protocol-fixture\xff\xd9",
            "read_start_monotonic_ns": started,
            "read_end_monotonic_ns": time.monotonic_ns(),
        }

    def close(self):
        pass

    def close_idle(self):
        pass


class AgentTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def command(self, operation, arguments=None, *, ident=None, epoch=None, owner="agent_test"):
        return self.lab.agent.command(
            operation,
            {
                "command_id": ident or uuid.uuid4().hex,
                "agent_session_id": owner,
                "authority_epoch": (0 if operation == "acquire" else self.lab.agent.epoch)
                if epoch is None
                else epoch,
                "arguments": arguments or {},
            },
        )

    def arm(self):
        self.start()
        self.lab.camera = TimedCamera()
        result = self.command(
            "acquire",
            {
                "confirm": "ARM AI CONTROL",
                "expected_generation": self.lab.generation,
                "camera_index": 0,
                "operator_present": True,
                "workspace_clear": True,
            },
        )
        self.assertEqual(result["status"], "completed")
        return result

    def observe(self):
        return self.command("observe")["observation"]

    def segment(self, observed, targets=None, duration=2.0, **extras):
        arguments = {
            "observation_id": observed["observation_id"],
            "observation_sha256": observed["observation_sha256"],
            "targets": targets or {"shoulder_pan": observed["joints"]["shoulder_pan"] + 3.0},
            "duration_seconds": duration,
        }
        arguments["units"] = {j: observed["units"][j] for j in arguments["targets"]}
        arguments.update(extras)
        return arguments

    def test_arming_never_starts_idle_or_implicit_motion(self):
        self.lab.camera = TimedCamera()
        initial = self.lab.generation
        with self.assertRaisesRegex(ValueError, "Start full-arm"):
            self.command(
                "acquire",
                {
                    "confirm": "ARM AI CONTROL",
                    "expected_generation": initial,
                    "camera_index": 0,
                    "operator_present": True,
                    "workspace_clear": True,
                },
            )
        self.assertEqual(self.buses, [])
        self.assertFalse(self.lab.torque)
        self.assertEqual(self.lab.generation, initial)

    def test_acquisition_is_exact_and_manual_ownership_cannot_overlap(self):
        self.arm()
        before = list(self.buses[-1].writes)
        self.assertEqual(self.lab.mode, "agent")
        with self.assertRaisesRegex(ValueError, "Start full-arm"):
            self.command(
                "acquire",
                {
                    "confirm": "ARM AI CONTROL",
                    "expected_generation": self.lab.generation,
                    "camera_index": 0,
                    "operator_present": True,
                    "workspace_clear": True,
                },
            )
        with self.assertRaisesRegex(ValueError, "manual control"):
            self.lab.jog({"joint": "shoulder_pan", "delta": 1})
        self.assertEqual(self.buses[-1].writes, before)

    def test_observation_is_fresh_hashed_mixed_units_and_read_interval_labeled(self):
        self.arm()
        a, b = self.observe(), self.observe()
        self.assertNotEqual(a["observation_id"], b["observation_id"])
        self.assertEqual(self.lab.camera.calls, 2)
        self.assertEqual(a["units"]["gripper"], "percent")
        self.assertEqual(a["units"]["elbow_flex"], "degrees")
        self.assertFalse(a["image_sensor_timestamp_verified"])
        self.assertEqual(a["sensor_time_policy"], "host_read_interval")
        original = dict(a)
        hashed = original.pop("observation_sha256")
        self.assertEqual(digest(original), hashed)
        self.assertEqual(a["calibration_sha256"], digest(self.lab.calibration))

    def test_original_observation_id_hash_epoch_and_state_are_admitted(self):
        self.arm()
        observed = self.observe()
        for changes in (
            {"observation_id": "invented"},
            {"observation_sha256": "a" * 64},
            {"units": {"shoulder_pan": "radians"}},
            {"targets": {"shoulder_pan": math.nan}},
            {"duration_seconds": True},
            {"duration_seconds": 0.1},
        ):
            args = self.segment(observed)
            args.update(changes)
            with self.assertRaises(ValueError):
                self.command("execute", args)
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.command("execute", self.segment(observed), epoch=self.lab.agent.epoch + 1)
        self.assertIsNone(self.lab.agent.segment)

    def test_expiry_drift_limits_speed_and_acceleration_reject_without_goals(self):
        self.arm()
        observed = self.observe()
        before = len(self.buses[-1].writes)
        args = self.segment(observed)
        with patch(
            "agent_control.time.monotonic_ns", return_value=observed["valid_until_monotonic_ns"] + 1
        ):
            with self.assertRaisesRegex(ValueError, "expired"):
                self.command("execute", args)
        with self.assertRaisesRegex(ValueError, "speed or acceleration"):
            self.command(
                "execute",
                self.segment(
                    observed, {"shoulder_pan": observed["joints"]["shoulder_pan"] + 5}, 0.5
                ),
            )
        with self.assertRaisesRegex(ValueError, "delta"):
            self.command(
                "execute",
                self.segment(observed, {"shoulder_pan": observed["joints"]["shoulder_pan"] + 6}, 3),
            )
        self.buses[-1].raw["shoulder_pan"] += 30
        with self.assertRaisesRegex(ValueError, "drift"):
            self.command("execute", args)
        self.assertEqual(len(self.buses[-1].writes), before)

    def test_segment_interpolates_bounded_goals_and_reports_measured_completion(self):
        self.arm()
        observed = self.observe()
        clock = [self.lab.agent.clock()]
        self.lab.agent.clock = lambda: clock[0]
        arguments = self.segment(observed)
        request_id = uuid.uuid4().hex
        result = self.command("execute", arguments, ident=request_id)
        self.assertEqual(result["status"], "started")
        previous = dict(observed["joints"])
        for _ in range(150):
            clock[0] += 0.025
            self.lab.agent.lease_deadline = self.lab.agent.operator_deadline = clock[0] + 2
            self.lab._tick()
            for j in JOINTS:
                self.assertLessEqual(abs(self.lab.action[j] - previous[j]), 10 * 0.025 + 0.1)
            previous = dict(self.lab.action)
            receipt = self.lab.agent.receipt(request_id)
            if receipt["status"] == "completed":
                break
        self.assertEqual(receipt["status"], "completed")
        self.assertAlmostEqual(
            receipt["final_measured_state"]["shoulder_pan"],
            arguments["targets"]["shoulder_pan"],
            delta=1,
        )
        self.assertEqual(
            receipt["requested_targets"]["shoulder_pan"], arguments["targets"]["shoulder_pan"]
        )
        self.assertFalse(receipt["task_success_verified"])
        writes = list(self.buses[-1].writes)
        duplicate = self.command("execute", arguments, ident=request_id)
        self.assertEqual(duplicate, receipt)
        self.assertEqual(self.buses[-1].writes, writes)
        changed = copy.deepcopy(arguments)
        changed["duration_seconds"] = 3
        with self.assertRaisesRegex(ValueError, "another payload"):
            self.command("execute", changed, ident=request_id)

    def test_single_flight_and_stop_revoke_before_more_goals(self):
        self.arm()
        observed = self.observe()
        receipt = self.command("execute", self.segment(observed))
        with self.assertRaisesRegex(ValueError, "Another segment"):
            self.command("execute", self.segment(observed))
        with self.assertRaisesRegex(ValueError, "Wait for"):
            self.observe()
        self.lab.stop()
        before = list(self.buses[-1].writes)
        self.lab._tick()
        self.assertEqual(self.buses[-1].writes, before)
        self.assertEqual(self.lab.agent.receipt(receipt["command_id"])["status"], "stopped")
        self.assertFalse(self.lab.torque)
        self.assertIsNone(self.lab.agent.owner)

    def test_held_pose_is_measured_and_health_cannot_be_renewed_by_network_alone(self):
        self.arm()
        self.lab._tick()
        self.lab.agent.last_health = self.lab.agent.clock() - 0.6
        self.assertTrue(self.lab.agent.expired())
        with self.assertRaisesRegex(HardwareError, "health expired"):
            self.command("heartbeat", {"nonce": 1, "operator_present": True})
        self.lab._tick()
        self.assertFalse(self.lab.agent.expired())
        self.command("heartbeat", {"nonce": 2, "operator_present": True})
        self.buses[-1].raw["shoulder_pan"] += 200
        with self.assertRaisesRegex(HardwareError, "hold tracking"):
            self.lab._tick()

    def test_heartbeat_is_owner_epoch_nonce_bound_and_duplicate_never_renews(self):
        self.arm()
        args = {"nonce": 1, "operator_present": True}
        ident = uuid.uuid4().hex
        self.command("heartbeat", args, ident=ident)
        deadline = self.lab.agent.lease_deadline
        time.sleep(0.01)
        self.command("heartbeat", args, ident=ident)
        self.assertEqual(self.lab.agent.lease_deadline, deadline)
        with self.assertRaisesRegex(ValueError, "fresh owner nonce"):
            self.command("heartbeat", args)
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.command("heartbeat", {"nonce": 2, "operator_present": True}, owner="old_owner")
        self.lab.agent.lease_deadline = self.lab.agent.clock() - 0.1
        with self.assertRaisesRegex(HardwareError, "lease expired"):
            self.command("heartbeat", {"nonce": 2, "operator_present": True})

    def test_stalled_tracking_fault_never_claims_completion(self):
        self.arm()
        self.buses[-1].follow_goals = False
        observed = self.observe()
        self.command(
            "execute",
            self.segment(observed, {"shoulder_pan": observed["joints"]["shoulder_pan"] + 5}, 2),
        )
        self.lab.agent.segment.last_command["shoulder_pan"] += 4
        with self.assertRaisesRegex(HardwareError, "tracking"):
            self.lab._tick()
        self.assertNotEqual(self.lab.agent.receipt(self.lab.agent.active_id)["status"], "completed")

    def test_stop_does_not_wait_for_camera_or_disk_before_torque_off(self):
        self.arm()
        entered, release = threading.Event(), threading.Event()
        self.lab.camera.on_capture = lambda: (entered.set(), release.wait(2))
        errors = []

        def observe():
            try:
                self.observe()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=observe)
        thread.start()
        self.assertTrue(entered.wait(1))
        self.lab.stop()
        self.assertFalse(self.lab.torque)
        self.assertFalse(self.buses[-1].opened)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)

    def test_stop_during_encoder_admission_cannot_resurrect_a_segment(self):
        self.arm()
        observed = self.observe()
        entered, release = threading.Event(), threading.Event()
        bus = self.lab.bus
        original = bus.positions
        errors = []

        def delayed_positions():
            raw = original()
            entered.set()
            release.wait(2)
            return raw

        bus.positions = delayed_positions

        def execute():
            try:
                self.command("execute", self.segment(observed))
            except Exception as error:
                errors.append(error)

        command = threading.Thread(target=execute)
        command.start()
        self.assertTrue(entered.wait(1))
        generation = self.lab.generation
        stop = threading.Thread(target=self.lab.stop)
        stop.start()
        deadline = time.monotonic() + 1
        while generation == self.lab.generation and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertGreater(self.lab.generation, generation)
        release.set()
        command.join(2)
        stop.join(2)
        self.assertFalse(command.is_alive() or stop.is_alive())
        self.assertTrue(errors)
        self.assertIsNone(self.lab.agent.segment)
        self.assertIsNone(self.lab.agent.active_id)
        self.assertFalse(self.lab.torque)

    def test_late_receipt_write_cannot_overwrite_stop_evidence(self):
        self.arm()
        observed = self.observe()
        entered, release = threading.Event(), threading.Event()
        command_id = uuid.uuid4().hex

        def delayed_write(path, record):
            if record.get("status") == "started":
                entered.set()
                release.wait(2)
            durable_json(path, record)

        with patch("agent_control.durable_json", side_effect=delayed_write):
            command = threading.Thread(
                target=lambda: self.command("execute", self.segment(observed), ident=command_id)
            )
            command.start()
            self.assertTrue(entered.wait(1))
            stop = threading.Thread(target=self.lab.stop)
            stop.start()
            deadline = time.monotonic() + 1
            while self.lab.torque and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertFalse(self.lab.torque)
            release.set()
            command.join(2)
            stop.join(2)
            self.assertFalse(command.is_alive() or stop.is_alive())
        stored = self.lab.agent.receipts_dir / (command_id + ".json")
        self.assertEqual(json.loads(stored.read_text())["status"], "stopped")
        self.assertEqual(self.lab.agent.receipt(command_id)["status"], "stopped")

    def test_parallel_duplicate_identity_is_claimed_once_before_usb_admission(self):
        self.arm()
        observed = self.observe()
        entered, release = threading.Event(), threading.Event()
        bus = self.lab.bus
        original = bus.positions
        command_id = uuid.uuid4().hex
        arguments = self.segment(observed)

        def delayed_positions():
            raw = original()
            entered.set()
            release.wait(2)
            return raw

        bus.positions = delayed_positions
        command = threading.Thread(
            target=lambda: self.command("execute", arguments, ident=command_id)
        )
        command.start()
        self.assertTrue(entered.wait(1))
        duplicate = self.command("execute", arguments, ident=command_id)
        self.assertEqual(duplicate["status"], "accepted")
        with self.assertRaisesRegex(ValueError, "another payload"):
            self.command("execute", dict(arguments, duration_seconds=3), ident=command_id)
        release.set()
        command.join(2)
        self.assertFalse(command.is_alive())
        self.assertEqual(self.lab.agent.command_revision, 1)
        self.assertEqual(self.lab.agent.receipt(command_id)["status"], "started")

    def test_stop_during_completion_write_cannot_publish_late_completed_receipt(self):
        self.arm()
        observed = self.observe()
        receipt = self.command(
            "execute",
            self.segment(
                observed, targets={"shoulder_pan": observed["joints"]["shoulder_pan"] + 0.1}
            ),
        )
        self.lab.agent.segment.progress_time = self.lab.agent.segment.duration
        entered, release = threading.Event(), threading.Event()
        original = self.lab.controller.command_raw

        def blocked(targets):
            sent = original(targets)
            entered.set()
            release.wait(2)
            return sent

        self.lab.controller.command_raw = blocked
        tick = threading.Thread(target=self.lab._tick)
        tick.start()
        self.assertTrue(entered.wait(1))
        generation = self.lab.generation
        stop = threading.Thread(target=self.lab.stop)
        stop.start()
        deadline = time.monotonic() + 1
        while generation == self.lab.generation and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertGreater(self.lab.generation, generation)
        release.set()
        tick.join(2)
        stop.join(2)
        self.assertFalse(tick.is_alive() or stop.is_alive())
        final = self.lab.agent.receipt(receipt["command_id"])
        self.assertEqual(final["status"], "stopped")
        self.assertFalse(final["measured_completion_verified"])
        self.assertFalse(self.lab.torque)

    def test_restart_keeps_identity_unknown_and_fences_old_acquire_generation(self):
        self.arm()
        observed = self.observe()
        receipt = self.command("execute", self.segment(observed))
        old_epoch = self.lab.agent.epoch
        recreated = AgentExecutor(self.lab)
        self.assertGreater(recreated.epoch, old_epoch)
        self.assertEqual(recreated.receipt(receipt["command_id"])["status"], "unknown")
        self.assertIsNone(recreated.owner)
        stored = self.lab.agent.root / "authority.json"
        self.assertEqual(stored.stat().st_mode & 0o777, 0o600)

    def test_old_manual_heartbeat_cannot_renew_agent_or_new_manual_generation(self):
        self.arm()
        old = self.lab.generation
        deadline = self.lab.agent.lease_deadline
        with self.assertRaisesRegex(ValueError, "cannot renew AI"):
            self.lab.control_heartbeat({"expected_generation": old})
        self.assertEqual(self.lab.agent.lease_deadline, deadline)
        self.lab.stop()
        self.start()
        with self.assertRaisesRegex(ValueError, "generation changed"):
            self.lab.control_heartbeat({"expected_generation": old})
        with self.assertRaises(ValueError):
            self.lab.control_heartbeat({"expected_generation": True})
        self.lab.control_heartbeat({"expected_generation": self.lab.generation})

    def test_stale_cleanup_stop_cannot_release_a_new_controller(self):
        self.start()
        old = self.lab.generation
        self.lab.stop()
        self.start()
        generation = self.lab.generation
        before = list(self.buses[-1].writes)
        with self.assertRaisesRegex(ValueError, "stale cleanup"):
            self.lab.stop(expected_generation=old)
        self.assertTrue(self.lab.torque)
        self.assertEqual(self.lab.generation, generation)
        self.assertEqual(self.buses[-1].writes, before)
        with self.assertRaises(ValueError):
            self.lab.stop(expected_generation=True)
        self.lab.stop(expected_generation=generation)
        self.assertFalse(self.lab.torque)
        # The unconditional emergency Stop remains available after any takeover.
        self.start()
        self.lab.stop()
        self.assertFalse(self.lab.torque)

    def test_manual_cleanup_fence_rejects_same_generation_agent_takeover(self):
        self.arm()
        generation = self.lab.generation
        with self.assertRaisesRegex(ValueError, "mode changed"):
            self.lab.stop(expected_generation=generation, expected_mode="manual")
        self.assertTrue(self.lab.torque)
        self.assertEqual(self.lab.mode, "agent")
        self.lab.stop()
        self.start()
        self.lab.stop(expected_generation=self.lab.generation, expected_mode="manual")
        self.assertFalse(self.lab.torque)

    def test_late_hardware_loop_error_cannot_stop_a_new_manual_controller(self):
        self.arm()
        entered, release = threading.Event(), threading.Event()

        def delayed_tick():
            entered.set()
            release.wait(2)
            self.lab.finished.set()
            raise HardwareError("Synthetic delayed old-controller error")

        self.lab._tick = delayed_tick
        worker = threading.Thread(target=self.lab._loop)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.lab.stop()
        self.start()
        generation = self.lab.generation
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.lab.generation, generation)
        self.assertEqual(self.lab.mode, "manual")
        self.assertTrue(self.lab.torque)

    def test_late_watchdog_expiry_cannot_stop_a_new_manual_controller(self):
        self.arm()
        entered, release = threading.Event(), threading.Event()

        def delayed_expiry():
            entered.set()
            release.wait(2)
            self.lab.finished.set()
            return True

        self.lab.agent.expired = delayed_expiry
        worker = threading.Thread(target=self.lab._watchdog)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.lab.stop()
        self.start()
        generation = self.lab.generation
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.lab.generation, generation)
        self.assertEqual(self.lab.mode, "manual")
        self.assertTrue(self.lab.torque)

    def test_local_watchdog_torque_off_does_not_depend_on_model_or_browser(self):
        self.arm()
        self.lab.agent.lease_deadline = self.lab.agent.clock() - 0.1
        thread = threading.Thread(target=self.lab._watchdog)
        thread.start()
        deadline = time.monotonic() + 1
        while self.lab.mode != "idle" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.lab.finished.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.lab.torque)
        self.assertIsNone(self.lab.agent.owner)
        self.assertIn("lease expired", self.lab.error)

    def test_wire_manual_heartbeat_cannot_renew_agent_and_null_cleanup_is_invalid(self):
        self.arm()
        token = "x" * 48
        server = create_server(self.lab, token, port=0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            deadline = self.lab.agent.lease_deadline
            for path, payload in (
                ("/v1/control/heartbeat", {}),
                ("/v1/stop", {"expected_generation": None}),
                ("/v1/stop", {"expected_mode": None}),
                ("/v1/stop", {"expected_mode": "agent"}),
            ):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request(
                    "POST",
                    path,
                    json.dumps(payload),
                    {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 422)
                response.read()
                connection.close()
            self.assertEqual(self.lab.agent.lease_deadline, deadline)
            self.assertTrue(self.lab.torque)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_duplicate_json_and_nonfinite_numbers_are_rejected_before_dispatch(self):
        for payload in (
            '{"authority_epoch":1,"authority_epoch":2}',
            '{"delta":NaN}',
            '{"delta":Infinity}',
        ):
            with self.assertRaises(ValueError):
                strict_json(payload)

    def test_long_tick_gap_cannot_create_a_large_catchup_step(self):
        initial = dict.fromkeys(JOINTS, 0.0)
        targets = dict(initial, shoulder_pan=5.0)
        motion = Segment(initial, targets, 2, 0)
        actual, arrived = motion.step(initial, 1)
        self.assertLess(actual["shoulder_pan"], 0.1)
        self.assertFalse(arrived)


if __name__ == "__main__":
    unittest.main()
