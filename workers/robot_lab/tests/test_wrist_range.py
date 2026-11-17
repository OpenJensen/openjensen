"""A follower wrist range recorded by hand holds every later command inside it."""

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_robot_lab as fixtures
from hardware import JOINTS, denormalized
from server import RequestError
from wrist_range import MARGIN, RangeRecorder, WristRanges, clamp


class RecorderTests(unittest.TestCase):
    def sweep(self, angles):
        recorder = RangeRecorder(angles[0])
        for angle in angles[1:]:
            recorder.sample(angle)
        return recorder

    def test_records_both_ends_a_margin_inside(self):
        recorder = self.sweep([0, -40, -120, -60, 0, 50, 90, 30])
        self.assertEqual(recorder.result(), (-120 + MARGIN, 90 - MARGIN))

    def test_follows_the_wrist_across_the_half_turn(self):
        """Turned past +180, the reading jumps to -180; the recorder keeps counting."""
        recorder = self.sweep([0, 90, 170, -170, -150, 100, 0, -100])
        snapshot = recorder.snapshot()
        self.assertEqual((snapshot["low"], snapshot["high"]), (-100, 210))
        # The part beyond +180 cannot be commanded without meaning the other side.
        self.assertEqual(recorder.result(), (-100 + MARGIN, 180 - MARGIN))

    def test_a_short_sweep_or_one_missing_the_working_pose_is_refused(self):
        with self.assertRaisesRegex(ValueError, "both of its ends"):
            self.sweep([0, 10, -10]).result()
        with self.assertRaisesRegex(ValueError, "working position"):
            self.sweep([40, 80, 160]).result()

    def test_clamp_holds_at_the_end_the_wrist_reached(self):
        self.assertEqual(clamp(2000, 1900, 1000, 3000), 2000)
        self.assertEqual(clamp(3500, 2900, 1000, 3000), 3000)
        # A leader past the half turn asks for the far side: stay at the end already reached.
        self.assertEqual(clamp(50, 3000, 1000, 3000), 3000)
        self.assertEqual(clamp(4000, 1000, 1000, 3000), 1000)
        self.assertEqual(clamp(4000, None, 1000, 3000), 3000)


class WristRangeTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def act(self, action, confirm=None):
        word = {"start": "RECORD", "save": "SAVE", "cancel": "CANCEL", "clear": "CLEAR"}[action]
        return self.lab.wrist_range_action({"confirm": confirm or f"{word} WRIST RANGE"}, action)

    def turn(self, bus, angles):
        for angle in angles:
            bus.raw["wrist_roll"] = denormalized("wrist_roll", angle, self.lab.calibration)
            self.lab.heartbeat = time.monotonic()
            self.lab._tick()

    def record(self, low=-120, high=90):
        self.start(mode="handguide")
        bus = self.buses[-1]
        self.turn(bus, [0])
        self.act("start")
        before = [w for w in bus.writes if w[0] == "Goal_Position"]
        self.turn(bus, [0, low / 2, low, 0, high / 2, high, 0])
        # Recording only reads: with torque off nothing is commanded.
        self.assertEqual([w for w in bus.writes if w[0] == "Goal_Position"], before)
        return self.act("save")

    def test_recorded_range_holds_the_leader_inside_it(self):
        saved = self.record()["session"]["wrist_range"]["saved"]
        self.assertAlmostEqual(saved["min"], -120 + MARGIN, delta=0.2)
        self.assertAlmostEqual(saved["max"], 90 - MARGIN, delta=0.2)
        self.lab.stop()
        self.start()
        bus = self.buses[-1]
        high = denormalized("wrist_roll", saved["max"], self.lab.calibration)
        low = denormalized("wrist_roll", saved["min"], self.lab.calibration)
        self.lab.mode = "leader"
        self.lab.leader = fixtures.FakeBus()
        self.lab.leader_calibration = self.lab.calibration
        for angle, expected in ((40, None), (150, high), (-170, high), (-150, high), (0, None)):
            self.lab.leader.raw = dict.fromkeys(JOINTS, 2050)
            leader_raw = denormalized("wrist_roll", angle, self.lab.calibration)
            self.lab.leader.raw["wrist_roll"] = leader_raw
            self.lab.heartbeat = time.monotonic()
            self.lab._tick()
            sent = self.lab.commanded_raw["wrist_roll"]
            if expected is None:
                self.assertTrue(low <= sent <= high)
                self.assertAlmostEqual(self.lab.action["wrist_roll"], angle, delta=0.5)
            else:
                # Past the end, and past the half turn, the wrist stays at the end it reached.
                self.assertEqual(sent, expected)
            self.assertEqual(bus.raw["wrist_roll"], sent)

    def test_manual_limits_show_the_range_and_clear_removes_it(self):
        self.record()
        self.lab.stop()
        status = self.start()
        limits = status["session"]["joint_limits"]["wrist_roll"]
        self.assertAlmostEqual(limits["min"], -120 + MARGIN, delta=0.2)
        self.assertAlmostEqual(limits["max"], 90 - MARGIN, delta=0.2)
        cleared = self.act("clear")
        self.assertIsNone(cleared["session"]["wrist_range"]["saved"])
        self.assertEqual(self.lab.controller.bridge.safe_ranges, {})
        self.lab.stop()
        self.assertIsNone(self.start()["session"]["wrist_range"]["saved"])

    def test_a_new_calibration_drops_the_old_range(self):
        self.record()
        changed = json.loads(json.dumps(self.lab.calibration))
        changed["wrist_roll"]["homing_offset"] = 300
        self.assertIsNone(WristRanges(self.lab.data_dir).load("test", changed))
        self.assertIsNotNone(WristRanges(self.lab.data_dir).load("test", self.lab.calibration))

    def test_recording_needs_hand_guiding_a_sweep_and_confirmation(self):
        self.start()
        with self.assertRaisesRegex(RequestError, "hand guiding"):
            self.act("start")
        with self.assertRaisesRegex(RequestError, "Record the wrist range first"):
            self.act("save")
        with self.assertRaises(RequestError):
            self.act("clear", confirm="yes")
        self.lab.stop()
        self.start(mode="handguide")
        self.act("start")
        self.turn(self.buses[-1], [0, 5, -5])
        with self.assertRaisesRegex(RequestError, "both of its ends"):
            self.act("save")
        self.act("cancel")
        self.assertIsNone(self.lab.wrist_recorder)
        self.act("start")
        self.lab.stop()
        self.assertIsNone(self.lab.wrist_recorder)


def path(*stops, step=10):
    """Angles a wrist passes through when turned by hand from stop to stop."""
    angles = [stops[0]]
    for target in stops[1:]:
        while angles[-1] != target:
            delta = target - angles[-1]
            angles.append(angles[-1] + max(-step, min(step, delta)))
    return angles


class CalibrationSweepTests(unittest.TestCase):
    """Calibration's own range sweep records the follower's wrist range automatically."""

    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown

    def calibrate(self, angles, role="follower"):
        self.lab.calibration_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "test",
                "role": role,
                "confirm": "START CALIBRATION",
            }
        )
        self.lab.calibration_center({})
        bus = self.lab.bus
        for angle in angles:
            # Raw ticks as the full-turn wrist reads them, wrapping past the half turn.
            bus.raw["wrist_roll"] = round(angle * 4095 / 360 + 4095 / 2) % 4096
            self.lab._tick()
        live = self.lab.status()["calibration"]["wrist_sweep"]
        self.lab.calibration_state["mins"] = dict.fromkeys(JOINTS, 100)
        self.lab.calibration_state["maxes"] = dict.fromkeys(JOINTS, 4000)
        result = self.lab.calibration_finish({"confirm": "SAVE CALIBRATION"})
        saved = json.loads(self.lab.calibration_path("test", role).read_text())
        return result, saved, live

    def test_sweeping_the_wrist_during_calibration_saves_its_range(self):
        result, saved, live = self.calibrate(path(0, -120, 0, 90, 0))
        self.assertEqual((live["low"], live["high"]), (-120, 90))
        wrist = result["calibration"]["wrist_range"]
        self.assertAlmostEqual(wrist["min"], -120 + MARGIN, delta=0.2)
        self.assertAlmostEqual(wrist["max"], 90 - MARGIN, delta=0.2)
        # The calibration itself keeps wrist roll a full turn, as LeRobot expects.
        self.assertEqual(
            (saved["wrist_roll"]["range_min"], saved["wrist_roll"]["range_max"]), (0, 4095)
        )
        # The next session with this calibration enforces the range.
        persisted = fixtures.FakeBus()
        persisted.calibration = saved
        self.lab.bus_factory = lambda *args: persisted
        status = self.lab.control_start(
            {
                "port": "/dev/cu.fake",
                "robot_id": "test",
                "mode": "manual",
                "confirm": "ENABLE MOTION",
            }
        )
        self.assertEqual(status["session"]["wrist_range"]["saved"], wrist)
        self.assertIn("wrist_roll", self.lab.controller.bridge.safe_ranges)

    def test_a_sweep_across_the_half_turn_stops_short_of_it(self):
        result, _, _ = self.calibrate(path(0, 200, 0, -60, 0))
        wrist = result["calibration"]["wrist_range"]
        self.assertAlmostEqual(wrist["min"], -60 + MARGIN, delta=0.2)
        self.assertEqual(wrist["max"], 180 - MARGIN)

    def test_no_range_without_a_wrist_sweep_or_for_a_leader(self):
        result, saved, _ = self.calibrate(path(0, 10, 0))
        self.assertIsNone(result["calibration"]["wrist_range"])
        self.assertIsNone(WristRanges(self.lab.data_dir).load("test", saved))
        result, saved, _ = self.calibrate(path(0, -120, 0, 90, 0), role="leader")
        self.assertIsNone(result["calibration"]["wrist_range"])
        self.assertIsNone(WristRanges(self.lab.data_dir).load("test", saved))


if __name__ == "__main__":
    unittest.main()
