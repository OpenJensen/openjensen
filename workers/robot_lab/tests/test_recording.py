"""Recording orchestration without camera, serial, or ML dependencies."""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_robot_lab as safety
from recording import CaptureRate, Collection
from server import RequestError


class RateTests(unittest.TestCase):
    def test_rate_is_limited_by_distinct_camera_frames(self):
        rate = CaptureRate(0, True)
        for index in range(161):
            now = index / 40
            rate.sample(now, (index // 4) / 10)
        self.assertEqual(rate.select(4), 8)

    def test_repeated_cached_image_does_not_establish_camera_fps(self):
        rate = CaptureRate(0, True)
        for index in range(161):
            rate.sample(index / 40, 0)
        self.assertIsNone(rate.select(4))

    def test_storage_is_a_separate_cadence_limit(self):
        rate = CaptureRate(0, False)
        rate.io_costs = [0.2] * 6
        for index in range(161):
            rate.sample(index / 40)
        self.assertEqual(rate.select(4), 4)

    def test_invalid_collection_fields_and_bounds(self):
        for config in (
            {"num_episodes": True},
            {"episode_time_s": 0},
            {"reset_time_s": -1},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                Collection(config)


class FakeWriter:
    def __init__(self, root, dataset_id, task, fps, image_shape):
        self.id, self.root, self.fps = dataset_id, root / dataset_id, fps
        self.root.mkdir()
        self.frames = self.total_frames = 0
        self.episodes = []
        self.episode_active = self.finalized = False
        self.actual_fps = None
        self.closed = False
        self.metadata_rates = [fps]

    def start_episode(self):
        if self.frames or self.episode_active:
            raise ValueError("pending episode")
        self.episode_active = True

    def add_frame(self, positions, action, image, captured_at):
        self.frames += 1

    def save_episode(self):
        if self.frames < 2:
            raise ValueError("two frames required")
        self.episodes.append({"length": self.frames})
        self.total_frames += self.frames
        self.discard_episode()

    def discard_episode(self):
        self.frames = 0
        self.episode_active = False

    def finalize(self):
        if self.frames:
            raise ValueError("pending")
        self.finalized = True

    def write_metadata(self):
        self.metadata_rates.append(self.fps)

    def close(self):
        self.closed = True


class CollectionTests(unittest.TestCase):
    setUp = safety.SafetyTests.setUp
    tearDown = safety.SafetyTests.tearDown
    start = safety.SafetyTests.start

    def create(self, *, fps=5, count=2, duration=2, countdown=1):
        self.start(mode="handguide")
        self.clock = 0.0
        patcher = patch("server.time.monotonic", side_effect=lambda: self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        with patch("server.DatasetWriter", FakeWriter):
            self.lab.recording_start(
                {
                    "dataset_id": "demo",
                    "task": "Pick up the pen",
                    "fps": fps,
                    "camera_index": None,
                    "collection": {
                        "episode_time_s": duration,
                        "reset_time_s": 2,
                        "num_episodes": count,
                        "countdown_s": countdown,
                    },
                }
            )
        return self.lab.writer

    def tick(self, now):
        self.clock = now
        self.lab._tick()

    def wait_io(self):
        if self.lab.recording_thread:
            self.lab.recording_thread.join(2)
            self.assertFalse(self.lab.recording_thread.is_alive())

    def test_timed_episodes_have_exact_frame_counts_and_reset_is_not_recorded(self):
        writer = self.create(count=2)
        self.lab.collection_action("start")
        self.tick(0.5)
        self.assertEqual(writer.frames, 0)
        for index in range(10):
            self.tick(1 + index * 0.2 + 1e-8)
        self.tick(3)
        self.tick(3.025)
        self.wait_io()
        self.assertEqual(writer.episodes, [{"length": 10}])
        self.assertEqual(self.lab.collection.phase, "reset")
        self.tick(4)
        self.assertEqual(writer.frames, 0)
        self.lab.collection_action("next")
        for index in range(10):
            self.tick(5 + index * 0.2 + 1e-8)
        self.tick(7)
        self.tick(7.025)
        self.wait_io()
        self.assertEqual(writer.episodes, [{"length": 10}, {"length": 10}])
        self.assertEqual(self.lab.collection.phase, "complete")
        self.assertEqual(self.lab.mode, "handguide")

    def test_return_to_rest_cannot_change_control_during_countdown(self):
        self.create()
        self.lab.collection_action("start")
        with self.assertRaisesRegex(RequestError, "Pause timed collection"):
            self.lab.rest_position({"confirm": "RETURN TO REST"})
        self.assertEqual(self.lab.collection.phase, "countdown")

    def test_stop_during_reset_revokes_automatic_progression(self):
        writer = self.create()
        self.lab.collection.transition("reset", "Reset", 0, 1)
        self.lab.stop()
        self.tick(20)
        self.assertEqual(self.lab.collection.phase, "paused")
        self.assertFalse(writer.episode_active)

    def test_stop_during_save_preserves_episode_without_restarting(self):
        writer = self.create()
        entered, release = threading.Event(), threading.Event()
        original = writer.save_episode

        def slow_save():
            entered.set()
            release.wait(2)
            original()

        writer.save_episode = slow_save
        self.lab.collection_action("start")
        self.tick(1)
        writer.frames = 10
        self.tick(3)
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(RequestError, "being saved"):
            self.lab.recording_action("discard")
        self.lab.stop()
        release.set()
        self.wait_io()
        self.assertEqual(len(writer.episodes), 1)
        self.assertEqual(self.lab.collection.phase, "paused")
        self.assertEqual(self.lab.collection.completed, 1)
        self.assertFalse(writer.episode_active)

    def test_pause_requires_disposition_and_rerecord_only_clears_pending(self):
        writer = self.create()
        writer.episodes.append({"length": 10})
        self.lab.collection_action("start")
        self.tick(1)
        self.tick(1.2)
        self.lab.collection_action("pause")
        with self.assertRaisesRegex(RequestError, "Save or discard"):
            self.lab.collection_action("resume")
        self.lab.collection_action("rerecord")
        self.wait_io()
        self.assertEqual(writer.frames, 0)
        self.assertEqual(writer.episodes, [{"length": 10}])
        self.assertEqual(self.lab.collection.phase, "countdown")

    def test_manual_save_of_partial_episode_does_not_complete_timed_goal(self):
        writer = self.create()
        self.lab.collection_action("start")
        self.tick(1)
        self.tick(1.2)
        self.lab.recording_action("save")
        self.assertEqual(len(writer.episodes), 1)
        self.assertEqual(self.lab.collection.completed, 0)
        self.assertEqual(self.lab.collection.phase, "paused")

    def test_save_failure_preserves_pending_and_pauses(self):
        writer = self.create()
        self.lab.collection_action("start")
        self.tick(1)
        writer.frames = 10
        with patch.object(writer, "save_episode", side_effect=OSError("disk full")):
            self.tick(3)
            self.wait_io()
        self.assertEqual(writer.frames, 10)
        self.assertEqual(self.lab.collection.phase, "paused")
        self.assertIn("disk full", self.lab.recording_error)
        self.assertEqual(self.lab.collection.completed, 0)

    def test_auto_rate_is_measured_and_frozen_before_first_frame(self):
        writer = self.create(fps=None)
        self.lab.collection_action("start")
        for index in range(121):
            self.tick(index / 40)
        self.wait_io()
        self.assertEqual(writer.fps, 33)  # 15% reserve; binary float conservative floor.
        self.assertEqual(writer.frames, 0)
        self.assertEqual(self.lab.recording_rate_status, "ready")
        selected = writer.fps
        self.tick(4)
        self.tick(4.03)
        self.assertGreater(writer.frames, 0)
        self.assertEqual(writer.fps, selected)

    def test_single_frame_timed_configuration_fails_before_capture(self):
        writer = self.create(fps=1, duration=1, countdown=0)
        self.lab.collection_action("start")
        self.tick(0)
        self.assertEqual(writer.frames, 0)
        self.assertEqual(self.lab.collection.phase, "paused")
        self.assertIn("at least two frames", self.lab.recording_error)

    def test_rate_metadata_failure_reports_error_instead_of_measuring_forever(self):
        writer = self.create(fps=None)
        with patch.object(writer, "write_metadata", side_effect=OSError("disk full")):
            for index in range(121):
                self.tick(index / 40)
            self.wait_io()
        self.assertEqual(self.lab.recording_rate_status, "error")
        self.assertIn("disk full", self.lab.recording_rate_message)
        self.assertFalse(writer.episode_active)
        with self.assertRaisesRegex(RequestError, "measurement failed"):
            self.lab.collection_action("start")
        with self.assertRaisesRegex(RequestError, "automatic frame rate"):
            self.lab.recording_action("start")

    def test_camera_preparation_does_not_hold_motor_lock_and_stop_revokes_attachment(self):
        from types import SimpleNamespace

        self.start(mode="handguide")
        entered, release = threading.Event(), threading.Event()
        errors = []

        def read(index):
            entered.set()
            release.wait(2)
            return SimpleNamespace(shape=(2, 2, 3))

        self.lab.camera.read = read

        def prepare():
            try:
                self.lab.recording_start(
                    {"dataset_id": "camera", "task": "Reach", "fps": 5, "camera_index": 0}
                )
            except Exception as error:
                errors.append(error)

        with patch("server.DatasetWriter", FakeWriter):
            thread = threading.Thread(target=prepare)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertTrue(self.lab.lock.acquire(timeout=0.1))
            self.lab.lock.release()
            self.lab.stop()
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(self.lab.writer)
        self.assertFalse(self.lab.recording_io)
        self.assertEqual(len(errors), 1)
        self.assertIn("Control stopped", str(errors[0]))

    def test_finalizing_writer_cannot_be_replaced_until_io_completes(self):
        writer = self.create()
        entered, release = threading.Event(), threading.Event()

        def finalize():
            writer.finalized = True
            entered.set()
            release.wait(2)

        writer.finalize = finalize
        thread = threading.Thread(target=lambda: self.lab.recording_action("finish"))
        thread.start()
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaisesRegex(RequestError, "files are busy"):
                self.lab.recording_start(
                    {"dataset_id": "replacement", "task": "Reach", "fps": 5, "camera_index": None}
                )
        finally:
            release.set()
            thread.join(2)
        self.assertIs(self.lab.writer, writer)
        self.assertFalse(self.lab.recording_io)

    def test_cached_camera_frame_is_not_recorded_twice(self):
        writer = self.create()
        self.lab.collection = None
        self.lab.camera_index = 0

        class Frame:
            def __getitem__(self, key):
                return self

            def copy(self):
                return self

        frame = Frame()
        captured = [0.0]
        self.lab.camera.read = lambda index: frame
        self.lab.camera.read_latest_timed = lambda index: (frame, captured[0])
        self.lab.recording_action("start")
        self.tick(0)
        self.tick(0.2)
        self.assertEqual(writer.frames, 1)
        captured[0] = 0.3
        self.tick(0.4)
        self.assertEqual(writer.frames, 2)

    def test_non_divisor_fps_retains_timing_phase(self):
        writer = self.create(fps=17, duration=2, countdown=0)
        self.lab.collection_action("start")
        for index in range(81):
            self.tick(index / 40 + 1e-8)
        self.wait_io()
        self.assertEqual(writer.episodes, [{"length": 34}])


if __name__ == "__main__":
    unittest.main()
