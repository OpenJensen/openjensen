"""Background frame freshness, bounded writes and independent motor control."""

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_robot_lab as fixtures
from camera import CameraPool
from image_writer import AsyncImageWriter


class Image:
    def __init__(self, value):
        self.value = value

    def copy(self):
        return Image(self.value)


class ImageWriterTests(unittest.TestCase):
    def test_slow_write_is_off_thread_bounded_and_copies_the_camera_buffer(self):
        entered, release = threading.Event(), threading.Event()
        saved = []

        def write(image, path):
            entered.set()
            release.wait(2)
            saved.append((image.value, path))

        writer = AsyncImageWriter(capacity=1, write=write)
        first = Image(1)
        try:
            writer.save_image(first, Path("first.png"))
            self.assertTrue(entered.wait(1))
            first.value = 99
            writer.save_image(Image(2), Path("second.png"))
            started = time.monotonic()
            with self.assertRaisesRegex(ValueError, "cannot keep up"):
                writer.save_image(Image(3), Path("third.png"))
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(writer.pending, 2)
            with self.assertRaisesRegex(ValueError, "still pending"):
                writer.wait_until_done(timeout=0.01)
            release.set()
            writer.wait_until_done()
            self.assertEqual(saved, [(1, Path("first.png")), (2, Path("second.png"))])
        finally:
            release.set()
            writer.stop(ignore_error=True)
        self.assertFalse(writer.thread.is_alive())
        with self.assertRaisesRegex(ValueError, "closed"):
            writer.save_image(Image(4), Path("four.png"))

    def test_failed_buffers_can_be_retried_without_losing_queued_images(self):
        entered, release = threading.Event(), threading.Event()
        first = [True]
        saved = []

        def write(image, path):
            if image.value == 1 and first[0]:
                entered.set()
                release.wait(2)
                first[0] = False
                raise OSError("temporary storage error")
            saved.append(image.value)

        writer = AsyncImageWriter(write=write)
        try:
            writer.save_image(Image(1), Path("one.png"))
            self.assertTrue(entered.wait(1))
            writer.save_image(Image(2), Path("two.png"))
            release.set()
            writer.wait_until_done(ignore_error=True)
            self.assertEqual(saved, [2])
            self.assertEqual(len(writer.failed), 1)
            writer.retry_failed()
            self.assertEqual(sorted(saved), [1, 2])
            self.assertFalse(writer.failed)
            self.assertIsNone(writer.error)
        finally:
            release.set()
            writer.stop(ignore_error=True)

    def test_background_disk_error_is_visible_and_discard_can_recover(self):
        write = Mock(side_effect=OSError("disk full"))
        writer = AsyncImageWriter(write=write)
        try:
            writer.save_image(Image(1), Path("one.png"))
            with self.assertRaisesRegex(ValueError, "disk full"):
                writer.wait_until_done()
            with self.assertRaisesRegex(ValueError, "disk full"):
                writer.save_image(Image(2), Path("two.png"))
            writer.wait_until_done(ignore_error=True)
            writer.clear_error()
            write.side_effect = None
            writer.save_image(Image(3), Path("three.png"))
            writer.wait_until_done()
            self.assertEqual(write.call_count, 2)
        finally:
            writer.stop(ignore_error=True)


class LatestCameraTests(unittest.TestCase):
    def fake_cv(self, capture):
        return SimpleNamespace(
            VideoCapture=lambda *a: capture,
            CAP_V4L2=1,
            CAP_AVFOUNDATION=2,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_BUFFERSIZE=5,
        )

    def test_latest_peek_does_not_wait_for_blocked_camera_and_rejects_stale_frames(self):
        entered, release = threading.Event(), threading.Event()
        first = object()
        reads = []

        def read():
            reads.append(1)
            if len(reads) > 1:
                entered.set()
                release.wait(2)
            return True, first

        capture = SimpleNamespace(
            isOpened=lambda: True, set=lambda *a: None, read=read, release=lambda: None
        )
        pool = CameraPool()
        with patch.dict(sys.modules, cv2=self.fake_cv(capture)):
            try:
                self.assertIs(pool.read(), first)
                pool.start_reader()
                self.assertTrue(entered.wait(1))
                started = time.monotonic()
                self.assertIs(pool.read_latest(), first)
                self.assertLess(time.monotonic() - started, 0.05)
                with pool.frame_lock:
                    pool.frame_times[0] = time.monotonic() - 1
                with self.assertRaisesRegex(TimeoutError, "stale"):
                    pool.read_latest()
            finally:
                release.set()
                pool.close()
        self.assertFalse(pool.handles or pool.readers or pool.last_frames)

    def test_repeated_capture_failures_are_bounded_and_visible(self):
        read = Mock(return_value=(False, None))
        released = []
        capture = SimpleNamespace(
            isOpened=lambda: True,
            set=lambda *a: None,
            read=read,
            release=lambda: released.append(1),
        )
        pool = CameraPool()
        with patch.dict(sys.modules, cv2=self.fake_cv(capture)):
            pool.start_reader()
            thread = pool.readers[0][0]
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(read.call_count, 10)
            with self.assertRaisesRegex(RuntimeError, "Camera capture failed"):
                pool.read_latest()
            pool.close()
        self.assertEqual(released, [1])

    def test_close_during_capture_does_not_reopen_or_publish_an_old_frame(self):
        entered, release = threading.Event(), threading.Event()
        opened = []

        def read():
            entered.set()
            release.wait(2)
            return True, object()

        capture = SimpleNamespace(
            isOpened=lambda: True, set=lambda *a: None, read=read, release=lambda: None
        )
        cv = self.fake_cv(capture)
        cv.VideoCapture = lambda *args: opened.append(1) or capture
        pool = CameraPool()
        with patch.dict(sys.modules, cv2=cv):
            pool.start_reader()
            self.assertTrue(entered.wait(1))
            close = threading.Thread(target=pool.close)
            close.start()
            deadline = time.monotonic() + 1
            while pool.readers and time.monotonic() < deadline:
                time.sleep(0.001)
            release.set()
            close.join(timeout=2)
            self.assertFalse(close.is_alive())
        self.assertEqual(opened, [1])
        self.assertFalse(pool.handles or pool.last_frames)

    def test_ai_timed_capture_still_uses_a_fresh_read_and_reports_host_bounds(self):
        pool = CameraPool()
        image = object()
        cv = SimpleNamespace(
            imencode=lambda *args: (True, SimpleNamespace(tobytes=lambda: b"jpeg")),
            IMWRITE_JPEG_QUALITY=1,
        )
        with (
            patch.dict(sys.modules, cv2=cv),
            patch.object(pool, "read", return_value=image) as read,
        ):
            result = pool.capture_timed(2)
        read.assert_called_once_with(2)
        self.assertFalse(result["image_sensor_timestamp_verified"])
        self.assertLessEqual(result["read_start_monotonic_ns"], result["read_end_monotonic_ns"])


class RecordingIsolationTests(unittest.TestCase):
    setUp = fixtures.SafetyTests.setUp
    tearDown = fixtures.SafetyTests.tearDown
    start = fixtures.SafetyTests.start

    def writer(self):
        return SimpleNamespace(
            episode_active=True,
            fps=10,
            finalized=False,
            actual_fps=None,
            id="test",
            frames=3,
            episodes=[],
            add_frame=Mock(),
        )

    def test_capture_failure_pauses_recording_and_keeps_control_until_explicit_stop(self):
        self.start()
        writer = self.writer()
        self.lab.writer = writer
        writer.add_frame.side_effect = OSError("disk full")
        self.lab._tick()
        self.assertFalse(writer.episode_active)
        self.assertIn("disk full", self.lab.status()["recording"]["error"])
        self.assertEqual(self.lab.mode, "manual")
        self.assertTrue(self.lab.torque)
        self.assertIsNone(self.lab.diagnostics.snapshot()["last_stop"])
        self.lab._tick()
        writer.add_frame.assert_called_once()
        self.lab.stop()
        self.assertFalse(self.lab.torque or self.buses[-1].opened)

    def test_stop_does_not_wait_for_a_blocked_image_writer(self):
        entered, release = threading.Event(), threading.Event()

        def write(image, path):
            entered.set()
            release.wait(2)

        images = AsyncImageWriter(write=write)
        try:
            self.start()
            self.lab.writer = self.writer()
            images.save_image(Image(1), Path("blocked.png"))
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            self.lab.stop()
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertFalse(self.lab.torque or self.buses[-1].opened)
        finally:
            release.set()
            images.stop(ignore_error=True)

    def test_control_cycle_accounts_for_work_time_and_does_not_accumulate_catchup_commands(self):
        for work, expected_wait in ((0.01, 0.015), (0.04, 0.0)):
            with self.subTest(work=work):
                clock = [0.0]
                waits = []

                class Finished:
                    done = False

                    def is_set(self):
                        return self.done

                    def set(self):
                        self.done = True

                    def wait(self, timeout):
                        waits.append(timeout)
                        clock[0] += timeout
                        self.done = len(waits) >= 3
                        return self.done

                def tick():
                    clock[0] += work

                original = self.lab.finished
                self.lab.finished = Finished()
                try:
                    with (
                        patch.object(self.lab, "_tick", side_effect=tick),
                        patch("server.time", SimpleNamespace(monotonic=lambda: clock[0])),
                    ):
                        self.lab._loop()
                finally:
                    self.lab.finished = original
                self.assertEqual(len(waits), 3)
                for delay in waits:
                    self.assertAlmostEqual(delay, expected_wait)

    def test_stale_recording_frame_does_not_become_a_motor_fault(self):
        self.start()
        writer = self.writer()
        self.lab.writer = writer
        self.lab.camera_index = 0
        self.lab.camera.read = Mock(side_effect=AssertionError("fresh camera I/O in motor loop"))
        self.lab.camera.read_latest = Mock(side_effect=TimeoutError("stale frame"))
        self.lab._tick()
        self.assertTrue(self.lab.torque)
        self.assertFalse(writer.episode_active)
        self.lab.camera.read.assert_not_called()
        writer.add_frame.assert_not_called()
        self.assertIn("stale frame", self.lab.recording_error)
        events = self.lab.diagnostics.snapshot(full=True)["events"]
        self.assertEqual(sum(e["event"] == "recording_paused" for e in events), 1)


if __name__ == "__main__":
    unittest.main()
