import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dispatch import MainThreadDispatcher, run_loop_pump


class DispatcherTests(unittest.TestCase):
    def test_requested_camera_open_runs_on_main_thread(self):
        dispatcher = MainThreadDispatcher()
        result = []
        worker = threading.Thread(
            target=lambda: result.append(dispatcher.call(threading.get_ident))
        )
        worker.start()
        dispatcher.pump_once()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [threading.main_thread().ident])
        dispatcher.close()

    def test_timeout_cancels_queued_job_without_camera_open(self):
        dispatcher = MainThreadDispatcher()
        calls, errors = [], []

        def worker():
            try:
                dispatcher.call(lambda: calls.append("opened"), timeout=0.02)
            except TimeoutError as error:
                errors.append(str(error))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(1)
        dispatcher.pump_once()
        self.assertEqual(calls, [])
        self.assertEqual(len(errors), 1)
        dispatcher.close()

    def test_shutdown_unblocks_waiting_camera_request(self):
        dispatcher = MainThreadDispatcher()
        errors = []

        def worker():
            try:
                dispatcher.call(lambda: "never opened", timeout=5)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=worker)
        thread.start()
        deadline = time.monotonic() + 1
        while dispatcher.queue.empty() and time.monotonic() < deadline:
            time.sleep(0.001)
        dispatcher.close()
        thread.join(0.2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            dispatcher.call(lambda: None)

    def test_timed_out_inflight_open_is_released_when_it_returns(self):
        dispatcher = MainThreadDispatcher()
        released, errors = [], []

        def action():
            time.sleep(0.06)
            return "handle"

        def worker():
            try:
                dispatcher.call(action, timeout=0.02, cleanup=released.append)
            except TimeoutError:
                errors.append("timeout")

        thread = threading.Thread(target=worker)
        thread.start()
        dispatcher.pump_once()
        thread.join(1)
        self.assertEqual(errors, ["timeout"])
        self.assertEqual(released, ["handle"])
        dispatcher.close()

    def test_open_exception_is_propagated_to_http_thread(self):
        dispatcher = MainThreadDispatcher()
        errors = []

        def action():
            raise ValueError("Camera permission denied")

        def worker():
            try:
                dispatcher.call(action)
            except ValueError as error:
                errors.append(str(error))

        thread = threading.Thread(target=worker)
        thread.start()
        dispatcher.pump_once()
        thread.join(1)
        self.assertEqual(errors, ["Camera permission denied"])
        dispatcher.close()

    def test_every_pump_lets_the_main_run_loop_handle_camera_notifications(self):
        """After a USB camera reset, reopening it returned no frames until the service
        restarted: the main thread never ran the loop that tells AVFoundation the camera
        came back."""
        dispatcher = MainThreadDispatcher()
        turns = []
        dispatcher.run_loop = lambda: turns.append(1)
        dispatcher.pump_once(timeout=0.01)
        dispatcher.pump_once(timeout=0.01)
        self.assertEqual(len(turns), 2)
        # A queued camera open still runs, after the loop's turn.
        result = []
        worker = threading.Thread(target=lambda: result.append(dispatcher.call(lambda: "opened")))
        worker.start()
        dispatcher.pump_once(timeout=1)
        worker.join(1)
        self.assertEqual(result, ["opened"])
        self.assertEqual(len(turns), 3)
        dispatcher.close()

    @unittest.skipUnless(sys.platform == "darwin", "the main run loop is a macOS facility")
    def test_the_macos_run_loop_turn_returns_at_once(self):
        turn = run_loop_pump()
        self.assertIsNotNone(turn)
        started = time.monotonic()
        for _ in range(20):
            turn()
        self.assertLess(time.monotonic() - started, 0.5)

    @unittest.skipIf(sys.platform == "darwin", "elsewhere there is no run loop to turn")
    def test_other_systems_have_no_run_loop_turn(self):
        self.assertIsNone(run_loop_pump())
        self.assertIsNone(MainThreadDispatcher().run_loop)


if __name__ == "__main__":
    unittest.main()
