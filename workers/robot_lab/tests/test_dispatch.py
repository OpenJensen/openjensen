import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dispatch import MainThreadDispatcher


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


if __name__ == "__main__":
    unittest.main()
