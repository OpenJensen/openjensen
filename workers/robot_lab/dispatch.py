"""Bounded main-thread work dispatch for AVFoundation's permission/open calls."""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
from dataclasses import dataclass, field


def run_loop_pump():
    """On macOS, a call that lets the main thread's run loop handle what is pending.

    AVFoundation learns that a camera disconnected and connected again (a USB camera
    resets when its cable is jostled) from notifications delivered on the main run
    loop. A main thread that only waits on a Python queue never runs that loop, so the
    process keeps the dead device: reopening it opens but returns no frames until the
    service restarts.
    """
    if sys.platform != "darwin":
        return None
    try:
        foundation = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        mode = ctypes.c_void_p.in_dll(foundation, "kCFRunLoopDefaultMode")
        run = foundation.CFRunLoopRunInMode
        run.restype = ctypes.c_int32
        run.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool]
    except OSError, AttributeError, ValueError:
        return None
    return lambda: run(mode, 0.0, True)


@dataclass
class Job:
    action: object
    cleanup: object = None
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None
    cancelled: bool = False


class MainThreadDispatcher:
    def __init__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Camera dispatcher must be created on the main thread")
        self.owner = threading.get_ident()
        self.queue = queue.Queue()
        self.lock = threading.RLock()
        self.closed = False
        self.run_loop = run_loop_pump()

    def call(self, action, timeout=8, cleanup=None):
        if threading.get_ident() == self.owner:
            if self.closed:
                raise RuntimeError("Camera dispatcher is shutting down")
            return action()
        job = Job(action, cleanup)
        with self.lock:
            if self.closed:
                raise RuntimeError("Camera dispatcher is shutting down")
            self.queue.put(job)
        if not job.done.wait(timeout):
            with self.lock:
                if not job.done.is_set():
                    job.cancelled = True
                    raise TimeoutError(
                        "Camera authorization/open timed out. Check camera permissions "
                        "on the robot computer and retry."
                    )
        if job.error is not None:
            raise job.error
        return job.result

    def pump_once(self, timeout=0.1):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Camera dispatcher pump must run on the main thread")
        if self.run_loop is not None:
            self.run_loop()  # camera connect and disconnect notifications arrive here
        try:
            job = self.queue.get(timeout=timeout)
        except queue.Empty:
            return
        with self.lock:
            cancelled = job.cancelled or self.closed
        if cancelled:
            job.error = RuntimeError("Camera request cancelled")
            job.done.set()
            return
        try:
            result = job.action()
            with self.lock:
                if job.cancelled or self.closed:
                    if job.cleanup is not None:
                        job.cleanup(result)
                    job.error = RuntimeError("Camera request cancelled")
                else:
                    job.result = result
        except BaseException as error:
            job.error = error
        finally:
            job.done.set()

    def close(self):
        with self.lock:
            self.closed = True
            while True:
                try:
                    job = self.queue.get_nowait()
                except queue.Empty:
                    break
                job.cancelled = True
                job.error = RuntimeError("Camera dispatcher is shutting down")
                job.done.set()
