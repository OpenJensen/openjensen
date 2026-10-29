"""Bounded asynchronous PNG saving, following LeRobot's image-writer pattern.

Reference: huggingface/lerobot datasets/image_writer.py at the revision recorded
in LEROBOT-PROVENANCE.json (Apache-2.0). This companion uses one daemon thread,
a bounded queue and explicit error propagation instead of importing Torch.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path


def save_png(image, path):
    from PIL import Image

    Image.fromarray(image).save(path, compress_level=1)


class AsyncImageWriter:
    def __init__(self, *, capacity=8, write=save_png):
        self.queue = queue.Queue(maxsize=capacity)
        self.write = write
        self.lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        self.pending = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="robot-lab-images", daemon=True)
        self.thread.start()

    def save_image(self, image, path: Path):
        # The camera can reuse a buffer after this method returns.
        copied = image.copy()
        with self.lock:
            if self.closed:
                raise ValueError("Image writer is closed")
            if self.error:
                raise ValueError("Image writing failed: " + self.error)
            try:
                self.queue.put_nowait((copied, path))
            except queue.Full:
                raise ValueError(
                    "Image writer cannot keep up; save or discard the paused episode"
                ) from None
            self.pending += 1
            self.idle.clear()

    def _run(self):
        while True:
            try:
                image, path = self.queue.get(timeout=0.1)
            except queue.Empty:
                with self.lock:
                    if self.closed:
                        return
                continue
            try:
                if self.error is None:
                    self.write(image, path)
            except Exception as error:
                with self.lock:
                    self.error = str(error)
            finally:
                with self.lock:
                    self.pending -= 1
                    if not self.pending:
                        self.idle.set()
                self.queue.task_done()

    def wait_until_done(self, timeout=5, *, ignore_error=False):
        if not self.idle.wait(timeout):
            raise ValueError("Image writes are still pending; saved episodes are preserved")
        if self.error and not ignore_error:
            raise ValueError("Image writing failed: " + self.error)

    def clear_error(self):
        with self.lock:
            if self.pending:
                raise ValueError("Wait for image writes before discarding")
            self.error = None

    def stop(self, timeout=2, *, ignore_error=False):
        with self.lock:
            self.closed = True
        try:
            self.wait_until_done(timeout, ignore_error=ignore_error)
        finally:
            self.thread.join(timeout=0.2)
