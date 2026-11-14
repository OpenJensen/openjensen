"""Lossless independent camera streams. The camera reader only enqueues references."""

import io
import json
import queue
import shutil
import threading
import zipfile


class CameraStreams:
    def __init__(self, root, roles):
        self.root = root
        self.roles = roles
        self.queue = queue.Queue(maxsize=32)
        self.error = None
        self.counts = dict.fromkeys(roles.values(), 0)
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.write, daemon=True, name="collection-cameras")
        self.thread.start()

    def capture(self, index, sequence, timestamp, frame):
        if index in self.roles and not self.closed.is_set():
            try:
                self.queue.put_nowait((self.roles[index], sequence, int(timestamp * 1e9), frame))
            except queue.Full:
                self.error = "Independent camera stream queue overflow"

    def write(self):
        try:
            from PIL import Image

            with (
                zipfile.ZipFile(self.root / "camera-streams.zip", "x", allowZip64=True) as archive,
                (self.root / "camera-streams.jsonl").open("x") as index,
            ):
                while not self.closed.is_set() or not self.queue.empty():
                    try:
                        role, sequence, timestamp, frame = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if (
                        sum(self.counts.values()) % 60 == 0
                        and shutil.disk_usage(self.root).free < 1024**3
                    ):
                        raise ValueError("Independent camera stream disk reserve reached")
                    name = f"{role}/{sequence}.png"
                    data = io.BytesIO()
                    Image.fromarray(frame[:, :, ::-1]).save(data, format="PNG", compress_level=1)
                    archive.writestr(name, data.getvalue())
                    row = {
                        "role": role,
                        "frame_id": f"{role}-{timestamp}",
                        "sequence": sequence,
                        "read_complete_robot_ns": timestamp,
                        "sensor_capture_ns": None,
                        "member": name,
                    }
                    index.write(json.dumps(row) + "\n")
                    self.counts[role] += 1
        except Exception as exc:
            self.error = str(exc)

    def finish(self):
        self.closed.set()
        self.thread.join()
