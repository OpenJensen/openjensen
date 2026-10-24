"""Shared camera handles, latest-frame preview and explicit permission errors."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import threading
import time


class CameraPool:
    def __init__(self, dispatcher=None):
        self.dispatcher = dispatcher
        self.lock = threading.RLock()
        self.handles = {}
        self.last_used = {}
        self.last_frames = {}

    def read(self, index=0, width=640, height=480):
        import cv2

        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= 8:
            raise ValueError("Camera index must be an integer from 0 to 8")
        with self.lock:
            if index not in self.handles:

                def open_capture():
                    # OpenCV must request AVFoundation authorization from the
                    # main thread; LaunchAgents otherwise cannot spin its loop.
                    capture = cv2.VideoCapture(
                        index,
                        cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY,
                    )
                    if not capture.isOpened():
                        capture.release()
                        raise RuntimeError(
                            "Camera could not open. In macOS Privacy & Security, allow Camera "
                            "access for the companion's Python process, then retry."
                        )
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, min(1920, max(160, int(width))))
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, min(1080, max(120, int(height))))
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return capture

                if self.dispatcher is not None:
                    capture = self.dispatcher.call(
                        open_capture, timeout=8, cleanup=lambda abandoned: abandoned.release()
                    )
                else:
                    capture = open_capture()
                self.handles[index] = capture
            self.last_used[index] = time.monotonic()
            ok, frame = self.handles[index].read()
            if not ok or frame is None:
                self.close(index)
                raise RuntimeError(
                    "Camera opened but did not return a frame; close other camera apps and retry"
                )
            self.last_frames[index] = frame
            return frame

    def jpeg(self, index=0):
        import cv2

        frame = self.read(index)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode camera preview")
        return encoded.tobytes()

    def discover(self):
        # Inventory only: do not open every camera or compete with recording.
        if platform.system() == "Darwin":
            executable = shutil.which("ffmpeg")
            if executable:
                result = subprocess.run(
                    [
                        executable,
                        "-hide_banner",
                        "-f",
                        "avfoundation",
                        "-list_devices",
                        "true",
                        "-i",
                        "",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                cameras, video = [], False
                for line in result.stderr.splitlines():
                    if "AVFoundation video devices" in line:
                        video = True
                    elif "AVFoundation audio devices" in line:
                        video = False
                    elif video:
                        match = re.search(r"\[(\d+)\] (.+)$", line)
                        if match and not match.group(2).startswith("Capture screen"):
                            cameras.append(
                                {
                                    "index": int(match.group(1)),
                                    "name": match.group(2),
                                    "width": 640,
                                    "height": 480,
                                    "verified": False,
                                }
                            )
                if cameras:
                    return cameras
            # macOS system_profiler is a read-only inventory fallback. Its list
            # order is not a verified AVFoundation index; preview verifies it.
            import json

            result = subprocess.run(
                ["/usr/sbin/system_profiler", "SPCameraDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            devices = json.loads(result.stdout).get("SPCameraDataType", [])
            return [
                {
                    "index": index,
                    "name": device.get("_name", f"Camera {index}"),
                    "width": 640,
                    "height": 480,
                    "verified": False,
                }
                for index, device in enumerate(devices)
            ]
        from pathlib import Path

        output = []
        for path in sorted(Path("/sys/class/video4linux").glob("video*")):
            index = int(path.name.removeprefix("video"))
            name_path = path / "name"
            output.append(
                {
                    "index": index,
                    "name": name_path.read_text().strip() if name_path.exists() else path.name,
                    "width": 640,
                    "height": 480,
                    "verified": False,
                }
            )
        return output

    def cached_jpeg(self, index=0):
        import cv2

        with self.lock:
            frame = self.last_frames.get(index)
            if frame is None:
                return self.jpeg(index)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("Could not encode camera preview")
            return encoded.tobytes()

    def close(self, index=None):
        with self.lock:
            for key in list(self.handles) if index is None else [index]:
                capture = self.handles.pop(key, None)
                if capture is not None:
                    capture.release()
                self.last_used.pop(key, None)
                self.last_frames.pop(key, None)

    def close_idle(self, max_age=30):
        with self.lock:
            for index, used in list(self.last_used.items()):
                if time.monotonic() - used > max_age:
                    self.close(index)
