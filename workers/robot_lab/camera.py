"""Shared camera handles, latest-frame preview and explicit permission errors."""

from __future__ import annotations

import os
import platform
import re
import struct
import threading
import time
from pathlib import Path

MAX_CAMERA_INDEX = 63


class CameraPool:
    def __init__(self, dispatcher=None):
        self.dispatcher = dispatcher
        self.lock = threading.RLock()
        self.handles = {}
        self.last_used = {}
        self.last_frames = {}
        self.frame_lock = threading.RLock()
        self.frame_times = {}
        self.readers = {}
        self.reader_errors = {}

    def read(self, index=0, width=640, height=480, *, _background=False, _stop=None):
        import cv2

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index <= MAX_CAMERA_INDEX
        ):
            raise ValueError(f"Camera index must be an integer from 0 to {MAX_CAMERA_INDEX}")
        with self.lock:
            if _stop is not None and _stop.is_set():
                raise RuntimeError("Camera reader stopped")
            if index not in self.handles:

                def open_capture():
                    # OpenCV must request AVFoundation authorization from the
                    # main thread; LaunchAgents otherwise cannot spin its loop.
                    capture = cv2.VideoCapture(
                        index,
                        cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_V4L2,
                    )
                    if not capture.isOpened():
                        capture.release()
                        if platform.system() == "Darwin":
                            message = (
                                "Camera could not open. In macOS Privacy & Security, allow Camera "
                                "access for the companion's Python process, then retry."
                            )
                        else:
                            message = (
                                f"Camera /dev/video{index} could not open. Check read/write "
                                "permissions (usually the video group), USB connection, and "
                                "whether another app is using it, then retry."
                            )
                        raise RuntimeError(message)
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
            if not _background:
                self.last_used[index] = time.monotonic()
            ok, frame = self.handles[index].read()
            if not ok or frame is None:
                if not _background:
                    self.close(index)
                raise RuntimeError(
                    "Camera opened but did not return a frame; close other camera apps and retry"
                )
            if _stop is not None and _stop.is_set():
                raise RuntimeError("Camera reader stopped")
            with self.frame_lock:
                self.last_frames[index] = frame
                self.frame_times[index] = time.monotonic()
            return frame

    def start_reader(self, index=0):
        """Keep a recent frame ready, as LeRobot OpenCVCamera.read_latest does."""
        if type(index) is not int or not 0 <= index <= MAX_CAMERA_INDEX:
            raise ValueError("Invalid camera index")
        with self.frame_lock:
            current = self.readers.get(index)
            if current and current[0].is_alive():
                if current[1].is_set():
                    raise RuntimeError("Camera is stopping; retry recording")
                return
            stop = threading.Event()
            thread = threading.Thread(
                target=self._read_loop,
                args=(index, stop),
                name=f"robot-lab-camera-{index}",
                daemon=True,
            )
            self.reader_errors.pop(index, None)
            self.readers[index] = (thread, stop)
            thread.start()

    def _read_loop(self, index, stop):
        failures = 0
        while not stop.is_set():
            try:
                self.read(index, _background=True, _stop=stop)
                failures = 0
                stop.wait(0.001)
            except Exception as error:
                failures += 1
                if failures >= 10:
                    with self.frame_lock:
                        self.reader_errors[index] = str(error)
                    return
                stop.wait(0.01)

    def read_latest(self, index=0, max_age=0.5):
        # Never acquire the device lock or wait on VideoCapture.read here.
        with self.frame_lock:
            reader = self.readers.get(index)
            error = self.reader_errors.get(index)
            frame, captured = self.last_frames.get(index), self.frame_times.get(index)
            if error:
                raise RuntimeError("Camera capture failed: " + error)
            if not reader or not reader[0].is_alive():
                raise RuntimeError("Camera background reader is not running")
            if frame is None or captured is None:
                raise RuntimeError("Camera has not captured a frame yet")
            if time.monotonic() - captured > max_age:
                raise TimeoutError("Camera frame is stale; recording paused")
            self.last_used[index] = time.monotonic()
            return frame

    def read_latest_timed(self, index=0, max_age=0.5):
        """Return one atomic host capture timestamp and its matching frame."""
        with self.frame_lock:
            frame = self.read_latest(index, max_age)
            return frame, self.frame_times[index]

    def jpeg(self, index=0):
        import cv2

        frame = self.read(index)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode camera preview")
        return encoded.tobytes()

    def capture_timed(self, index=0):
        """Read a fresh frame with explicit host interval, never a cached preview.

        OpenCV does not supply portable exposure timestamps for these cameras.
        These are host read bounds, not a claim of sensor-clock synchronization.
        """
        import cv2

        started = time.monotonic_ns()
        frame = self.read(index)
        ended = time.monotonic_ns()
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode admitted camera observation")
        return {
            "jpeg": encoded.tobytes(),
            "read_start_monotonic_ns": started,
            "read_end_monotonic_ns": ended,
            "image_sensor_timestamp_verified": False,
        }

    def latest_timed_jpeg(self, index=0, max_age=0.15):
        """Encode the background reader's latest atomic host-timestamped frame."""
        import cv2

        frame, captured = self.read_latest_timed(index, max_age)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode policy camera observation")
        return {"jpeg": encoded.tobytes(), "captured_monotonic": captured}

    def discover(self):
        # Inventory only: do not open every camera or compete with recording.
        if platform.system() == "Darwin":
            # Enumerate in the capture process, using the same AVFoundation
            # devices and unique-ID ordering as OpenCV. FFmpeg and
            # system_profiler use different device sets and index orders.
            if self.dispatcher is not None:
                return self.dispatcher.call(macos_cameras, timeout=5)
            return macos_cameras()
        return linux_cameras()

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
        with self.frame_lock:
            selected = list(set(self.handles) | set(self.readers)) if index is None else [index]
            readers = [self.readers.pop(key) for key in selected if key in self.readers]
            for _, stop in readers:
                stop.set()
        with self.lock:
            for key in selected:
                capture = self.handles.pop(key, None)
                if capture is not None:
                    capture.release()
                self.last_used.pop(key, None)
                with self.frame_lock:
                    self.last_frames.pop(key, None)
                    self.frame_times.pop(key, None)
                    self.reader_errors.pop(key, None)
        for thread, _ in readers:
            if thread is not threading.current_thread():
                thread.join(timeout=0.5)

    def close_idle(self, max_age=30):
        with self.lock:
            for index, used in list(self.last_used.items()):
                if time.monotonic() - used > max_age:
                    self.close(index)


def linux_cameras(sysfs=Path("/sys/class/video4linux"), devices=Path("/dev")):
    """Query V4L2 capabilities without starting capture or requesting frames."""
    import fcntl

    output = []
    for path in sysfs.glob("video*"):
        match = re.fullmatch(r"video(\d+)", path.name)
        if not match or int(match[1]) > MAX_CAMERA_INDEX:
            continue
        index = int(match[1])
        try:
            descriptor = os.open(devices / path.name, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            # Keep an inaccessible camera visible so preview can explain permissions.
            descriptor = None
        except OSError:
            continue  # Disconnected between sysfs enumeration and query.
        if descriptor is not None:
            try:
                capability = bytearray(104)  # struct v4l2_capability, Linux UAPI.
                fcntl.ioctl(descriptor, 0x80685600, capability)  # VIDIOC_QUERYCAP
                caps, device_caps = struct.unpack_from("=II", capability, 84)
                if caps & 0x80000000:  # V4L2_CAP_DEVICE_CAPS
                    caps = device_caps
                if not caps & (0x00000001 | 0x00001000):  # VIDEO_CAPTURE / MPLANE
                    continue  # Metadata-only/output nodes are not selectable cameras.
            except OSError:
                continue
            finally:
                os.close(descriptor)
        try:
            name = (path / "name").read_text().strip()
        except OSError:
            name = path.name
        output.append(
            {"index": index, "name": name, "width": 640, "height": 480, "verified": False}
        )
    return sorted(output, key=lambda camera: camera["index"])


def macos_cameras():
    """Inventory only; do not open cameras or request camera authorization."""
    return [
        {
            "index": index,
            "name": name,
            "unique_id": unique_id,
            "width": 640,
            "height": 480,
            "verified": False,
        }
        for index, (name, unique_id) in enumerate(_avfoundation_devices()[: MAX_CAMERA_INDEX + 1])
    ]


def _avfoundation_devices():
    """Match OpenCV's Video + Muxed device list and NSString uniqueID compare.

    Native APIs are loaded lazily on macOS, so Linux needs no Apple libraries
    or additional Python packages. No device is opened by this inventory.
    """
    import ctypes
    from functools import cmp_to_key

    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    foundation = ctypes.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value

    def send(obj, selector, result=ctypes.c_void_p, types=(), args=()):
        # Separate typed function pointers avoid mutating a shared msgSend
        # signature while another camera request is running.
        function = ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p, *types)(address)
        return function(obj, objc.sel_registerName(selector.encode("ascii")), *args)

    def string(obj):
        value = send(obj, "UTF8String", ctypes.c_char_p)
        if value is None:
            raise RuntimeError("Camera returned an unavailable native identifier")
        return value.decode("utf-8")

    pool = send(send(objc.objc_getClass(b"NSAutoreleasePool"), "alloc"), "init")
    try:
        cls = objc.objc_getClass(b"AVCaptureDevice")
        devices = []
        for media_type in ("AVMediaTypeVideo", "AVMediaTypeMuxed"):
            media = ctypes.c_void_p.in_dll(foundation, media_type).value
            array = send(cls, "devicesWithMediaType:", types=(ctypes.c_void_p,), args=(media,))
            count = send(array, "count", ctypes.c_ulong)
            if count > 128:
                raise RuntimeError("Camera inventory exceeds the supported device limit")
            for index in range(count):
                device = send(array, "objectAtIndex:", types=(ctypes.c_ulong,), args=(index,))
                devices.append((device, send(device, "uniqueID")))

        def compare(left, right):
            return send(left[1], "compare:", ctypes.c_long, (ctypes.c_void_p,), (right[1],))

        devices.sort(key=cmp_to_key(compare))
        return [(string(send(device, "localizedName")), string(uid)) for device, uid in devices]
    finally:
        send(pool, "drain", None)
