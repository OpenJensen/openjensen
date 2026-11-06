"""Native Linux camera modes and stable USB-port settings identities."""

from __future__ import annotations

import errno
import os
import re
import struct
from pathlib import Path


def camera_identity(index, sysfs=Path("/sys/class/video4linux")):
    device = (sysfs / f"video{index}" / "device").resolve()
    for parent in (device, *device.parents):
        try:
            vendor = (parent / "idVendor").read_text().strip().lower()
            product = (parent / "idProduct").read_text().strip().lower()
        except OSError:
            continue
        # Identical USB webcams may share a firmware serial. Include their USB
        # port so independent view settings never overwrite one another.
        return f"usb:{vendor}:{product}:{parent.name}"
    return f"video:{index}:{device}"


def validate_mode(mode):
    if not isinstance(mode, dict) or set(mode) != {"fourcc", "width", "height", "fps"}:
        raise ValueError("Choose a format, resolution and frame rate")
    if not isinstance(mode["fourcc"], str) or not re.fullmatch(r"[A-Z0-9]{4}", mode["fourcc"]):
        raise ValueError("Invalid camera format")
    for key in ("width", "height"):
        if type(mode[key]) is not int or not 1 <= mode[key] <= 8192:
            raise ValueError("Camera dimensions must be integers from 1 to 8192")
    if type(mode["fps"]) not in (int, float) or not 1 <= mode["fps"] <= 240:
        raise ValueError("Camera frame rate must be from 1 to 240 fps")
    return dict(mode)


def linux_modes(index, devices=Path("/dev")):
    """Enumerate advertised discrete modes without starting a video stream."""
    import fcntl

    result = []
    descriptor = os.open(devices / f"video{index}", os.O_RDONLY | os.O_NONBLOCK)
    try:

        def query(request, value):
            try:
                fcntl.ioctl(descriptor, request, value, True)
                return True
            except OSError as error:
                if error.errno == errno.EINVAL:
                    return False
                raise

        for format_index in range(32):
            value = bytearray(64)
            struct.pack_into("=II", value, 0, format_index, 1)
            if not query(0xC0405602, value):
                break
            code = struct.unpack_from("=I", value, 44)[0]
            fourcc = code.to_bytes(4, "little").decode("ascii", errors="replace")
            for size_index in range(64):
                size = bytearray(44)
                struct.pack_into("=II", size, 0, size_index, code)
                if not query(0xC02C564A, size):
                    break
                if struct.unpack_from("=I", size, 8)[0] != 1:
                    continue
                width, height = struct.unpack_from("=II", size, 12)
                for rate_index in range(64):
                    interval = bytearray(52)
                    struct.pack_into("=IIII", interval, 0, rate_index, code, width, height)
                    if not query(0xC034564B, interval):
                        break
                    if struct.unpack_from("=I", interval, 16)[0] != 1:
                        continue
                    numerator, denominator = struct.unpack_from("=II", interval, 20)
                    if numerator and denominator:
                        mode = {
                            "fourcc": fourcc,
                            "width": width,
                            "height": height,
                            "fps": round(denominator / numerator, 6),
                        }
                        try:
                            validate_mode(mode)
                        except ValueError:
                            continue
                        if mode not in result:
                            result.append(mode)
    finally:
        os.close(descriptor)
    return sorted(
        result, key=lambda mode: (mode["fourcc"], mode["width"] * mode["height"], mode["fps"])
    )
