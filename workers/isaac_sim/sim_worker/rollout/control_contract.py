"""Explicit simulator-native admission, independent of the legacy physical calibration."""

import hashlib
import os
import stat
from pathlib import Path

from .control_schema import canonical, validate


def envelope(value):
    if not isinstance(value, dict) or set(value) != {"record", "sha256"}:
        raise ValueError("Control contract requires exact record and SHA256")
    record = validate(value["record"])
    if hashlib.sha256(canonical(record)).hexdigest() != value["sha256"]:
        raise ValueError("Simulator control contract SHA256 mismatch")
    return record


def scene_digest(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 512 * 1024**2:
            raise ValueError("Simulator scene must be a bounded regular file")
        digest, count = hashlib.sha256(), 0
        while block := stream.read(1024 * 1024):
            count += len(block)
            if count > before.st_size:
                raise ValueError("Simulator scene changed during admission")
            digest.update(block)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if count != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError("Simulator scene changed during admission")
    return digest.hexdigest()


def admit(value, sim):
    record = envelope(value)
    if (
        type(sim.fps) is not int
        or list(sim.joints) != record["joint_order"]
        or sim.fps != record["action_fps"]
        or (sim.camera, sim.width, sim.height)
        != (record["camera"]["prim"], record["camera"]["width"], record["camera"]["height"])
    ):
        raise ValueError("Simulator joint order, action FPS or camera differs from policy contract")
    if scene_digest(Path(sim.scene)) not in record["source"]["scene_sha256"]:
        raise ValueError("Simulator root scene differs from recorded policy source")
    return record
