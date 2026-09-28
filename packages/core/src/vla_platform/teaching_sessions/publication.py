"""Bounded capture copy and atomic no-replace publication after worker reaping."""

import hashlib
import os
import re
import stat
import time
from pathlib import Path

from vla_platform.datasets import recordings
from vla_platform.datasets.local_preview import _open_beneath
from vla_platform.datasets.snapshots import _publish

from .config import TeachingError

MAX_ENTRIES = 50000
PROOF_LIMIT = 16 * 1024**2


class PublicationUncertain(TeachingError):
    """Publication failed without proof that the owned capture stayed private."""


def new_json(path: Path, value: dict) -> None:
    raw = recordings.canonical(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync(path.parent)


def sync(path: Path) -> None:
    with recordings.directory(path) as fd:
        os.fsync(fd)


def inventory(root: Path, maximum: int) -> dict[str, str]:
    result = {}
    deadline = time.monotonic() + 60
    total = count = 0
    with recordings.directory(root):
        pass
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            if time.monotonic() > deadline:
                raise TeachingError("Capture inventory exceeded its time limit")
            count += 1
            if count > MAX_ENTRIES:
                raise TeachingError("Capture exceeds its entry limit")
            item = Path(folder) / name
            info = item.lstat()
            if stat.S_ISDIR(info.st_mode):
                with recordings.directory(item):
                    pass
                continue
            if not stat.S_ISREG(info.st_mode):
                raise TeachingError("Capture contains a linked or special file")
            total += info.st_size
            if total > maximum or info.st_size > 2 * 1024**3:
                raise TeachingError("Capture exceeds its byte limit")
            relative = item.relative_to(root)
            with _open_beneath(root, relative) as source:
                before = os.fstat(source.fileno())
                if stamp(before) != stamp(info):
                    raise TeachingError("Capture changed before reading")
                digest = hashlib.sha256()
                remaining = before.st_size
                while remaining:
                    if time.monotonic() > deadline:
                        raise TeachingError("Capture inventory exceeded its time limit")
                    chunk = source.read(min(1024**2, remaining))
                    if not chunk:
                        raise TeachingError("Capture changed while reading")
                    remaining -= len(chunk)
                    digest.update(chunk)
                if source.read(1) or stamp(before) != stamp(os.fstat(source.fileno())):
                    raise TeachingError("Capture changed while reading")
            result[relative.as_posix()] = digest.hexdigest()
    return result


def stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def proof(raw: Path, receipt: Path, maximum: int) -> dict:
    value = recordings.decode(recordings.read(receipt, PROOF_LIMIT))
    if set(value) != {
        "schema_version",
        "session_id",
        "session_sha256",
        "inventory",
        "inventory_sha256",
        "episodes",
        "origin",
        "lineage_group",
    }:
        raise TeachingError("Invalid capture verification receipt")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise TeachingError("Unsupported capture verification schema")
    expected = value["inventory"]
    if not isinstance(expected, dict) or not 1 <= len(expected) <= MAX_ENTRIES:
        raise TeachingError("Invalid capture inventory")
    for name, digest in expected.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            raise TeachingError("Invalid capture file identity")
        if name != "session.json" and not re.fullmatch(
            r"[a-f0-9]{32}/(?:started.json|episode.json|trajectory.jsonl|events.jsonl|frames/[0-9]{6}.rgb)",
            name,
        ):
            raise TeachingError("Capture contains an unexpected file")
    if (
        recordings.digest(recordings.canonical(expected)) != value["inventory_sha256"]
        or inventory(raw, maximum) != expected
    ):
        raise TeachingError("Capture inventory no longer matches verification")
    metadata = recordings.metadata(recordings.decode(recordings.read(raw / "session.json")))
    for key in ("session_id", "origin", "lineage_group"):
        if value[key] != metadata[key]:
            raise TeachingError("Capture verification metadata differs")
    if value["session_sha256"] != expected.get("session.json"):
        raise TeachingError("Capture session hash differs")
    return value


def stage_capture(raw: Path, stage: Path, value: dict, maximum: int) -> None:
    """Copy into a private sibling outside the catalog, retaining original bytes."""
    stage.mkdir(mode=0o700, exist_ok=False)
    expected = value["inventory"]
    deadline = time.monotonic() + 60
    total = 0
    for name in sorted(expected):
        relative = Path(name)
        destination = stage / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _open_beneath(raw, relative) as source, destination.open("xb") as target:
            before = os.fstat(source.fileno())
            digest = hashlib.sha256()
            remaining = before.st_size
            total += remaining
            if total > maximum:
                raise TeachingError("Capture file exceeds its bound")
            while remaining:
                if time.monotonic() > deadline:
                    raise TeachingError("Capture copy exceeded its time limit")
                chunk = source.read(min(1024**2, remaining))
                if not chunk:
                    raise TeachingError("Capture changed while copying")
                remaining -= len(chunk)
                digest.update(chunk)
                target.write(chunk)
            if (
                source.read(1)
                or stamp(before) != stamp(os.fstat(source.fileno()))
                or digest.hexdigest() != expected[name]
            ):
                raise TeachingError("Capture changed while copying")
            target.flush()
            os.fsync(target.fileno())
    if inventory(raw, maximum) != expected or inventory(stage, maximum) != expected:
        raise TeachingError("Capture source or copy changed before publication")
    for folder, _directories, _files in os.walk(stage, topdown=False):
        sync(Path(folder))
    sync(stage.parent)


def publish(stage: Path, root: Path, session_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise TeachingError("Invalid published capture identity")
    with recordings.directory(root):
        pass
    destination = root / session_id
    owned = stage.stat(follow_symlinks=False)
    try:
        _publish(stage, destination)
    except BaseException as error:
        # Rename may have succeeded before directory fsync failed. Roll back only
        # our own inode, never a concurrent replacement or pre-existing capture.
        def is_owned(path: Path) -> bool:
            try:
                current = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                return False
            return (owned.st_dev, owned.st_ino) == (current.st_dev, current.st_ino)

        try:
            if is_owned(destination) and not os.path.lexists(stage):
                _publish(destination, stage)
        except OSError:
            pass  # Verify the outcome below, including a failed rollback fsync.
        try:
            private = is_owned(stage) and not is_owned(destination)
        except OSError:
            private = False
        if not private:
            raise PublicationUncertain(
                "Capture publication or rollback could not be verified; "
                "a verified capture may remain in the catalog. "
                "Inspect retained evidence before retrying."
            ) from error
        raise
    return destination
