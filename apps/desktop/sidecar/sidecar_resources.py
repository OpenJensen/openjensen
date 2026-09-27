"""Bounded static-resource identity checks; application data is never written here."""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_MANIFEST = 4 * 1024**2
MAX_FILES = 10000
MAX_STATIC_BYTES = 512 * 1024**2


def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def json_object(raw: bytes):
    def finite(value):
        raise ValueError("Nonfinite JSON")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=finite)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def absolute_directory(path: Path, *, existing: bool = True) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("A trusted absolute real directory is required")
    if existing and not path.is_dir():
        raise ValueError("Directory is missing")
    return path.resolve()


def read_regular(path: Path, maximum: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError("Resource is not a bounded regular file")
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Resource exceeded its byte limit")
    return raw


@dataclass(frozen=True)
class Resources:
    root: Path
    web: Path
    build_id: str
    app_version: str
    manifest_sha256: str


def file_identity(path: Path, size: int) -> str:
    """Hash at most the declared size plus one, without allocating an asset-sized buffer."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError("Static resource size changed")
        digest, seen = hashlib.sha256(), 0
        while chunk := source.read(min(65536, size + 1 - seen)):
            seen += len(chunk)
            digest.update(chunk)
            if seen > size:
                raise ValueError("Static resource grew while checking")
        if seen != size:
            raise ValueError("Static resource shrank while checking")
    return digest.hexdigest()


def load_resources(root: Path) -> Resources:
    root = absolute_directory(root)
    raw = read_regular(root / "resources.json", MAX_MANIFEST)
    manifest = json_object(raw)
    if (
        set(manifest) != {"schema_version", "build_id", "app_version", "files"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["build_id"], str)
        or not re.fullmatch(r"[a-f0-9]{40}", manifest["build_id"])
        or manifest["app_version"] != "0.1.0"
        or not isinstance(manifest["files"], dict)
        or not 1 <= len(manifest["files"]) <= MAX_FILES
        or "index.html" not in manifest["files"]
    ):
        raise ValueError("Invalid static resource manifest")
    web = absolute_directory(root / "web")
    expected = manifest["files"]
    total = 0
    for name, info in expected.items():
        rel = PurePosixPath(name)
        if (
            not name
            or rel.is_absolute()
            or rel.as_posix() != name
            or any(part in {".", ".."} for part in rel.parts)
            or "\\" in name
            or ":" in name
            or not isinstance(info, dict)
            or set(info) != {"bytes", "sha256"}
            or type(info["bytes"]) is not int
            or not 0 <= info["bytes"] <= MAX_STATIC_BYTES
            or not isinstance(info["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", info["sha256"])
        ):
            raise ValueError("Invalid static resource inventory entry")
        total += info["bytes"]
        if total > MAX_STATIC_BYTES:
            raise ValueError("Static resource payload exceeds its bound")
    actual = set()
    for directory, dirs, files in os.walk(web, followlinks=False):
        for name in [*dirs, *files]:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not (
                stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
            ):
                raise ValueError("Static resources cannot contain links or special files")
        for name in files:
            path = Path(directory) / name
            key = path.relative_to(web).as_posix()
            if key not in expected:
                raise ValueError("Unregistered static resource")
            actual.add(key)
            info = expected[key]
            # Read exactly the recorded limit plus one; changing files fail closed.
            if file_identity(path, info["bytes"]) != info["sha256"]:
                raise ValueError("Static resource hash or size changed")
    if actual != set(expected):
        raise ValueError("Static resource is missing")
    return Resources(
        root, web, manifest["build_id"], manifest["app_version"], hashlib.sha256(raw).hexdigest()
    )
