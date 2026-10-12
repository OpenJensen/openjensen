"""Fixed reviewed model identity and bounded POSIX no-follow reads."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from .contracts import LICENSE, DecisionError

# Raw bytes from the pinned Hugging Face commit; no remote code is vendored here.
FILES = {
    "README.md": (2802, "2a3268033869cc811a5c192a9d09920a4ed004e9c679f5e8d0aad5eb21bd1ee9"),
    "config.json": (556, "e08acb934be8c02767197a561e92b47501e9ca85a1381081e98467ae92cc3b7b"),
    "decision_model.py": (2082, "e0f3174d8578f643019552364bde9f5038894953d9e159efba05dd64b096ba01"),
    "merges.txt": (217070, "4cc053b0552311c2a05ccd4451295fec8747d8d00c1ec37d87d1181b2a128f5b"),
    "model.py": (7577, "7e743f2bdb5534359bb38d951bb98d8d28d9a6d5c04eeb6a7bd81a02e775d6b9"),
    "model.safetensors": (
        203051616,
        "3d81f0712ea1e9495a5996258dbc9a41e2fc2e1912dd87464683b20950e5c76f",
    ),
    "vocab.json": (374931, "8c7e0f4021640cb024648d11831329b98a9f3f7c8ca340aa9915ae5a11746747"),
}


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_file(path: Path, limit: int, *, exact_size: int | None = None) -> bytes:
    # dir_fd + O_NOFOLLOW prevent symlinks in every ancestor, not just the final file.
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise DecisionError("This worker requires POSIX no-follow file access (Linux or macOS).")
    path = Path(path)
    if ".." in path.parts:
        raise DecisionError("Parent traversal is not allowed in input paths.")
    path = path.absolute()
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    descriptor = None
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise DecisionError("Input must be a bounded regular file.")
        if exact_size is not None and before.st_size != exact_size:
            raise DecisionError("Pinned model file size does not match.", "model_integrity")
        remaining = before.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                raise DecisionError("Input changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) or _stamp(before) != _stamp(os.fstat(descriptor)):
            raise DecisionError("Input changed while being read.")
        return b"".join(chunks)
    except OSError as exc:
        raise DecisionError(
            "Cannot read a regular input file without following symlinks.", "input_file"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def verified_files(model_dir: Path, accepted_license: str | None):
    if accepted_license != LICENSE:
        raise DecisionError(
            f"Explicit --accept-license {LICENSE} is required; commercial use is restricted."
        )
    verified = {}
    for name, (size, digest) in FILES.items():
        raw = read_file(Path(model_dir) / name, size, exact_size=size)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise DecisionError(
                f"Pinned checksum mismatch for {name}; nothing was loaded.", "model_integrity"
            )
        verified[name] = raw
    return verified
