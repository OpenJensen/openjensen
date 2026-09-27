"""Explicit ACT export download: immutable GCS source to a new local job artifact.

The child runs in the existing credentialed SkyPilot interpreter. It downloads no
unlisted paths, launches no compute, and never changes the registered descriptor.
"""

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath

if __package__:
    from . import cloud_storage
else:  # The isolated SkyPilot Python executes this exact server-owned script.
    import cloud_storage

MAX_FILES = 128
MAX_FILE_BYTES = 2 * 1024**3
MAX_BYTES = 4 * 1024**3
MAX_METADATA = 8 * 1024**2
RESERVED = "firebird_cloud_source.json"
REQUIRED = {
    "checkpoint/manifest.json",
    "checkpoint/pretrained_model/model.safetensors",
    "checkpoint/pretrained_model/config.json",
    "checkpoint/recipe.json",
    "verification.json",
}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate cloud metadata field")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("Non-finite cloud metadata")


def read_bytes(path, limit=MAX_METADATA):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("Cloud metadata must be a bounded regular file")
        data = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        if len(data) != before.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Cloud metadata changed while reading")
        return data


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite cloud metadata")
    return number


def decode(data):
    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
        parse_float=_float,
    )


def descriptor(directory, expected_sha):
    """Offline admission before a job or GCS request; never trust mutable labels."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Cloud checkpoint must be a registered nonsymlink directory")
    content = read_bytes(directory / "manifest.json")
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("Registered cloud checkpoint manifest changed")
    manifest = decode(content)
    pointer = decode(read_bytes(directory / "remote.json", 16384))
    files = cloud_storage.checked_files(manifest)
    metadata = manifest.get("metadata", {})
    dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or not isinstance(metadata, dict)
        or metadata.get("architecture") != "act"
        or metadata.get("training_backend") != "lerobot"
        or metadata.get("method") != "full"
        or metadata.get("reload_verified") is not True
        or not REQUIRED.issubset(files)
        or not isinstance(dataset, dict)
        or dataset.get("source") != "huggingface"
        or not re.fullmatch(r"[a-f0-9]{40}", str(dataset.get("revision", "")))
    ):
        raise ValueError("Choose the completed, reload-verified native ACT checkpoint")
    if not isinstance(pointer, dict) or pointer.get("manifest_sha256") != expected_sha:
        raise ValueError("Cloud checkpoint pointer differs from its manifest")
    uri = pointer.get("uri")
    if (
        not isinstance(uri, str)
        or uri != metadata.get("remote_uri")
        or metadata.get("storage") != "gcs"
    ):
        raise ValueError("Cloud checkpoint storage identity differs")
    cloud_storage.split_uri(uri)
    if type(pointer.get("file_bytes")) is not int or not 0 < pointer["file_bytes"] <= MAX_BYTES:
        raise ValueError("ACT checkpoint exceeds the 4 GiB download limit")
    # Reserve one receipt entry; bounded depth/name rules match the ACT consumer.
    entries = set(files)
    for name in files:
        path = PurePosixPath(name)
        if len(name) > 1024 or any(
            part.startswith(".") or part.endswith((".tmp", ".partial")) or ":" in part
            for part in path.parts
        ):
            raise ValueError("Unsafe ACT checkpoint filename")
        entries.update(str(parent) for parent in path.parents if str(parent) != ".")
    if len(entries) + 2 > MAX_FILES or RESERVED in files:
        raise ValueError("ACT checkpoint has too many entries or a reserved receipt")
    return content, pointer, manifest


def publish_new(source, destination):
    """Atomic no-replace rename; an existing empty destination is not replaceable."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(source), os.fsencode(destination), 4)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise ValueError("Atomic checkpoint publication is unsupported on this host")
    if result:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code))


def _write(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def copy_remote(source, output, expected_sha, artifact_id):
    """Generation-pinned, bounded file copy; no extraction or original mutation."""
    source, output = Path(source), Path(output)
    original, pointer, manifest = descriptor(source, expected_sha)
    if output.exists() or output.is_symlink() or output.parent.is_symlink():
        raise ValueError("Checkpoint destination must be new in its job directory")
    if not output.parent.is_dir() or not output.is_absolute():
        raise ValueError("Checkpoint destination parent must already exist")
    if shutil.disk_usage(output.parent).free < pointer["file_bytes"] + 64 * 1024**2:
        raise ValueError("Not enough local disk space for the completed ACT checkpoint")
    remote_manifest, objects = cloud_storage.archive_plan(source)
    if remote_manifest != original:
        raise ValueError("Remote cloud checkpoint manifest differs from registration")
    if sum(blob.size for _, blob, _ in objects) != pointer["file_bytes"] or any(
        type(blob.size) is not int or not 0 <= blob.size <= MAX_FILE_BYTES for _, blob, _ in objects
    ):
        raise ValueError("ACT checkpoint file exceeds its committed byte limits")
    generations = {}
    for name, blob, _ in objects:
        generation = getattr(blob, "generation", None)
        if isinstance(generation, bool) or not re.fullmatch(r"[0-9]+", str(generation or "")):
            raise ValueError("Cloud checkpoint object generation is not pinned")
        generations[name] = str(generation)
    staging = Path(tempfile.mkdtemp(prefix=".act-checkpoint-", dir=output.parent))
    try:
        total = 0
        for name, blob, expected in objects:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            checksum, remaining = hashlib.sha256(), blob.size
            with (
                blob.open(
                    "rb",
                    chunk_size=cloud_storage.GCS_READ_AHEAD_BYTES,
                    timeout=60,
                    raw_download=True,
                ) as remote,
                path.open("xb") as stream,
            ):
                while remaining:
                    block = remote.read(min(cloud_storage.STREAM_CHUNK_BYTES, remaining))
                    if not block or len(block) > remaining:
                        raise ValueError("Cloud checkpoint file differs from its committed size")
                    remaining -= len(block)
                    total += len(block)
                    if total > pointer["file_bytes"] or total > MAX_BYTES:
                        raise ValueError("ACT checkpoint exceeds its committed total bytes")
                    checksum.update(block)
                    stream.write(block)
                if remote.read(1) or checksum.hexdigest() != expected:
                    raise ValueError("Cloud checkpoint checksum or size mismatch")
                stream.flush()
                os.fsync(stream.fileno())
        if total != pointer["file_bytes"]:
            raise ValueError("Cloud checkpoint total differs from its descriptor")
        if read_bytes(source / "manifest.json") != original:
            raise ValueError("Registered checkpoint changed during download")
        _, final_pointer, _ = descriptor(source, expected_sha)
        if final_pointer != pointer:
            raise ValueError("Registered checkpoint pointer changed during download")
        receipt = {
            "schema_version": 1,
            "source_artifact_id": artifact_id,
            "source_manifest_sha256": expected_sha,
            "remote_uri": pointer["uri"],
            "file_bytes": total,
            "object_generations": generations,
            "payload_files": manifest["files"],
        }
        evidence = cloud_storage.encoded(receipt)
        _write(staging / RESERVED, evidence)
        metadata = {
            key: value
            for key, value in manifest["metadata"].items()
            if key not in {"storage", "remote", "remote_uri"}
        }
        metadata["cloud_source"] = {
            key: receipt[key]
            for key in ("source_artifact_id", "source_manifest_sha256", "remote_uri", "file_bytes")
        }
        local_manifest = {
            **manifest,
            "metadata": metadata,
            "files": {**manifest["files"], RESERVED: hashlib.sha256(evidence).hexdigest()},
        }
        _write(staging / "manifest.json", cloud_storage.encoded(local_manifest))
        publish_new(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


if __name__ == "__main__":
    try:
        if len(sys.argv) != 5:
            raise ValueError("Invalid checkpoint download invocation")
        copy_remote(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
    except Exception:
        # Cloud SDK paths/credential diagnostics never enter browser-visible logs.
        print(
            "Completed ACT checkpoint download failed; "
            "verify private storage access, integrity and local disk space.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
