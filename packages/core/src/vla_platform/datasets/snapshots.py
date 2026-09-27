"""Complete local LeRobot snapshots. Native readers stay outside the app process."""

import ctypes
import errno
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .local_preview import PreviewError, _identity, _open_beneath, reader_python

MANIFEST = "firebird-snapshot.json"
HEX = re.compile(r"[0-9a-f]{64}\Z")
FILE = re.compile(
    r"(?:meta/(?:info|stats|firebird-lineage|firebird-demonstrations)\.json|"
    r"meta/tasks\.parquet|(?:data|meta/episodes)/chunk-[0-9]{3,}/file-[0-9]{3,}\.parquet|"
    r"videos/[a-zA-Z0-9_.-]+/chunk-[0-9]{3,}/file-[0-9]{3,}\.mp4|README\.md|\.gitattributes)\Z"
)


class SnapshotError(ValueError):
    """Actionable validation failure before any training submission."""


@dataclass(frozen=True)
class SnapshotLimits:
    max_files: int = 4096
    max_bytes: int = 16 * 1024**3
    max_file_bytes: int = 4 * 1024**3
    max_rows: int = 2_000_000
    max_episodes: int = 20_000
    max_decoded_bytes: int = 256 * 1024**2
    timeout_seconds: int = 120

    def __post_init__(self):
        ceilings = {
            "max_files": 4096,
            "max_bytes": 16 * 1024**3,
            "max_file_bytes": 4 * 1024**3,
            "max_rows": 2_000_000,
            "max_episodes": 20_000,
            "max_decoded_bytes": 256 * 1024**2,
            "timeout_seconds": 120,
        }
        for name, maximum in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise SnapshotError(f"{name} must be an integer in 1..{maximum}")


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(root, name, maximum=8 * 1024**2):
    with _open_beneath(root, Path(name)) as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise SnapshotError(f"Metadata exceeds byte limit: {name}")
    try:
        return json.loads(raw, object_pairs_hook=_unique), raw
    except (ValueError, UnicodeError) as exc:
        raise SnapshotError(f"Invalid JSON: {name}") from exc


def _inventory(root, limits, *, manifest=False):
    # lstat rejects special nodes before any open; descriptor-relative opens below
    # recheck every ancestor so a directory swap cannot follow a symlink.
    found = {}
    pending = [root]
    total = 0
    directories = 0
    while pending:
        directory = pending.pop()
        if directory.is_symlink():
            raise SnapshotError("Dataset directories must not be symlinks")
        for path in directory.iterdir():
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                pending.append(path)
                directories += 1
                if directories > limits.max_files:
                    raise SnapshotError("Dataset has too many directories")
                continue
            name = path.relative_to(root).as_posix()
            if not stat.S_ISREG(mode):
                raise SnapshotError(f"Dataset contains a symlink or special file: {name}")
            if name == MANIFEST and manifest:
                continue
            if not FILE.fullmatch(name):
                raise SnapshotError(f"Unsupported or unfinished dataset file: {name}")
            with _open_beneath(root, Path(name)) as handle:
                identity = _identity(handle)
            if identity[2] <= 0 or identity[2] > limits.max_file_bytes:
                raise SnapshotError(f"Empty or oversized dataset file: {name}")
            total += identity[2]
            found[name] = identity
            if len(found) > limits.max_files or total > limits.max_bytes:
                raise SnapshotError("Dataset exceeds snapshot file or byte budget")
    return found


def _copy(root, destination, inventory):
    files = []
    for name, identity in sorted(inventory.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with _open_beneath(root, Path(name)) as source, target.open("xb") as output:
            if _identity(source) != identity:
                raise SnapshotError(f"Dataset changed during snapshot: {name}")
            remaining = identity[2]
            while remaining:
                chunk = source.read(min(1024**2, remaining))
                if not chunk:
                    raise SnapshotError(f"Dataset truncated during snapshot: {name}")
                remaining -= len(chunk)
                digest.update(chunk)
                output.write(chunk)
            if source.read(1):
                raise SnapshotError(f"Dataset grew during snapshot: {name}")
            if _identity(source) != identity:
                raise SnapshotError(f"Dataset changed during snapshot: {name}")
            output.flush()
            os.fsync(output.fileno())
        files.append({"path": name, "size": identity[2], "sha256": digest.hexdigest()})
    return files


@contextmanager
def _defer_signals():
    # The isolated intake worker invokes this on its main thread. Preserve its
    # SIGTERM handler while protecting the Popen-handle and cleanup windows.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    pending = []
    try:
        for number in handlers:
            signal.signal(number, lambda signum, frame: pending.append(signum))
        yield
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        if pending:
            handler = handlers[pending[0]]
            if callable(handler):
                handler(pending[0], None)
            elif handler == signal.SIG_DFL:
                signal.raise_signal(pending[0])


def _validate(root, limits, python):
    command = [
        str(python or reader_python()),
        "-I",
        "-B",
        str(Path(__file__).with_name("snapshot_reader.py")),
        str(root),
    ]
    process = None
    collector = None
    output = []
    try:
        with _defer_signals():
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            def collect():
                output.append(process.stdout.read(4 * 1024**2 + 1))
                if len(output[0]) > 4 * 1024**2:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            collector = threading.Thread(target=collect, daemon=True)
            collector.start()
        process.stdin.write(canonical(asdict(limits)))
        process.stdin.close()
        process.wait(timeout=limits.timeout_seconds)
        collector.join(timeout=1)
    except subprocess.TimeoutExpired as exc:
        raise SnapshotError("Complete dataset validation exceeded its wall-clock budget") from exc
    except (OSError, BrokenPipeError) as exc:
        raise SnapshotError(
            "Isolated dataset reader is unavailable or exited before reading input"
        ) from exc
    finally:
        with _defer_signals():
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                if collector is not None:
                    collector.join(timeout=5)
                process.stdout.close()
                process.stdin.close()
    if not output or len(output[0]) > 4 * 1024**2:
        raise SnapshotError("Dataset validation result exceeds byte budget")
    try:
        result = json.loads(output[0])
    except (ValueError, UnicodeError) as exc:
        raise SnapshotError(
            "Isolated dataset validator failed; check CPU reader installation"
        ) from exc
    if not isinstance(result, dict):
        raise SnapshotError("Invalid isolated dataset validation result")
    if process.returncode or "error" in result:
        raise SnapshotError(result.get("error", "Isolated dataset validator failed"))
    return result


def descriptor(manifest, digest):
    return {
        "schema_version": 1,
        "id": "sha256:" + digest,
        "manifest_sha256": digest,
        "format": "lerobot_v3",
        "total_bytes": sum(item["size"] for item in manifest["files"]),
        "file_count": len(manifest["files"]),
        **{
            key: manifest[key]
            for key in ("total_episodes", "total_frames", "lineage_validated", "warnings")
        },
    }


def verify_snapshot(root, expected_sha256, *, limits=SnapshotLimits()):
    """Rehash the entire package; a caller-provided location is never accepted by the API."""
    if not isinstance(expected_sha256, str) or not HEX.fullmatch(expected_sha256):
        raise SnapshotError("Invalid snapshot identity")
    root = Path(root)
    manifest, raw = read_json(root, MANIFEST)
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or canonical(manifest) != raw:
        raise SnapshotError("Snapshot manifest identity mismatch")
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise SnapshotError("Unsupported snapshot schema")
    inventory = _inventory(root, limits, manifest=True)
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != len(inventory):
        raise SnapshotError("Snapshot file inventory mismatch")
    expected = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise SnapshotError("Invalid snapshot inventory entry")
        name = item["path"]
        if not isinstance(name, str) or name in expected or name not in inventory:
            raise SnapshotError("Snapshot file inventory mismatch")
        if type(item["size"]) is not int or item["size"] != inventory[name][2]:
            raise SnapshotError(f"Snapshot file size mismatch: {name}")
        digest = hashlib.sha256()
        with _open_beneath(root, Path(name)) as handle:
            before = _identity(handle)
            if before != inventory[name]:
                raise SnapshotError(f"Snapshot changed before reading: {name}")
            remaining = item["size"]
            while remaining:
                chunk = handle.read(min(1024**2, remaining))
                if not chunk:
                    raise SnapshotError(f"Snapshot truncated while reading: {name}")
                remaining -= len(chunk)
                digest.update(chunk)
            if handle.read(1) or before != _identity(handle):
                raise SnapshotError(f"Snapshot changed while reading: {name}")
        if digest.hexdigest() != item["sha256"]:
            raise SnapshotError(f"Snapshot file hash mismatch: {name}")
        expected[name] = item
    if _inventory(root, limits, manifest=True) != inventory:
        raise SnapshotError("Snapshot changed while verifying")
    return manifest


def stage_snapshot(source, destination, expected_sha256, *, limits=SnapshotLimits()):
    """Copy only validated bytes into a new job-owned directory, never hardlink originals."""
    source, destination = Path(source), Path(destination)
    manifest = verify_snapshot(source, expected_sha256, limits=limits)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        files = _copy(source, destination, _inventory(source, limits, manifest=True))
        if files != manifest["files"]:
            raise SnapshotError("Snapshot changed during staging")
        (destination / MANIFEST).write_bytes(canonical(manifest))
        verify_snapshot(destination, expected_sha256, limits=limits)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination


def training_split(root, digest, validation_fraction, seed):
    """Admission-time lineage partition; duplicated in the isolated native worker."""
    manifest = verify_snapshot(root, digest)
    return split_lineage(manifest, validation_fraction, seed)


def resolve_snapshot(store, value):
    digest = value.get("manifest_sha256")
    if (
        not isinstance(digest, str)
        or not HEX.fullmatch(digest)
        or value.get("id") != "sha256:" + digest
    ):
        raise SnapshotError("Invalid snapshot descriptor")
    root = Path(store) / digest
    if descriptor(verify_snapshot(root, digest), digest) != value:
        raise SnapshotError("Snapshot descriptor differs from immutable manifest")
    return root


def _publish(source, destination):
    """Atomic no-replace publication on the supported secure-open platforms."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(source), os.fsencode(destination), 4)
    elif sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise SnapshotError("Atomic no-replace dataset publication is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise SnapshotError("Atomic local dataset publication requires Linux or macOS")
    if result:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code))
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_snapshot(allowed_root, dataset_path, store, *, python=None, limits=SnapshotLimits()):
    """Freeze, validate and publish one local dataset without altering its original files."""
    allowed_root, dataset_path, store = Path(allowed_root), Path(dataset_path), Path(store)
    if not allowed_root.is_absolute() or ".." in allowed_root.parts or ".." in dataset_path.parts:
        raise SnapshotError("Dataset must be contained by the configured local root")
    root = dataset_path if dataset_path.is_absolute() else allowed_root / dataset_path
    if not root.is_relative_to(allowed_root):
        raise SnapshotError("Dataset is outside the configured local root")
    store.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=store))
    try:
        inventory = _inventory(root, limits)
        files = _copy(root, staging, inventory)
        if _inventory(root, limits) != inventory:
            raise SnapshotError("Dataset changed during snapshot; finish recording and retry")
        validation = _validate(staging, limits, python)
        if _inventory(root, limits) != inventory:
            raise SnapshotError("Dataset changed during snapshot validation")
        manifest = {"schema_version": 1, "format": "lerobot_v3", "files": files, **validation}
        raw = canonical(manifest)
        digest = hashlib.sha256(raw).hexdigest()
        with (staging / MANIFEST).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        verify_snapshot(staging, digest, limits=limits)
        destination = store / digest
        try:
            _publish(staging, destination)
        except FileExistsError:
            verify_snapshot(destination, digest, limits=limits)
        return descriptor(manifest, digest)
    except PreviewError as exc:
        raise SnapshotError(str(exc)) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def split_lineage(manifest, fraction, seed):
    if type(fraction) not in (float, int) or not 0 < fraction < 1 or type(seed) is not int:
        raise ValueError("Invalid local dataset split controls")
    groups = {}
    lineage = manifest.get("lineage")
    count = manifest.get("total_episodes")
    if not isinstance(lineage, list) or type(count) is not int or len(lineage) != count:
        raise ValueError("Incomplete dataset lineage")
    for index, row in enumerate(lineage):
        if (
            row.get("episode_index") != index
            or type(row.get("episode_index")) is not int
            or not isinstance(row.get("lineage_group"), str)
        ):
            raise ValueError("Invalid dataset lineage")
        groups.setdefault(row["lineage_group"], []).append(index)
    if len(groups) < 2:
        raise ValueError("Training needs at least two distinct lineage groups for a held-out split")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    validation = set(names[: max(1, min(len(names) - 1, math.ceil(len(names) * fraction)))])
    return {
        "train": sorted(
            index
            for group, indices in groups.items()
            if group not in validation
            for index in indices
        ),
        "validation": sorted(
            index for group, indices in groups.items() if group in validation for index in indices
        ),
        "lineage_validated": manifest["lineage_validated"],
        "train_groups": sorted(set(groups) - validation),
        "validation_groups": sorted(validation),
    }
