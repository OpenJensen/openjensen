"""Provisional local-only preview; deliberately outside the public worker protocol.

The core imports only the standard library. The same file is a fixed subprocess
entry point in the isolated CPU reader environment. No dataset files are copied.
"""

import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

READER_VERSION = "25.0.1"
FRAME_COLUMNS = (
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "action",
    "observation.state",
)
EPISODE_COLUMNS = (
    "episode_index",
    "tasks",
    "length",
    "dataset_from_index",
    "dataset_to_index",
    "data/chunk_index",
    "data/file_index",
)
WARNINGS = [
    "Bounded rows from one Parquet file only; not a complete dataset or episode validation.",
    "File hashes identify the bytes read, not an immutable full-dataset snapshot.",
    "Action units, reference frames, controller semantics and calibration are not verified.",
    "Policy, robot and simulator compatibility is not established.",
    "Videos and image columns are not decoded; omitted files are not checked for existence.",
]


class PreviewError(ValueError):
    """Stable local error codes, pending shared-contract review."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PreviewLimits:
    max_rows: int = 10
    max_file_bytes: int = 4 * 1024 * 1024
    max_read_bytes: int = 16 * 1024 * 1024
    max_decoded_bytes: int = 8 * 1024 * 1024
    max_output_bytes: int = 64 * 1024
    timeout_seconds: float = 10.0

    def __post_init__(self):
        ceilings = {
            "max_rows": 100,
            "max_file_bytes": 64 * 1024 * 1024,
            "max_read_bytes": 128 * 1024 * 1024,
            "max_decoded_bytes": 64 * 1024 * 1024,
            "max_output_bytes": 1024 * 1024,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            minimum = 1024 if name == "max_output_bytes" else 1
            if type(value) is not int or not minimum <= value <= ceiling:
                raise PreviewError("invalid_limits", f"{name} must be {minimum}..{ceiling}")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 30
        ):
            raise PreviewError("invalid_limits", "timeout_seconds must be finite and in (0, 30]")


def reader_python() -> Path:
    """Source-checkout default; installed applications must configure an explicit path."""
    root = Path(__file__).resolve().parents[5]
    return (
        root
        / "workers/_cpu_readers/.venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode()


def preview_local(
    dataset_path: str | Path,
    allowed_root: str | Path | None,
    *,
    kind: Literal["frames", "episodes"] = "frames",
    parquet_path: str | None = None,
    limits: PreviewLimits | None = None,
    python: str | Path | None = None,
) -> dict:
    """Read a deterministic file prefix into a provisional JSON-compatible structure.

    ``python`` is trusted application configuration, never a dataset field. Failure
    returns no preview. This does not change metadata-only intake or persist jobs.
    """
    limits = limits or PreviewLimits()
    if not allowed_root:
        raise PreviewError("local_disabled", "Configure FIREBIRD_LOCAL_DATA_ROOT first")
    _require_secure_opens()
    if kind not in ("frames", "episodes"):
        raise PreviewError("invalid_request", "kind must be frames or episodes")
    selected = parquet_path or (
        "data/chunk-000/file-000.parquet"
        if kind == "frames"
        else "meta/episodes/chunk-000/file-000.parquet"
    )
    request = {
        "root": str(allowed_root),
        "dataset": str(dataset_path),
        "kind": kind,
        "path": selected,
        "limits": asdict(limits),
    }
    # Reject malformed paths before launching a native reader as well as inside it.
    _paths(request)
    payload = _json_bytes(request)
    if len(payload) > 16384:
        raise PreviewError("invalid_request", "Preview request exceeds 16 KiB")
    executable = Path(python) if python is not None else reader_python()
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [str(executable), "-I", "-B", str(Path(__file__).resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise PreviewError("reader_unavailable", "Install the isolated CPU reader first") from exc
    output: list[bytes] = []

    def collect():
        data = process.stdout.read(limits.max_output_bytes + 1)
        output.append(data)
        if len(data) > limits.max_output_bytes:
            process.kill()

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()
    try:
        process.stdin.write(payload)
        process.stdin.close()
        remaining = limits.timeout_seconds - (time.monotonic() - started)
        process.wait(timeout=max(0, remaining))
        collector.join(timeout=max(0, limits.timeout_seconds - (time.monotonic() - started)))
        if collector.is_alive() or time.monotonic() - started > limits.timeout_seconds:
            raise subprocess.TimeoutExpired(process.args, limits.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise PreviewError("timeout", "Local preview exceeded its wall-clock budget") from exc
    except BrokenPipeError as exc:
        raise PreviewError(
            "reader_failed", "Isolated reader exited before accepting input"
        ) from exc
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        collector.join()
        process.stdout.close()
        process.stdin.close()
    raw = output[0]
    if len(raw) > limits.max_output_bytes:
        raise PreviewError("output_limit", "Reader response exceeds the output byte budget")
    try:
        response = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise PreviewError(
            "reader_failed", "Isolated reader returned no valid JSON result"
        ) from exc
    if not isinstance(response, dict):
        raise PreviewError("reader_failed", "Invalid isolated reader result")
    if "error" in response:
        raise PreviewError(response["error"]["code"], response["error"]["message"])
    if process.returncode or response.get("preview_schema_version") != 1:
        raise PreviewError("reader_failed", "Isolated reader failed or returned an unknown schema")
    return response


def _parts(path: str) -> Path:
    if not path or len(path) > 4096 or "\\" in path or "\0" in path:
        raise PreviewError("unsafe_path", "Invalid local path")
    parsed = Path(path)
    if ".." in parsed.parts:
        raise PreviewError("unsafe_path", "Traversal is not allowed")
    return parsed


def _paths(request: dict) -> tuple[Path, Path, Path]:
    root = _parts(request["root"])
    if not root.is_absolute():
        raise PreviewError("unsafe_path", "The configured local root must be absolute")
    dataset = _parts(request["dataset"])
    if dataset.is_absolute():
        try:
            dataset = dataset.relative_to(root)
        except ValueError as exc:
            raise PreviewError("unsafe_path", "Dataset must remain within the local root") from exc
    parquet = _parts(request["path"])
    prefix = ("data",) if request["kind"] == "frames" else ("meta", "episodes")
    if parquet.is_absolute() or parquet.parts[: len(prefix)] != prefix:
        raise PreviewError("unsafe_path", "Parquet path must stay in its declared dataset area")
    if parquet.suffix != ".parquet":
        raise PreviewError("unsafe_path", "Expected a .parquet file")
    return root, dataset, parquet


def _require_secure_opens():
    # Reject unsupported hosts before launching or depending on an installed reader.
    if (
        os.open not in os.supports_dir_fd
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise PreviewError("unsupported_platform", "Secure no-follow opens are unavailable")


def _open_beneath(root: Path, relative: Path):
    # Walk from the filesystem anchor; never resolve away a symlink before checking.
    _require_secure_opens()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root.anchor, directory_flags)
    try:
        components = (*root.parts[1:], *relative.parts)
        for component in components[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        fd = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise PreviewError("unsafe_path", "Only regular files may be previewed")
        return os.fdopen(fd, "rb", buffering=0)
    except FileNotFoundError as exc:
        raise PreviewError(
            "missing_file", "Requested dataset metadata or Parquet file is missing"
        ) from exc
    except OSError as exc:
        raise PreviewError("unsafe_path", "Unreadable path or symlink component rejected") from exc
    finally:
        os.close(descriptor)


class _BudgetedFile(io.RawIOBase):
    def __init__(self, handle, budget: dict):
        self.handle = handle
        self.budget = budget

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.handle.tell()

    def seek(self, offset, whence=0):
        return self.handle.seek(offset, whence)

    def read(self, size=-1):
        if size < 0:
            size = os.fstat(self.handle.fileno()).st_size - self.tell()
        if size > self.budget["remaining"]:
            raise PreviewError("read_limit", "Aggregate input read byte budget exceeded")
        data = self.handle.read(size)
        self.budget["remaining"] -= len(data)
        return data


def _identity(handle) -> tuple:
    info = os.fstat(handle.fileno())
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_preview(request: dict) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if pa.__version__ != READER_VERSION:
        raise PreviewError("reader_version", f"Expected pyarrow=={READER_VERSION}")
    limits = PreviewLimits(**request["limits"])
    root, dataset, path = _paths(request)
    budget = {"remaining": limits.max_read_bytes}
    with _open_beneath(root, dataset / "meta/info.json") as metadata:
        before = _identity(metadata)
        if before[2] > min(limits.max_file_bytes, 2 * 1024 * 1024):
            raise PreviewError("file_limit", "Metadata exceeds the file byte budget")
        raw = _BudgetedFile(metadata, budget).read(before[2])
        if _identity(metadata) != before:
            raise PreviewError("file_changed", "Metadata changed during preview")
    try:
        info = json.loads(raw)
        features = info["features"]
        if (
            not re.fullmatch(r"v3\.\d+(\.\d+)?", info["codebase_version"])
            or not isinstance(features, dict)
            or "action" not in features
            or not any(key.startswith("observation.") for key in features)
        ):
            raise ValueError("Expected LeRobot v3 feature metadata")
        for name, feature in features.items():
            if (
                not isinstance(feature, dict)
                or not isinstance(feature.get("dtype"), str)
                or not isinstance(feature.get("shape"), list)
                or any(type(n) is not int or n <= 0 for n in feature["shape"])
            ):
                raise ValueError(f"Invalid feature definition: {name}")
        if any(
            type(info[name]) is not int or info[name] < 0
            for name in ("total_episodes", "total_frames")
        ):
            raise ValueError("Invalid source-declared counts")
        if (
            type(info["fps"]) not in (int, float)
            or not math.isfinite(info["fps"])
            or info["fps"] <= 0
        ):
            raise ValueError("Invalid source-declared frame rate")
    except (ValueError, TypeError, KeyError) as exc:
        raise PreviewError("invalid_metadata", "Expected LeRobot v3 meta/info.json") from exc
    with _open_beneath(root, dataset / path) as handle:
        before = _identity(handle)
        if before[2] > limits.max_file_bytes:
            raise PreviewError("file_limit", "Parquet exceeds the file byte budget")
        bounded = _BudgetedFile(handle, budget)
        digest = hashlib.sha256()
        left = before[2]
        while left:
            block = bounded.read(min(left, 65536))
            if not block:
                raise PreviewError("file_changed", "Parquet was truncated during hashing")
            digest.update(block)
            left -= len(block)
        bounded.seek(0)
        parquet = pq.ParquetFile(
            bounded,
            memory_map=False,
            pre_buffer=False,
            arrow_extensions_enabled=False,
            thrift_string_size_limit=65536,
            thrift_container_size_limit=10000,
        )
        # iter_batches bounds rows, but compressed pages can expand first. Reject
        # files whose declared uncompressed column chunks exceed the decode budget.
        decoded = sum(
            parquet.metadata.row_group(g).column(c).total_uncompressed_size
            for g in range(parquet.metadata.num_row_groups)
            for c in range(parquet.metadata.row_group(g).num_columns)
        )
        if decoded < 0 or decoded > limits.max_decoded_bytes:
            raise PreviewError("decoded_limit", "Declared uncompressed Parquet exceeds the budget")
        names = parquet.schema_arrow.names
        if len(names) != len(set(names)):
            raise PreviewError("unsupported_columns", "Duplicate column names are unsupported")
        wanted = FRAME_COLUMNS if request["kind"] == "frames" else EPISODE_COLUMNS
        columns = [name for name in wanted if name in names]
        required = (
            {"episode_index", "frame_index", "action"}
            if request["kind"] == "frames"
            else {
                "episode_index",
                "length",
                "dataset_from_index",
                "dataset_to_index",
            }
        )
        if not required.issubset(columns):
            raise PreviewError("unsupported_columns", "Required preview columns are missing")
        for name in columns:
            dtype = parquet.schema_arrow.field(name).type
            if pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype):
                dtype = dtype.value_type
            if not (
                pa.types.is_integer(dtype)
                or pa.types.is_floating(dtype)
                or pa.types.is_boolean(dtype)
                or pa.types.is_string(dtype)
            ):
                raise PreviewError(
                    "unsupported_columns", "Preview supports scalar or flat-list values"
                )
        batches = parquet.iter_batches(
            batch_size=limits.max_rows,
            columns=columns,
            use_threads=False,
        )
        batch = next(batches, None)
        if batch is not None and batch.nbytes > limits.max_decoded_bytes:
            raise PreviewError("decoded_limit", "Decoded batch exceeds the byte budget")
        rows = [] if batch is None else batch.to_pylist()
        if _identity(handle) != before:
            raise PreviewError("file_changed", "Parquet changed during preview")
        result = {
            "preview_schema_version": 1,
            "inspection_scope": "bounded_parquet_rows",
            "source": "local",
            "format": "lerobot_v3",
            "kind": request["kind"],
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "file": {
                "path": path.as_posix(),
                "size_bytes": before[2],
                "sha256": digest.hexdigest(),
            },
            "reader": f"pyarrow=={READER_VERSION}",
            "total_file_rows": parquet.metadata.num_rows,
            "row_offset": 0,
            "rows": rows,
            "returned_rows": len(rows),
            "truncated": len(rows) < parquet.metadata.num_rows,
            "columns": columns,
            "omitted_columns": [name for name in names if name not in columns],
            "read_bytes": limits.max_read_bytes - budget["remaining"],
            "declared_decoded_bytes": decoded,
            "limits": asdict(limits),
            "warnings": WARNINGS,
        }
        parquet.close()
    return result


def _main():
    try:
        request = json.loads(sys.stdin.buffer.read(32768))
        result = _read_preview(request)
        raw = _json_bytes(result)
        if len(raw) > request["limits"]["max_output_bytes"]:
            raise PreviewError("output_limit", "Preview exceeds the JSON output byte budget")
    except PreviewError as exc:
        raw = _json_bytes({"error": {"code": exc.code, "message": str(exc)[:300]}})
    except ImportError:
        raw = _json_bytes(
            {"error": {"code": "reader_unavailable", "message": "Install pyarrow==25.0.1"}}
        )
    except Exception:
        raw = _json_bytes(
            {"error": {"code": "corrupt_payload", "message": "Invalid Parquet or row payload"}}
        )
    sys.stdout.buffer.write(raw)


if __name__ == "__main__":
    _main()
