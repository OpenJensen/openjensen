"""Explicit bounded local Parquet preview; independent of metadata-only API intake."""

import argparse
import hashlib
import json
import math
import os
import sys

from vla_platform.contracts import IntakeRequest
from vla_platform.datasets.inspect import profile
from vla_platform.datasets.local import confined_path, declared_files, read_local_metadata

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_FOOTER_BYTES = 256 * 1024
MAX_ROW_GROUP_BYTES = 4 * 1024 * 1024
MAX_ROW_GROUP_ROWS = 50_000
MAX_COLUMN_VALUES = 100_000
MAX_ROWS = 16
MAX_COLUMNS = 8
MAX_VECTOR_VALUES = 128
MAX_JSON_BYTES = 96 * 1024
DEFAULT_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "action",
    "observation.state",
    "index",
    "task_index",
)


def encode_preview(result: dict) -> str:
    encoded = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("Preview JSON exceeds the 96 KiB output limit")
    return encoded


def _value(value):
    if value is None or type(value) in {int, bool}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= 512:
        return value
    if isinstance(value, list) and len(value) <= MAX_VECTOR_VALUES:
        if all(type(item) in {int, float, bool} or item is None for item in value):
            return [_value(item) for item in value]
    raise ValueError("Unsupported, non-finite or oversized preview value")


def _supported_type(pa, dtype) -> bool:
    def primitive(kind):
        return pa.types.is_boolean(kind) or pa.types.is_integer(kind) or pa.types.is_floating(kind)

    if primitive(dtype) or pa.types.is_string(dtype):
        return True
    if pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype):
        return primitive(dtype.value_type) and (
            not pa.types.is_fixed_size_list(dtype) or dtype.list_size <= MAX_VECTOR_VALUES
        )
    return False


def _read_preview(target, columns, max_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with target.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > MAX_FILE_BYTES:
            raise ValueError("Parquet file exceeds the 32 MiB preview limit")
        snapshot = handle.read(MAX_FILE_BYTES + 1)
        if len(snapshot) > MAX_FILE_BYTES:
            raise ValueError("Parquet file exceeds the 32 MiB preview limit")
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(
            snapshot
        ) != before.st_size:
            raise ValueError("Parquet file changed during preview; retry with a stable source")
        if len(snapshot) < 12 or snapshot[:4] != b"PAR1":
            raise ValueError("Unsupported or invalid Parquet file header")
        footer = snapshot[-8:]
        footer_size = int.from_bytes(footer[:4], "little")
        if footer[4:] != b"PAR1" or not 0 < footer_size <= min(
            MAX_FOOTER_BYTES, before.st_size - 12
        ):
            raise ValueError("Invalid or oversized Parquet footer (limit 256 KiB)")
        # Hash and decode exactly the same bounded immutable bytes. No copy is
        # written to disk and source growth cannot turn hashing into an unbounded read.
        digest = hashlib.sha256(snapshot).hexdigest()
        parquet = pq.ParquetFile(
            pa.BufferReader(snapshot),
            memory_map=False,
            pre_buffer=False,
            buffer_size=0,
            thrift_string_size_limit=MAX_FOOTER_BYTES,
            thrift_container_size_limit=16_384,
            arrow_extensions_enabled=False,
        )
        metadata = parquet.metadata
        if metadata.num_columns > 128 or metadata.num_row_groups > 1024:
            raise ValueError("Parquet schema or row-group count exceeds the preview limit")
        schema = parquet.schema_arrow
        if len(set(schema.names)) != len(schema.names) or any(
            len(name) > 128 for name in schema.names
        ):
            raise ValueError("Duplicate or oversized Parquet column names")
        selected = columns or [name for name in DEFAULT_COLUMNS if name in schema.names]
        if not selected or len(selected) > MAX_COLUMNS or len(set(selected)) != len(selected):
            raise ValueError("Select one to eight unique preview columns")
        for name in selected:
            if name not in schema.names or not _supported_type(pa, schema.field(name).type):
                raise ValueError(f"Missing or unsupported preview column: {name}")
        omitted = [name for name in schema.names if name not in selected]
        rows = []
        group_rows = 0
        if metadata.num_row_groups:
            group = metadata.row_group(0)
            group_rows = group.num_rows
            if not 0 <= group.num_rows <= MAX_ROW_GROUP_ROWS or not (
                0 <= group.total_byte_size <= MAX_ROW_GROUP_BYTES
            ):
                raise ValueError("First Parquet row group exceeds the decoder budget")
            # Validate ALL column chunks of this group before opening a decoder. A tiny
            # batch size alone does not bound dictionary/page allocations.
            total_bytes = 0
            for index in range(group.num_columns):
                chunk = group.column(index)
                if chunk.file_path:
                    raise ValueError("External Parquet column files are unsupported")
                if not 0 <= chunk.num_values <= MAX_COLUMN_VALUES or not (
                    0 <= chunk.total_uncompressed_size <= MAX_ROW_GROUP_BYTES
                ):
                    raise ValueError("Parquet column chunk exceeds the decoder budget")
                total_bytes += chunk.total_uncompressed_size
            if total_bytes > MAX_ROW_GROUP_BYTES:
                raise ValueError("First Parquet row group exceeds the decoder budget")
            batch = next(
                parquet.iter_batches(
                    batch_size=max_rows, row_groups=[0], columns=selected, use_threads=False
                ),
                None,
            )
            if batch is not None:
                rows = [
                    {name: _value(value) for name, value in row.items()}
                    for row in batch.to_pylist()
                ]
        return {
            "file_sha256": digest,
            "file_bytes": before.st_size,
            "file_rows": metadata.num_rows,
            "row_group": 0 if metadata.num_row_groups else None,
            "row_group_rows": group_rows,
            "columns": selected,
            "omitted_columns": omitted,
            "rows": rows,
            "sample_truncated": len(rows) < metadata.num_rows,
            "decoder": f"pyarrow/{pa.__version__}",
        }


def preview_local(
    path: str,
    allowed_root: str | None,
    parquet_path: str,
    *,
    max_rows: int = 8,
    columns: list[str] | None = None,
) -> dict:
    """Read the first bounded row group of one explicit local file, without scanning."""
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise ValueError("Preview row limit must be between 1 and 16")
    if columns is not None and (
        not isinstance(columns, list)
        or not 1 <= len(columns) <= MAX_COLUMNS
        or any(not isinstance(name, str) or len(name) > 128 for name in columns)
    ):
        raise ValueError("Select one to eight bounded preview column names")
    dataset, raw, info = read_local_metadata(path, allowed_root)
    digest = hashlib.sha256(raw).hexdigest()
    metadata_profile = profile(
        raw, IntakeRequest(source="local", path=path), f"metadata-sha256:{digest}"
    )
    target = confined_path(dataset, parquet_path)
    checks = declared_files(dataset, info)
    result = {
        "schema_version": 1,
        "inspection_scope": "bounded_local_parquet_preview",
        "dataset_path": str(dataset),
        "format": metadata_profile.format,
        "metadata_sha256": digest,
        "file": str(target.relative_to(dataset)).replace("\\", "/"),
        "declared_file_checks": checks,
        "file_check_scope": "Initial declared shard and up to eight video paths only; no inventory",
        "video_decode": "unsupported",
        "semantics": "unverified: action units, reference frames, controller and calibration",
        "compatibility": "not established",
        "warnings": [
            "Rows are a bounded sample, not a complete episode or dataset validation.",
            "No video, image or binary payload is decoded; no policy compatibility is inferred.",
        ],
    }
    if not target.is_file():
        result.update(status="missing", rows=[], warnings=["Selected Parquet file is missing"])
    else:
        if target.suffix.lower() != ".parquet":
            raise ValueError("Preview requires an explicit .parquet file")
        result.update(_read_preview(target, columns, max_rows))
        result["missing_feature_columns"] = [
            key
            for key, value in info["features"].items()
            if value["dtype"] != "video"
            and key not in result["columns"] + result["omitted_columns"]
        ]
        episodes = [row.get("episode_index") for row in result["rows"]]
        known_episodes = bool(episodes) and all(
            type(value) is int and value >= 0 for value in episodes
        )
        result["episode_indices"] = sorted(set(episodes)) if known_episodes else []
        result["episode_identity"] = "sample_column" if known_episodes else "unavailable"
        result["status"] = (
            "previewed"
            if result["rows"]
            and not result["missing_feature_columns"]
            and not any(check["status"] == "missing" for check in checks)
            else "incomplete"
        )
        if not known_episodes:
            result["warnings"].append("Episode identity is unavailable in the selected sample")
    encode_preview(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Dataset path beneath the allowed root")
    parser.add_argument("--root", default=os.getenv("FIREBIRD_LOCAL_DATA_ROOT"))
    parser.add_argument("--parquet", required=True, help="Explicit file path relative to dataset")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--column", action="append", dest="columns")
    args = parser.parse_args()
    try:
        result = preview_local(
            args.path, args.root, args.parquet, max_rows=args.rows, columns=args.columns
        )
        print(encode_preview(result))
        return 0 if result["status"] == "previewed" else 2
    except (OSError, ValueError, NotImplementedError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)[:500]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
