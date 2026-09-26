"""Fixed isolated Hub Parquet decoder. Executed as a script, never imported by core.

Only bounded bytes arrive on stdin. There are no dataset paths, imports, network
requests, or commands in its protocol. PyArrow runs in the CPU reader environment.
"""

import json
import math
import sys
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

READER_VERSION = "25.0.1"
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024

MAX_INDEX_ROWS = 100_000
MAX_SAMPLE_ROWS = 1_000_000
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_PROJECTED_VALUES = 4_000_000
MAX_INDEX_MATERIALIZED_BYTES = 8 * 1024 * 1024
MAX_TASKS_PER_EPISODE = 20
MAX_TASK_TEXT_BYTES = 4096
MAX_VECTOR_LENGTH = 1024
SAMPLE_COUNT = 5


class DecodeError(ValueError):
    pass


def integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise DecodeError(f"Dataset metadata has an invalid {field}")
    return value


def number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecodeError(f"Dataset metadata has an invalid {field}")
    if not math.isfinite(value) or value < 0:
        raise DecodeError(f"Dataset metadata has an invalid {field}")
    return float(value)


def parquet_file(raw: bytes) -> pq.ParquetFile:
    options = dict(
        arrow_extensions_enabled=False,
        thrift_string_size_limit=2 * 1024 * 1024,
        thrift_container_size_limit=100_000,
    )
    file = pq.ParquetFile(pa.BufferReader(raw), **options)
    if sum(file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups)) > (
        MAX_UNCOMPRESSED_BYTES
    ):
        raise DecodeError("Parquet data exceeds the decoded preview size limit")
    # Preserve encoded text dictionaries: repeated large strings must not be expanded
    # before their logical materialization cost is checked below.
    dictionaries = [column.path for column in file.schema if column.physical_type == "BYTE_ARRAY"]
    return pq.ParquetFile(pa.BufferReader(raw), read_dictionary=dictionaries, **options)


def check_projected_values(file: pq.ParquetFile, columns: list[str]) -> None:
    # Encoded row-group byte sizes do not bound RLE/dictionary expansion. Leaf-value
    # counts bound Arrow's allocation even when one list row contains huge repeated data.
    fields = file.schema_arrow
    leaves = []
    for index, column in enumerate(file.schema):
        for name in columns:
            if column.path == name or (
                column.path.startswith(name + ".") and is_list(fields.field(name).type)
            ):
                leaves.append(index)
                break
    count = sum(
        file.metadata.row_group(group).column(index).num_values
        for group in range(file.num_row_groups)
        for index in leaves
    )
    if count > MAX_PROJECTED_VALUES:
        raise DecodeError("Parquet columns exceed the decoded preview value limit")


def is_list(kind: pa.DataType) -> bool:
    return (
        pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind)
    )


def numeric(kind: pa.DataType) -> bool:
    return pa.types.is_integer(kind) or pa.types.is_floating(kind)


def list_values(array: pa.Array, limit: int, label: str) -> pa.Array:
    if not is_list(array.type):
        raise DecodeError(f"Dataset {label} must be a list")
    longest = (
        array.type.list_size
        if pa.types.is_fixed_size_list(array.type)
        else pc.max(pc.list_value_length(array)).as_py() or 0
    )
    if longest > limit:
        raise DecodeError(f"Dataset {label} exceeds the preview length limit")
    return array.flatten()


def task_materialization_bytes(array: pa.Array) -> int:
    values = list_values(array, MAX_TASKS_PER_EPISODE, "task list")
    strings = values.dictionary if pa.types.is_dictionary(values.type) else values
    if not (pa.types.is_string(strings.type) or pa.types.is_large_string(strings.type)):
        raise DecodeError("Dataset task descriptions must be text")
    lengths = pc.binary_length(strings)
    if (pc.max(lengths).as_py() or 0) > MAX_TASK_TEXT_BYTES:
        raise DecodeError("Dataset task text exceeds the preview length limit")
    if pa.types.is_dictionary(values.type):
        lengths = pc.take(lengths, values.indices)
    # Count repetitions, not dictionary storage, and include conservative Python
    # string/list overhead. This check runs before conversion to Python objects.
    return (pc.sum(lengths).as_py() or 0) * 4 + len(values) * 80 + len(array) * 64


def index_rows(raw: bytes) -> list[dict[str, Any]]:
    file = parquet_file(raw)
    if file.metadata.num_rows > MAX_INDEX_ROWS:
        raise DecodeError("Episode index exceeds the preview row limit")
    columns = [
        name
        for name in file.schema_arrow.names
        if name in {"episode_index", "length", "tasks", "data/chunk_index", "data/file_index"}
        or (
            name.startswith("videos/")
            and name.rsplit("/", 1)[-1]
            in {"chunk_index", "file_index", "from_timestamp", "to_timestamp"}
        )
    ]
    for name in columns:
        kind = file.schema_arrow.field(name).type
        if name == "tasks":
            if not is_list(kind):
                raise DecodeError("Episode tasks must be a list of text")
            value_kind = kind.value_type
            if pa.types.is_dictionary(value_kind):
                value_kind = value_kind.value_type
            if not (pa.types.is_string(value_kind) or pa.types.is_large_string(value_kind)):
                raise DecodeError("Episode tasks must be a list of text")
        elif not numeric(kind):
            raise DecodeError("Episode index contains an invalid scalar column")
    check_projected_values(file, columns)
    result = []
    materialized = 0
    # Nested task dictionaries can differ between row groups. Arrow cannot
    # coalesce those lists into one batch; decode each group separately while
    # keeping one cumulative materialization budget for the entire index.
    for group in range(file.num_row_groups):
        for batch in file.iter_batches(
            batch_size=32, row_groups=[group], columns=columns, use_threads=False
        ):
            materialized += batch.num_rows * (128 + len(columns) * 64)
            if "tasks" in columns:
                materialized += task_materialization_bytes(batch.column("tasks"))
            if materialized > MAX_INDEX_MATERIALIZED_BYTES:
                raise DecodeError("Episode index exceeds the materialized preview size limit")
            result.extend(batch.to_pylist())
    return result


def vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) > MAX_VECTOR_LENGTH
        or not all(
            not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item)
            for item in value
        )
    ):
        raise DecodeError("Sample data contains an invalid action or state vector")
    return [float(item) for item in value]


def frame_samples(raw: bytes, episode_index: int) -> list[dict[str, Any]]:
    file = parquet_file(raw)
    if file.metadata.num_rows > MAX_SAMPLE_ROWS:
        raise DecodeError("Frame shard exceeds the preview row limit")
    names = file.schema_arrow.names
    required = {"episode_index", "frame_index", "timestamp"}
    if not required.issubset(names):
        raise DecodeError("Frame data is missing episode, frame or timestamp columns")
    columns = [name for name in names if name in required | {"action", "observation.state"}]
    for name in columns:
        kind = file.schema_arrow.field(name).type
        if name in {"action", "observation.state"}:
            if not pa.types.is_null(kind) and (not is_list(kind) or not numeric(kind.value_type)):
                raise DecodeError("Sample vectors must contain numeric values")
        elif not numeric(kind):
            raise DecodeError("Frame data contains an invalid scalar column")
    check_projected_values(file, columns)
    result = []
    for batch in file.iter_batches(batch_size=32, columns=columns, use_threads=False):
        for name in {"action", "observation.state"}.intersection(columns):
            if not pa.types.is_null(batch.column(name).type):
                list_values(batch.column(name), MAX_VECTOR_LENGTH, "sample vector")
        batch = batch.filter(pc.equal(batch.column("episode_index"), episode_index))
        batch = batch.slice(0, SAMPLE_COUNT - len(result))
        for row in batch.to_pylist():
            result.append(
                dict(
                    frame_index=integer(row["frame_index"], "frame index"),
                    timestamp=number(row["timestamp"], "frame timestamp"),
                    action=vector(row.get("action")),
                    state=vector(row.get("observation.state")),
                )
            )
            if len(result) == SAMPLE_COUNT:
                return result
    return result


def main() -> None:
    try:
        if pa.__version__ != READER_VERSION:
            raise DecodeError(f"Expected pyarrow=={READER_VERSION}")
        operation = sys.argv[1]
        if operation not in {"index", "frames"}:
            raise DecodeError("Unsupported preview operation")
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise DecodeError("Dataset file exceeds the preview input size limit")
        rows = index_rows(raw) if operation == "index" else frame_samples(raw, int(sys.argv[2]))
        result = {"schema_version": 1, "rows": rows}
        output = json.dumps(result, allow_nan=False, ensure_ascii=True).encode()
        if len(output) > MAX_OUTPUT_BYTES:
            raise DecodeError("Reader response exceeds the output byte budget")
    except DecodeError as exc:
        output = json.dumps({"error": str(exc)[:300]}).encode()
    except Exception:
        output = b'{"error":"Dataset Parquet data is invalid or unsupported"}'
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()
