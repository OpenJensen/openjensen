"""Small, on-demand views of public LeRobot datasets at an inspected Hub commit.

Metadata intake remains metadata-only. Exploration downloads bounded metadata and
Parquet files; camera URLs point directly at the pinned Hub video for browser streaming.
"""

import asyncio
import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import PurePosixPath
from string import Formatter
from typing import Any
from urllib.parse import quote, urljoin

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vla_platform.contracts import (
    CameraPreview,
    DatasetProfile,
    EpisodePage,
    EpisodePreview,
    EpisodeSummary,
    FrameSample,
    Job,
)

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_PARQUET_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 48 * 1024 * 1024
MAX_INDEX_FILES = 16
MAX_INDEX_ROWS = 100_000
MAX_SAMPLE_ROWS = 1_000_000
MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_PROJECTED_VALUES = 4_000_000
MAX_INDEX_MATERIALIZED_BYTES = 8 * 1024 * 1024
MAX_TASKS_PER_EPISODE = 20
MAX_TASK_TEXT_BYTES = 4096
MAX_VECTOR_LENGTH = 1024
SAMPLE_COUNT = 5


class ExplorationError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def source_profile(job: Job) -> DatasetProfile:
    if job.status != "succeeded" or job.result is None:
        raise ExplorationError(
            "Finish a successful dataset inspection before exploring episodes", 409
        )
    result = job.result
    if result.source != "huggingface":
        raise ExplorationError(
            "Episode previews currently support public Hugging Face datasets only"
        )
    if not result.repo_id or not re.fullmatch(r"[\w.-]+/[\w.-]+", result.repo_id):
        raise ExplorationError("The inspection has no valid Hugging Face repository")
    if any(part in {".", ".."} for part in result.repo_id.split("/")):
        raise ExplorationError("The inspection has no valid Hugging Face repository")
    if not re.fullmatch(r"[a-f0-9]{40}", result.revision):
        raise ExplorationError("Episode previews require an immutable inspected dataset revision")
    return result


def safe_path(path: str) -> str:
    if (
        not isinstance(path, str)
        or not path
        or len(path) > 1024
        or path.startswith("/")
        or any(part in {".", "..", ""} for part in path.split("/"))
        or any(char in path for char in "\\?#%:")
        or any(ord(char) < 32 for char in path)
    ):
        raise ExplorationError("Dataset metadata contains an unsafe file path")
    return str(PurePosixPath(path))


def render_path(template: Any, **values: Any) -> str:
    if not isinstance(template, str) or len(template) > 512:
        raise ExplorationError("Dataset metadata has no supported file path template")
    try:
        for _, name, spec, conversion in Formatter().parse(template):
            if name is not None and (
                name not in values
                or conversion is not None
                or not re.fullmatch(r"(?:0?[1-9]?[0-9]?d)?", spec)
            ):
                raise ValueError("unsupported placeholder")
        return safe_path(template.format(**values))
    except (ValueError, KeyError, IndexError) as exc:
        raise ExplorationError("Dataset metadata has an unsupported file path template") from exc


def hub_url(source: DatasetProfile, path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{quote(source.repo_id or '', safe='/')}/"
        f"resolve/{source.revision}/{quote(safe_path(path), safe='/')}"
    )


def allowed_url(url: str | httpx.URL) -> bool:
    parsed = httpx.URL(url)
    host = parsed.host
    return (
        parsed.scheme == "https"
        and parsed.port in {None, 443}
        and not parsed.username
        and not parsed.password
        and (
            host == "huggingface.co"
            or host.endswith(".huggingface.co")
            or host == "hf.co"
            or host.endswith(".hf.co")
        )
    )


@dataclass
class Budget:
    used: int = 0


class HubReader:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.cache_size = 0

    async def read(self, url: str, limit: int, budget: Budget, *, head: bool = False) -> bytes:
        original = url
        if not head and url in self.cache:
            data = self.cache[url]
            self.cache.move_to_end(url)
            if len(data) > limit:
                raise ExplorationError("Dataset file exceeds the preview size limit")
            budget.used += len(data)
            if budget.used > MAX_REQUEST_BYTES:
                raise ExplorationError("Dataset preview exceeds the total download limit")
            return data
        for _ in range(6):
            if not allowed_url(url):
                raise ExplorationError("Dataset file redirects outside Hugging Face storage")
            async with self.client.stream("HEAD" if head else "GET", url) as response:
                if response.is_redirect:
                    url = urljoin(str(response.url), response.headers.get("location", ""))
                    continue
                if response.status_code in {401, 403}:
                    raise ExplorationError("This preview is gated or private; use a public dataset")
                if response.status_code == 404:
                    raise ExplorationError("Dataset preview file was not found", 404)
                response.raise_for_status()
                if head:
                    return b""
                if int(response.headers.get("content-length", "0")) > limit:
                    raise ExplorationError("Dataset file exceeds the preview size limit")
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    data.extend(chunk)
                    budget.used += len(chunk)
                    if len(data) > limit or budget.used > MAX_REQUEST_BYTES:
                        raise ExplorationError("Dataset preview exceeds the download size limit")
                result = bytes(data)
                # Keep small, immutable metadata/Parquet files only, with a total memory cap.
                if len(result) <= MAX_INDEX_BYTES:
                    # Another request may have cached this URL while this download awaited I/O.
                    previous = self.cache.pop(original, None)
                    if previous is not None:
                        self.cache_size -= len(previous)
                    while self.cache and self.cache_size + len(result) > MAX_PARQUET_BYTES:
                        _, old = self.cache.popitem(last=False)
                        self.cache_size -= len(old)
                    self.cache[original] = result
                    self.cache_size += len(result)
                return result
        raise ExplorationError("Dataset file exceeded the redirect limit")


def integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ExplorationError(f"Dataset metadata has an invalid {field}")
    return value


def number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExplorationError(f"Dataset metadata has an invalid {field}")
    if not math.isfinite(value) or value < 0:
        raise ExplorationError(f"Dataset metadata has an invalid {field}")
    return float(value)


def parquet_file(raw: bytes) -> pq.ParquetFile:
    options = dict(
        thrift_string_size_limit=2 * 1024 * 1024,
        thrift_container_size_limit=100_000,
    )
    file = pq.ParquetFile(pa.BufferReader(raw), **options)
    if sum(file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups)) > (
        MAX_UNCOMPRESSED_BYTES
    ):
        raise ExplorationError("Parquet data exceeds the decoded preview size limit")
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
        raise ExplorationError("Parquet columns exceed the decoded preview value limit")


def is_list(kind: pa.DataType) -> bool:
    return (
        pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind)
    )


def numeric(kind: pa.DataType) -> bool:
    return pa.types.is_integer(kind) or pa.types.is_floating(kind)


def list_values(array: pa.Array, limit: int, label: str) -> pa.Array:
    if not is_list(array.type):
        raise ExplorationError(f"Dataset {label} must be a list")
    longest = (
        array.type.list_size
        if pa.types.is_fixed_size_list(array.type)
        else pc.max(pc.list_value_length(array)).as_py() or 0
    )
    if longest > limit:
        raise ExplorationError(f"Dataset {label} exceeds the preview length limit")
    return array.flatten()


def task_materialization_bytes(array: pa.Array) -> int:
    values = list_values(array, MAX_TASKS_PER_EPISODE, "task list")
    strings = values.dictionary if pa.types.is_dictionary(values.type) else values
    if not (pa.types.is_string(strings.type) or pa.types.is_large_string(strings.type)):
        raise ExplorationError("Dataset task descriptions must be text")
    lengths = pc.binary_length(strings)
    if (pc.max(lengths).as_py() or 0) > MAX_TASK_TEXT_BYTES:
        raise ExplorationError("Dataset task text exceeds the preview length limit")
    if pa.types.is_dictionary(values.type):
        lengths = pc.take(lengths, values.indices)
    # Count repetitions, not dictionary storage, and include conservative Python
    # string/list overhead. This check runs before conversion to Python objects.
    return (pc.sum(lengths).as_py() or 0) * 4 + len(values) * 80 + len(array) * 64


def index_rows(raw: bytes) -> list[dict[str, Any]]:
    file = parquet_file(raw)
    if file.metadata.num_rows > MAX_INDEX_ROWS:
        raise ExplorationError("Episode index exceeds the preview row limit")
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
                raise ExplorationError("Episode tasks must be a list of text")
            value_kind = kind.value_type
            if pa.types.is_dictionary(value_kind):
                value_kind = value_kind.value_type
            if not (pa.types.is_string(value_kind) or pa.types.is_large_string(value_kind)):
                raise ExplorationError("Episode tasks must be a list of text")
        elif not numeric(kind):
            raise ExplorationError("Episode index contains an invalid scalar column")
    check_projected_values(file, columns)
    result = []
    materialized = 0
    for batch in file.iter_batches(batch_size=32, columns=columns, use_threads=False):
        materialized += batch.num_rows * (128 + len(columns) * 64)
        if "tasks" in columns:
            materialized += task_materialization_bytes(batch.column("tasks"))
        if materialized > MAX_INDEX_MATERIALIZED_BYTES:
            raise ExplorationError("Episode index exceeds the materialized preview size limit")
        result.extend(batch.to_pylist())
    return result


def summary(row: dict[str, Any], fps: float) -> EpisodeSummary:
    index = integer(row.get("episode_index"), "episode index")
    length = integer(row.get("length"), "episode length")
    tasks = row.get("tasks", [])
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ExplorationError("Dataset metadata has invalid episode task descriptions")
    return EpisodeSummary(
        episode_index=index,
        frame_count=length,
        duration_seconds=length / fps,
        tasks=tasks[:20],
    )


def feature_names(features: dict[str, Any], key: str) -> list[str]:
    names = features.get(key, {}).get("names")
    # v2 datasets can encode names by axis rather than as a flat list.
    if isinstance(names, dict):
        names = next((value for value in names.values() if isinstance(value, list)), [])
    return names if isinstance(names, list) and all(isinstance(n, str) for n in names) else []


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
        raise ExplorationError("Sample data contains an invalid action or state vector")
    return [float(item) for item in value]


def frame_samples(raw: bytes, episode_index: int) -> list[FrameSample]:
    file = parquet_file(raw)
    if file.metadata.num_rows > MAX_SAMPLE_ROWS:
        raise ExplorationError("Frame shard exceeds the preview row limit")
    names = file.schema_arrow.names
    required = {"episode_index", "frame_index", "timestamp"}
    if not required.issubset(names):
        raise ExplorationError("Frame data is missing episode, frame or timestamp columns")
    columns = [name for name in names if name in required | {"action", "observation.state"}]
    for name in columns:
        kind = file.schema_arrow.field(name).type
        if name in {"action", "observation.state"}:
            if not pa.types.is_null(kind) and (not is_list(kind) or not numeric(kind.value_type)):
                raise ExplorationError("Sample vectors must contain numeric values")
        elif not numeric(kind):
            raise ExplorationError("Frame data contains an invalid scalar column")
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
                FrameSample(
                    frame_index=integer(row["frame_index"], "frame index"),
                    timestamp=number(row["timestamp"], "frame timestamp"),
                    action=vector(row.get("action")),
                    state=vector(row.get("observation.state")),
                )
            )
            if len(result) == SAMPLE_COUNT:
                return result
    return result


class DatasetExplorer:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=20, follow_redirects=False)
        self.reader = HubReader(self.client)
        self.slots = asyncio.Semaphore(2)

    async def close(self) -> None:
        await self.client.aclose()

    async def metadata(self, source: DatasetProfile, budget: Budget) -> dict[str, Any]:
        raw = await self.reader.read(hub_url(source, "meta/info.json"), MAX_JSON_BYTES, budget)
        if hashlib.sha256(raw).hexdigest() != source.metadata_sha256:
            raise ExplorationError("Dataset metadata no longer matches the inspected snapshot")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ExplorationError("Dataset metadata is not an object")
        return result

    async def episodes(
        self,
        source: DatasetProfile,
        budget: Budget,
        start: int,
        count: int,
        *,
        selected: int | None = None,
    ) -> list[dict[str, Any]]:
        if source.format == "lerobot_v2":
            raw = await self.reader.read(
                hub_url(source, "meta/episodes.jsonl"), MAX_INDEX_BYTES, budget
            )
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len(rows) > MAX_INDEX_ROWS or not all(isinstance(row, dict) for row in rows):
                raise ExplorationError("Episode index exceeds the preview row limit or is invalid")
            rows.sort(key=lambda row: integer(row.get("episode_index"), "episode index"))
            return (
                [row for row in rows if row["episode_index"] == selected]
                if (selected is not None)
                else rows[start : start + count]
            )
        repo = quote(source.repo_id or "", safe="/")
        listing_url = (
            f"https://huggingface.co/api/datasets/{repo}/tree/{source.revision}/"
            "meta/episodes?recursive=true&limit=1000"
        )
        listing = json.loads(await self.reader.read(listing_url, MAX_JSON_BYTES, budget))
        if not isinstance(listing, list):
            raise ExplorationError("Episode index listing is invalid")
        files = sorted(
            item["path"]
            for item in listing
            if isinstance(item, dict)
            and item.get("type") == "file"
            and isinstance(item.get("path"), str)
            and re.fullmatch(r"meta/episodes/chunk-\d+/file-\d+\.parquet", item["path"])
        )
        if not files:
            raise ExplorationError("No LeRobot v3 episode index was found", 404)
        collected: list[dict[str, Any]] = []
        scanned = 0
        for path in files[:MAX_INDEX_FILES]:
            raw = await self.reader.read(hub_url(source, path), MAX_INDEX_BYTES, budget)
            rows = await asyncio.to_thread(index_rows, raw)
            rows.sort(key=lambda row: integer(row.get("episode_index"), "episode index"))
            if selected is not None:
                found = [row for row in rows if row["episode_index"] == selected]
                if found:
                    return found
            else:
                collected.extend(rows[max(0, start - scanned) : max(0, start + count - scanned)])
                if len(collected) >= count:
                    return collected[:count]
            scanned += len(rows)
        if len(files) > MAX_INDEX_FILES:
            raise ExplorationError(
                "This episode is beyond the bounded preview index; use a smaller dataset"
            )
        return collected

    async def page(self, job: Job, offset: int = 0, limit: int = 12) -> EpisodePage:
        source = source_profile(job)
        if offset < 0 or not 1 <= limit <= 24:
            raise ExplorationError("Episode page requires offset >= 0 and limit between 1 and 24")
        async with self.slots, asyncio.timeout(60):
            budget = Budget()
            rows = []
            if offset < source.total_episodes:
                await self.metadata(source, budget)
                rows = await self.episodes(source, budget, offset, limit)
            return EpisodePage(
                repo_id=source.repo_id,
                revision=source.revision,
                total_episodes=source.total_episodes,
                offset=offset,
                limit=limit,
                episodes=[summary(row, source.fps) for row in rows],
                warnings=[]
                if rows or offset >= source.total_episodes
                else ["No episodes were found in the source episode index."],
            )

    async def preview(self, job: Job, episode_index: int) -> EpisodePreview:
        source = source_profile(job)
        if episode_index < 0 or episode_index >= source.total_episodes:
            raise ExplorationError("Episode not found", 404)
        async with self.slots, asyncio.timeout(60):
            budget = Budget()
            info = await self.metadata(source, budget)
            rows = await self.episodes(source, budget, 0, 1, selected=episode_index)
            if not rows:
                raise ExplorationError("Episode not found in the source index", 404)
            row = rows[0]
            item = summary(row, source.fps)
            warnings: list[str] = []
            cameras = await self.cameras(source, info, row, item, budget, warnings)
            samples: list[FrameSample] = []
            try:
                path = self.data_path(source, info, row)
                raw = await self.reader.read(hub_url(source, path), MAX_PARQUET_BYTES, budget)
                samples = await asyncio.to_thread(frame_samples, raw, episode_index)
                if not samples:
                    warnings.append("The selected episode has no sample rows in its frame shard.")
            except (ValueError, httpx.HTTPError, pa.ArrowException) as exc:
                warnings.append(f"Sample rows unavailable: {self.friendly_error(exc)}")
            return EpisodePreview(
                **item.model_dump(),
                repo_id=source.repo_id,
                revision=source.revision,
                cameras=cameras,
                action_names=feature_names(source.features, "action"),
                state_names=feature_names(source.features, "observation.state"),
                samples=samples,
                warnings=warnings,
            )

    @staticmethod
    def friendly_error(exc: Exception) -> str:
        return (
            str(exc) if isinstance(exc, ExplorationError) else "The source file could not be read."
        )

    @staticmethod
    def data_path(source: DatasetProfile, info: dict, row: dict) -> str:
        if source.format == "lerobot_v2":
            chunks = integer(info.get("chunks_size", 1000), "chunk size")
            if chunks == 0:
                raise ExplorationError("Dataset chunk size must be positive")
            return render_path(
                info.get("data_path"),
                episode_index=row["episode_index"],
                episode_chunk=row["episode_index"] // chunks,
            )
        return render_path(
            info.get("data_path"),
            chunk_index=integer(row.get("data/chunk_index"), "data chunk"),
            file_index=integer(row.get("data/file_index"), "data file"),
        )

    async def cameras(
        self,
        source: DatasetProfile,
        info: dict,
        row: dict,
        item: EpisodeSummary,
        budget: Budget,
        warnings: list[str],
    ) -> list[CameraPreview]:
        result = []
        camera_features = [
            (key, feature)
            for key, feature in source.features.items()
            if feature.get("dtype") in {"video", "image"}
        ]
        for key, feature in camera_features[:8]:
            if feature.get("dtype") == "image":
                warnings.append(f"{key}: embedded image previews are not available yet.")
                continue
            try:
                if source.format == "lerobot_v2":
                    chunks = integer(info.get("chunks_size", 1000), "chunk size")
                    if chunks == 0:
                        raise ExplorationError("Dataset chunk size must be positive")
                    path = render_path(
                        info.get("video_path"),
                        video_key=key,
                        episode_chunk=item.episode_index // chunks,
                        episode_index=item.episode_index,
                    )
                    start, end = 0.0, item.duration_seconds
                else:
                    prefix = f"videos/{key}/"
                    path = render_path(
                        info.get("video_path"),
                        video_key=key,
                        chunk_index=integer(row.get(prefix + "chunk_index"), "video chunk"),
                        file_index=integer(row.get(prefix + "file_index"), "video file"),
                    )
                    start = number(row.get(prefix + "from_timestamp"), "video start time")
                    end = number(row.get(prefix + "to_timestamp"), "video end time")
                    if end <= start:
                        raise ExplorationError("Video segment has invalid time boundaries")
                url = hub_url(source, path)
                await self.reader.read(url, 0, budget, head=True)
                video_info = feature.get("info") or feature.get("video_info") or {}
                if not isinstance(video_info, dict):
                    raise ExplorationError("Camera metadata is not a valid object")
                shape = feature.get("shape", [])
                axes = feature.get("names", [])
                dimensions = (
                    dict(zip(axes, shape))
                    if isinstance(axes, list) and all(isinstance(axis, str) for axis in axes)
                    else {}
                )
                width = video_info.get("video.width", dimensions.get("width"))
                height = video_info.get("video.height", dimensions.get("height"))
                fps = number(video_info.get("video.fps", source.fps), "camera FPS")
                if fps <= 0:
                    raise ExplorationError("Camera FPS must be positive")
                result.append(
                    CameraPreview(
                        key=key,
                        url=url,
                        start_seconds=start,
                        end_seconds=end,
                        width=width if type(width) is int and width > 0 else None,
                        height=height if type(height) is int and height > 0 else None,
                        fps=fps,
                    )
                )
            except (ValueError, httpx.HTTPError) as exc:
                warnings.append(f"{key}: {self.friendly_error(exc)}")
        if len(camera_features) > 8:
            warnings.append("The preview shows the first eight cameras.")
        if not camera_features:
            warnings.append("This dataset does not declare camera images or videos.")
        return result
