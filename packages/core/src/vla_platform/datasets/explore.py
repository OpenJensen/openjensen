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
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from vla_platform.contracts import (
    CameraPreview,
    DatasetProfile,
    EpisodePage,
    EpisodePreview,
    EpisodeSummary,
    FrameSample,
    Job,
)
from vla_platform.datasets.hub_parquet import ReaderError, read_parquet

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_PARQUET_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 48 * 1024 * 1024
MAX_INDEX_FILES = 16
MAX_INDEX_ROWS = 100_000


class ExplorationError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def source_profile(job: Job) -> DatasetProfile:
    if job.kind != "dataset.inspect":
        raise ExplorationError("Episode previews require a dataset inspection", 422)
    if job.status != "succeeded" or job.result is None:
        raise ExplorationError(
            "Finish a successful dataset inspection before exploring episodes", 409
        )
    result = job.result
    if not isinstance(result, DatasetProfile):
        raise ExplorationError("Episode previews require a dataset inspection", 422)
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


async def index_rows(raw: bytes, *, python: str | Path | None = None) -> list[dict[str, Any]]:
    try:
        return await read_parquet(raw, "index", python=python)
    except ReaderError as exc:
        raise ExplorationError(str(exc)) from exc


async def frame_samples(
    raw: bytes, episode_index: int, *, python: str | Path | None = None
) -> list[FrameSample]:
    try:
        rows = await read_parquet(raw, "frames", episode_index=episode_index, python=python)
        return [FrameSample.model_validate(row) for row in rows]
    except ReaderError as exc:
        raise ExplorationError(str(exc)) from exc


class DatasetExplorer:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        reader_python: str | Path | None = None,
        cover_cache: Path | None = None,
    ):
        self.client = client or httpx.AsyncClient(timeout=20, follow_redirects=False)
        self.reader = HubReader(self.client)
        self.reader_python = reader_python
        self.slots = asyncio.Semaphore(2)
        self.cover_cache = cover_cache
        self.covers: OrderedDict[str, EpisodePreview] = OrderedDict()
        self.cover_locks: dict[str, asyncio.Lock] = {}

    async def close(self) -> None:
        await self.client.aclose()

    async def cover(self, job: Job) -> EpisodePreview:
        """First camera only: no frame shard downloads or repeated index parsing.

        The cache is addressed by the immutable source and verified metadata hash,
        so a new inspection revision never inherits another revision's preview.
        """
        source = source_profile(job)
        identity = json.dumps([source.repo_id, source.revision, source.metadata_sha256])
        key = hashlib.sha256(identity.encode()).hexdigest()
        async with self.cover_locks.setdefault(key, asyncio.Lock()):
            if key in self.covers:
                self.covers.move_to_end(key)
                return self.covers[key].model_copy(deep=True)
            path = self.cover_cache / f"{key}.json" if self.cover_cache else None
            if path and path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536:
                try:
                    cached = EpisodePreview.model_validate_json(path.read_bytes())
                    if (cached.repo_id, cached.revision) != (source.repo_id, source.revision):
                        raise ValueError("Cached cover identity differs")
                    if (
                        cached.episode_index >= source.total_episodes
                        or len(cached.cameras) > 1
                        or cached.samples
                        or any(
                            not allowed_url(c.url)
                            or c.end_seconds <= c.start_seconds
                            or c.key not in source.features
                            for c in cached.cameras
                        )
                    ):
                        raise ValueError("Invalid cached cover")
                    self.covers[key] = cached
                    while len(self.covers) > 128:
                        old_key, _ = self.covers.popitem(last=False)
                        if not self.cover_locks[old_key].locked():
                            self.cover_locks.pop(old_key, None)
                    return cached.model_copy(deep=True)
                except ValueError, OSError:
                    pass
            async with self.slots, asyncio.timeout(60):
                budget = Budget()
                info = await self.metadata(source, budget)
                rows = await self.episodes(source, budget, 0, 1)
                if not rows:
                    raise ExplorationError("No preview episode was found", 404)
                row = rows[0]
                item = summary(row, source.fps)
                warnings: list[str] = []
                cameras = await self.cameras(
                    source, info, row, item, budget, warnings, limit=1, verify=False
                )
                result = EpisodePreview(
                    **item.model_dump(),
                    repo_id=source.repo_id,
                    revision=source.revision,
                    cameras=cameras,
                    samples=[],
                    action_names=[],
                    state_names=[],
                    warnings=warnings,
                )
            self.covers[key] = result
            while len(self.covers) > 128:
                old_key, _ = self.covers.popitem(last=False)
                if not self.cover_locks[old_key].locked():
                    self.cover_locks.pop(old_key, None)
            if path:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Only this generated cache directory is bounded; original data is untouched.
                    owned = sorted(
                        path.parent.glob("[0-9a-f]" * 64 + ".json"), key=lambda p: p.stat().st_mtime
                    )
                    for old in owned[:-127]:
                        if old != path and not old.is_symlink():
                            old.unlink()
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(result.model_dump_json())
                    temporary.replace(path)
                except OSError:
                    pass  # A cache failure must not turn a valid preview into an error.
            return result.model_copy(deep=True)

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
            rows = await index_rows(raw, python=self.reader_python)
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
                samples = await frame_samples(raw, episode_index, python=self.reader_python)
                if not samples:
                    warnings.append("The selected episode has no sample rows in its frame shard.")
            except (ValueError, httpx.HTTPError) as exc:
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
        *,
        limit: int = 8,
        verify: bool = True,
    ) -> list[CameraPreview]:
        result = []
        camera_features = [
            (key, feature)
            for key, feature in source.features.items()
            if feature.get("dtype") in {"video", "image"}
        ]
        for key, feature in camera_features[:limit]:
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
                if verify:
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
