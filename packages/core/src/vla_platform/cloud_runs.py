"""Bounded, read-only snapshots produced by a separate trusted cloud monitor."""

import json
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    field_validator,
)

MAX_SNAPSHOT_BYTES = 768 * 1024
MAX_LOG_BYTES = 48 * 1024
MAX_RUNS = 50
MAX_DIRECTORY_ENTRIES = 200
STALE_AFTER_SECONDS = 90
RUN_ID = r"^[a-z0-9][a-z0-9_-]{0,63}$"
RunId = Annotated[StrictStr, Field(pattern=RUN_ID)]


class SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CloudRunLogs(SnapshotModel):
    isaac: str = Field(strict=True, max_length=MAX_LOG_BYTES)
    vla: str = Field(strict=True, max_length=MAX_LOG_BYTES)

    @field_validator("isaac", "vla")
    @classmethod
    def bounded_utf8(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_LOG_BYTES:
            raise ValueError("Log exceeds UTF-8 byte limit")
        return value


class CloudRunOutcomes(SnapshotModel):
    rollout_completed: StrictBool | None = None
    pickup_success: StrictBool | None = None
    calibration: Literal["unknown", "unverified", "verified"] = "unknown"


class CloudRunSnapshot(SnapshotModel):
    schema_version: Literal[1]
    run_id: RunId
    label: str = Field(strict=True, min_length=1, max_length=120)
    cluster: str = Field(strict=True, min_length=1, max_length=100)
    job_id: str | None = Field(default=None, strict=True, pattern=r"^[0-9]{1,64}$")
    collected_at: AwareDatetime | None
    status: str = Field(strict=True, min_length=1, max_length=40)
    collection_error: str | None = Field(default=None, strict=True, max_length=2000)
    logs: CloudRunLogs
    outcomes: CloudRunOutcomes

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported snapshot version")
        return value

    @field_validator("collected_at", mode="before")
    @classmethod
    def timestamp_string(cls, value):
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("Collection time must be a timezone-aware timestamp")
        return value


class CloudRunView(CloudRunSnapshot):
    age_seconds: float | None
    stale: bool


class CloudFeedIssue(SnapshotModel):
    run_id: RunId | None = None
    message: str


class CloudRunsFeed(SnapshotModel):
    enabled: bool
    server_time: AwareDatetime
    stale_after_seconds: int = STALE_AFTER_SECONDS
    runs: list[CloudRunView] = Field(default_factory=list)
    errors: list[CloudFeedIssue] = Field(default_factory=list)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _directory_identity(directory: Path) -> tuple[int, int]:
    # Do not resolve before checking: that would hide a configured symlink.
    for path in (directory, *directory.parents):
        if path.is_symlink():
            raise ValueError("Snapshot directory must not use symlinks")
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("Snapshot directory is unavailable")
    return _identity(info)


def _read_snapshot(path: Path) -> CloudRunSnapshot:
    # An atomic publisher may replace a regular file between inspection and open.
    # Retry that specific race once, without relaxing identity or size checks.
    for attempt in range(2):
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Snapshot must be a regular file, without symlinks")
        if before.st_size > MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds the file size limit")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("Snapshot must be a regular file")
            if _identity(before) != _identity(opened):
                if attempt == 0:
                    continue
                raise ValueError("Snapshot changed while opening; retry collection")
            if opened.st_size > MAX_SNAPSHOT_BYTES:
                raise ValueError("Snapshot exceeds the file size limit")
            raw = handle.read(MAX_SNAPSHOT_BYTES + 1)
            break
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Snapshot exceeds the file size limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
        snapshot = CloudRunSnapshot.model_validate(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        # Never return validation inputs or raw snapshot content in error messages.
        raise ValueError("Snapshot contains invalid versioned JSON data") from exc
    if snapshot.run_id != path.stem:
        raise ValueError("Snapshot run ID does not match its filename")
    return snapshot


def read_cloud_runs(directory: Path | None) -> CloudRunsFeed:
    now = datetime.now(UTC)
    feed = CloudRunsFeed(enabled=directory is not None, server_time=now)
    if directory is None:
        return feed
    try:
        directory = directory.absolute()
        original = _directory_identity(directory)
        names: list[str] = []
        with os.scandir(directory) as entries:
            for count, entry in enumerate(entries, 1):
                if count > MAX_DIRECTORY_ENTRIES:
                    raise ValueError("Snapshot directory exceeds the entry limit")
                if entry.name.endswith(".json"):
                    names.append(entry.name)
                if len(names) > MAX_RUNS:
                    raise ValueError("Snapshot directory exceeds the 50-run limit")
        for name in sorted(names):
            run_id = name[:-5]
            if not re.fullmatch(RUN_ID, run_id):
                feed.errors.append(
                    CloudFeedIssue(message="Snapshot filename has an invalid run ID")
                )
                continue
            try:
                snapshot = _read_snapshot(directory / name)
                age = (
                    (now - snapshot.collected_at).total_seconds()
                    if snapshot.collected_at is not None
                    else None
                )
                if age is not None and age < -5:
                    raise ValueError("Snapshot collection time is in the future")
                feed.runs.append(
                    CloudRunView(
                        **snapshot.model_dump(),
                        age_seconds=max(0, age) if age is not None else None,
                        stale=age is None or age > STALE_AFTER_SECONDS,
                    )
                )
            except (OSError, ValueError, ValidationError) as exc:
                message = str(exc) if isinstance(exc, ValueError) else "Snapshot could not be read"
                feed.errors.append(CloudFeedIssue(run_id=run_id, message=message))
        if _directory_identity(directory) != original:
            raise ValueError("Snapshot directory changed during collection")
    except (OSError, ValueError) as exc:
        feed.runs.clear()
        feed.errors = [
            CloudFeedIssue(
                message=str(exc)
                if isinstance(exc, ValueError)
                else "Snapshot directory is unavailable"
            )
        ]
    feed.runs.sort(
        key=lambda run: (run.collected_at or datetime.min.replace(tzinfo=UTC), run.run_id),
        reverse=True,
    )
    return feed
