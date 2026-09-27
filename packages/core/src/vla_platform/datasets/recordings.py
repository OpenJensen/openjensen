"""Project-owned capture catalog and one supervised conversion → snapshot job.

Only the fixed isolated writer imports LeRobot. Operator roots must be quiescent;
source inventories, not an assumed closed-session flag, enforce publication.
"""

import asyncio
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vla_platform.datasets.local_preview import PreviewError, _open_beneath
from vla_platform.datasets.recording_contracts import RecordingPreparation, RecordingSummary

WORKER_FILES = ("__init__.py", "prepare_dataset.py", "dataset.py", "contracts.py", "journal.py")
HEX = re.compile(r"[a-f0-9]{64}\Z")
ID = re.compile(r"[a-f0-9]{32}\Z")
META_LIMIT = 2 * 1024**2
CATALOG_BYTES = 16 * 1024**2
CATALOG_ENTRIES = 1000
RESULT_LIMIT = 16 * 1024**2


class RecordingError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecordingError("Duplicate JSON keys are not supported")
        result[key] = value
    return result


def finite(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise RecordingError("Nonfinite JSON numbers are not supported")
    return value


def decode(raw):
    def invalid(_):
        raise RecordingError("Nonfinite JSON numbers are not supported")

    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid, parse_float=finite
    )
    if not isinstance(value, dict):
        raise RecordingError("Expected a JSON object")
    return value


def path(value):
    item = Path(value)
    if not item.is_absolute() or ".." in item.parts or "\\" in str(item) or "\0" in str(item):
        raise RecordingError("Operator paths must be absolute without traversal")
    return item


def read(item, maximum=META_LIMIT):
    item = path(item)
    with _open_beneath(Path(item.anchor), item.relative_to(item.anchor)) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > maximum:
            raise RecordingError("Recording metadata exceeds its byte limit")
        raw = stream.read(min(before.st_size, maximum) + 1)
        after = os.fstat(stream.fileno())

        def stamp(s):
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

        if len(raw) != before.st_size or stamp(before) != stamp(after):
            raise RecordingError("Recording metadata changed during read", 409)
        return raw


@contextmanager
def directory(item):
    item = path(item)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(item.anchor, flags)
    try:
        for name in item.parts[1:]:
            child = os.open(name, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def entries(item, maximum):
    result = []
    with directory(item) as descriptor, os.scandir(descriptor) as listing:
        for entry in listing:
            if len(result) >= maximum:
                raise RecordingError("Capture catalog exceeds its entry limit; narrow its root")
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not (
                stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
            ):
                raise RecordingError("Capture catalog contains a symlink or special node")
            result.append((entry.name, stat.S_ISDIR(info.st_mode)))
    return sorted(result)


class ProjectRoot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    project_id: str = Field(min_length=1, max_length=200)
    capture_root: str = Field(min_length=1, max_length=4096)


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    dataset_python: str
    worker_root: str
    ffmpeg: str
    ffprobe: str
    projects: list[ProjectRoot] = Field(min_length=1, max_length=100)

    @field_validator("schema_version", mode="before")
    @classmethod
    def schema(cls, value):
        if type(value) is not int:
            raise ValueError("Configuration schema must be integer 1")
        return value

    @model_validator(mode="after")
    def disjoint(self):
        ids = [p.project_id for p in self.projects]
        roots = [path(p.capture_root) for p in self.projects]
        if len(set(ids)) != len(ids):
            raise ValueError("Project mappings must be unique")
        for index, root in enumerate(roots):
            for other in roots[index + 1 :]:
                if root.is_relative_to(other) or other.is_relative_to(root):
                    raise ValueError("Project recording roots must be disjoint")
        return self


@dataclass(frozen=True)
class Loaded:
    config_path: Path
    value: Configuration
    identity: str

    def root(self, project):
        matched = next(
            (p.capture_root for p in self.value.projects if p.project_id == project), None
        )
        if matched is None:
            raise RecordingError("No recording source is mapped to this project", 404)
        return path(matched)


def executable_identity(value):
    # Virtualenv executables normally symlink to their base interpreter. Preserve
    # that invocation path; bind the resolved operator-controlled target as well.
    invoked = path(value)
    with directory(invoked.parent):
        pass
    target = invoked.resolve(strict=True)
    info = target.stat()
    if not stat.S_ISREG(info.st_mode) or not os.access(invoked, os.X_OK):
        raise RecordingError("A configured recording executable is unavailable")
    return {
        "invoked": str(invoked),
        "target": str(target),
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def load_config(config_path):
    if config_path is None:
        raise RecordingError("Configure FIREBIRD_RECORDING_CONFIG to prepare recordings")
    raw = read(config_path, 65536)
    value = Configuration.model_validate(decode(raw))
    workers = path(value.worker_root) / "firebird_teaching"
    isaac = path(value.worker_root).parent / "isaac_sim" / "sim_worker"
    identity = {
        "configuration_sha256": digest(raw),
        "contracts_files": {
            name: digest(read(isaac / name))
            for name in ("__init__.py", "rollout/__init__.py", "rollout/contracts.py")
        },
        "worker_files": {name: digest(read(workers / name)) for name in WORKER_FILES},
        "executables": {
            name: executable_identity(getattr(value, name))
            for name in ("dataset_python", "ffmpeg", "ffprobe")
        },
    }
    for project in value.projects:
        with directory(project.capture_root):
            pass
    return Loaded(path(config_path), value, digest(canonical(identity)))


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise RecordingError("Invalid recording count or dimensions")
    return value


def metadata(value):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise RecordingError("Unsupported capture schema")
    if not isinstance(value.get("session_id"), str) or not ID.fullmatch(value["session_id"]):
        raise RecordingError("Invalid capture session identity")
    for name, expected in {
        "controller": "joint_position_targets",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "camera_key": "observation.images.front",
    }.items():
        if value.get(name) != expected:
            raise RecordingError("Unsupported capture coordinates or camera contract")
    if not isinstance(value.get("origin"), str) or value["origin"] not in {"recorded", "synthetic"}:
        raise RecordingError("Capture origin must distinguish recorded and synthetic data")
    group = value.get("lineage_group")
    if not isinstance(group, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", group):
        raise RecordingError("Invalid capture lineage group")
    names = value.get("joint_names")
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= 32
        or any(not isinstance(n, str) or not n.isidentifier() for n in names)
        or len(set(names)) != len(names)
    ):
        raise RecordingError("Invalid capture joint order")
    for name in ("width", "height"):
        if integer(value.get(name), 2, 1920) % 2:
            raise RecordingError("Capture dimensions must be even")
    integer(value.get("fps"), 1, 60)
    integer(value.get("physics_hz"), value["fps"], 1000)
    if not isinstance(value.get("scene_sha256"), str) or not HEX.fullmatch(value["scene_sha256"]):
        raise RecordingError("Invalid capture scene hash")
    # Camera prims are scene identifiers, not local filesystem paths.
    camera = value.get("camera_prim")
    if not isinstance(camera, str) or not 1 <= len(camera) <= 256 or not camera.isprintable():
        raise RecordingError("Invalid capture camera prim")
    return {
        key: value[key]
        for key in (
            "session_id",
            "origin",
            "lineage_group",
            "controller",
            "state_units",
            "action_units",
            "timebase",
            "camera_key",
            "joint_names",
            "width",
            "height",
            "fps",
            "physics_hz",
            "scene_sha256",
            "camera_prim",
        )
    }


def _catalog(loaded, project):
    root = loaded.root(project)
    remaining, count = CATALOG_BYTES, 0
    captures, locations = [], {}

    def bounded_read(item):
        nonlocal remaining
        raw = read(item, min(remaining, META_LIMIT))
        remaining -= len(raw)
        return raw

    for folder, is_dir in entries(root, 100):
        if not is_dir:
            raise RecordingError("Published capture roots may contain session directories only")
        capture = root / folder
        raw = bounded_read(capture / "session.json")
        meta = metadata(decode(raw))
        ident = meta["session_id"]
        if ident in locations:
            raise RecordingError("Duplicate session identity in recording catalog")
        locations[ident] = capture
        episodes = []
        for name, child_dir in entries(capture, CATALOG_ENTRIES - count):
            count += 1
            if not child_dir:
                continue
            if not ID.fullmatch(name):
                raise RecordingError("Invalid episode directory identity")
            try:
                receipt_raw = bounded_read(capture / name / "episode.json")
            except PreviewError as error:
                if error.code != "missing_file":
                    raise
                continue  # An unfinished episode is never selectable.
            receipt = decode(receipt_raw)
            if (
                type(receipt.get("schema_version")) is not int
                or receipt["schema_version"] != 1
                or receipt.get("episode_id") != name
                or receipt.get("finalized") is not True
                or not isinstance(receipt.get("outcome"), str)
                or receipt["outcome"] not in {"unknown", "operator_reported_failure"}
                or not isinstance(receipt.get("termination"), str)
                or receipt["termination"] not in {"finish", "reset", "step_limit", "shutdown"}
            ):
                raise RecordingError("Invalid finalized recording receipt")
            episodes.append(
                {
                    "episode_id": name,
                    "receipt_sha256": digest(receipt_raw),
                    "frames": integer(receipt.get("frames"), 1, 3600),
                    "outcome": receipt["outcome"],
                    "termination": receipt["termination"],
                }
            )
        if episodes:
            captures.append(
                {
                    **meta,
                    "session_sha256": digest(raw),
                    "episodes": episodes,
                    "content_verified": False,
                }
            )
    value = {
        "configuration_sha256": loaded.identity,
        "captures": captures,
        "message": (
            "Finalized metadata only. Keep capture folders quiescent; "
            "full content is checked during preparation."
        ),
    }
    if len(canonical(value)) > 1024**2:
        raise RecordingError("Capture catalog exceeds its response limit")
    return value, locations


def catalog(loaded, project):
    return _catalog(loaded, project)[0]


def resolve_selection(loaded, project, request: RecordingPreparation):
    if loaded.identity != request.configuration_sha256:
        raise RecordingError("Recording configuration changed; review it again", 409)
    public, locations = _catalog(loaded, project)
    found = {capture["session_id"]: capture for capture in public["captures"]}
    selected = []
    for capture in request.captures:
        current = found.get(capture.session_id)
        if current is None:
            raise RecordingError("Selected capture is not available in this project", 404)
        if capture.session_sha256 != current["session_sha256"]:
            raise RecordingError("Selected capture metadata changed; review it again", 409)
        receipts = {e["episode_id"]: e for e in current["episodes"]}
        for episode in capture.episodes:
            if episode.episode_id not in receipts:
                raise RecordingError("Selected finalized episode is unavailable", 404)
            if episode.receipt_sha256 != receipts[episode.episode_id]["receipt_sha256"]:
                raise RecordingError("Selected episode receipt changed; review it again", 409)
        selected.append({**capture.model_dump(), "path": str(locations[capture.session_id])})
    return selected


def options(config_path, project):
    try:
        loaded = load_config(config_path)
        loaded.root(project)
    except OSError, ValueError:
        return {
            "configured": False,
            "runtime_verified": False,
            "configuration_sha256": None,
            "setup_message": (
                "Configure a project recording root and the isolated pinned writer "
                "with FFmpeg/ffprobe."
            ),
        }
    return {
        "configured": True,
        "runtime_verified": False,
        "configuration_sha256": loaded.identity,
        "max_episodes": 100,
        "max_source_bytes": 8 * 1024**3,
        "setup_message": (
            "Configured; writer compatibility is checked in the preparation job. "
            "No simulator connection is required."
        ),
    }


def inventory(root, *, deadline, stop, budget=None, with_sizes=False):
    """Bounded cancellable standard-library hashing, without importing the writer."""
    root = path(root)
    result, pending = {}, [root]
    budget = [0, 0] if budget is None else budget

    def checkpoint():
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("Recording verification stopped at its deadline")

    while pending:
        checkpoint()
        current = pending.pop()
        for name, is_dir in entries(current, 50000 - budget[1]):
            checkpoint()
            budget[1] += 1
            item = current / name
            if is_dir:
                pending.append(item)
                continue
            with _open_beneath(root, item.relative_to(root)) as stream:
                before = os.fstat(stream.fileno())
                budget[0] += before.st_size
                if before.st_size > 2 * 1024**3 or budget[0] > 8 * 1024**3:
                    raise RecordingError("Recording inventory exceeds its byte limit")
                hashed, remaining = hashlib.sha256(), before.st_size
                while remaining:
                    if stop.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("Recording verification stopped at its deadline")
                    part = stream.read(min(1024**2, remaining))
                    if not part:
                        raise RecordingError("Recording file shrank during verification")
                    hashed.update(part)
                    remaining -= len(part)
                after = os.fstat(stream.fileno())
                if (
                    stream.read(1)
                    or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_ctime_ns != after.st_ctime_ns
                ):
                    raise RecordingError("Recording file changed during verification")
                result[item.relative_to(root).as_posix()] = (
                    {"sha256": hashed.hexdigest(), "size": before.st_size}
                    if with_sizes
                    else hashed.hexdigest()
                )
    return result


async def checked_io(function, *args, **kwargs):
    """Drain our read task on cancellation; the hash loop observes the stop flag."""
    stop = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(function, *args, stop=stop, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stop.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


def new_json(item, value):
    with directory(item.parent):
        pass
    with item.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def checked_conversion(response, job, selection, destination, *, deadline, stop):
    expected = {
        "schema_version",
        "job_id",
        "configuration_sha256",
        "selection_sha256",
        "conversion",
        "files",
    }
    recipe = job.request.recordings
    selection_hash = digest(canonical(recipe.model_dump()))
    if (
        set(response) != expected
        or type(response["schema_version"]) is not int
        or response["schema_version"] != 1
        or response["job_id"] != job.id
        or response["configuration_sha256"] != recipe.configuration_sha256
        or response["selection_sha256"] != selection_hash
    ):
        raise RecordingError("Preparation response identity did not match its job")
    conversion = response["conversion"]
    if not isinstance(conversion, dict) or set(conversion) != {
        "status",
        "path",
        "episodes",
        "frames",
        "readback_verified",
        "task_success_verified",
        "source_preserved",
        "sources",
        "lineage_groups",
    }:
        raise RecordingError("Invalid preparation conversion result")
    if (
        conversion["status"] != "finalized"
        or conversion["path"] != str(destination)
        or conversion["readback_verified"] is not True
        or conversion["source_preserved"] is not True
        or conversion["task_success_verified"] is not False
    ):
        raise RecordingError("Preparation did not verify its exact owned output")
    files = inventory(destination, deadline=deadline, stop=stop)
    if response["files"] != files:
        raise RecordingError("Prepared dataset inventory does not match its files")
    provenance = decode(read(destination / "meta/firebird-demonstrations.json", 8 * 1024**2))
    lineage = decode(read(destination / "meta/firebird-lineage.json"))
    expected_lineage, expected_details, frames = [], [], 0
    expected_sources, source_budget = [], [0, 0]
    common = None
    for source in selection:
        root = Path(source["path"])
        source_files = inventory(root, deadline=deadline, stop=stop, budget=source_budget)
        if source_files.get("session.json") != source["session_sha256"]:
            raise RecordingError("Selected recording metadata changed")
        meta = metadata(decode(read(root / "session.json")))
        if meta["session_id"] != source["session_id"]:
            raise RecordingError("Selected recording identity changed")
        schema = {
            key: meta[key]
            for key in (
                "joint_names",
                "camera_key",
                "camera_prim",
                "width",
                "height",
                "fps",
                "physics_hz",
                "controller",
                "state_units",
                "action_units",
                "timebase",
            )
        }
        if common is not None and schema != common:
            raise RecordingError("Selected capture coordinate/camera schemas differ")
        common = schema
        expected_sources.append(
            {
                "capture_path": str(root),
                "source_session_id": source["session_id"],
                "scene_sha256": meta["scene_sha256"],
                "lineage_group": meta["lineage_group"],
                "origin": meta["origin"],
                "source_files": source_files,
            }
        )
        for episode in source["episodes"]:
            name = episode["episode_id"] + "/episode.json"
            if source_files.get(name) != episode["receipt_sha256"]:
                raise RecordingError("Selected recording receipt changed")
            receipt = decode(read(root / name))
            index = len(expected_lineage)
            frames += integer(receipt.get("frames"), 1, 3600)
            expected_lineage.append(
                {
                    "episode_index": index,
                    "origin": meta["origin"],
                    "lineage_group": meta["lineage_group"],
                }
            )
            expected_details.append(
                {
                    "episode_index": index,
                    "capture_episode_id": episode["episode_id"],
                    "source_session_id": source["session_id"],
                    "reset_id": episode["episode_id"],
                    "outcome": receipt["outcome"],
                    "termination": receipt["termination"],
                    "events": receipt.get("events", []),
                    "source_trajectory_sha256": receipt["trajectory_sha256"],
                }
            )
    expected_coordinates = {
        key: common[key]
        for key in ("controller", "state_units", "action_units", "timebase", "camera_prim")
    }
    expected_coordinates.update(
        joint_order=common["joint_names"],
        action_column="action",
        requested_action_column="teaching.requested_action",
    )
    if (
        type(provenance.get("schema_version")) is not int
        or provenance["schema_version"] != 1
        or type(lineage.get("schema_version")) is not int
        or any(provenance.get(key) != value for key, value in expected_coordinates.items())
        or provenance.get("sources") != expected_sources
        or provenance.get("episodes") != expected_details
        or provenance.get("task_success_verified") is not False
        or provenance.get("writer_upstream_revision") != "e595b7902714ba51f91e47523f66f89c5181b649"
        or provenance.get("lerobot_version") != "0.6.2"
        or lineage != {"schema_version": 1, "episodes": expected_lineage}
    ):
        raise RecordingError("Prepared dataset lost its exact recording provenance")
    groups = len({e["lineage_group"] for e in expected_lineage})
    for key, expected in {
        "episodes": len(expected_lineage),
        "frames": frames,
        "sources": len(selection),
        "lineage_groups": groups,
    }.items():
        if type(conversion[key]) is not int or conversion[key] != expected:
            raise RecordingError("Prepared dataset counts differ from selected recordings")
    return files, RecordingSummary(
        job_id=job.id,
        selection_sha256=selection_hash,
        source_count=len(selection),
        lineage_group_count=groups,
    )


def checked_snapshot(store, snapshot, files, *, deadline, stop):
    """Recheck the native reader's content-addressed result with cancellable I/O."""
    from vla_platform.datasets.snapshots import MANIFEST, descriptor
    from vla_platform.datasets.snapshots import canonical as snapshot_canonical

    identity = snapshot.manifest_sha256
    if not HEX.fullmatch(identity) or snapshot.id != "sha256:" + identity:
        raise RecordingError("Invalid prepared snapshot identity")
    root = store / identity
    raw = read(root / MANIFEST, RESULT_LIMIT)
    manifest = decode(raw)
    if digest(raw) != identity or snapshot_canonical(manifest) != raw:
        raise RecordingError("Prepared snapshot manifest changed")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise RecordingError("Invalid prepared snapshot manifest")
    actual = inventory(root, deadline=deadline, stop=stop, with_sizes=True)
    if actual.pop(MANIFEST, None) != {"sha256": identity, "size": len(raw)}:
        raise RecordingError("Prepared snapshot changed during verification")
    expected = {}
    for entry in manifest.get("files", []):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size"}
            or not isinstance(entry["path"], str)
            or entry["path"] in expected
            or type(entry["size"]) is not int
        ):
            raise RecordingError("Invalid prepared snapshot inventory")
        expected[entry["path"]] = {"sha256": entry["sha256"], "size": entry["size"]}
    if expected != actual or {key: value["sha256"] for key, value in actual.items()} != files:
        raise RecordingError("Snapshot bytes differ from the prepared dataset")
    if descriptor(manifest, identity) != snapshot.model_dump():
        raise RecordingError("Prepared snapshot descriptor changed")
    return manifest


class Recordings:
    def __init__(self, execution):
        self.execution = execution

    async def validate(self, project, request):
        try:
            loaded = await asyncio.to_thread(load_config, self.execution.settings.recording_config)
        except (OSError, ValueError) as error:
            raise RecordingError(
                "Recording runtime configuration is unavailable; review setup"
            ) from error
        selected = await asyncio.to_thread(resolve_selection, loaded, project, request.recordings)
        return loaded, selected

    async def stage(self, job, message):
        from vla_platform.contracts import TERMINAL

        async with self.execution.lock:
            current = await self.execution.get(job.id)
            if current is None or current.status in TERMINAL:
                raise asyncio.CancelledError
            current.stage = message
            await self.execution.save(current)
        await self.execution.lifecycle.event(job, "recording_preparation", message)

    async def execute(self, argv, cwd, env, log_path):
        from vla_platform.lifecycle.native_replay import error_tail

        process, drain = None, None
        lifecycle = self.execution.lifecycle
        try:
            process = await lifecycle.act_spawn_owned(
                *argv,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                grace_seconds=8,
            )
            drain = asyncio.create_task(error_tail(process.stderr))
            code = await process.wait()
        finally:

            async def cleanup():
                failure = None
                try:
                    if process is not None:
                        await lifecycle.act_stop_owned(process, grace_seconds=8)
                except Exception as error:
                    failure = error
                try:
                    if drain is not None:
                        tail = await asyncio.wait_for(drain, 2)
                        new_json(log_path, {"stderr_tail": tail})
                except Exception as error:
                    failure = failure or error
                if failure is not None:
                    raise RecordingError(
                        "Recording preparation cleanup is unverified; inspect private job evidence"
                    ) from failure

            # Do not let repeated caller cancellation hide an ownership failure.
            # The independent cleanup task always checks both the process and pipe.
            finishing = asyncio.create_task(cleanup())
            interrupted = False
            while not finishing.done():
                try:
                    await asyncio.shield(finishing)
                except asyncio.CancelledError:
                    interrupted = True
            finishing.result()
            if interrupted:
                raise asyncio.CancelledError
        if code:
            # Do not place native diagnostics/private paths into public Job.error.
            raise RecordingError("Recording preparation worker failed; inspect its private log")

    async def run(self, job):
        from vla_platform.contracts import TERMINAL, IntakeRequest, WorkerRequest, WorkerResult
        from vla_platform.frozen_commands import intake_command

        deadline = time.monotonic() + job.request.recordings.timeout_seconds
        try:
            async with self.execution.native_slots:
                deadline = time.monotonic() + job.request.recordings.timeout_seconds
                async with self.execution.lock:
                    current = await self.execution.get(job.id)
                    if current is None or current.status in TERMINAL:
                        return
                    current.status = "running"
                    await self.execution.save(current)
                async with asyncio.timeout(job.request.recordings.timeout_seconds):
                    await self.stage(job, "Verifying recording selection and writer")
                    loaded, selection = await self.validate(job.project_id, job.request)
                    directory_path = (
                        self.execution.settings.data_dir / "jobs" / job.id / "recording-preparation"
                    )
                    directory_path.mkdir(parents=True, exist_ok=False)
                    with directory(directory_path):
                        pass
                    destination = directory_path / "dataset"
                    request_path, response_path = (
                        directory_path / "request.json",
                        directory_path / "result.json",
                    )
                    worker_request = {
                        "schema_version": 1,
                        "job_id": job.id,
                        "configuration_sha256": loaded.identity,
                        "selection_sha256": digest(canonical(job.request.recordings.model_dump())),
                        "captures": selection,
                        "output": str(destination),
                        "ffmpeg": loaded.value.ffmpeg,
                        "ffprobe": loaded.value.ffprobe,
                    }
                    new_json(request_path, worker_request)
                    scratch = directory_path / "environment"
                    scratch.mkdir()
                    env = {
                        "PATH": os.pathsep.join(
                            dict.fromkeys(
                                [
                                    str(Path(loaded.value.ffmpeg).parent),
                                    str(Path(loaded.value.ffprobe).parent),
                                    str(Path(loaded.value.dataset_python).parent),
                                    "/usr/bin",
                                    "/bin",
                                ]
                            )
                        ),
                        "HOME": str(scratch),
                        "TMPDIR": str(scratch),
                        "HF_HOME": str(scratch / "huggingface"),
                        "HF_HUB_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "PYTHONPATH": os.pathsep.join(
                            [
                                loaded.value.worker_root,
                                str(Path(loaded.value.worker_root).parent / "isaac_sim"),
                            ]
                        ),
                    }
                    await self.stage(job, "Converting selected recordings with the pinned writer")
                    await self.execute(
                        [
                            loaded.value.dataset_python,
                            "-B",
                            "-m",
                            "firebird_teaching.prepare_dataset",
                            "--request",
                            str(request_path),
                            "--result",
                            str(response_path),
                        ],
                        directory_path,
                        env,
                        directory_path / "writer-log.json",
                    )
                    response = decode(read(response_path, RESULT_LIMIT))
                    files, summary = await checked_io(
                        checked_conversion, response, job, selection, destination, deadline=deadline
                    )
                    await self.stage(job, "Verifying immutable dataset snapshot")
                    intake_path, intake_result = (
                        directory_path / "snapshot-request.json",
                        directory_path / "snapshot-result.json",
                    )
                    normalized = IntakeRequest(
                        source="local", path=str(destination), snapshot_for_training=True
                    )
                    new_json(
                        intake_path,
                        WorkerRequest(
                            intake=normalized,
                            local_root=str(directory_path),
                            snapshot_store=str(
                                self.execution.settings.data_dir / "dataset-snapshots"
                            ),
                            reader_python=loaded.value.dataset_python,
                        ).model_dump(),
                    )
                    core_env = {**env, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
                    await self.execute(
                        intake_command(intake_path, intake_result),
                        directory_path,
                        core_env,
                        directory_path / "snapshot-log.json",
                    )
                    result = WorkerResult.model_validate(decode(read(intake_result, 4 * 1024**2)))
                    if result.result is None or result.result.snapshot is None:
                        raise RecordingError(
                            "Prepared dataset did not pass complete immutable intake"
                        )
                    profile = result.result
                    await checked_io(
                        checked_snapshot,
                        self.execution.settings.data_dir / "dataset-snapshots",
                        profile.snapshot,
                        files,
                        deadline=deadline,
                    )
                    if (
                        profile.total_episodes != response["conversion"]["episodes"]
                        or profile.total_frames != response["conversion"]["frames"]
                    ):
                        raise RecordingError("Snapshot counts differ from the prepared dataset")
                    # Recheck source identities and full capture bytes after the reader.
                    await checked_io(
                        checked_conversion, response, job, selection, destination, deadline=deadline
                    )
                    if (
                        await asyncio.to_thread(load_config, loaded.config_path)
                    ).identity != loaded.identity:
                        raise RecordingError(
                            "Recording runtime configuration changed during preparation"
                        )
                    profile.recording_preparation = summary
                    if summary.lineage_group_count < 2:
                        profile.warnings.append(
                            "Recordings share one lineage group; training needs at least "
                            "two distinct groups for a held-out split."
                        )
                    new_json(
                        directory_path / "verification.json",
                        {
                            "schema_version": 1,
                            "job_id": job.id,
                            "configuration_sha256": loaded.identity,
                            "recording_preparation": summary.model_dump(),
                            "snapshot": profile.snapshot.model_dump(),
                        },
                    )
                    await self.stage(
                        job, "Prepared dataset verified; task success remains unverified"
                    )
                    await self.execution.finish(job.id, WorkerResult(result=profile))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            message = (
                str(error)
                if isinstance(error, RecordingError)
                else (
                    "Recording preparation failed or exceeded its deadline; "
                    "inspect the private job evidence."
                )
            )
            async with self.execution.lock:
                current = await self.execution.get(job.id)
                if current is not None:
                    if current.status not in TERMINAL:
                        current.status, current.result = "failed", None
                    # Cleanup failures must remain visible even if cancellation
                    # already made this job terminal. They are never success.
                    current.error = ((current.error + " ") if current.error else "") + message
                    await self.execution.save(current)
