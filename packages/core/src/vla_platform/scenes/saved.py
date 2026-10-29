"""Verify saved reconstruction inputs before a new simulation-only Job."""

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from vla_platform.datasets.explore import source_profile
from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import Proposal, Review

MAX_FILES = 2048
MAX_BYTES = 256 * 1024**2
CORE_GATES = ("source_evidence", "agent_review", "usd_structure")
SOURCE_FIELDS = (
    "repo_id",
    "revision",
    "metadata_sha256",
    "scope",
    "total_episodes",
    "catalog_sha256",
    "trajectory_sha256",
    "inspected_episodes",
    "replay_episode",
    "replay_frames",
)
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class SavedScene:
    parent: object
    bundle: Path
    files: dict
    catalog_hash: str
    trajectory_hash: str


def regular(path, root):
    if not path.is_relative_to(root) or any(part.is_symlink() for part in (path, *path.parents)):
        raise SceneError("Saved scene contains linked or invalid inputs.")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise SceneError("Saved scene input is not a regular file.")
    return info.st_size


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            result.update(part)
    return result.hexdigest()


def read_json(path, root, limit=2 * 1024**2):
    if regular(path, root) > limit:
        raise SceneError("Saved scene metadata exceeds its size bound.")

    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("Duplicate metadata key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite metadata")

    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def relative(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 512
        or "\\" in name
        or any(ord(char) < 32 for char in name)
        or PurePosixPath(name).is_absolute()
        or str(PurePosixPath(name)) != name
        or any(part in {"", ".", "..", ".codex-runtime"} for part in name.split("/"))
    ):
        raise SceneError("Saved scene manifest contains an invalid asset path.")
    return name


def verify(scenes, project, request, source):
    """Read-only validation; no provider, runtime process, allocation or writes."""
    try:
        parent = scenes.get(request.source_scene_id)
        profile = source_profile(source)
        if (
            parent.project_id != project
            or parent.project_id != source.project_id
            or parent.request.source_job_id != request.source_job_id
            or parent.status not in {"failed", "interrupted"}
            or not parent.job_id
            or parent.id in scenes.executions
            or parent.id in scenes.tasks
            and not scenes.tasks[parent.id].done()
            or parent.source.get("scope") != "entire_dataset"
            or parent.source.get("repo_id") != profile.repo_id
            or parent.source.get("revision") != profile.revision
            or parent.source.get("metadata_sha256") != profile.metadata_sha256
            or parent.source.get("total_episodes") != profile.total_episodes
            or any(parent.gates.get(key) is not True for key in CORE_GATES)
            or parent.agent is None
            or request.model != parent.agent.model
            or request.reasoning_effort != parent.agent.reasoning_effort
            or parent.request.episode_index is not None
            or not 1 <= parent.attempt <= 3
        ):
            raise SceneError(
                "Choose a failed scene from this dataset and project with completed "
                "reconstruction checks and matching model settings."
            )
        root = scenes.folder(parent.id)
        bundle = root / f"attempt-{parent.attempt}" / "bundle"
        if any(part.is_symlink() for part in (bundle, *bundle.parents)) or not bundle.is_dir():
            raise SceneError("The saved scene bundle is missing or linked.")
        manifest = read_json(bundle / "manifest.json", root)
        files = manifest.get("files")
        if (
            not isinstance(files, dict)
            or not 1 <= len(files) <= MAX_FILES
            or not isinstance(manifest.get("source"), dict)
            or any(manifest["source"].get(key) != parent.source.get(key) for key in SOURCE_FIELDS)
            or manifest.get("episode_index") != parent.source.get("replay_episode")
        ):
            raise SceneError("Saved scene manifest identity could not be verified.")
        required = {
            "scene.usda",
            "structure.json",
            "proposal.json",
            "review.json",
            "evidence.json",
            "dataset-catalog.json",
            "trajectory.json",
        }
        if not required.issubset(files):
            raise SceneError("The saved scene bundle is incomplete.")
        found, total = set(), 0
        for path in bundle.rglob("*"):
            if path.is_symlink():
                raise SceneError("Saved scene assets contain a linked path.")
            if path.is_dir():
                continue
            name = str(path.relative_to(bundle))
            if name == "manifest.json":
                continue
            name = relative(name)
            total += regular(path, root)
            if total > MAX_BYTES or len(found) >= MAX_FILES:
                raise SceneError("Saved scene assets exceed their bounded snapshot size.")
            expected = files.get(name)
            if (
                not isinstance(expected, str)
                or not SHA256.fullmatch(expected)
                or sha(path) != expected
            ):
                raise SceneError("A saved scene asset no longer matches its manifest.")
            found.add(name)
        if found != set(files):
            raise SceneError("The saved scene manifest has missing or unexpected assets.")
        if read_json(bundle / "structure.json", root).get("passed") is not True:
            raise SceneError("The saved scene did not pass structural validation.")
        proposal = Proposal.model_validate(read_json(bundle / "proposal.json", root))
        review = Review.model_validate(read_json(bundle / "review.json", root))
        if (
            not proposal.matches_template
            or not review.supported
            or not review.consistent_with_images
        ):
            raise SceneError("Saved agent review does not support this reconstruction.")

        catalog_hash = parent.source.get("catalog_sha256")
        trajectory_hash = parent.source.get("trajectory_sha256")
        for name, expected in (
            ("dataset-catalog.json", catalog_hash),
            ("trajectory.json", trajectory_hash),
        ):
            if (
                not isinstance(expected, str)
                or not SHA256.fullmatch(expected)
                or files[name] != expected
                or regular(root / name, root) > 8 * 1024**2
                or sha(root / name) != expected
            ):
                raise SceneError("Saved dataset evidence hashes no longer match.")
        catalog = read_json(bundle / "dataset-catalog.json", root)
        episodes = catalog.get("episodes")
        if (
            catalog.get("scope") != "entire_dataset"
            or catalog.get("total_episodes") != profile.total_episodes
            or not isinstance(episodes, list)
            or len(episodes) != profile.total_episodes
            or sorted(ep["episode_index"] for ep in episodes) != list(range(profile.total_episodes))
        ):
            raise SceneError("The saved source catalog does not cover the complete dataset.")
        catalog_source = catalog.get("source")
        expected_cameras = {
            key for key, value in profile.features.items() if value.get("dtype") == "video"
        }
        if not isinstance(catalog_source, dict) or any(
            catalog_source.get(key) != parent.source[key]
            for key in ("repo_id", "revision", "metadata_sha256")
        ):
            raise SceneError("The saved camera catalog belongs to a different source.")
        video_prefix = (
            f"https://huggingface.co/datasets/{profile.repo_id}/resolve/{profile.revision}/"
        )
        total_frames = 0
        for episode in episodes:
            count = episode.get("frame_count")
            cameras = episode.get("cameras")
            if (
                type(count) is not int
                or count < 1
                or not isinstance(cameras, list)
                or {camera["key"] for camera in cameras} != expected_cameras
                or any(
                    not isinstance(camera.get("url"), str)
                    or not camera["url"].startswith(video_prefix)
                    for camera in cameras
                )
            ):
                raise SceneError("The saved dataset catalog has incomplete episode videos.")
            total_frames += count
        if total_frames != profile.total_frames:
            raise SceneError("The saved dataset catalog has incomplete recorded frames.")
        selected = parent.source.get("replay_episode")
        episode = next((ep for ep in episodes if ep["episode_index"] == selected), None)
        trajectory = read_json(bundle / "trajectory.json", root, 8 * 1024**2)
        frames = trajectory.get("frames")
        expected_frames = parent.source.get("replay_frames")
        if (
            type(selected) is not int
            or episode is None
            or type(expected_frames) is not int
            or not 2 <= expected_frames <= 18000
            or episode.get("frame_count") != expected_frames
            or trajectory.get("episode_index") != selected
            or trajectory.get("fps") != profile.fps
            or trajectory.get("mapping_status") != "uncalibrated_dataset_normalized"
            or not isinstance(frames, list)
            or len(frames) != expected_frames
        ):
            raise SceneError("The saved replay does not contain its complete source episode.")
        for index, frame in enumerate(frames):
            stamp, positions = frame["timestamp"], frame["positions"]
            if (
                type(stamp) not in {int, float}
                or not math.isfinite(stamp)
                or abs(stamp - index / profile.fps) > 0.5 / profile.fps
                or not isinstance(positions, list)
                or len(positions) != 6
                or any(
                    type(value) not in {int, float} or not math.isfinite(value) or abs(value) > 200
                    for value in positions
                )
            ):
                raise SceneError("The saved replay contains invalid recorded joint states.")
        return SavedScene(parent, bundle, files, catalog_hash, trajectory_hash)
    except SceneError:
        raise
    except OSError, ValueError, KeyError, TypeError, AttributeError:
        raise SceneError("The saved scene inputs are unavailable or invalid.") from None


def copy_verified(origin, target, expected, root):
    regular(origin, root)
    descriptor = os.open(origin, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as source, target.open("xb") as output:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise SceneError("Saved scene input changed while copying.")
        size = 0
        while part := source.read(1024**2):
            size += len(part)
            if size > MAX_BYTES:
                raise SceneError("Saved scene input exceeds its copy bound.")
            digest.update(part)
            output.write(part)
    if digest.hexdigest() != expected:
        target.unlink(missing_ok=True)
        raise SceneError("Saved scene input changed while copying.")


def restore(scenes, run, saved):
    """Copy a verified parent into this new run; the parent stays unchanged."""
    from vla_platform.scenes.service import write

    root = scenes.folder(run.id)
    parent_root = scenes.folder(saved.parent.id)
    total = sum(regular(saved.bundle / name, parent_root) for name in saved.files)
    if shutil.disk_usage(root).free < 10 * 1024**3 + 2 * total:
        raise SceneError("Not enough disk space to copy the saved scene and retain headroom.")
    work = root / "attempt-1"
    bundle = work / "bundle"
    bundle.mkdir(parents=True, mode=0o700)
    for name, expected in saved.files.items():
        destination = bundle / relative(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_verified(saved.bundle / name, destination, expected, parent_root)
    for name in ("dataset-catalog.json", "trajectory.json", "evidence.json"):
        copy_verified(bundle / name, root / name, saved.files[name], root)
    for name in ("proposal.json", "review.json"):
        copy_verified(bundle / name, work / name, saved.files[name], root)

    run.attempt = 1
    run.agent = saved.parent.agent.model_copy(deep=True)
    run.source = copy.deepcopy(saved.parent.source)
    run.source.pop("resource_cleanup", None)
    run.source["resumed_from_scene_id"] = saved.parent.id
    run.gates = {key: key in CORE_GATES for key in run.gates}
    run.media = []
    run.artifacts = [
        "dataset-catalog.json",
        "trajectory.json",
        "evidence.json",
        "attempt-1/proposal.json",
        "attempt-1/review.json",
        "attempt-1/bundle/structure.json",
    ]
    for media in saved.parent.media:
        if media.role != "source_image":
            continue
        name = relative(media.path)
        if "/" in name or name not in saved.files:
            raise SceneError("Saved source image is outside the verified scene bundle.")
        copy_verified(bundle / name, root / name, saved.files[name], root)
        run.media.append(media.model_copy(deep=True))
        run.artifacts.append(name)
    write(
        root / "saved-scene.json",
        {
            "source_scene_id": saved.parent.id,
            "parent_job_id": saved.parent.job_id,
            "source_manifest_sha256": sha(saved.bundle / "manifest.json"),
            "catalog_sha256": saved.catalog_hash,
            "trajectory_sha256": saved.trajectory_hash,
        },
    )
    run.artifacts.append("saved-scene.json")
    return bundle


async def restore_owned(scenes, run, saved):
    """Cancellation drains this run's copy before publishing its terminal state."""
    task = asyncio.create_task(asyncio.to_thread(restore, scenes, run, saved))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            if not cancelled:
                raise
    if cancelled:
        task.exception()
        raise asyncio.CancelledError
    return task.result()
