"""Read verified Firebird local snapshots without a dependency on the application."""

import hashlib
import json
import math
import os
import random
import re
import stat
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

MANIFEST = "firebird-snapshot.json"
MAX_BYTES = 16 * 1024**3


def _open(root, name):
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise ValueError("Local snapshot training requires secure POSIX file opens")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not relative.parts:
        raise ValueError("Unsafe dataset snapshot path")
    root = Path(root).absolute()
    if ".." in root.parts:
        raise ValueError("Unsafe dataset snapshot root")
    fd = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = (*root.parts[1:], *relative.parts)
        for component in parts[:-1]:
            new = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        handle = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            os.close(handle)
            raise ValueError("Dataset snapshot contains a special file")
        return os.fdopen(handle, "rb")
    finally:
        os.close(fd)


def verify_local_snapshot(pointer):
    if not isinstance(pointer, dict) or set(pointer) != {"path", "manifest_sha256", "id"}:
        raise ValueError("Missing verified local snapshot pointer")
    digest = pointer["manifest_sha256"]
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or pointer["id"] != "sha256:" + digest
    ):
        raise ValueError("Invalid dataset snapshot identity")
    root = Path(pointer["path"])
    with _open(root, MANIFEST) as handle:
        raw = handle.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Dataset snapshot manifest identity mismatch")
    manifest = json.loads(raw)
    canonical = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if (
        raw != canonical
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("format") != "lerobot_v3"
    ):
        raise ValueError("Unsupported dataset snapshot manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 4096:
        raise ValueError("Dataset snapshot file count exceeds budget")
    expected = set()
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ValueError("Invalid dataset snapshot entry")
        name, size = item["path"], item["size"]
        if (
            not isinstance(name, str)
            or name in expected
            or name == MANIFEST
            or type(size) is not int
            or not 0 < size <= 4 * 1024**3
        ):
            raise ValueError("Invalid dataset snapshot inventory")
        expected.add(name)
        total += size
        if total > MAX_BYTES:
            raise ValueError("Dataset snapshot exceeds byte budget")
        digest = hashlib.sha256()
        with _open(root, name) as handle:
            before = os.fstat(handle.fileno())
            if before.st_size != size:
                raise ValueError("Dataset snapshot size mismatch")
            remaining = size
            while remaining:
                chunk = handle.read(min(1024**2, remaining))
                if not chunk:
                    raise ValueError("Dataset snapshot truncated during read")
                remaining -= len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, key) != getattr(after, key) for key in fields) or handle.read(1):
                raise ValueError("Dataset snapshot changed during read")
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Dataset snapshot file hash mismatch")
    actual = set()
    pending = [root]
    directories = 0
    while pending:
        directory = pending.pop()
        if directory.is_symlink():
            raise ValueError("Dataset snapshot directory must not be a symlink")
        for path in directory.iterdir():
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                directories += 1
                if directories > 4096:
                    raise ValueError("Dataset snapshot directory budget exceeded")
                pending.append(path)
            elif stat.S_ISREG(mode):
                actual.add(path.relative_to(root).as_posix())
                if len(actual) > 4097:
                    raise ValueError("Dataset snapshot file count exceeds budget")
            else:
                raise ValueError("Dataset snapshot contains a symlink or special file")
    if actual != expected | {MANIFEST}:
        raise ValueError("Dataset snapshot file inventory mismatch")
    return root, manifest


def bind_local_recipe(job, recipe):
    pointer = job.get("dataset_snapshot")
    _, manifest = verify_local_snapshot(pointer)
    data = job["dataset"]
    if data.get("format") != "lerobot_v3" or data.get("features") != manifest["features"]:
        raise ValueError("Local dataset profile differs from the verified snapshot")
    recipe.update(
        {
            "dataset_source": "local",
            "dataset_id": "firebird/local-" + pointer["manifest_sha256"][:16],
            "dataset_revision": pointer["id"],
            "dataset_manifest_sha256": pointer["manifest_sha256"],
            "dataset_lineage_validated": manifest["lineage_validated"],
        }
    )
    recipe["dataset_splits"] = split_lineage(
        manifest, recipe["validation_fraction"], recipe["seed"]
    )
    return recipe


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


def load_operation_snapshot(operation, recipe):
    pointer = json.loads((Path(operation) / "dataset-snapshot.json").read_text())
    root, manifest = verify_local_snapshot(pointer)
    if pointer["manifest_sha256"] != recipe.get("dataset_manifest_sha256") or pointer[
        "id"
    ] != recipe.get("dataset_revision"):
        raise ValueError("Saved recipe local dataset identity mismatch")
    if split_lineage(manifest, recipe["validation_fraction"], recipe["seed"]) != recipe.get(
        "dataset_splits"
    ):
        raise ValueError("Saved recipe dataset lineage split mismatch")
    return root, manifest


@contextmanager
def offline_dataset_loading():
    """Prevent upstream's missing-local-file recovery from silently contacting the Hub."""

    def reject(*args, **kwargs):
        raise ValueError("Verified local dataset is incomplete; Hub fallback is disabled")

    with (
        patch("lerobot.datasets.dataset_metadata.snapshot_download", reject),
        patch("lerobot.datasets.lerobot_dataset.snapshot_download", reject),
    ):
        yield


def make_local_datasets(cfg, recipe):
    from lerobot.datasets.factory import make_dataset

    train_cfg, eval_cfg = deepcopy(cfg), deepcopy(cfg)
    train_cfg.dataset.episodes = recipe["dataset_splits"]["train"]
    eval_cfg.dataset.episodes = recipe["dataset_splits"]["validation"]
    eval_cfg.dataset.image_transforms.enable = False
    with offline_dataset_loading():
        training = make_dataset(train_cfg)
        validation = make_dataset(eval_cfg)
    return training, validation
