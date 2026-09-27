"""Snapshot identity and lineage gates run without Torch or network access."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from firebird_vla.lerobot_application import cli_arguments, resolve_recipe
from firebird_vla.local_dataset import (
    load_operation_snapshot,
    make_local_datasets,
    offline_dataset_loading,
    split_lineage,
    verify_local_snapshot,
)
from firebird_vla.native_profiles import NATIVE_PROFILES


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@pytest.fixture
def local_job(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.images.front": {"dtype": "video", "shape": [32, 32, 3]},
        "observation.state": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [6]},
    }
    data = b'{"fixture":"identity test only, not a complete dataset"}'
    (root / "meta/info.json").write_bytes(data)
    manifest = {
        "schema_version": 1,
        "format": "lerobot_v3",
        "files": [
            {
                "path": "meta/info.json",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
        "total_episodes": 4,
        "total_frames": 12,
        "features": features,
        "lineage_validated": True,
        "lineage": [
            {
                "episode_index": i,
                "origin": "recorded" if i % 2 == 0 else "augmented",
                "lineage_group": f"root:{i // 2}",
            }
            for i in range(4)
        ],
        "warnings": [],
    }
    raw = canonical(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    (root / "firebird-snapshot.json").write_bytes(raw)
    profile = NATIVE_PROFILES["act"]
    return {
        "parameters": {
            "training_method": "full",
            "training": {
                "model_id": profile["model_id"],
                "model_revision": profile["model_revision"],
                "steps": 2,
                "camera_keys": ["observation.images.front"],
            },
        },
        "dataset": {
            "source": "local",
            "repo_id": "original-name",
            "revision": "metadata-sha256:old",
            "format": "lerobot_v3",
            "features": features,
        },
        "dataset_snapshot": {
            "path": str(root),
            "manifest_sha256": digest,
            "id": "sha256:" + digest,
        },
    }


def test_local_recipe_binds_content_identity_not_original_path(local_job):
    recipe, profile = resolve_recipe(local_job)
    assert recipe["dataset_source"] == "local"
    assert recipe["dataset_revision"] == local_job["dataset_snapshot"]["id"]
    assert recipe["dataset_manifest_sha256"] == local_job["dataset_snapshot"]["manifest_sha256"]
    assert local_job["dataset_snapshot"]["path"] not in json.dumps(recipe)
    splits = recipe["dataset_splits"]
    assert set(splits["train_groups"]).isdisjoint(splits["validation_groups"])
    assert set(splits["train"]) in ({0, 1}, {2, 3})
    args = cli_arguments(recipe, profile, output="out", dataset_root="inputs/dataset")
    assert "--dataset.root=inputs/dataset" in args


@pytest.mark.parametrize(
    "case", ["mutation", "extra", "symlink", "fifo", "metadata", "missingpointer", "identity"]
)
def test_invalid_local_input_rejected_before_training(local_job, case):
    pointer = local_job["dataset_snapshot"]
    root = Path(pointer["path"])
    if case == "mutation":
        (root / "meta/info.json").write_bytes(b"bad")
    elif case == "extra":
        (root / "extra").write_text("unexpected")
    elif case == "symlink":
        (root / "link").symlink_to(root / "meta/info.json")
    elif case == "fifo":
        os.mkfifo(root / "pipe")
    elif case == "metadata":
        local_job["dataset"]["features"] = {}
    elif case == "missingpointer":
        del local_job["dataset_snapshot"]
    elif case == "identity":
        pointer["id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        resolve_recipe(local_job)


def test_saved_recipe_and_operation_pointer_must_agree(local_job, tmp_path):
    recipe, _ = resolve_recipe(local_job)
    (tmp_path / "dataset-snapshot.json").write_text(json.dumps(local_job["dataset_snapshot"]))
    root, manifest = load_operation_snapshot(tmp_path, recipe)
    assert root == Path(local_job["dataset_snapshot"]["path"])
    recipe["dataset_splits"]["train"] = [0]
    with pytest.raises(ValueError, match="lineage split"):
        load_operation_snapshot(tmp_path, recipe)
    manifest["lineage"] = [{**row, "lineage_group": "same-root"} for row in manifest["lineage"]]
    with pytest.raises(ValueError, match="two distinct"):
        split_lineage(manifest, 0.2, 42)


def test_group_split_is_reproducible_and_unknown_ancestry_stays_unvalidated(local_job):
    _, manifest = verify_local_snapshot(local_job["dataset_snapshot"])
    manifest["lineage_validated"] = False
    assert split_lineage(manifest, 0.2, 42) == split_lineage(manifest, 0.2, 42)
    assert split_lineage(manifest, 0.2, 42)["lineage_validated"] is False


def test_local_factory_selects_exact_groups_and_disables_validation_augmentation(
    local_job, monkeypatch
):
    import sys
    from types import ModuleType

    factory = ModuleType("lerobot.datasets.factory")
    factory.make_dataset = Mock(side_effect=lambda cfg: cfg)
    metadata = ModuleType("lerobot.datasets.dataset_metadata")
    metadata.snapshot_download = Mock()
    dataset = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset.snapshot_download = Mock()
    for name, module in {
        "lerobot": ModuleType("lerobot"),
        "lerobot.datasets": ModuleType("lerobot.datasets"),
        "lerobot.datasets.factory": factory,
        "lerobot.datasets.dataset_metadata": metadata,
        "lerobot.datasets.lerobot_dataset": dataset,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    recipe, _ = resolve_recipe(local_job)
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(episodes=None, image_transforms=SimpleNamespace(enable=True))
    )
    training, validation = make_local_datasets(cfg, recipe)
    assert training.dataset.episodes == recipe["dataset_splits"]["train"]
    assert validation.dataset.episodes == recipe["dataset_splits"]["validation"]
    assert training.dataset.image_transforms.enable is True
    assert validation.dataset.image_transforms.enable is False
    assert cfg.dataset.episodes is None
    with offline_dataset_loading(), pytest.raises(ValueError, match="fallback is disabled"):
        metadata.snapshot_download("must-not-fetch")
    metadata.snapshot_download.assert_not_called()
