"""Filesystem recovery tests; fixture bytes are not a claim of native model reload."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from firebird_vla.checkpoint import (
    publish_checkpoint,
    resolve_checkpoint,
    sha256,
    validate_resume_recipe,
    validate_resume_state,
    verify_training_checkpoint,
    write_json,
)
from firebird_vla.config import TrainConfig


def bundle(path, step=1):
    path.mkdir()
    for name in (
        "recipe.json",
        "stats.json",
        "splits.json",
        "training.pt",
        "probe.safetensors",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "policy/config.json",
    ):
        target = path / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"CPU integrity fixture, not a serialized model")
    write_json(path / "recipe.json", TrainConfig().to_dict())
    write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "step": step,
            "files": {
                p.relative_to(path).as_posix(): sha256(p) for p in path.rglob("*") if p.is_file()
            },
        },
    )
    return path


@pytest.mark.parametrize("pointer", [None, "{", "{}", "stale", "unsafe"])
def test_recovers_latest_complete_checkpoint_independently_of_pointer(tmp_path, pointer):
    previous = bundle(tmp_path / "checkpoint-000001")
    latest = bundle(tmp_path / "checkpoint-000002", 2)
    bundle(tmp_path / ".checkpoint-uncommitted", 3)
    partial = bundle(tmp_path / "checkpoint-000004", 4)
    (partial / "training.pt").unlink()
    if pointer == "stale":
        write_json(tmp_path / "latest.json", {"checkpoint": previous.name, "step": 1})
    elif pointer == "unsafe":
        write_json(tmp_path / "latest.json", {"checkpoint": "../../outside", "step": 100})
    elif pointer is not None:
        (tmp_path / "latest.json").write_text(pointer)
    assert resolve_checkpoint(tmp_path) == latest
    assert resolve_checkpoint(previous) == previous


@pytest.mark.parametrize("phase", ["before_bundle_rename", "before_pointer_replace"])
def test_killed_publication_recovers_last_committed_bundle(tmp_path, phase):
    previous = publish_checkpoint(
        bundle(tmp_path / ".checkpoint-old"), tmp_path / "checkpoint-000001"
    )
    staging = bundle(tmp_path / ".checkpoint-new", 2)
    destination = tmp_path / "checkpoint-000002"
    ready = tmp_path / "ready"
    # Stop a real subprocess at the filesystem commit boundary. All bundle writes,
    # hashes, fsyncs and any preceding rename execute normally in the child.
    program = """
import os, sys, time
from pathlib import Path
from firebird_vla.checkpoint import publish_checkpoint
staging, destination, ready, phase = sys.argv[1:]
operation = 'rename' if phase == 'before_bundle_rename' else 'replace'
original = getattr(os, operation)
def pause(source, target):
    if str(target).endswith('checkpoint-000002' if operation == 'rename' else 'latest.json'):
        Path(ready).write_text('ready')
        while True:
            time.sleep(1)
    return original(source, target)
setattr(os, operation, pause)
publish_checkpoint(staging, destination)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(staging), str(destination), str(ready), phase]
    )
    try:
        import time

        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), f"Publication did not reach {phase}; exit={process.poll()}"
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert json.loads((tmp_path / "latest.json").read_text()) == {
        "checkpoint": previous.name,
        "step": 1,
    }
    expected = previous if phase == "before_bundle_rename" else destination
    assert resolve_checkpoint(tmp_path) == expected
    assert verify_training_checkpoint(expected)["step"] == (1 if expected == previous else 2)


def test_publication_refuses_to_replace_existing_evidence(tmp_path):
    destination = publish_checkpoint(
        bundle(tmp_path / ".checkpoint-a"), tmp_path / "checkpoint-000001"
    )
    manifest = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_checkpoint(bundle(tmp_path / ".checkpoint-b"), destination)
    assert (destination / "manifest.json").read_bytes() == manifest


def test_incomplete_bundle_is_not_published(tmp_path):
    staging = bundle(tmp_path / ".checkpoint-new")
    manifest = json.loads((staging / "manifest.json").read_text())
    (staging / "training.pt").unlink()
    del manifest["files"]["training.pt"]
    write_json(staging / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Incomplete training checkpoint"):
        publish_checkpoint(staging, tmp_path / "checkpoint-000001")
    assert not (tmp_path / "checkpoint-000001").exists()
    assert not (tmp_path / "latest.json").exists()


@pytest.mark.parametrize("mutation", ["corrupt", "step", "manifest", "symlink", "internal_symlink"])
def test_explicit_bad_checkpoint_never_silently_falls_back(tmp_path, mutation):
    bundle(tmp_path / "checkpoint-000001")
    latest = bundle(tmp_path / "checkpoint-000002", 2)
    if mutation == "corrupt":
        (latest / "training.pt").write_bytes(b"broken")
    elif mutation == "step":
        manifest = json.loads((latest / "manifest.json").read_text())
        manifest["step"] = 1
        write_json(latest / "manifest.json", manifest)
    elif mutation == "manifest":
        (latest / "manifest.json").write_text("{")
    elif mutation == "symlink":
        link = tmp_path / "checkpoint-000003"
        link.symlink_to(latest, target_is_directory=True)
        latest = link
    else:
        (latest / "linked").symlink_to(latest / "adapter", target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_checkpoint(latest)


def test_no_complete_checkpoint_has_clear_error(tmp_path):
    candidate = bundle(tmp_path / "checkpoint-000001")
    (candidate / "manifest.json").unlink()
    with pytest.raises(ValueError, match="No complete, integrity-verified training checkpoint"):
        resolve_checkpoint(tmp_path)


@pytest.mark.parametrize("state", [None, {}, {"step": True}, {"step": 2}, {"step": 0}])
def test_resume_rejects_inconsistent_manifest_step(state):
    with pytest.raises(ValueError, match="training state"):
        validate_resume_state(state, {"step": 1}, TrainConfig())


@pytest.mark.parametrize("consumed", [None, -1, True, 7, 8.0])
def test_resume_rejects_inconsistent_batch_cursor(consumed):
    with pytest.raises(ValueError, match="batch cursor"):
        validate_resume_state({"step": 1, "consumed_batches": consumed}, {"step": 1}, TrainConfig())


def test_resume_retains_valid_cursor_and_rejects_completed_recipe():
    state = {"step": 1, "consumed_batches": 8}
    validate_resume_state(state, {"step": 1}, TrainConfig())
    with pytest.raises(ValueError, match="already completed"):
        validate_resume_state(state, {"step": 1}, replace(TrainConfig(), steps=1))


def test_latest_valid_but_incompatible_checkpoint_never_falls_back(tmp_path):
    bundle(tmp_path / "checkpoint-000001")
    latest = bundle(tmp_path / "checkpoint-000002", 2)
    original = TrainConfig.load(latest / "recipe.json")
    selected = resolve_checkpoint(tmp_path)
    assert selected == latest
    with pytest.raises(ValueError, match="original recipe"):
        validate_resume_recipe(selected, replace(original, seed=original.seed + 1))
    validate_resume_recipe(selected, replace(original, output_dir="another-run"))


def test_invalid_json_never_replaces_previous_pointer(tmp_path):
    path = tmp_path / "latest.json"
    write_json(path, {"step": 1})
    with pytest.raises(ValueError):
        write_json(path, {"step": float("nan")})
    assert json.loads(path.read_text()) == {"step": 1}
