"""Application snapshot admission uses real files; no training quality claims."""

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_dataset_snapshots import build_dataset, mutate_table
from test_dataset_snapshots import dataset as dataset
from vla_platform.api import create_app
from vla_platform.compute_settings import ComputeSettings, ComputeSettingsUpdate
from vla_platform.contracts import (
    DatasetProfile,
    DatasetSnapshot,
    IntakeRequest,
    WorkerRequest,
    WorkerResult,
)
from vla_platform.datasets.worker import run
from vla_platform.settings import Settings

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Secure snapshots require POSIX")


def wait(client, ident):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        job = client.get("/api/v1/jobs/" + ident).json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.04)
    raise AssertionError("Snapshot job did not finish")


def test_snapshot_is_full_copy_and_not_reused_after_row_change(dataset, tmp_path):
    settings = Settings(data_dir=tmp_path / "workspace", local_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        pid = client.post("/api/v1/projects", json={"name": "Snapshot fixture"}).json()["id"]
        route = f"/api/v1/projects/{pid}/intakes"
        payload = {"source": "local", "path": str(dataset), "snapshot_for_training": True}
        accepted = client.post(route, json=payload)
        assert accepted.status_code == 202, accepted.text
        first = wait(client, accepted.json()["id"])
        assert first["status"] == "succeeded", first
        assert first["result"]["inspection_scope"] == "complete_snapshot"
        assert not any("Metadata-only" in warning for warning in first["result"]["warnings"])
        snapshot = first["result"]["snapshot"]
        assert "path" not in snapshot
        assert snapshot["total_frames"] == 12
        assert snapshot["lineage_validated"] is False
        mutate_table(dataset, "data/chunk-000/file-000.parquet", "rows[0]['action']=[0.5,1.]")
        second = wait(client, client.post(route, json=payload).json()["id"])
        assert second["status"] == "succeeded", second
        assert second["id"] != first["id"]
        assert second["result"]["snapshot"]["id"] != snapshot["id"]
        old = settings.data_dir / "dataset-snapshots" / snapshot["manifest_sha256"]
        assert old.is_dir()
        # A normal metadata check may not silently reuse the stronger, older receipt.
        metadata = wait(
            client,
            client.post(route, json={**payload, "snapshot_for_training": False}).json()["id"],
        )
        assert metadata["status"] == "succeeded"
        assert "snapshot" not in metadata["result"]
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/jobs/" + first["id"]).json()["result"]["snapshot"] == snapshot


def save_to_library(client, project, name, source):
    """Save a folder the way a finished robot recording is saved: upload, then finish."""
    entry = client.post(f"/api/v1/projects/{project}/dataset-uploads", json={"name": name}).json()
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        uploaded = client.put(
            f"/api/v1/dataset-uploads/{entry['id']}/file",
            params={"path": path.relative_to(source).as_posix()},
            content=path.read_bytes(),
        )
        assert uploaded.status_code == 200, uploaded.text
    saved = client.post(f"/api/v1/dataset-uploads/{entry['id']}/finish")
    assert saved.status_code == 200, saved.text
    assert saved.json()["status"] == "ready", saved.json()
    return saved.json()


def test_single_episode_recording_opens_and_says_why_it_cannot_train(tmp_path):
    """Looking at a dataset must never depend on it being ready to train on.

    Regression: opening a saved dataset asks for a training copy by default, and a
    one-episode recording made that whole inspection fail with a validator error.
    """
    settings = Settings(data_dir=tmp_path / "workspace", local_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        pid = client.post("/api/v1/projects", json={"name": "One episode"}).json()["id"]
        saved = save_to_library(
            client, pid, "First take", build_dataset(tmp_path / "take", episodes=1)
        )
        route = f"/api/v1/projects/{pid}/intakes"
        # Exactly what the Datasets page sends when a saved dataset is opened.
        payload = {
            "source": "local",
            "library_id": saved["id"],
            "revision": "main",
            "snapshot_for_training": True,
        }
        job = wait(client, client.post(route, json=payload).json()["id"])
        assert job["status"] == "succeeded", job
        assert job.get("error") is None
        result = job["result"]
        assert result["total_episodes"] == 1 and result["total_frames"] == 3
        assert "snapshot" not in result and result["inspection_scope"] != "complete_snapshot"
        blocker = result["training_blocker"]
        assert "at least 2" in blocker and "has 1" in blocker
        assert "Dataset validation failed" not in blocker
        # Nothing half-made is left where training would look for it.
        assert not list((settings.data_dir / "dataset-snapshots").glob("[a-f0-9]" * 64))
        # The recorded frames the page plays back are served for this dataset.
        frames = client.get(f"/api/v1/datasets/{saved['id']}/samples")
        assert frames.status_code == 200, frames.text
        assert {frame["episode_index"] for frame in frames.json()} == {0}


def test_one_more_episode_makes_the_same_folder_trainable(tmp_path):
    source = build_dataset(tmp_path / "source", episodes=1)
    settings = Settings(data_dir=tmp_path / "workspace", local_root=tmp_path)
    with TestClient(create_app(settings)) as client:
        pid = client.post("/api/v1/projects", json={"name": "Growing"}).json()["id"]
        route = f"/api/v1/projects/{pid}/intakes"
        payload = {"source": "local", "path": str(source), "snapshot_for_training": True}
        first = wait(client, client.post(route, json=payload).json()["id"])
        assert first["status"] == "succeeded", first
        assert "has 1" in first["result"]["training_blocker"]
        shutil.rmtree(source)
        build_dataset(source, episodes=2)
        # A blocked result is never reused: the next inspection reads the source again.
        ready = wait(client, client.post(route, json=payload).json()["id"])
        assert ready["status"] == "succeeded", ready
        assert ready["id"] != first["id"]
        assert "training_blocker" not in ready["result"]
        assert ready["result"]["snapshot"]["total_episodes"] == 2


def test_snapshot_worker_reports_missing_media_without_publishing(dataset, tmp_path):
    next(dataset.glob("videos/**/*.mp4")).unlink()
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(
        WorkerRequest(
            intake=IntakeRequest(source="local", path=str(dataset), snapshot_for_training=True),
            local_root=str(tmp_path),
            snapshot_store=str(tmp_path / "store"),
        ).model_dump_json()
    )
    asyncio.run(run(request, result))
    response = WorkerResult.model_validate_json(result.read_bytes())
    # The metadata is readable, so the dataset opens; it is simply not trainable.
    assert response.error is None and response.result.snapshot is None
    assert response.result.training_blocker
    assert response.result.inspection_scope != "complete_snapshot"
    assert not list((tmp_path / "store").glob("[a-f0-9]" * 64))


def test_unreadable_dataset_still_fails_inspection(tmp_path):
    (tmp_path / "broken" / "meta").mkdir(parents=True)
    (tmp_path / "broken" / "meta" / "info.json").write_text("{not json")
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(
        WorkerRequest(
            intake=IntakeRequest(
                source="local", path=str(tmp_path / "broken"), snapshot_for_training=True
            ),
            local_root=str(tmp_path),
            snapshot_store=str(tmp_path / "store"),
        ).model_dump_json()
    )
    asyncio.run(run(request, result))
    response = WorkerResult.model_validate_json(result.read_bytes())
    assert response.result is None and response.error


def test_profile_cannot_claim_a_training_copy_and_a_blocker():
    snapshot = dict(
        schema_version=1,
        id="sha256:" + "a" * 64,
        manifest_sha256="a" * 64,
        format="lerobot_v3",
        total_bytes=1,
        file_count=1,
        total_episodes=2,
        total_frames=3,
        lineage_validated=False,
        warnings=[],
    )
    with pytest.raises(ValueError, match="no training blocker"):
        DatasetProfile(
            source="local",
            revision="metadata-sha256:" + "b" * 64,
            format="lerobot_v3",
            total_episodes=2,
            total_frames=3,
            fps=10,
            features={"action": {}},
            metadata_sha256="b" * 64,
            inspected_at="2026-01-01T00:00:00Z",
            warnings=[],
            inspection_scope="complete_snapshot",
            snapshot=snapshot,
            training_blocker="Not enough episodes",
        )


def test_snapshot_contract_rejects_caller_paths_and_hf_request():
    with pytest.raises(ValueError):
        IntakeRequest(repo_id="org/data", snapshot_for_training=True)
    with pytest.raises(ValueError):
        DatasetSnapshot(
            schema_version=1,
            id="sha256:" + "a" * 64,
            manifest_sha256="b" * 64,
            format="lerobot_v3",
            total_bytes=1,
            file_count=1,
            total_episodes=2,
            total_frames=3,
            lineage_validated=False,
            warnings=[],
        )


def training_runtime(tmp_path):
    """One CUDA runtime that offers SmolVLA; nothing is ever executed on it."""
    config = tmp_path / "runtimes.json"
    config.write_text(
        json.dumps(
            {
                "runtimes": [
                    {
                        "id": "fixture",
                        "label": "protocol fixture",
                        "python": sys.executable,
                        "worker_root": str(Path(__file__).parent / "fixtures/native_worker"),
                        "vendor": str(tmp_path),
                        "build": str(tmp_path),
                        "device": "cuda",
                        "training_python": sys.executable,
                        "training_root": "fixture",
                        "training_model_ids": ["smolvla"],
                    }
                ],
                "sources": [],
            }
        )
    )
    settings = Settings(data_dir=tmp_path / "workspace", local_root=tmp_path, runtime_config=config)
    ComputeSettings(settings.data_dir).update(
        ComputeSettingsUpdate.model_validate({"local": {"enabled": True}})
    )
    return settings


@pytest.mark.parametrize("episodes", [1, 4])
def test_training_refusals_on_a_recorded_dataset_say_what_to_do(tmp_path, episodes):
    """ "I cannot run LoRA on SmolVLA": the old refusal named neither cause nor remedy."""
    source = build_dataset(tmp_path / "source", episodes=episodes)
    with TestClient(create_app(training_runtime(tmp_path))) as client:
        pid = client.post("/api/v1/projects", json={"name": "Refusals"}).json()["id"]
        payload = {"source": "local", "path": str(source), "snapshot_for_training": True}
        dataset = wait(
            client, client.post(f"/api/v1/projects/{pid}/intakes", json=payload).json()["id"]
        )
        assert dataset["status"] == "succeeded", dataset
        refused = client.post(
            f"/api/v1/projects/{pid}/policy-jobs",
            json={
                "operation": "policy.finetune",
                "runtime_id": "fixture",
                "dataset_job_id": dataset["id"],
                "training_method": "lora",
            },
        )
        assert refused.status_code == 422, refused.text
        detail = refused.json()["detail"]
        if episodes == 1:
            # Too few episodes: say that, with the count, before anything about models.
            assert "not ready for training" in detail and "has 1" in detail
        else:
            # Enough data, but this method cannot read a recorded dataset.
            assert "SmolVLA LORA trains from a Hugging Face dataset" in detail
            assert "Publish this dataset to Hugging Face" in detail
        assert "native LeRobot model" not in detail
