"""Application snapshot admission uses real files; no training quality claims."""

import asyncio
import os
import time

import pytest
from fastapi.testclient import TestClient
from test_dataset_snapshots import dataset as dataset
from test_dataset_snapshots import mutate_table
from vla_platform.api import create_app
from vla_platform.contracts import (
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


def test_snapshot_worker_rejects_missing_media_without_publishing(dataset, tmp_path):
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
    assert response.result is None and response.error
    assert not list((tmp_path / "store").glob("[a-f0-9]" * 64))


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
