import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest, WorkerRequest
from vla_platform.datasets.inspect import MAX_METADATA_BYTES, bounded_get, inspect_local, profile
from vla_platform.settings import Settings


@pytest.fixture
def local_dataset(tmp_path: Path) -> Path:
    directory = tmp_path / "datasets" / "fixture" / "meta"
    directory.mkdir(parents=True)
    (directory / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "test_fixture",
                "total_episodes": 2,
                "total_frames": 20,
                "fps": 10,
                "features": {
                    "action": {"dtype": "float32", "shape": [6]},
                    "observation.state": {"dtype": "float32", "shape": [6]},
                },
            }
        )
    )
    return directory.parent


def wait_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = client.get(f"/api/v1/jobs/{job_id}").json()
        if record["status"] not in {"queued", "running"}:
            return record
        time.sleep(0.05)
    raise AssertionError("Inspection did not terminate")


def test_real_subprocess_local_intake_persists_across_restart(tmp_path, local_dataset):
    settings = Settings(data_dir=tmp_path / "workspace", local_root=local_dataset.parent)
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/v1/projects", json={"name": "Fixture project"}).json()
        response = client.post(
            f"/api/v1/projects/{project['id']}/intakes",
            json={
                "source": "local",
                "path": "fixture",
            },
        )
        assert response.status_code == 202
        record = wait_job(client, response.json()["id"])
        assert record["status"] == "succeeded", record
        assert record["result"]["total_frames"] == 20
        assert record["result"]["inspection_scope"] == "metadata_only"
        assert "metadata-sha256:" in record["result"]["revision"]
        assert any("not established" in warning for warning in record["result"]["warnings"])
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/projects").json() == [project]
        assert client.get(f"/api/v1/projects/{project['id']}/jobs").json() == [record]


def test_invalid_and_disabled_requests(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.post("/api/v1/projects", json={"name": "  "}).status_code == 422
        project = client.post("/api/v1/projects", json={"name": "Test"}).json()
        endpoint = f"/api/v1/projects/{project['id']}/intakes"
        assert client.post(endpoint, json={"source": "local", "path": "."}).status_code == 422
        assert (
            client.post(endpoint, json={"repo_id": "https://attacker.invalid"}).status_code == 422
        )
        assert (
            client.post("/api/v1/projects/missing/intakes", json={"repo_id": "x/y"}).status_code
            == 404
        )
        assert (
            client.post(
                "/api/v1/projects",
                json={"name": "bad origin"},
                headers={
                    "Origin": "https://attacker.invalid",
                },
            ).status_code
            == 403
        )
        assert (
            client.get("/api/v1/projects", headers={"Host": "attacker.invalid"}).status_code == 400
        )
        capabilities = client.get("/api/v1/capabilities").json()
        assert [c["stage"] for c in capabilities if c["status"] == "available"] == ["Dataset"]


def test_failed_local_job_does_not_publish_result(tmp_path, local_dataset):
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "state", local_root=local_dataset.parent))
    ) as client:
        project = client.post("/api/v1/projects", json={"name": "Failure"}).json()
        response = client.post(
            f"/api/v1/projects/{project['id']}/intakes",
            json={
                "source": "local",
                "path": "does-not-exist",
            },
        )
        job = wait_job(client, response.json()["id"])
        assert job["status"] == "failed"
        assert job["result"] is None and job["error"]


def test_local_path_escape_and_malformed_metadata(tmp_path, local_dataset):
    request = IntakeRequest(source="local", path=str(local_dataset))
    with pytest.raises(ValueError, match="within"):
        inspect_local(request, str(tmp_path / "datasets" / "fixture" / "meta"))
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        profile(b'{"codebase_version":"unknown"}', request, "test")
    with pytest.raises(ValueError, match="2 MiB"):
        profile(b"x" * (MAX_METADATA_BYTES + 1), request, "test")
    with pytest.raises(ValueError):
        WorkerRequest.model_validate({"schema_version": 99, "intake": request.model_dump()})


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_hub_errors_are_not_success(status):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status, json={"error": "unavailable"})
            )
        ) as client:
            with pytest.raises((ValueError, httpx.HTTPStatusError)):
                await bounded_get(client, "https://huggingface.co/api/datasets/test/data")

    asyncio.run(run())


def test_hub_download_limit():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=b"x" * (MAX_METADATA_BYTES + 1))
            )
        ) as client:
            with pytest.raises(ValueError, match="2 MiB"):
                await bounded_get(client, "https://huggingface.co/test")

    asyncio.run(run())
