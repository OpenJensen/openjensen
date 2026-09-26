"""Application integration with real subprocess fixtures, not ML/quality claims."""

import io
import json
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.settings import Settings


@pytest.fixture
def configured(tmp_path):
    root = Path(__file__).parent / "fixtures/native_worker"
    config = tmp_path / "runtimes.json"
    config.write_text(
        json.dumps(
            {
                "runtimes": [
                    {
                        "id": "fixture",
                        "label": "protocol fixture",
                        "python": sys.executable,
                        "worker_root": str(root.resolve()),
                        "vendor": str(tmp_path),
                        "build": str(tmp_path),
                        "simulator_lane": str(tmp_path),
                    }
                ],
                "sources": [
                    {
                        "id": "source",
                        "label": "Synthetic fixture",
                        "path": str(tmp_path / "source"),
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    return Settings(data_dir=tmp_path / "workspace", runtime_config=config)


def wait(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get("/api/v1/jobs/" + job_id).json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.025)
    raise AssertionError("Fixture job did not finish")


def project(client):
    return client.post("/api/v1/projects", json={"name": "Protocol fixture"}).json()["id"]


def submit(client, project_id, **fields):
    payload = {
        "operation": "policy.workflow",
        "runtime_id": "fixture",
        "source_id": "source",
        **fields,
    }
    response = client.post(f"/api/v1/projects/{project_id}/policy-jobs", json=payload)
    assert response.status_code == 202, response.text
    return response.json()["id"]


def set_label(settings, label):
    catalog = json.loads(settings.runtime_config.read_text())
    catalog["runtimes"][0]["label"] = label
    settings.runtime_config.write_text(json.dumps(catalog))


def test_workflow_persists_artifacts_lineage_events_and_defaults(configured):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        jid = submit(client, pid)
        job = wait(client, jid)
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == "diagnostics_only"
        artifacts = client.get(f"/api/v1/projects/{pid}/artifacts").json()
        assert len(artifacts) == 3
        assert artifacts[1]["parent_ids"] == [artifacts[0]["id"]]
        assert all(x["metadata"]["fixture_only"] for x in artifacts)
        events = client.get(f"/api/v1/jobs/{jid}/events").json()
        assert len(events) >= 10
        assert any(x["message"] == "Optimizer step 1" for x in events)
        # An API caller cannot smuggle an executable into a worker request.
        bad = client.post(
            f"/api/v1/projects/{pid}/policy-jobs",
            json={
                "operation": "policy.import",
                "runtime_id": "fixture",
                "source_id": "source",
                "command": ["sh"],
            },
        )
        assert bad.status_code == 422
        other = project(client)
        bad = client.post(
            f"/api/v1/projects/{other}/policy-jobs",
            json={
                "operation": "policy.quantize",
                "runtime_id": "fixture",
                "artifact_id": artifacts[0]["id"],
            },
        )
        assert bad.status_code == 422
    with TestClient(create_app(configured)) as client:
        assert client.get(f"/api/v1/projects/{pid}/artifacts").json() == artifacts
        assert client.get(f"/api/v1/jobs/{jid}").json()["status"] == "succeeded"


def test_default_recipe_depends_on_target_and_training_is_a_method_choice(configured):
    with TestClient(create_app(configured)) as client:
        options = client.get("/api/v1/policy-options").json()
        assert {x["id"] for x in options["training_methods"]} == {"lora", "qlora"}
        assert options["default_training_method"] == "lora"
        assert options["quantization_defaults"]["cuda"]["language"] == "Q4_0"
        assert options["quantization_defaults"]["cpu"]["language"] == "Q8_0"
        pid = project(client)
        imported = wait(client, submit(client, pid, operation="policy.import"))
        aid = imported["result"]["artifacts"][0]["id"]
        response = client.post(
            f"/api/v1/projects/{pid}/policy-jobs",
            json={"operation": "policy.quantize", "runtime_id": "fixture", "artifact_id": aid},
        )
        result = wait(client, response.json()["id"])
        assert result["result"]["artifacts"][0]["metadata"]["precision"] == {
            "language": "Q8_0",
            "vision": None,
        }


@pytest.mark.parametrize(
    "label,decision",
    [
        ("protocol fixture", "validated"),
        ("failed holdout fixture", "no_feasible_candidate"),
        ("failed q4 fixture", "validated"),
    ],
)
def test_quality_selection_requires_unused_final_evaluation(configured, label, decision):
    set_label(configured, label)
    with TestClient(create_app(configured)) as client:
        jid = submit(client, project(client), evaluation={"mode": "libero"}, limits={})
        job = wait(client, jid)
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == decision
        reports = job["result"]["reports"]
        assert any(x["stage"] == "final-reference" for x in reports)
        assert any(x["stage"] == "final-evaluation" for x in reports)
        if decision == "no_feasible_candidate":
            assert job["result"]["selected_artifact_id"] is None
            assert not any(x["format"] == "deployment_package" for x in job["result"]["artifacts"])
        else:
            assert job["result"]["selected_artifact_id"]


def test_cancellation_stops_native_worker_and_never_publishes_late_result(configured):
    set_label(configured, "slow fixture")
    with TestClient(create_app(configured)) as client:
        jid = submit(client, project(client), operation="policy.import")
        started = configured.data_dir / "jobs" / jid / "operation/started"
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists()
        response = client.post(f"/api/v1/jobs/{jid}/cancel")
        assert response.json()["status"] == "cancelled"
        assert wait(client, jid)["result"] is None
        assert not (started.parent / "result.json").exists()


def test_corrupt_registered_manifest_is_rejected(configured):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        job = wait(client, submit(client, pid, operation="policy.import"))
        artifact = job["result"]["artifacts"][0]
        (configured.data_dir / artifact["path"] / "manifest.json").write_text("{}")
        response = client.post(
            f"/api/v1/projects/{pid}/policy-jobs",
            json={
                "operation": "policy.quantize",
                "runtime_id": "fixture",
                "artifact_id": artifact["id"],
            },
        )
        failed = wait(client, response.json()["id"])
        assert failed["status"] == "failed"
        assert "manifest changed" in failed["error"]


def test_unconfigured_and_overlapping_evaluation_are_rejected(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        pid = project(client)
        body = {"operation": "policy.workflow", "runtime_id": "missing", "source_id": "source"}
        assert client.post(f"/api/v1/projects/{pid}/policy-jobs", json=body).status_code == 422
        body["evaluation"] = {"initial_states": [0], "final_states": [0]}
        assert client.post(f"/api/v1/projects/{pid}/policy-jobs", json=body).status_code == 422


def test_download_is_concurrent_and_revalidates_inventory(configured):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        job = wait(client, submit(client, pid, operation="policy.import"))
        artifact = job["result"]["artifacts"][0]
        url = f"/api/v1/projects/{pid}/artifacts/{artifact['id']}/download"
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: client.get(url), range(2)))
        for response in replies:
            assert response.status_code == 200
            with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
                assert archive.extractfile("policy/model").read() == b"x" * 100
                assert "policy/manifest.json" in archive.getnames()
        (configured.data_dir / artifact["path"] / "model").write_bytes(b"changed")
        assert client.get(url).status_code == 422


def test_capabilities_require_a_training_environment(configured):
    with TestClient(create_app(configured)) as client:
        status = {x["operation"]: x["status"] for x in client.get("/api/v1/capabilities").json()}
        assert status["policy.quantize"] == "available"
        assert status["policy.finetune"] == "planned"
        assert status["policy.distill"] == "planned"
