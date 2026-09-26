"""Compute preferences gate configured workers without provisioning resources."""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform.api import create_app
from vla_platform.compute_settings import ComputePreferences, ComputeSettings
from vla_platform.lifecycle.runtime import Runtime
from vla_platform.settings import Settings


@pytest.fixture
def configured_compute(tmp_path):
    root = str((Path(__file__).parent / "fixtures/native_worker").resolve())
    config = tmp_path / "runtimes.json"
    config.write_text(
        json.dumps(
            {
                "runtimes": [
                    {
                        "id": "local-gpu",
                        "label": "RTX 3070",
                        "device": "cuda",
                        "gpu_name": "NVIDIA GeForce RTX 3070",
                        "gpu_memory_mib": 8192,
                        "python": sys.executable,
                        "worker_root": root,
                        "training_python": sys.executable,
                        "training_root": root,
                        "vendor": "private-vendor-path",
                        "build": "private-build-path",
                        "env": {"PRIVATE_TOKEN": "private-secret"},
                    }
                ],
                "sources": [
                    {
                        "id": "fixture",
                        "label": "Fixture source",
                        "path": "/private/source/path",
                        "sha256": "a" * 64,
                    }
                ],
            }
        )
    )
    return Settings(data_dir=tmp_path / "workspace", runtime_config=config)


def test_local_preferences_start_disabled_and_keep_safe_public_fields(
    configured_compute,
):
    with TestClient(create_app(configured_compute)) as client:
        response = client.get("/api/v1/compute-settings")
        assert response.status_code == 200
        result = response.json()
        assert result["local"] == {"enabled": False, "label": "Local machine"}
        assert result["runtimes"][0]["provider"] == "local"
        assert result["runtimes"][0]["region"] is None
        assert result["runtimes"][0]["enabled"] is False
        assert result["runtimes"][0]["provider_label"] == "Local machine"
        assert not any(value in response.text for value in ("private-", "PRIVATE_TOKEN"))
        assert not any(
            key in result["runtimes"][0]
            for key in ("python", "worker_root", "training_root", "env", "build", "vendor")
        )
        options = client.get("/api/v1/policy-options").json()
        assert options["compute"] == {"local": result["local"], "gcp": result["gcp"]}
        assert result["gcp"]["enabled"] is True
        assert result["gcp_status"]["configured"] is False
        assert options["runtimes"] == result["runtimes"]
        assert options["training_models"][0]["runtime_ids"] == []


def test_settings_persist_label_and_disable_model_availability(configured_compute):
    payload = {"local": {"enabled": False, "label": "  Robot lab desktop  "}}
    with TestClient(create_app(configured_compute)) as client:
        result = client.put("/api/v1/compute-settings", json=payload).json()
        assert result["local"] == {"enabled": False, "label": "Robot lab desktop"}
        assert result["runtimes"][0]["provider_label"] == "Robot lab desktop"
        assert result["runtimes"][0]["label"] == "RTX 3070"
        assert result["runtimes"][0]["enabled"] is False
        options = client.get("/api/v1/policy-options").json()
        assert not any(model["available"] for model in options["training_models"])
        assert options["sources"] == [
            {"id": "fixture", "label": "Fixture source", "task": "unverified"}
        ]
    saved = configured_compute.data_dir / "compute-settings.json"
    assert json.loads(saved.read_text()) == {
        "version": 1,
        "local": result["local"],
        "gcp": result["gcp"],
    }
    if os.name == "posix":
        assert saved.stat().st_mode & 0o077 == 0
    with TestClient(create_app(configured_compute)) as client:
        assert client.get("/api/v1/compute-settings").json() == result
        response = client.put(
            "/api/v1/compute-settings",
            json={"local": {**result["local"], "enabled": True}},
        )
        assert response.json()["runtimes"][0]["enabled"] is True
        assert client.get("/api/v1/policy-options").json()["training_models"][0]["available"]


def test_disabling_local_keeps_declared_cloud_workers_available(configured_compute):
    config = json.loads(configured_compute.runtime_config.read_text())
    config["runtimes"].append(
        {**config["runtimes"][0], "id": "gcp-gpu", "provider": "gcp", "region": "us-central1"}
    )
    configured_compute.runtime_config.write_text(json.dumps(config))
    with TestClient(create_app(configured_compute)) as client:
        result = client.put(
            "/api/v1/compute-settings", json={"local": {"enabled": False, "label": "Desktop"}}
        ).json()
        assert [runtime["enabled"] for runtime in result["runtimes"]] == [False, True]
        assert result["runtimes"][1]["provider_label"] == "Google Cloud"
        assert result["runtimes"][1]["region"] == "us-central1"
        assert client.get("/api/v1/policy-options").json()["training_models"][0]["runtime_ids"] == [
            "gcp-gpu"
        ]


@pytest.mark.parametrize(
    "payload",
    [
        {"local": {"label": ""}},
        {"local": {"label": "   "}},
        {"local": {"label": "x" * 101}},
        {"local": {"label": "bad\nlabel"}},
        {"local": {"enabled": "false"}},
        {"local": {"enabled": 1}},
        {"local": {"command": "run anything"}},
        {"provider": "gcp"},
        {"runtimes": [{"python": "arbitrary-executable"}]},
    ],
)
def test_invalid_settings_cannot_mutate_preferences(configured_compute, payload):
    with TestClient(create_app(configured_compute)) as client:
        previous = client.get("/api/v1/compute-settings").json()
        assert client.put("/api/v1/compute-settings", json=payload).status_code == 422
        assert client.get("/api/v1/compute-settings").json() == previous
    assert not (configured_compute.data_dir / "compute-settings.json").exists()


@pytest.mark.parametrize("operation", ["policy.import", "policy.finetune"])
def test_disabled_local_is_rejected_before_policy_submission(configured_compute, operation):
    with TestClient(create_app(configured_compute)) as client:
        pid = client.post("/api/v1/projects", json={"name": "Compute gating"}).json()["id"]
        client.put("/api/v1/compute-settings", json={"local": {"enabled": False}})
        request = {"operation": operation, "runtime_id": "local-gpu"}
        request.update(
            {"source_id": "fixture"}
            if operation == "policy.import"
            else {"dataset_job_id": "dataset"}
        )
        response = client.post(f"/api/v1/projects/{pid}/policy-jobs", json=request)
        assert response.status_code == 422
        assert "Local runs are disabled" in response.json()["detail"]
        assert client.get(f"/api/v1/projects/{pid}/jobs").json() == []


def test_queued_worker_rechecks_preferences_before_process_start(configured_compute, monkeypatch):
    start = AsyncMock(side_effect=AssertionError("A disabled worker must never start"))
    monkeypatch.setattr("vla_platform.lifecycle.service.asyncio.create_subprocess_exec", start)
    app = create_app(configured_compute)
    with TestClient(app) as client:
        client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
        pid = client.post("/api/v1/projects", json={"name": "Queued compute"}).json()["id"]
        client.portal.call(app.state.execution.native_slots.acquire)
        try:
            response = client.post(
                f"/api/v1/projects/{pid}/policy-jobs",
                json={
                    "operation": "policy.import",
                    "runtime_id": "local-gpu",
                    "source_id": "fixture",
                },
            )
            assert response.status_code == 202
            jid = response.json()["id"]
            assert client.get(f"/api/v1/jobs/{jid}").json()["status"] == "queued"
            client.put("/api/v1/compute-settings", json={"local": {"enabled": False}})
        finally:
            client.portal.call(app.state.execution.native_slots.release)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{jid}").json()
            if job["status"] == "failed":
                break
            time.sleep(0.01)
        assert job["status"] == "failed"
        assert "Local runs are disabled" in job["error"]
        start.assert_not_awaited()


def test_failed_atomic_save_preserves_active_and_saved_preferences(tmp_path, monkeypatch):
    settings = ComputeSettings(tmp_path)
    original = ComputePreferences.model_validate({"local": {"enabled": False, "label": "Lab"}})
    settings.update(original)
    saved = settings.path.read_text()

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("vla_platform.compute_settings.os.replace", fail)
    with pytest.raises(OSError):
        settings.update(ComputePreferences())
    assert settings.preferences() == original
    assert settings.path.read_text() == saved
    assert not list(tmp_path.glob(".compute-settings-*"))


def test_corrupt_settings_cannot_silently_reenable_local_runs(tmp_path):
    (tmp_path / "compute-settings.json").write_text('{"version": 2, "local": {"enabled": false}}')
    with pytest.raises(ValidationError):
        ComputeSettings(tmp_path)


@pytest.mark.parametrize(
    "field,value", [("provider", "unknown"), ("provider", "aws"), ("region", "bad\nregion")]
)
def test_operator_provider_metadata_is_validated(field, value):
    with pytest.raises(ValidationError):
        Runtime.model_validate(
            {
                "id": "fixture",
                "label": "Fixture",
                "worker_root": "worker",
                "vendor": "vendor",
                "build": "build",
                field: value,
            }
        )


def test_compute_settings_put_is_allowed_for_local_frontend(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        response = client.options(
            "/api/v1/compute-settings",
            headers={
                "origin": "http://localhost:3000",
                "access-control-request-method": "PUT",
                "access-control-request-headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert "PUT" in response.headers["access-control-allow-methods"]
