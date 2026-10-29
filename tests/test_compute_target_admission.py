"""Real job acceptance and private target bindings without worker/provider execution."""

import hashlib
import json
import os
import stat
import sys

import pytest
from fastapi.testclient import TestClient
from vla_platform import cloud_compute_catalog
from vla_platform.api import create_app
from vla_platform.compute_targets import ComputeTarget, ComputeTargets
from vla_platform.execution import Execution
from vla_platform.lifecycle import sky_runner
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.settings import Settings


@pytest.fixture
def admission(tmp_path, monkeypatch, request):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "runtimes.json"
    config.write_text(
        json.dumps(
            {
                "runtimes": []
                if getattr(request, "param", None) == "empty"
                else [
                    {
                        "id": "local-worker",
                        "label": "Registered CPU worker",
                        "python": sys.executable,
                        "worker_root": str(tmp_path / "worker"),
                        "vendor": str(tmp_path),
                        "build": str(tmp_path),
                        "device": "cpu",
                    }
                ],
                "sources": [],
            }
        )
    )
    state = {
        "endpoint": "https://first.example.invalid",
        "captures": [],
        "launched": [],
        "validated": [],
    }

    async def validate(self, project, request):
        state["validated"].append(request.runtime_id)
        runtime = self.runtime(request.runtime_id)
        assert runtime is not None
        self.require_compute(request, runtime)
        return None

    async def endpoint(executable):
        state["captures"].append(state["endpoint"])
        return state["endpoint"]

    monkeypatch.setattr(Lifecycle, "validate", validate)
    monkeypatch.setattr(Execution, "_schedule", lambda self, ident: state["launched"].append(ident))
    monkeypatch.setattr(ComputeTargets, "sky_executable", staticmethod(lambda: "sky"))
    monkeypatch.setattr(sky_runner, "executable", lambda: "sky")
    monkeypatch.setattr(cloud_compute_catalog, "configured_sky_endpoint", endpoint)
    app = create_app(Settings(data_dir=workspace, runtime_config=config))
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "Target admission"}).json()["id"]
        yield client, workspace, project, state


def save_target(client, **changes):
    definition = ComputeTarget.model_validate(
        {
            "id": "remote-gpu",
            "label": "Remote GPU",
            "adapter": "skypilot",
            "infrastructure": "ssh",
            "runtime_id": "portable-policy",
            "location": "research-pool",
            "enabled": True,
            "resources": {"accelerator": "L4", "gpu_count": 1},
            **changes,
        }
    )
    response = client.put(
        "/api/v1/compute-targets/" + definition.id, json=definition.model_dump(mode="json")
    )
    assert response.status_code == 200, response.text
    return definition


def submit(
    client, project, *, target="remote-gpu", runtime="portable-policy", key="target-admission"
):
    body = {
        "operation": "policy.run",
        "runtime_id": runtime,
        "artifact_id": "fixture-artifact",
        "timeout_seconds": 30,
    }
    if target is not None:
        body["execution_target_id"] = target
    return client.post(
        f"/api/v1/projects/{project}/policy-jobs", json=body, headers={"Idempotency-Key": key}
    )


@pytest.mark.parametrize("admission", ["empty"], indirect=True)
def test_remote_admission_freezes_private_connection_without_local_ml(admission):
    client, workspace, project, state = admission
    save_target(client)
    response = submit(client, project)
    assert response.status_code == 202, response.text
    job = response.json()
    binding = job["execution"]
    assert binding["adapter"] == "skypilot-target" and binding["target_id"] == "remote-gpu"
    assert binding["target_snapshot"]["definition"]["runtime_id"] == "portable-policy"
    connection = workspace / "jobs" / job["id"] / "execution-connection.json"
    assert not connection.is_symlink()
    assert hashlib.sha256(connection.read_bytes()).hexdigest() == binding["connection_sha256"]
    assert json.loads(connection.read_bytes())["sky_api_endpoint"] == state["endpoint"]
    if os.name == "posix":
        assert stat.S_IMODE(connection.stat().st_mode) == 0o600
    assert state["endpoint"] not in response.text
    assert "sky_api_endpoint" not in response.text
    assert "authorization" not in response.text.lower()
    assert state["launched"] == [job["id"]]
    assert client.app.state.execution.lifecycle.catalog.runtimes == []
    assert client.app.state.execution.lifecycle.catalog.runtime("portable-policy") is None
    assert client.app.state.execution.lifecycle.runtime("portable-policy") is not None


def test_idempotent_repeat_retains_original_connection_after_global_change(admission):
    client, workspace, project, state = admission
    save_target(client)
    first = submit(client, project)
    assert first.status_code == 202, first.text
    job = first.json()
    connection = workspace / "jobs" / job["id"] / "execution-connection.json"
    original = connection.read_bytes()
    state["endpoint"] = "https://changed.example.invalid"
    again = submit(client, project)
    assert again.status_code == 202, again.text
    assert again.json()["id"] == job["id"]
    assert connection.read_bytes() == original
    assert state["captures"] == ["https://first.example.invalid"]
    assert state["validated"] == ["portable-policy"]
    assert state["launched"] == [job["id"]]


@pytest.mark.skipif(os.name != "posix", reason="Tests POSIX symbolic-link storage boundary")
@pytest.mark.parametrize("linked", ["jobs", "job-directory"])
def test_connection_storage_rejects_symlink_parent(admission, tmp_path, monkeypatch, linked):
    from uuid import UUID

    import vla_platform.execution as execution_module

    client, workspace, project, state = admission
    save_target(client)
    fixed = UUID("11111111-1111-4111-8111-111111111111")
    monkeypatch.setattr(execution_module, "uuid4", lambda: fixed)
    outside = tmp_path / "outside"
    outside.mkdir()
    jobs = workspace / "jobs"
    if linked == "jobs":
        assert not jobs.exists()
        jobs.symlink_to(outside, target_is_directory=True)
    else:
        jobs.mkdir(exist_ok=True)
        (jobs / str(fixed)).symlink_to(outside, target_is_directory=True)
    response = submit(client, project)
    assert response.status_code == 422, response.text
    assert "symbolic link" in response.json()["detail"]
    assert list(outside.iterdir()) == []
    assert state["launched"] == []
    assert client.get(f"/api/v1/projects/{project}/jobs").json() == []


def test_connection_reference_rejects_credentials_before_job_acceptance(admission):
    client, workspace, project, state = admission
    private_value = "never-persist-this-value"
    (workspace / "compute-connections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "connections": {
                    "remote": {
                        "sky_api_endpoint": "https://compute.example.invalid",
                        "token": private_value,
                    }
                },
            }
        )
    )
    save_target(client, connection_ref="remote")
    response = submit(client, project)
    assert response.status_code == 422, response.text
    assert private_value not in response.text
    assert state["captures"] == [] and state["launched"] == []
    assert client.get(f"/api/v1/projects/{project}/jobs").json() == []
    assert not list(workspace.glob("jobs/*/execution-connection.json"))


def test_enabled_local_target_overrides_legacy_disabled_setting(admission):
    client, workspace, project, state = admission
    assert client.get("/api/v1/compute-settings").json()["local"]["enabled"] is False
    save_target(
        client,
        id="local-cpu",
        label="CPU worker",
        adapter="local",
        infrastructure="local",
        runtime_id="local-worker",
        location=None,
        resources={},
    )
    options = client.get("/api/v1/policy-options")
    assert options.status_code == 200, options.text
    entry = next(
        runtime for runtime in options.json()["runtimes"] if runtime["id"] == "local-worker"
    )
    assert entry["enabled"] and entry["launchable"]
    assert options.json()["compute"]["local"]["enabled"] is False
    response = submit(client, project, target="local-cpu", runtime="local-worker")
    assert response.status_code == 202, response.text
    assert response.json()["execution"]["target_id"] == "local-cpu"
    assert state["captures"] == []
    assert len(state["launched"]) == 1


def test_legacy_idempotent_job_repeat_does_not_revalidate_disabled_runtime(admission):
    client, workspace, project, state = admission
    enabled = client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
    assert enabled.status_code == 200, enabled.text
    first = submit(client, project, target=None, runtime="local-worker", key="legacy-accepted")
    assert first.status_code == 202, first.text
    disabled = client.put("/api/v1/compute-settings", json={"local": {"enabled": False}})
    assert disabled.status_code == 200, disabled.text
    again = submit(client, project, target=None, runtime="local-worker", key="legacy-accepted")
    assert again.status_code == 202, again.text
    assert again.json()["id"] == first.json()["id"]
    assert state["validated"] == ["local-worker"]
    assert state["captures"] == [] and len(state["launched"]) == 1


def test_gcp_reference_without_workspace_uses_project_workspace_at_acceptance(admission):
    client, workspace, project, state = admission
    (workspace / "compute-connections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "connections": {
                    "cloud-api": {
                        "sky_api_endpoint": "https://compute.example.invalid",
                        "project_id": "sample-project",
                    }
                },
            }
        )
    )
    save_target(client, infrastructure="gcp", location="us-central1", connection_ref="cloud-api")
    response = submit(client, project)
    assert response.status_code == 202, response.text
    job = response.json()
    connection = json.loads(
        (workspace / "jobs" / job["id"] / "execution-connection.json").read_bytes()
    )
    assert connection["workspace"] == cloud_compute_catalog.sky_workspace_name("sample-project")
    assert connection["project_id"] == "sample-project"
    assert connection["region"] == "us-central1"
    assert state["captures"] == []
    assert "https://compute.example.invalid" not in response.text


@pytest.mark.parametrize(
    "explicit,configured,expected",
    [
        (None, "us-east1", "us-east1"),
        ("us-central1", "us-east1", "us-central1"),
    ],
)
def test_gcp_region_selection_is_frozen_with_explicit_target_priority(
    admission, explicit, configured, expected
):
    client, workspace, project, state = admission
    (workspace / "cloud-connections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {"gcp": {"project_id": "sample-project", "region": configured}},
            }
        )
    )
    save_target(client, infrastructure="gcp", location=explicit)
    response = submit(client, project)
    assert response.status_code == 202, response.text
    job = response.json()
    path = workspace / "jobs" / job["id"] / "execution-connection.json"
    connection = json.loads(path.read_bytes())
    assert connection["region"] == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() == job["execution"]["connection_sha256"]


def test_gcp_missing_region_is_rejected_before_acceptance(admission):
    client, workspace, project, state = admission
    (workspace / "compute-connections.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "connections": {
                    "cloud-api": {
                        "sky_api_endpoint": "https://compute.example.invalid",
                        "project_id": "sample-project",
                    }
                },
            }
        )
    )
    save_target(client, infrastructure="gcp", location=None, connection_ref="cloud-api")
    response = submit(client, project)
    assert response.status_code == 422, response.text
    assert "Choose a Google Cloud region" in response.json()["detail"]
    assert state["launched"] == []
    assert client.get(f"/api/v1/projects/{project}/jobs").json() == []
    assert not list(workspace.glob("jobs/*/execution-connection.json"))
