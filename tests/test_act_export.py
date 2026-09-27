"""ACT application routing and lineage contracts; protocol fixtures do not prove ML quality."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import insert
from test_lifecycle import wait
from vla_platform.api import create_app
from vla_platform.contracts import Job, now
from vla_platform.lifecycle.contracts import LifecycleResult, PolicyArtifact, PolicyRequest
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog, command
from vla_platform.local_worker_registry import LocalWorkerRegistry
from vla_platform.settings import Settings
from vla_platform.storage import jobs

FIXTURE = Path(__file__).parent / "fixtures/act_worker"


def replace_catalog(app, catalog):
    lifecycle = app.state.execution.lifecycle
    lifecycle._operator_catalog = catalog
    lifecycle.local_workers = LocalWorkerRegistry(lifecycle.settings.data_dir, catalog)


def runtime(**changes):
    return Runtime.model_validate(
        {
            "id": "act-cpu",
            "label": "CPU export fixture",
            "worker_root": "/never/used",
            "vendor": "/never/vendor",
            "build": "/never/build",
            "python": "/never/python",
            "act_export_python": sys.executable,
            "act_export_root": str(FIXTURE.resolve()),
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"act_export_root": None},
        {"act_export_python": None},
        {"provider": "gcp"},
        {"execution": "skypilot"},
    ],
)
def test_export_config_requires_complete_local_pair(changes):
    with pytest.raises(ValidationError):
        runtime(**changes)


def test_fixed_export_command_never_uses_gpu_training_image_or_request_module(tmp_path):
    configured = runtime(
        device="cuda",
        image="unrelated",
        training_image="unrelated",
        env={"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "0"},
    )
    argv, cwd, env = command(
        configured,
        tmp_path / "request",
        tmp_path / "result",
        tmp_path,
        "container",
        act_export=True,
    )
    assert argv[:3] == [sys.executable, "-m", "firebird_act.application"]
    assert cwd == str(FIXTURE.resolve())
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["HF_HUB_OFFLINE"] == "1"
    assert "docker" not in argv and "policykit.application" not in argv


@pytest.fixture
def application(tmp_path):
    config = tmp_path / "runtimes.json"
    config.write_text(RuntimeCatalog(runtimes=[runtime()]).model_dump_json())
    app = create_app(Settings(data_dir=tmp_path / "data", runtime_config=config))
    with TestClient(app) as client:
        client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
        pid = client.post("/api/v1/projects", json={"name": "ACT protocol"}).json()["id"]
        root = app.state.execution.settings.data_dir / "source"
        policy = root / "checkpoint/pretrained_model"
        policy.mkdir(parents=True)
        (policy / "model.safetensors").write_bytes(b"protocol fixture; no actual model")
        (policy / "config.json").write_text(
            json.dumps(
                {"chunk_size": 100, "n_action_steps": 100, "n_obs_steps": 1, "use_vae": True}
            )
        )
        (root / "checkpoint/manifest.json").write_text(json.dumps({"step": 1}))
        metadata = {
            "architecture": "act",
            "method": "full",
            "training_backend": "lerobot",
            "dataset": {"source": "huggingface", "repo_id": "fixture/robot", "revision": "a" * 40},
            "camera_keys": ["observation.images.front"],
        }
        (root / "manifest.json").write_text(json.dumps({"metadata": metadata}))
        artifact = PolicyArtifact(
            id="source:step1",
            project_id=pid,
            job_id="source",
            label="ACT step 1",
            format="training_checkpoint",
            path="source",
            file_bytes=1,
            manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            metadata=metadata,
        )
        record = Job(
            id="source",
            project_id=pid,
            kind="policy.finetune",
            status="succeeded",
            request=PolicyRequest(
                operation="policy.finetune", runtime_id="act-cpu", dataset_job_id="fixture"
            ),
            result=LifecycleResult(artifacts=[artifact]),
            created_at=now(),
            updated_at=now(),
        )

        async def seed():
            async with app.state.execution.storage.engine.begin() as connection:
                await connection.execute(
                    insert(jobs).values(
                        id=record.id,
                        project_id=pid,
                        status=record.status,
                        record=record.model_dump(),
                    )
                )

        client.portal.call(seed)
        yield app, client, pid, artifact, root


def submit(client, pid, artifact, **updates):
    return client.post(
        f"/api/v1/projects/{pid}/policy-jobs",
        json={
            "operation": "policy.export",
            "runtime_id": "act-cpu",
            "artifact_id": artifact.id,
            **updates,
        },
    )


def test_exports_persists_reloads_downloads_distinct_artifact(application):
    app, client, pid, source, source_path = application
    before = {str(p): p.read_bytes() for p in source_path.rglob("*") if p.is_file()}
    options = client.get("/api/v1/policy-options").json()
    assert options["runtimes"][0]["act_export"] is True
    assert "act_export_python" not in json.dumps(options)
    response = submit(client, pid, source)
    assert response.status_code == 202, response.text
    completed = wait(client, response.json()["id"])
    assert completed["status"] == "succeeded", completed
    result = completed["result"]
    assert result["decision"] == "completed"
    artifact = result["artifacts"][0]
    assert artifact["format"] == "inference_export" and artifact["parent_ids"] == [source.id]
    assert artifact["metadata"]["source_manifest_sha256"] == source.manifest_sha256
    assert artifact["metadata"]["task_success"] is None
    download = client.get(f"/api/v1/projects/{pid}/artifacts/{artifact['id']}/download")
    assert download.status_code == 200, download.text
    with tarfile.open(fileobj=io.BytesIO(download.content)) as archive:
        assert "policy/policy/manifest.json" in archive.getnames()
    assert before == {str(p): p.read_bytes() for p in source_path.rglob("*") if p.is_file()}
    denied = submit(
        client,
        pid,
        PolicyArtifact.model_validate(artifact),
        operation="policy.finetune",
        dataset_job_id="fixture",
    )
    assert denied.status_code == 422 and "not training" in denied.text


@pytest.mark.parametrize(
    "kind",
    [
        "remote",
        "missing-model",
        "no-runtime",
        "disabled",
        "wrong-format",
        "quantize",
        "workflow",
        "request-command",
        "local-dataset",
    ],
)
def test_incompatible_export_fails_before_worker(application, kind):
    app, client, pid, source, root = application
    changes = {}
    if kind == "remote":
        (root / "remote.json").write_text("{}")
    elif kind == "missing-model":
        (root / "checkpoint/pretrained_model/model.safetensors").unlink()
    elif kind == "no-runtime":
        catalog = app.state.execution.lifecycle.catalog
        catalog.runtimes[0].act_export_python = None
        replace_catalog(app, catalog)
    elif kind == "disabled":
        client.put("/api/v1/compute-settings", json={"local": {"enabled": False}})
    elif kind == "wrong-format":
        record = client.portal.call(app.state.execution.get, "source")
        record.result.artifacts[0].format = "native_checkpoint"
        client.portal.call(app.state.execution.save, record)
    elif kind == "local-dataset":
        record = client.portal.call(app.state.execution.get, "source")
        record.result.artifacts[0].metadata["dataset"] = {
            "source": "local",
            "revision": "sha256:" + "a" * 64,
        }
        client.portal.call(app.state.execution.save, record)
    elif kind in {"quantize", "workflow"}:
        changes["operation"] = "policy." + kind
    else:
        changes["act_export_python"] = "arbitrary"
    response = submit(client, pid, source, **changes)
    assert response.status_code == 422, response.text
    assert len(client.get(f"/api/v1/projects/{pid}/jobs").json()) == 1


@pytest.mark.parametrize("fault", ["source", "format", "reload", "step", "gpu", "no-artifact"])
def test_bad_export_receipt_never_registers(application, fault):
    app, client, pid, source, _ = application
    catalog = app.state.execution.lifecycle.catalog
    catalog.runtimes[0].env["FIXTURE_FAULT"] = fault
    replace_catalog(app, catalog)
    accepted = submit(client, pid, source)
    assert accepted.status_code == 202, accepted.text
    completed = wait(client, accepted.json()["id"])
    assert completed["status"] == "failed", completed
    assert [a["id"] for a in client.get(f"/api/v1/projects/{pid}/artifacts").json()] == [source.id]


def test_act_training_cannot_start_an_unsupported_gguf_workflow(application):
    app, client, pid, _, _ = application
    catalog = app.state.execution.lifecycle.catalog
    configured = catalog.runtimes[0]
    configured.training_python = sys.executable
    configured.training_root = str(FIXTURE)
    configured.training_module = "firebird_vla.lerobot_application"
    configured.training_model_ids = ["act"]
    replace_catalog(app, catalog)
    response = client.post(
        f"/api/v1/projects/{pid}/policy-jobs",
        json={
            "operation": "policy.workflow",
            "runtime_id": "act-cpu",
            "dataset_job_id": "fixture",
            "training_method": "full",
            "training": {"model_id": "code://lerobot/act"},
        },
    )
    assert response.status_code == 422 and "GGUF workflow" in response.text
    assert len(client.get(f"/api/v1/projects/{pid}/jobs").json()) == 1


def test_changed_source_cannot_omit_temporal_export_claims(application):
    _, client, pid, artifact, root = application
    config = root / "checkpoint/pretrained_model/config.json"
    config.write_text(
        json.dumps({"chunk_size": 8, "n_action_steps": 3, "n_obs_steps": 1, "use_vae": True})
    )
    response = submit(client, pid, artifact)
    assert response.status_code == 202
    record = wait(client, response.json()["id"])
    assert record["status"] == "failed"
    assert "temporal claims" in record["error"]
    assert not record.get("result")
