"""Native ACT packed-package admission; fixtures never represent robot quality."""

import hashlib
import io
import json
import shutil
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
from vla_platform.lifecycle.native_quantization import command, source_info
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog
from vla_platform.settings import Settings
from vla_platform.storage import jobs


def request(**updates):
    return PolicyRequest.model_validate(
        {
            "operation": "policy.quantize",
            "runtime_id": "native-cpu",
            "artifact_id": "source:operation",
            "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
            **updates,
        }
    )


def test_request_survives_persistence_and_defaults():
    value = request()
    assert value.timeout_seconds == 600
    assert PolicyRequest.model_validate(value.model_dump()) == value


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "policy.run"},
        {"source_id": "configured"},
        {"training": {}},
        {"dataset_job_id": "dataset"},
        {"resume_job_id": "resume"},
        {"precision": {"language": "Q8_0"}},
        {"candidates": [{"language": "Q4_0"}]},
        {"evaluation": {"repetitions": 20}},
        {"limits": {}},
        {"training_method": "full"},
        {"timeout_seconds": 601},
        {"timeout_seconds": 30.0},
        {"native_quantization": {"format": "firebird_quant", "bits": 8.0, "group_size": 64}},
        {"native_quantization": {"format": "firebird_quant", "bits": True, "group_size": 64}},
        {"native_quantization": {"format": "firebird_quant", "bits": 4, "group_size": 64.0}},
    ],
)
def test_native_request_rejects_conflicting_or_coerced_options(changes):
    with pytest.raises(ValidationError):
        request(**changes)


def runtime(root, **updates):
    return Runtime.model_validate(
        {
            "id": "native-cpu",
            "label": "CPU packed ACT",
            "native_quantization_only": True,
            "native_quantization_python": sys.executable,
            "native_quantization_root": str(root),
            **updates,
        }
    )


def test_public_capability_requires_installed_worker_without_paths(tmp_path):
    configured = runtime(tmp_path)
    catalog = RuntimeCatalog(runtimes=[configured])
    assert not catalog.public()["runtimes"][0]["native_quantization"]
    module = tmp_path / "src/firebird_quant/native_application.py"
    module.parent.mkdir(parents=True)
    module.write_text("# fixed worker")
    act = tmp_path.parent / "act_optimizer/src/firebird_act/application.py"
    act.parent.mkdir(parents=True)
    act.write_text("# fixed consumer")
    public = catalog.public()["runtimes"][0]
    assert public["native_quantization"]
    assert not public["run"] and not public["engine_evaluation"]
    assert "native_quantization_python" not in public


@pytest.mark.parametrize(
    "updates",
    [
        {"provider": "gcp"},
        {"device": "cuda"},
        {"execution": "skypilot"},
        {"native_quantization_python": None},
        {"native_quantization_root": None},
        {"training_python": "python"},
        {"simulator_lane": "libero"},
        {"act_export_python": sys.executable, "act_export_root": "/operator/export"},
    ],
)
def test_runtime_rejects_nonlocal_or_incomplete_configuration(tmp_path, updates):
    with pytest.raises(ValidationError):
        runtime(tmp_path, **updates)


def save_manifest(root, metadata):
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "metadata": metadata, "files": files})
    )
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


@pytest.fixture
def application(tmp_path):
    worker = tmp_path / "workers/firebird_quant"
    shutil.copytree(Path(__file__).parent / "fixtures/native_quant_worker", worker)
    act = worker.parent / "act_optimizer/src/firebird_act/application.py"
    act.parent.mkdir(parents=True)
    act.write_text("# fixed consumer marker; no ML in protocol fixture")
    catalog = tmp_path / "runtimes.json"
    catalog.write_text(RuntimeCatalog(runtimes=[runtime(worker)]).model_dump_json())
    app = create_app(Settings(data_dir=tmp_path / "data", runtime_config=catalog))
    with TestClient(app) as client:
        client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
        pid = client.post("/api/v1/projects", json={"name": "Native ACT packing protocol"}).json()[
            "id"
        ]
        root = app.state.execution.settings.data_dir / "jobs/source/policy"
        policy = root / "policy"
        policy.mkdir(parents=True)
        config = {
            "type": "act",
            "use_vae": False,
            "chunk_size": 100,
            "n_action_steps": 100,
            "input_features": {
                "observation.images.front": {"shape": [3, 32, 32]},
                "observation.state": {"shape": [6]},
            },
            "output_features": {"action": {"shape": [6]}},
        }
        (policy / "config.json").write_text(json.dumps(config))
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (policy / name).write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "registry_name": "normalizer_processor",
                                "state_file": "stats.safetensors",
                            }
                        ]
                    }
                )
            )
        (policy / "stats.safetensors").write_bytes(b"protocol stats")
        (policy / "model.safetensors").write_bytes(b"protocol floating weights")
        metadata = {"architecture": "act", "inference_only": True}
        digest = save_manifest(root, metadata)
        artifact = PolicyArtifact(
            id="source:operation",
            project_id=pid,
            job_id="source",
            label="Protocol ACT",
            format="inference_export",
            path="jobs/source/policy",
            manifest_sha256=digest,
            metadata=metadata,
            file_bytes=0,
        )
        record = Job(
            id="source",
            project_id=pid,
            kind="policy.export",
            status="succeeded",
            request=PolicyRequest(
                operation="policy.export", runtime_id="native-cpu", artifact_id="training:operation"
            ),
            result=LifecycleResult(artifacts=[artifact]),
            created_at=now(),
            updated_at=now(),
        )

        async def seed():
            async with app.state.execution.storage.engine.begin() as conn:
                await conn.execute(
                    insert(jobs).values(
                        id=record.id,
                        project_id=pid,
                        status=record.status,
                        record=record.model_dump(),
                    )
                )

        client.portal.call(seed)
        yield app, client, pid, artifact, root


def update_source(application, *, fault=None, config=None, metadata=None, format=None):
    app, client, _, source, root = application
    if fault or config:
        path = root / "policy/config.json"
        value = json.loads(path.read_text())
        value.update(config or {})
        if fault:
            value["fixture_fault"] = fault
        path.write_text(json.dumps(value))
    record = client.portal.call(app.state.execution.get, "source")
    source = record.result.artifacts[0]
    if metadata:
        source.metadata.update(metadata)
    if format:
        source.format = format
    source.manifest_sha256 = save_manifest(root, source.metadata)
    client.portal.call(app.state.execution.save, record)
    return source


def submit(application, **updates):
    _, client, pid, _, _ = application
    return client.post(f"/api/v1/projects/{pid}/policy-jobs", json=request(**updates).model_dump())


def test_native_package_persist_download_and_no_run(application):
    app, client, pid, source, root = application
    before = source_info(source, app.state.execution.settings.data_dir)
    response = submit(application)
    assert response.status_code == 202, response.text
    job = wait(client, response.json()["id"])
    assert job["status"] == "succeeded", job
    artifact = job["result"]["artifacts"][0]
    assert artifact["format"] == "native_quantized" and artifact["parent_ids"] == [source.id]
    assert artifact["metadata"]["quality_verified"] is False
    assert source_info(source, app.state.execution.settings.data_dir) == before
    with tarfile.open(
        fileobj=io.BytesIO(
            client.get(f"/api/v1/projects/{pid}/artifacts/{artifact['id']}/download").content
        )
    ) as tar:
        assert any(name.endswith("policy/model.fbq") for name in tar.getnames())
        assert not any(name.endswith("model.safetensors") for name in tar.getnames())
    stored = client.portal.call(app.state.execution.get, job["id"])
    assert stored.request.native_quantization.bits == 8
    denied = client.post(
        f"/api/v1/projects/{pid}/policy-jobs",
        json={"operation": "policy.run", "runtime_id": "native-cpu", "artifact_id": artifact["id"]},
    )
    assert denied.status_code == 422


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "lineage",
        "quality",
        "incomplete",
        "reload",
        "source",
        "float-master",
        "size",
        "drift",
        "model",
        "encoding-bool",
        "encoding-float",
        "lineage-bool",
    ],
)
def test_worker_evidence_is_checked_before_registration(application, fault):
    update_source(application, fault=fault)
    response = submit(application)
    assert response.status_code == 202, response.text
    job = wait(application[1], response.json()["id"])
    assert job["status"] == "failed", job
    assert not job.get("result") or not job["result"].get("artifacts")


@pytest.mark.parametrize(
    "change",
    [
        "vae",
        "shape",
        "chunk",
        "smol",
        "training",
        "tampered",
        "missing-stats",
        "disabled",
        "remote",
        "cross-project",
    ],
)
def test_invalid_native_source_never_starts(application, change):
    app, client, _, _, root = application
    if change == "vae":
        update_source(application, config={"use_vae": True})
    elif change == "shape":
        update_source(application, config={"output_features": {"action": {"shape": [7]}}})
    elif change == "chunk":
        update_source(application, config={"chunk_size": 50})
    elif change == "smol":
        update_source(application, metadata={"architecture": "smolvla"})
    elif change == "training":
        update_source(application, format="training_checkpoint")
    elif change == "tampered":
        (root / "policy/model.safetensors").write_bytes(b"changed")
    elif change == "missing-stats":
        (root / "policy/stats.safetensors").unlink()
        update_source(application)
    elif change == "disabled":
        client.put("/api/v1/compute-settings", json={"local": {"enabled": False}})
    elif change == "remote":
        update_source(application, metadata={"storage": "gcs"})
    elif change == "cross-project":
        pid = client.post("/api/v1/projects", json={"name": "other"}).json()["id"]
        response = client.post(f"/api/v1/projects/{pid}/policy-jobs", json=request().model_dump())
        assert response.status_code == 422
        return
    response = submit(application)
    assert response.status_code == 422, response.text
    assert len(list((app.state.execution.settings.data_dir / "jobs").iterdir())) == 1


def test_worker_environment_excludes_credentials_and_runtime_overrides(application, monkeypatch):
    app = application[0]
    configured = app.state.execution.lifecycle.catalog.runtimes[0]
    configured.env = {"PROVIDER_KEY": "private", "CUDA_VISIBLE_DEVICES": "0"}
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/private/key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "private")
    argv, _, env = command(configured, Path("/tmp/request"), Path("/tmp/response"))
    assert argv[1:3] == ["-m", "firebird_quant.native_application"]
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert not set(env) & {"PROVIDER_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "OPENROUTER_API_KEY"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX worker process owner")
@pytest.mark.parametrize("fault", ["hang", "detached-child"])
def test_cancelled_native_worker_is_reaped_without_publishing(application, fault):
    import os
    import time

    update_source(application, fault=fault)
    app, client, _, source, root = application
    before = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    response = submit(application)
    assert response.status_code == 202
    ident = response.json()["id"]
    pid_path = app.state.execution.settings.data_dir / "jobs" / ident / "operation/pid"
    end = time.monotonic() + 10
    while not pid_path.exists() and time.monotonic() < end:
        time.sleep(0.01)
    assert pid_path.is_file()
    pid = int(pid_path.read_text())
    if fault == "detached-child":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(client.post, f"/api/v1/jobs/{ident}/cancel")
            deadline = time.monotonic() + 5
            while not (pid_path.parent / "cleaning").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (pid_path.parent / "cleaning").is_file()

            async def interrupt_again():
                import asyncio

                task = app.state.execution.tasks[ident]
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0.01)

            client.portal.call(interrupt_again)
            assert response.result(timeout=10).status_code == 200
    else:
        assert client.post(f"/api/v1/jobs/{ident}/cancel").status_code == 200
    assert wait(client, ident)["status"] == "cancelled"
    end = time.monotonic() + 10
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("Owned native packing process survived cancellation")
    if fault == "detached-child":
        assert (pid_path.parent / "child-reaped").read_text() == "yes"
        with pytest.raises(ProcessLookupError):
            os.kill(int((pid_path.parent / "child-pid").read_text()), 0)
    assert before == {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert not client.get(f"/api/v1/jobs/{ident}").json().get("result")


@pytest.mark.parametrize(
    "kind",
    [
        "symlink",
        "fifo",
        "oversize",
        "extra-file",
        "metadata",
        "duplicate-json",
        "nonfinite-json",
        "utf16",
    ],
)
def test_boundaries_reject_source_without_spawn(application, kind):
    app, client, _, artifact, root = application
    target = root / "policy/model.safetensors"
    if kind == "symlink":
        target.unlink()
        target.symlink_to(root / "policy/stats.safetensors")
    elif kind == "fifo":
        if not hasattr(__import__("os"), "mkfifo"):
            pytest.skip("POSIX FIFO")
        target.unlink()
        __import__("os").mkfifo(target)
    elif kind == "oversize":
        with target.open("r+b") as stream:
            stream.truncate(512 * 1024**2 + 1)
    elif kind == "extra-file":
        (root / "extra").write_text("unexpected")
    elif kind == "metadata":
        record = client.portal.call(app.state.execution.get, "source")
        record.result.artifacts[0].metadata["inference_only"] = False
        client.portal.call(app.state.execution.save, record)
    else:
        config = root / "policy/config.json"
        if kind == "duplicate-json":
            config.write_text('{"type":"act","type":"smolvla"}')
        elif kind == "nonfinite-json":
            config.write_text('{"type":"act","unused":1e309}')
        else:
            config.write_bytes('{"type":"act"}'.encode("utf-16"))
        update_source(application)
    response = submit(application)
    assert response.status_code == 422, response.text


def test_existing_gguf_request_and_export_runtime_contract_unchanged():
    legacy = PolicyRequest(
        operation="policy.quantize", runtime_id="cpp", artifact_id="gguf:operation"
    )
    assert legacy.timeout_seconds == 7200 and legacy.native_quantization is None
    assert PolicyRequest.model_validate(legacy.model_dump()) == legacy


@pytest.mark.parametrize("bad", [None, [], 1, "invalid"])
@pytest.mark.parametrize("part", ["state", "action", "step"])
def test_malformed_source_objects_return_422(application, bad, part):
    root = application[4]
    config_path = root / "policy/config.json"
    config = json.loads(config_path.read_text())
    if part == "state":
        config["input_features"]["observation.state"] = bad
        config_path.write_text(json.dumps(config))
    elif part == "action":
        config["output_features"]["action"] = bad
        config_path.write_text(json.dumps(config))
    else:
        (root / "policy/policy_preprocessor.json").write_text(json.dumps({"steps": [bad]}))
    update_source(application)
    response = submit(application)
    assert response.status_code == 422, response.text


def test_failed_worker_retains_bounded_diagnostic(application):
    update_source(application, fault="stderr")
    response = submit(application)
    assert response.status_code == 202
    job = wait(application[1], response.json()["id"])
    assert job["status"] == "failed"
    assert "specific offline failure:" in job["error"]
    assert len(job["error"]) < 4300


@pytest.mark.parametrize("missing", ["manifest.json", "policy"])
def test_missing_source_returns_admission_error(application, missing):
    path = application[4] / missing
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    response = submit(application)
    assert response.status_code == 422, response.text
