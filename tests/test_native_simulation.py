"""Application admission/persistence tests; fixture execution is not robot evidence."""

import asyncio
import io
import json
import shutil
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform.api import create_app
from vla_platform.lifecycle import simulation
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.settings import Settings


def wait(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.02)
    raise AssertionError("Fixture job did not finish")


@pytest.fixture
def app(tmp_path, monkeypatch):
    calls = []
    profile = SimpleNamespace(
        id="cup-scene",
        label="Fixture cup scene",
        identity_hash=lambda: "a" * 64,
        public=lambda: {
            "id": "cup-scene",
            "label": "Fixture cup scene",
            "experimental": True,
            "architectures": ["act", "smolvla"],
            "task_object": "cup",
        },
    )
    monkeypatch.setattr(simulation, "profiles", lambda _: (profile,))

    async def resolve(lifecycle, profile, source, destination, receipt_path, *, archive=False):
        destination.mkdir()
        if archive:
            family = source.read_text()
            if family not in {"act", "smolvla"}:
                raise ValueError("Invalid fixture archive")
            model = destination / "contents"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"type": family}))
            (model / "model.safetensors").write_bytes(b"fixture-only-not-model-weights")
        else:
            shutil.copytree(source, destination / "contents")
            model = next((destination / "contents").rglob("config.json")).parent
            family = json.loads((model / "config.json").read_text())["type"]
        receipt = {
            "schema_version": 1,
            "directory": model.relative_to(destination).as_posix(),
            "checkpoint": {"policy_type": family, "model_id": "sha256:" + "b" * 64},
            "source_sha256": simulation.sha(source) if archive else None,
            "files": {
                p.relative_to(destination).as_posix(): {
                    "sha256": simulation.sha(p),
                    "bytes": p.stat().st_size,
                }
                for p in destination.rglob("*")
                if p.is_file()
            },
        }
        receipt_path.write_text(json.dumps(receipt))
        return simulation.check_receipt(destination, receipt_path)

    async def execute(profile, checkpoint, directory, event, timeout, **kwargs):
        calls.append({"model": checkpoint, "timeout": timeout, **kwargs})
        directory.mkdir()
        await event("running", "Fixture only; no GPU or simulation", {"fixture_only": True})
        records = []
        for name in simulation.RECORD_FILES:
            file = directory / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"fixture-only-not-real-simulation")
            records.append(
                {
                    "path": name,
                    "sha256": simulation.sha(file),
                    "bytes": file.stat().st_size,
                    "generation": "123",
                }
            )
        report = {
            "execution_status": "succeeded",
            "model_id": kwargs["expected_model_id"],
            "profile_sha256": kwargs["expected_profile_sha256"],
            "profile_id": profile.id,
            "run_id": "c" * 32,
            "calibration_verified": False,
            "task_success": None,
            "artifacts": records,
            "fixture_only": True,
        }
        (directory / "report.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(simulation, "resolve", resolve)
    runner = SimpleNamespace(run=execute, admit=lambda p, m: {"model_id": m["model_id"]})
    monkeypatch.setitem(sys.modules, "vla_platform.lifecycle.isaac_runner", runner)
    import vla_platform.lifecycle

    monkeypatch.setattr(vla_platform.lifecycle, "isaac_runner", runner, raising=False)
    settings = Settings(data_dir=tmp_path / "workspace")
    with TestClient(create_app(settings)) as client:
        pid = client.post("/api/v1/projects", json={"name": "Native policy fixture"}).json()["id"]
        yield client, pid, settings, profile, calls


def upload(client, pid, family):
    response = client.post(
        f"/api/v1/projects/{pid}/model-imports?profile_id=cup-scene",
        content=family.encode(),
        headers={"Content-Type": "application/x-tar"},
    )
    assert response.status_code == 202, response.text
    return wait(client, response.json()["id"])


@pytest.mark.parametrize("family", ["act", "smolvla"])
def test_both_native_families_import_and_run_through_owned_jobs(app, family):
    client, pid, settings, profile, calls = app
    imported = upload(client, pid, family)
    assert imported["status"] == "succeeded", imported
    artifact = imported["result"]["artifacts"][0]
    assert artifact["metadata"]["architecture"] == family
    assert artifact["metadata"]["training_resume_supported"] is False
    assert artifact["metadata"]["runtime_verified"] is False
    assert artifact["metadata"]["task_success"] is None
    assert imported["simulation_target"] is None and imported["compute_target"] is None
    assert not list((settings.data_dir / "model-uploads").glob("*/archive.tar"))
    response = client.post(
        f"/api/v1/projects/{pid}/policy-jobs",
        json={
            "operation": "policy.run",
            "runtime_id": profile.id,
            "artifact_id": artifact["id"],
            "simulation": {"profile_id": profile.id, "experimental": True},
        },
    )
    assert response.status_code == 202, response.text
    job = wait(client, response.json()["id"])
    assert job["status"] == "succeeded", job
    assert job["compute_target"] is None
    assert job["simulation_target"]["accelerators"] == ["L4", "H100"]
    assert job["simulation_target"]["source_manifest_sha256"] == artifact["manifest_sha256"]
    assert job["result"]["decision"] == "diagnostics_only"
    assert job["result"]["reports"][0]["task_success"] is None
    assert len(calls) == 1
    record = job["result"]["artifacts"][0]
    assert record["format"] == "simulation_record"
    video_url = f"/api/v1/jobs/{job['id']}/simulation-media/video"
    assert client.get(video_url).content == b"fixture-only-not-real-simulation"
    download = client.get(f"/api/v1/projects/{pid}/artifacts/{record['id']}/download")
    assert download.status_code == 200, download.text
    with tarfile.open(fileobj=io.BytesIO(download.content)) as archive:
        assert "policy/artifacts/outputs/video.mp4" in archive.getnames()
    (settings.data_dir / record["path"] / "artifacts/outputs/video.mp4").write_bytes(b"changed")
    assert client.get(video_url).status_code == 422


def test_import_failure_is_a_visible_job_with_no_dispatch(app):
    client, pid, _, _, calls = app
    job = upload(client, pid, "bad")
    assert job["status"] == "failed"
    assert not calls
    assert client.get(f"/api/v1/projects/{pid}/artifacts").json() == []


def test_cross_project_artifact_and_tamper_fail_before_dispatch(app):
    client, pid, settings, profile, calls = app
    artifact = upload(client, pid, "act")["result"]["artifacts"][0]
    other = client.post("/api/v1/projects", json={"name": "Another project"}).json()["id"]
    payload = {
        "operation": "policy.run",
        "runtime_id": profile.id,
        "artifact_id": artifact["id"],
        "simulation": {"profile_id": profile.id, "experimental": True},
    }
    assert client.post(f"/api/v1/projects/{other}/policy-jobs", json=payload).status_code == 422
    policy = next((settings.data_dir / artifact["path"]).rglob("model.safetensors"))
    policy.write_bytes(b"modified")
    assert client.post(f"/api/v1/projects/{pid}/policy-jobs", json=payload).status_code == 422
    assert not calls


@pytest.mark.parametrize(
    "patch",
    [
        {"operation": "policy.evaluate"},
        {"operation": "policy.quantize"},
        {"simulation": {"profile_id": "cup-scene", "experimental": False}},
        {"runtime_id": "another"},
        {"training": {"steps": 10}},
        {"timeout_seconds": 7201},
        {"simulation": {"profile_id": "cup-scene", "experimental": True, "command": "injected"}},
    ],
)
def test_unsupported_simulation_operations_and_caller_commands_are_rejected(patch):
    with pytest.raises(ValidationError):
        PolicyRequest.model_validate(
            {
                "operation": "policy.run",
                "runtime_id": "cup-scene",
                "artifact_id": "fixture",
                "simulation": {"profile_id": "cup-scene", "experimental": True},
                **patch,
            }
        )


def test_upload_limits_origin_and_unknown_profile_leave_no_jobs(app):
    client, pid, settings, _, calls = app
    path = f"/api/v1/projects/{pid}/model-imports?profile_id=cup-scene"
    assert client.post(path, content=b"act").status_code == 415
    assert (
        client.post(path, content=b"", headers={"Content-Type": "application/x-tar"}).status_code
        == 413
    )
    assert (
        client.post(
            path,
            content=b"act",
            headers={"Content-Type": "application/x-tar", "Origin": "https://untrusted.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            path.replace("cup-scene", "unknown"),
            content=b"act",
            headers={"Content-Type": "application/x-tar"},
        ).status_code
        == 422
    )
    assert client.get(f"/api/v1/projects/{pid}/jobs").json() == []
    assert not (settings.data_dir / "model-uploads").exists()
    assert not calls


def test_changed_profile_fails_accepted_job_without_gpu_dispatch(app):
    client, pid, _, profile, calls = app
    artifact = upload(client, pid, "act")["result"]["artifacts"][0]
    identities = iter(["a" * 64, "d" * 64])
    profile.identity_hash = lambda: next(identities)
    response = client.post(
        f"/api/v1/projects/{pid}/policy-jobs",
        json={
            "operation": "policy.run",
            "runtime_id": profile.id,
            "artifact_id": artifact["id"],
            "simulation": {"profile_id": profile.id, "experimental": True},
        },
    )
    assert response.status_code == 202
    job = wait(client, response.json()["id"])
    assert job["status"] == "failed"
    assert "changed after submission" in job["error"]
    assert not calls


def test_cancellation_drains_owned_write_or_submission_before_cleanup():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        evidence = []

        async def write_or_submit():
            entered.set()
            await release.wait()
            evidence.append("completed")

        task = asyncio.create_task(simulation.finish_owned(write_or_submit()))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert evidence == ["completed"]

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["hash", "missing", "path", "model", "success"])
def test_unverified_simulation_outputs_are_never_published(app, monkeypatch, change):
    client, pid, _, profile, calls = app
    artifact = upload(client, pid, "act")["result"]["artifacts"][0]
    from vla_platform.lifecycle import isaac_runner

    original = isaac_runner.run

    async def corrupt(*args, **kwargs):
        report = await original(*args, **kwargs)
        if change == "hash":
            report["artifacts"][0]["sha256"] = "0" * 64
        elif change == "missing":
            report["artifacts"].pop()
        elif change == "path":
            report["artifacts"][0]["path"] = "../credentials"
        elif change == "model":
            report["model_id"] = "sha256:" + "0" * 64
        else:
            report["task_success"] = True
        (args[2] / "report.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(isaac_runner, "run", corrupt)
    response = client.post(
        f"/api/v1/projects/{pid}/policy-jobs",
        json={
            "operation": "policy.run",
            "runtime_id": profile.id,
            "artifact_id": artifact["id"],
            "simulation": {"profile_id": profile.id, "experimental": True},
        },
    )
    job = wait(client, response.json()["id"])
    assert job["status"] == "failed"
    assert job["result"] is None
    assert client.get(f"/api/v1/jobs/{job['id']}/simulation-media/video").status_code == 422
    assert len(calls) == 1


@pytest.mark.parametrize("location", ["outer", "checkpoint"])
def test_run_rejects_registered_coordinates_missing_from_resolved_package(app, location):
    client, pid, settings, profile, calls = app
    imported = upload(client, pid, "act")
    execution = client.app.state.execution
    record = client.portal.call(execution.get, imported["id"])
    artifact = record.result.artifacts[0]
    metadata = artifact.metadata
    if location == "checkpoint":
        metadata = metadata["checkpoint"]
    metadata["control_contract_sha256"] = "e" * 64
    root = settings.data_dir / artifact.path
    (root / "manifest.json").unlink()
    artifact.manifest_sha256, artifact.file_bytes = simulation.manifest(root, artifact.metadata)
    client.portal.call(execution.save, record)
    response = client.post(
        f"/api/v1/projects/{pid}/policy-jobs",
        json={
            "operation": "policy.run",
            "runtime_id": profile.id,
            "artifact_id": artifact.id,
            "simulation": {"profile_id": profile.id, "experimental": True},
        },
    )
    assert response.status_code == 202, response.text
    job = wait(client, response.json()["id"])
    assert job["status"] == "failed", job
    assert "control metadata differs" in job["error"]
    assert not calls
    assert len(client.get(f"/api/v1/projects/{pid}/artifacts").json()) == 1
