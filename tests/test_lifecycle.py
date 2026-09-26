"""Application integration with real subprocess fixtures, not ML/quality claims."""

import asyncio
import hashlib
import io
import json
import os
import signal
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
        jid = submit(client, pid, candidates=[{"language": "Q8_0"}, {"language": "Q4_0"}])
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_default_recipe_requires_validation_and_training_is_a_method_choice(configured, device):
    catalog = json.loads(configured.runtime_config.read_text())
    catalog["runtimes"][0]["device"] = device
    configured.runtime_config.write_text(json.dumps(catalog))
    with TestClient(create_app(configured)) as client:
        options = client.get("/api/v1/policy-options").json()
        assert {x["id"] for x in options["training_methods"]} == {"lora", "qlora"}
        assert options["default_training_method"] == "lora"
        assert options["quantization_defaults"]["cuda"]["language"] == "Q8_0"
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
        jid = submit(
            client,
            project(client),
            evaluation={"mode": "libero"},
            limits={},
            candidates=[{"language": "Q8_0"}, {"language": "Q4_0"}],
        )
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


def test_failed_conversions_cannot_promote_the_only_runnable_reference(configured):
    set_label(configured, "failed packed candidates fixture")
    with TestClient(create_app(configured)) as client:
        jid = submit(client, project(client), evaluation={"mode": "libero"}, limits={})
        job = wait(client, jid)
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == "diagnostics_only"
        assert job["result"]["selected_artifact_id"] is None
        assert len(job["result"]["artifacts"]) == 1
        assert not any(x["stage"] == "final-evaluation" for x in job["result"]["reports"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_stop_kills_descendant_when_leader_exits_on_term(tmp_path):
    from vla_platform.lifecycle.service import Lifecycle

    ready, survived = tmp_path / "ready", tmp_path / "survived"
    child = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).touch(); time.sleep(1); Path({str(survived)!r}).touch(); "
        "time.sleep(60)"
    )
    leader = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    )

    async def exercise():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            leader,
            start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(10):
                while not ready.exists():
                    await asyncio.sleep(0.01)
            await Lifecycle.__new__(Lifecycle).stop(process, None)
            await asyncio.sleep(1.1)
            assert not survived.exists(), "Cancelled worker left a native descendant running"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    asyncio.run(exercise())


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


@pytest.mark.parametrize("training", [False, True])
def test_capabilities_require_a_training_environment(configured, training):
    if training:
        catalog = json.loads(configured.runtime_config.read_text())
        catalog["runtimes"][0].update(training_python=sys.executable, training_root="fixture")
        configured.runtime_config.write_text(json.dumps(catalog))
    with TestClient(create_app(configured)) as client:
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        assert all(x["support"] == [] for x in response.json())
        status = {x["operation"]: x["status"] for x in response.json()}
        assert status["policy.quantize"] == "untested"
        assert status["policy.finetune"] == ("untested" if training else "planned")
        assert status["policy.distill"] == "planned"


def test_restart_preserves_completed_native_stages_without_adopting_late_results(configured):
    set_label(configured, "slow evaluation fixture")
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        jid = submit(client, pid)
        started = configured.data_dir / "jobs" / jid / "baseline/started"
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists()
        original = client.get(f"/api/v1/projects/{pid}/artifacts").json()
        assert len(original) == 1
    with TestClient(create_app(configured)) as client:
        job = client.get(f"/api/v1/jobs/{jid}").json()
        assert job["status"] == "interrupted"
        assert client.get(f"/api/v1/projects/{pid}/artifacts").json() == original
        assert not (started.parent / "result.json").exists()
        assert job["result"]["selected_artifact_id"] is None


def completed_checkpoint(directory, step):
    checkpoint = directory / f"checkpoint-{step:06d}"
    files = {}
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
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint fixture")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (checkpoint / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "step": step, "files": files})
    )
    return checkpoint


def test_resume_checkpoint_is_project_scoped_and_contained(configured):
    from vla_platform.lifecycle.contracts import PolicyRequest

    app = create_app(configured)
    with TestClient(app) as client:
        pid = project(client)
        jid = submit(client, pid, operation="policy.import")
        wait(client, jid)
        execution = app.state.execution
        job = client.portal.call(execution.get, jid)
        job.status, job.kind, job.result = "interrupted", "policy.finetune", None
        job.request = PolicyRequest(
            operation="policy.finetune", runtime_id="fixture", dataset_job_id="fixture-intake"
        )
        client.portal.call(execution.save, job)
        directory = configured.data_dir / "jobs" / jid / "operation/training"
        checkpoint = completed_checkpoint(directory, 1)
        (directory / "latest.json").write_text(json.dumps({"checkpoint": checkpoint.name}))
        assert client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid) == checkpoint
        with pytest.raises(ValueError, match="in this project"):
            client.portal.call(execution.lifecycle.resume_checkpoint, project(client), jid)
        (directory / "latest.json").write_text(json.dumps({"checkpoint": "../../outside"}))
        # The pointer is advisory and is never followed, even when it escapes.
        assert client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid) == checkpoint
        job.status = "running"
        client.portal.call(execution.save, job)
        with pytest.raises(ValueError, match="interrupted training job"):
            client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid)


@pytest.mark.parametrize(
    "damage", ["truncated", "changed_payload", "changed_manifest", "extra", "duplicate", "symlink"]
)
def test_corrupt_cached_download_is_rebuilt_and_verified(configured, damage):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        job = wait(client, submit(client, pid, operation="policy.import"))
        artifact = job["result"]["artifacts"][0]
        url = f"/api/v1/projects/{pid}/artifacts/{artifact['id']}/download"
        original = client.get(url)
        assert original.status_code == 200
        cached = next((configured.data_dir / "exports").rglob("*.tar"))
        if damage == "truncated":
            cached.write_bytes(b"corrupt")
        else:
            with tarfile.open(fileobj=io.BytesIO(original.content)) as archive:
                members = [
                    (member, archive.extractfile(member).read() if member.isfile() else None)
                    for member in archive
                ]
            with tarfile.open(cached, "w") as archive:
                for member, payload in members:
                    if damage == "changed_payload" and member.name == "policy/model":
                        payload = b"z" * len(payload)
                    if damage == "changed_manifest" and member.name == "policy/manifest.json":
                        payload = b" " * len(payload)
                    if damage == "symlink" and member.name == "policy/model":
                        member.type, member.linkname, member.size, payload = (
                            tarfile.SYMTYPE,
                            "/outside",
                            0,
                            None,
                        )
                    archive.addfile(member, io.BytesIO(payload) if payload is not None else None)
                if damage in {"extra", "duplicate"}:
                    member = tarfile.TarInfo(
                        "policy/extra" if damage == "extra" else "policy/model"
                    )
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"!"))
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda _: client.get(url), range(3)))
        for response in responses:
            assert response.status_code == 200, response.text
            with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
                assert archive.extractfile("policy/model").read() == b"x" * 100
                assert sorted(archive.getnames()) == [
                    "policy",
                    "policy/manifest.json",
                    "policy/model",
                ]
                assert (
                    archive.extractfile("policy/manifest.json").read()
                    == (configured.data_dir / artifact["path"] / "manifest.json").read_bytes()
                )
        assert client.get(url).content == responses[0].content


@pytest.mark.parametrize(
    "tail", [b'{"sequence":99,"stage":', b'{"sequence":99,"message":"\xf0\x9f']
)
def test_restart_preserves_event_prefix_and_reports_interrupted_write(configured, tail):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        jid = submit(client, pid, operation="policy.import")
        assert wait(client, jid)["status"] == "succeeded"
        prefix = client.get(f"/api/v1/jobs/{jid}/events").json()
    path = configured.data_dir / "jobs" / jid / "events.jsonl"
    valid = path.read_bytes()
    path.write_bytes(valid + tail)
    app = create_app(configured)
    with TestClient(app) as client:
        response = client.get(f"/api/v1/jobs/{jid}/events")
        assert response.status_code == 200
        events = response.json()
        assert events[:-1] == prefix
        assert events[-1]["stage"] == "recovery"
        assert "incomplete" in events[-1]["message"].lower()
        assert events[-1]["sequence"] == prefix[-1]["sequence"] + 1
        job = client.portal.call(app.state.execution.get, jid)
        client.portal.call(app.state.execution.lifecycle.event, job, "test", "After recovery")
        repaired = client.get(f"/api/v1/jobs/{jid}/events").json()
        assert repaired[:-1] == events
        assert repaired[-1]["message"] == "After recovery"
        assert repaired[-1]["sequence"] == events[-1]["sequence"] + 1
        assert path.read_bytes().startswith(valid)
        assert client.get(
            f"/api/v1/jobs/{jid}/events", params={"after": events[-1]["sequence"]}
        ).json() == [repaired[-1]]
    with TestClient(create_app(configured)) as client:
        assert client.get(f"/api/v1/jobs/{jid}/events").json() == repaired


@pytest.mark.parametrize("tail", [b'{"sequence":99}\n', b'{"sequence":', b'{"sequence":\n'])
def test_interior_or_completed_event_corruption_is_not_hidden(configured, tail):
    with TestClient(create_app(configured)) as client:
        jid = submit(client, project(client), operation="policy.import")
        assert wait(client, jid)["status"] == "succeeded"
    path = configured.data_dir / "jobs" / jid / "events.jsonl"
    valid = path.read_bytes()
    # Even an unterminated fragment becomes interior corruption when more records follow.
    path.write_bytes(valid + tail + valid.splitlines(keepends=True)[-1])
    with TestClient(create_app(configured), raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/jobs/{jid}/events")
        assert response.status_code == 422
        assert "corrupt" in response.json()["detail"].lower()
    assert path.read_bytes() == valid + tail + valid.splitlines(keepends=True)[-1]


def test_complete_event_without_newline_is_retained_when_appending(configured):
    app = create_app(configured)
    with TestClient(app) as client:
        jid = submit(client, project(client), operation="policy.import")
        assert wait(client, jid)["status"] == "succeeded"
        prefix = client.get(f"/api/v1/jobs/{jid}/events").json()
        path = configured.data_dir / "jobs" / jid / "events.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        job = client.portal.call(app.state.execution.get, jid)
        client.portal.call(app.state.execution.lifecycle.event, job, "test", "After valid record")
        events = client.get(f"/api/v1/jobs/{jid}/events").json()
        assert events[:-1] == prefix
        assert events[-1]["sequence"] == prefix[-1]["sequence"] + 1


@pytest.mark.parametrize("pointer", [None, '{"checkpoint":', '{"checkpoint":"checkpoint-000001"}'])
def test_resume_recovers_highest_verified_checkpoint_without_trusting_pointer(configured, pointer):
    from vla_platform.lifecycle.contracts import PolicyRequest

    app = create_app(configured)
    with TestClient(app) as client:
        pid = project(client)
        jid = submit(client, pid, operation="policy.import")
        wait(client, jid)
        execution = app.state.execution
        job = client.portal.call(execution.get, jid)
        job.status, job.kind, job.result = "interrupted", "policy.finetune", None
        job.request = PolicyRequest(
            operation="policy.finetune", runtime_id="fixture", dataset_job_id="fixture-intake"
        )
        client.portal.call(execution.save, job)
        directory = configured.data_dir / "jobs" / jid / "operation/training"
        old = completed_checkpoint(directory, 1)
        latest = completed_checkpoint(directory, 2)
        partial = directory / "checkpoint-000003"
        partial.mkdir()
        (partial / "manifest.json").write_text("{}")
        corrupt = completed_checkpoint(directory, 4)
        (corrupt / "training.pt").write_bytes(b"corrupt")
        mismatch = completed_checkpoint(directory, 5)
        mismatch.rename(directory / "checkpoint-000006")
        (directory / ".checkpoint-000007").mkdir()
        if pointer is not None:
            (directory / "latest.json").write_text(pointer)
        assert client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid) == latest
        (latest / "adapter/adapter_model.safetensors").unlink()
        assert client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid) == old
        (old / "training.pt").unlink()
        with pytest.raises(ValueError, match="completed checkpoint"):
            client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid)


@pytest.mark.skipif(os.name != "posix", reason="Symlink creation needs privileges on Windows")
def test_resume_rejects_checkpoint_symlink_to_another_run(configured):
    from vla_platform.lifecycle.contracts import PolicyRequest

    app = create_app(configured)
    with TestClient(app) as client:
        pid = project(client)
        jid = submit(client, pid, operation="policy.import")
        wait(client, jid)
        execution = app.state.execution
        job = client.portal.call(execution.get, jid)
        job.status, job.kind, job.result = "interrupted", "policy.finetune", None
        job.request = PolicyRequest(
            operation="policy.finetune", runtime_id="fixture", dataset_job_id="fixture-intake"
        )
        client.portal.call(execution.save, job)
        other = completed_checkpoint(configured.data_dir / "another-run", 1)
        directory = configured.data_dir / "jobs" / jid / "operation/training"
        directory.mkdir(parents=True)
        (directory / "checkpoint-000001").symlink_to(other, target_is_directory=True)
        (directory / "latest.json").write_text('{"checkpoint":"checkpoint-000001"}')
        with pytest.raises(ValueError, match="completed checkpoint"):
            client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid)


def test_corrupt_open_cache_does_not_block_fresh_generation(configured, monkeypatch):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        job = wait(client, submit(client, pid, operation="policy.import"))
        artifact = job["result"]["artifacts"][0]
        url = f"/api/v1/projects/{pid}/artifacts/{artifact['id']}/download"
        assert client.get(url).status_code == 200
        cached = next((configured.data_dir / "exports").rglob("*.tar"))
        cached.write_bytes(b"corrupt")
        unlink = Path.unlink

        def windows_reader_holds_file(path, *args, **kwargs):
            if path == cached:
                raise PermissionError("Existing Windows reader")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", windows_reader_holds_file)
        response = client.get(url)
        assert response.status_code == 200
        with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
            assert archive.extractfile("policy/model").read() == b"x" * 100
        assert cached.read_bytes() == b"corrupt"
        assert len(list(cached.parent.glob("*.tar"))) == 2
        assert client.get(url).content == response.content


def test_export_changed_during_archiving_is_not_published(configured, monkeypatch):
    with TestClient(create_app(configured)) as client:
        pid = project(client)
        job = wait(client, submit(client, pid, operation="policy.import"))
        artifact = job["result"]["artifacts"][0]
        model = configured.data_dir / artifact["path"] / "model"
        add = tarfile.TarFile.add

        def mutate_before_archiving(archive, name, *args, **kwargs):
            if Path(name) == model:
                model.write_bytes(b"z" * 100)
            return add(archive, name, *args, **kwargs)

        monkeypatch.setattr(tarfile.TarFile, "add", mutate_before_archiving)
        response = client.get(f"/api/v1/projects/{pid}/artifacts/{artifact['id']}/download")
        assert response.status_code == 422
        assert "hash mismatch" in response.text
        assert not list((configured.data_dir / "exports").rglob("*.tar"))
        assert not list((configured.data_dir / "exports").rglob("*.tmp"))


def test_completed_invalid_final_event_is_rejected(configured):
    with TestClient(create_app(configured)) as client:
        jid = submit(client, project(client), operation="policy.import")
        assert wait(client, jid)["status"] == "succeeded"
        path = configured.data_dir / "jobs" / jid / "events.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"sequence":99,"stage":\n')
        with pytest.raises(ValueError, match="Corrupt event record"):
            client.get(f"/api/v1/jobs/{jid}/events")


@pytest.mark.parametrize("schema", [True, 1.0, "1"])
def test_resume_checkpoint_requires_integer_schema_version(configured, schema):
    from vla_platform.lifecycle.contracts import PolicyRequest

    app = create_app(configured)
    with TestClient(app) as client:
        pid = project(client)
        jid = submit(client, pid, operation="policy.import")
        wait(client, jid)
        execution = app.state.execution
        job = client.portal.call(execution.get, jid)
        job.status, job.kind, job.result = "interrupted", "policy.finetune", None
        job.request = PolicyRequest(
            operation="policy.finetune", runtime_id="fixture", dataset_job_id="fixture-intake"
        )
        client.portal.call(execution.save, job)
        directory = configured.data_dir / "jobs" / jid / "operation/training"
        verified = completed_checkpoint(directory, 1)
        invalid = completed_checkpoint(directory, 2) / "manifest.json"
        manifest = json.loads(invalid.read_text())
        manifest["schema_version"] = schema
        invalid.write_text(json.dumps(manifest))
        assert client.portal.call(execution.lifecycle.resume_checkpoint, pid, jid) == verified


@pytest.mark.parametrize(
    "stage", ["baseline", "final-reference", "final-evaluation", "package-and-reload"]
)
@pytest.mark.parametrize(
    "field,value",
    [("complete_episodes", 1), ("p95_ms", 0), ("peak_device_mib", -1), ("success_rate", 1.2)],
)
def test_incomplete_or_invalid_measurements_never_qualify(configured, stage, field, value):
    set_label(
        configured, "invalid-report:" + json.dumps({"stage": stage, "field": field, "value": value})
    )
    with TestClient(create_app(configured)) as client:
        job = wait(
            client, submit(client, project(client), evaluation={"mode": "libero"}, limits={})
        )
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == "no_feasible_candidate"
        assert job["result"]["selected_artifact_id"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_worker_json_fails_without_corrupting_persisted_job(configured, value):
    set_label(
        configured,
        "invalid-report:"
        + json.dumps({"stage": "final-evaluation", "field": "p95_ms", "value": value}),
    )
    with TestClient(create_app(configured)) as client:
        job = wait(
            client, submit(client, project(client), evaluation={"mode": "libero"}, limits={})
        )
        assert job["status"] == "failed", job
        assert "non-finite number" in job["error"]
        assert job["result"]["selected_artifact_id"] is None
        assert not any(x["stage"] == "package-and-reload" for x in job["result"]["reports"])


def test_workflow_starts_with_q8_only_and_keeps_explicit_comparisons():
    from vla_platform.lifecycle.contracts import PolicyRequest

    fields = {"operation": "policy.workflow", "runtime_id": "fixture", "source_id": "source"}
    request = PolicyRequest(**fields)
    assert [candidate.language for candidate in request.candidates] == ["Q8_0"]
    assert len(PolicyRequest(**fields, candidates=[{"language": "Q8_0"}]).candidates) == 1
    explicit = PolicyRequest(**fields, candidates=[{"language": "Q8_0"}, {"language": "Q4_0"}])
    assert [candidate.language for candidate in explicit.candidates] == ["Q8_0", "Q4_0"]


@pytest.mark.parametrize(
    "stage", ["baseline", "final-reference", "final-evaluation", "package-and-reload"]
)
def test_incomplete_memory_coverage_never_qualifies(configured, stage):
    set_label(
        configured,
        "invalid-report:"
        + json.dumps(
            {
                "stage": stage,
                "field": "memory_coverage",
                "value": {"complete": False},
            }
        ),
    )
    with TestClient(create_app(configured)) as client:
        job = wait(
            client, submit(client, project(client), evaluation={"mode": "libero"}, limits={})
        )
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == "no_feasible_candidate"
        assert job["result"]["selected_artifact_id"] is None


def test_overflow_worker_json_fails_before_publishing_report(configured):
    set_label(configured, "overflow-json:final-evaluation")
    with TestClient(create_app(configured)) as client:
        job = wait(
            client, submit(client, project(client), evaluation={"mode": "libero"}, limits={})
        )
        assert job["status"] == "failed", job
        assert "non-finite number" in job["error"]
        assert job["result"]["selected_artifact_id"] is None
        assert not any(x["stage"] == "final-evaluation" for x in job["result"]["reports"])
