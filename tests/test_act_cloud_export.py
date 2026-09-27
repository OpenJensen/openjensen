"""Explicit cloud ACT export uses a real bounded copy, never a real cloud service."""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from test_act_export import application as _application
from test_act_export import submit
from test_cloud_storage import Blob
from test_cloud_storage import cloud as _cloud
from test_lifecycle import wait
from vla_platform.lifecycle import cloud_materialize as copy
from vla_platform.lifecycle import cloud_storage as storage
from vla_platform.lifecycle.runtime import Runtime

application = _application
cloud = _cloud


def prepare(application, cloud, tmp_path, monkeypatch):
    app, client, pid, source, root = application
    (root / "checkpoint/pretrained_model/config.json").write_text("{}")
    (root / "checkpoint/recipe.json").write_text("{}")
    (root / "verification.json").write_text('{"reload_verified":true}')
    metadata = {
        **source.metadata,
        "reload_verified": True,
        "dataset": {
            **source.metadata["dataset"],
            "format": "lerobot_v3",
            "features": {"action": {"shape": [6]}},
        },
    }
    storage.write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "metadata": metadata,
            "files": {
                path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"
            }
            | {
                "checkpoint/manifest.json": hashlib.sha256(
                    (root / "checkpoint/manifest.json").read_bytes()
                ).hexdigest()
            },
        },
    )
    descriptor = storage.upload_artifact(
        root, "gs://test-bucket/jobs/complete/artifact", kind="training_checkpoint"
    )
    remote = root.parent / "remote"
    storage.install_descriptor(remote, descriptor)
    source = source.model_copy(
        update={
            "path": "remote",
            "manifest_sha256": descriptor["manifest_sha256"],
            "metadata": descriptor["manifest"]["metadata"],
            "file_bytes": descriptor["file_bytes"],
        }
    )
    record = client.portal.call(app.state.execution.get, "source")
    record.result.artifacts = [source]
    client.portal.call(app.state.execution.save, record)
    store = tmp_path / "fake-objects"
    for name, content in cloud.objects.items():
        path = store / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setenv("FIREBIRD_TEST_GCS_STORE", str(store))
    from vla_platform import cloud_compute_catalog

    monkeypatch.setattr(cloud_compute_catalog, "sky_python", lambda _: sys.executable)
    from vla_platform.lifecycle import sky_runner

    monkeypatch.setattr(sky_runner, "executable", lambda: "never-called")
    monkeypatch.setattr(
        copy, "__file__", str(Path(__file__).parent / "fixtures/cloud_copy_worker.py")
    )
    monkeypatch.setattr(Blob, "generation", 7, raising=False)
    return app, client, pid, source, remote, store


def test_remote_completed_checkpoint_is_copied_exported_registered_and_original_unchanged(
    application, cloud, tmp_path, monkeypatch
):
    app, client, pid, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in remote.iterdir()}
    accepted = submit(client, pid, source)
    assert accepted.status_code == 202, accepted.text
    completed = wait(client, accepted.json()["id"])
    assert completed["status"] == "succeeded", completed
    local, exported = completed["result"]["artifacts"]
    assert local["format"] == "training_checkpoint" and local["parent_ids"] == [source.id]
    assert exported["parent_ids"] == [local["id"]]
    assert exported["metadata"]["dataset"] == {
        key: source.metadata["dataset"][key] for key in ("source", "repo_id", "revision")
    }
    assert local["metadata"]["dataset"] == source.metadata["dataset"]
    assert local["metadata"]["cloud_source"]["source_manifest_sha256"] == source.manifest_sha256
    assert "storage" not in local["metadata"] and "remote_uri" not in local["metadata"]
    assert (
        client.get(f"/api/v1/projects/{pid}/artifacts/{exported['id']}/download").status_code == 200
    )
    assert before == {p.name: p.read_bytes() for p in remote.iterdir()}
    assert len(cloud.downloads) == 0  # only the real fixture child performed reads


@pytest.mark.parametrize(
    "damage", ["running", "failed", "periodic", "cross-project", "manifest-change"]
)
def test_bad_cloud_source_is_rejected_before_any_child(
    application, cloud, tmp_path, monkeypatch, damage
):
    app, client, pid, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    record = client.portal.call(app.state.execution.get, "source")
    if damage in {"running", "failed"}:
        record.status = damage
    elif damage == "periodic":
        record.result.artifacts[0].metadata["reload_verified"] = False
    elif damage == "cross-project":
        pid = client.post("/api/v1/projects", json={"name": "Other project"}).json()["id"]
    else:
        (remote / "manifest.json").write_bytes((remote / "manifest.json").read_bytes() + b" ")
    client.portal.call(app.state.execution.save, record)
    response = submit(client, pid, source)
    assert response.status_code == 422, response.text
    assert not list((app.state.execution.settings.data_dir / "jobs").glob("*/checkpoint-download"))


def test_failed_copy_or_parity_never_registers_local_child(
    application, cloud, tmp_path, monkeypatch
):
    app, client, pid, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    app.state.execution.lifecycle.catalog.runtimes[0].env["FIXTURE_FAULT"] = "reload"
    accepted = submit(client, pid, source)
    completed = wait(client, accepted.json()["id"])
    assert completed["status"] == "failed"
    assert [a["id"] for a in client.get(f"/api/v1/projects/{pid}/artifacts").json()] == [source.id]
    assert not (
        app.state.execution.settings.data_dir / "jobs" / completed["id"] / "checkpoint-download"
    ).exists()
    assert (remote / "remote.json").exists()


def test_cancelling_live_download_reaps_child_and_removes_only_owned_staging(
    application, cloud, tmp_path, monkeypatch
):
    app, client, pid, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    started = tmp_path / "child-pid"
    monkeypatch.setenv("FIREBIRD_TEST_COPY_HANG", str(started))
    accepted = submit(client, pid, source).json()

    async def observe():
        async with asyncio.timeout(10):
            while not started.exists():
                await asyncio.sleep(0.02)

    client.portal.call(observe)
    child_pid = int(started.read_text())
    cancelled = client.post(f"/api/v1/jobs/{accepted['id']}/cancel")
    assert cancelled.status_code == 200
    completed = wait(client, accepted["id"])
    assert completed["status"] == "cancelled"

    async def cleaned():
        async with asyncio.timeout(10):
            while (
                app.state.execution.settings.data_dir
                / "jobs"
                / accepted["id"]
                / "checkpoint-download"
            ).exists():
                await asyncio.sleep(0.02)

    client.portal.call(cleaned)
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    assert (remote / "remote.json").exists()
    assert [a["id"] for a in client.get(f"/api/v1/projects/{pid}/artifacts").json()] == [source.id]


def test_export_only_runtime_needs_no_fake_engine_paths_and_cannot_run_engine_jobs(application):
    app, client, pid, source, _ = application
    current = app.state.execution.lifecycle.catalog.runtimes[0]
    configured = Runtime.model_validate(
        {
            "id": "act-cpu",
            "label": "CPU ACT exporter",
            "export_only": True,
            "act_export_python": current.act_export_python,
            "act_export_root": current.act_export_root,
        }
    )
    assert configured.vendor is None and configured.build is None and configured.worker_root is None
    app.state.execution.lifecycle.catalog.runtimes = [configured]
    options = client.get("/api/v1/policy-options").json()["runtimes"]
    assert options[0]["export_only"] is True and options[0]["training"] is False
    for operation in ("policy.run", "policy.quantize", "policy.finetune", "policy.workflow"):
        rejected = submit(client, pid, source, operation=operation, dataset_job_id="fixture")
        assert rejected.status_code == 422 and "export only" in rejected.text
    assert wait(client, submit(client, pid, source).json()["id"])["status"] == "succeeded"
    capabilities = client.get("/api/v1/capabilities").json()
    assert next(c for c in capabilities if c["operation"] == "policy.run")["status"] == "planned"


@pytest.mark.parametrize(
    "damage", ["corrupt-late", "short", "extra", "generation", "disk", "per-file", "pointer-change"]
)
def test_materialization_failures_leave_source_untouched_and_no_partial_output(
    application, cloud, tmp_path, monkeypatch, damage
):
    _, _, _, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in remote.iterdir()}
    output = tmp_path / "completed-copy"
    original_open = Blob.open
    if damage == "generation":
        monkeypatch.setattr(Blob, "generation", None)
    elif damage == "disk":
        from types import SimpleNamespace

        monkeypatch.setattr(copy.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    elif damage == "per-file":
        monkeypatch.setattr(copy, "MAX_FILE_BYTES", 1)
    else:
        import io

        def faulty_open(blob, *args, **kwargs):
            if blob.name.endswith("verification.json"):
                raw = blob.bucket.objects[blob.name]
                if damage == "corrupt-late":
                    raw = b"x" + raw[1:]
                elif damage == "short":
                    raw = raw[:-1]
                elif damage == "extra":
                    raw += b"x"
                elif damage == "pointer-change":
                    pointer = json.loads((remote / "remote.json").read_bytes())
                    pointer["extra"] = "concurrent change"
                    (remote / "remote.json").write_text(json.dumps(pointer))
                return io.BytesIO(raw)
            return original_open(blob, *args, **kwargs)

        monkeypatch.setattr(Blob, "open", faulty_open)
    with pytest.raises(ValueError):
        copy.copy_remote(remote, output, source.manifest_sha256, source.id)
    assert not output.exists() and not list(tmp_path.glob(".act-checkpoint-*"))
    if damage != "pointer-change":
        assert before == {p.name: p.read_bytes() for p in remote.iterdir()}


def test_no_replace_publication_preserves_existing_and_racing_destinations(
    application, cloud, tmp_path, monkeypatch
):
    _, _, _, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    output = tmp_path / "completed-copy"
    output.mkdir()
    with pytest.raises(ValueError, match="new"):
        copy.copy_remote(remote, output, source.manifest_sha256, source.id)
    output.rmdir()
    publish = copy.publish_new

    def raced(staging, destination):
        destination.mkdir()
        (destination / "other-owner").write_text("preserve")
        publish(staging, destination)

    monkeypatch.setattr(copy, "publish_new", raced)
    with pytest.raises(FileExistsError):
        copy.copy_remote(remote, output, source.manifest_sha256, source.id)
    assert (output / "other-owner").read_text() == "preserve"
    assert not list(tmp_path.glob(".act-checkpoint-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO safety boundary")
def test_descriptor_fifo_is_rejected_without_blocking(application, cloud, tmp_path, monkeypatch):
    _, _, _, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    (remote / "remote.json").unlink()
    os.mkfifo(remote / "remote.json")
    with pytest.raises(ValueError, match="regular"):
        copy.descriptor(remote, source.manifest_sha256)


def test_bad_download_bytes_fail_api_without_registering_or_retaining_staging(
    application, cloud, tmp_path, monkeypatch
):
    app, client, pid, source, remote, store = prepare(application, cloud, tmp_path, monkeypatch)
    path = store / "jobs/complete/artifact/verification.json"
    raw = path.read_bytes()
    path.write_bytes(b"x" + raw[1:])
    accepted = submit(client, pid, source).json()
    completed = wait(client, accepted["id"])
    assert completed["status"] == "failed" and "download failed" in completed["error"]
    assert [a["id"] for a in client.get(f"/api/v1/projects/{pid}/artifacts").json()] == [source.id]
    assert not (
        app.state.execution.settings.data_dir / "jobs" / accepted["id"] / "checkpoint-download"
    ).exists()
    assert (remote / "remote.json").exists()


@pytest.mark.parametrize("window", ["spawn", "cleanup", "completed-cleanup"])
@pytest.mark.parametrize("stage", ["download", "export"])
def test_repeated_cancellation_reaps_actual_owned_child_before_staging_cleanup(
    application, cloud, tmp_path, monkeypatch, window, stage
):
    app, client, pid, source, remote, _ = prepare(application, cloud, tmp_path, monkeypatch)
    started = tmp_path / "cancel-child-pid"
    if window != "completed-cleanup":
        monkeypatch.setenv(
            "FIREBIRD_TEST_COPY_HANG" if stage == "download" else "FIREBIRD_TEST_EXPORT_HANG",
            str(started),
        )
    lifecycle = app.state.execution.lifecycle
    original_spawn, original_stop = asyncio.create_subprocess_exec, lifecycle.stop

    async def exercise():
        arrived, release = asyncio.Event(), asyncio.Event()
        children = []

        async def spawn(*args, **kwargs):
            child = await original_spawn(*args, **kwargs)
            children.append(child)
            if window == "spawn" and (stage == "download" or len(children) == 2):
                arrived.set()
                await release.wait()
            return child

        async def stop(child, container):
            if window != "spawn" and (stage == "download" or len(children) == 2):
                arrived.set()
                await release.wait()
            await original_stop(child, container)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(lifecycle, "stop", stop)
        from vla_platform.lifecycle.contracts import PolicyRequest

        original = await app.state.execution.get("source")
        job = original.model_copy(
            deep=True,
            update={
                "id": "owned-cancel-" + window,
                "kind": "policy.export",
                "status": "running",
                "request": PolicyRequest(
                    operation="policy.export", runtime_id="act-cpu", artifact_id=source.id
                ),
            },
        )
        task = asyncio.create_task(lifecycle.run(job))
        async with asyncio.timeout(10):
            if window == "cleanup":
                while not started.exists():
                    await asyncio.sleep(0.01)
                task.cancel()
            await arrived.wait()
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            # The child must remain owned until its creation/cleanup handle returns.
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(children) == (1 if stage == "download" else 2)
        assert all(child.returncode is not None for child in children)
        assert not (
            app.state.execution.settings.data_dir / "jobs" / job.id / "checkpoint-download"
        ).exists()
        if os.name == "posix":
            for child in children:
                with pytest.raises(ProcessLookupError):
                    os.kill(child.pid, 0)

    client.portal.call(exercise)
    assert (remote / "remote.json").exists()
    assert [a["id"] for a in client.get(f"/api/v1/projects/{pid}/artifacts").json()] == [source.id]


@pytest.mark.parametrize(
    "value", [b'{"value":NaN}', b'{"value":1e309}', b'{"value":-1e309}', b'{"value":1,"value":2}']
)
def test_malformed_metadata_fails_before_cloud_access(value):
    with pytest.raises(ValueError):
        copy.decode(value)
