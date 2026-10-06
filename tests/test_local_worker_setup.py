"""Installer lifecycle, admission and registration checks without downloading ML packages."""

import asyncio
import json
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from vla_platform import local_worker_discovery as discovery
from vla_platform import local_worker_setup as installer
from vla_platform.api import create_app
from vla_platform.settings import Settings


@pytest.fixture
def host(tmp_path, monkeypatch):
    root = tmp_path / "worker"
    root.mkdir()
    (root / "requirements-smolvla-linux.txt").write_text("generated install fixture")
    monkeypatch.setattr(discovery, "_worker_root", lambda: root)
    monkeypatch.setattr(installer, "_worker_root", lambda: root)
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    gpu = ("0", "GPU-1234-abcd", "NVIDIA RTX 3070", 8192)
    monkeypatch.setattr(installer, "_hardware", lambda env: gpu)
    monkeypatch.setattr(discovery, "_hardware", lambda env: gpu)
    monkeypatch.setattr(installer.shutil, "which", lambda name: sys.executable)
    monkeypatch.setattr(
        installer.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3)
    )
    monkeypatch.setattr(
        discovery,
        "_run_bounded",
        lambda *args, **kwargs: json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "reason": None,
                "gpu": {"name": gpu[2], "memory_mib": gpu[3], "uuid": gpu[1], "index": 0},
            }
        ).encode(),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []

    async def install(self, arguments, stage):
        calls.append((arguments, stage))
        self.state.stage = stage
        self._save()
        await asyncio.sleep(0.02)
        python = self.directory / "smolvla-cu126/bin/python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("generated installed interpreter fixture")
        python.chmod(0o700)

    monkeypatch.setattr(installer.LocalWorkerSetup, "_command", install)
    return Settings(data_dir=tmp_path / "workspace"), calls


def terminal(client):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get("/api/v1/compute-settings/local/setup").json()
        if state["status"] != "running":
            return state
        time.sleep(0.01)
    raise AssertionError("Setup did not become terminal")


def test_get_is_read_only_and_install_is_explicit_idempotent_and_registers(host):
    settings, calls = host
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/compute-settings/local/setup").json()["status"] == "idle"
        assert calls == [] and not (settings.data_dir / "local-environments").exists()
        first = client.post("/api/v1/compute-settings/local/setup")
        assert first.status_code == 202
        again = client.post("/api/v1/compute-settings/local/setup")
        assert first.json()["id"] == again.json()["id"]
        result = terminal(client)
        assert result["status"] == "succeeded" and result["runtime_id"]
        assert len(calls) == 2
        assert "--python" in calls[0][0] and calls[0][0][-1].endswith("smolvla-cu126")
        compute = client.get("/api/v1/compute-settings").json()
        assert compute["local"]["enabled"] is False
        assert compute["runtimes"][0]["training_model_ids"] == ["smolvla"]
        assert "workspace" not in json.dumps(result) and "GPU-1234" not in json.dumps(result)
        assert client.post("/api/v1/compute-settings/local/setup").json()["status"] == "succeeded"
        assert len(calls) == 2  # Ready workers are not reinstalled.
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/compute-settings/local/setup").json()["status"] == "succeeded"


def test_rejects_browser_paths_and_insufficient_disk(host, monkeypatch):
    settings, calls = host
    with TestClient(create_app(settings)) as client:
        for payload in [
            {"command": "arbitrary"},
            {"python": "/tmp/foreign"},
            {"candidate_id": "elsewhere"},
        ]:
            assert (
                client.post("/api/v1/compute-settings/local/setup", json=payload).status_code == 422
            )
        monkeypatch.setattr(installer.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
        response = client.post("/api/v1/compute-settings/local/setup")
        assert response.status_code == 409 and "24 GB" in response.text
        assert calls == []


def test_cancellation_and_restart_never_claim_ready(host, monkeypatch):
    settings, calls = host

    async def wait(self, *args):
        await asyncio.sleep(60)

    monkeypatch.setattr(installer.LocalWorkerSetup, "_command", wait)
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/compute-settings/local/setup").status_code == 202
        result = client.post("/api/v1/compute-settings/local/setup/cancel").json()
        assert result["status"] == "cancelled"
        assert not (settings.data_dir / "local-workers.json").exists()
        assert client.post("/api/v1/compute-settings/local/setup").status_code == 202
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/compute-settings/local/setup").json()["status"] == "interrupted"


def test_failed_install_has_safe_error_and_retry(host, monkeypatch):
    settings, _ = host

    async def failure(self, *args):
        raise OSError("PRIVATE_CREDENTIAL /private/install/path")

    monkeypatch.setattr(installer.LocalWorkerSetup, "_command", failure)
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/compute-settings/local/setup").status_code == 202
        state = terminal(client)
        assert state["status"] == "failed" and "PRIVATE" not in json.dumps(state)
        assert not (settings.data_dir / "local-workers.json").exists()


def test_real_owned_installer_timeout_retires_process(tmp_path, monkeypatch):
    lifecycle = SimpleNamespace(settings=Settings(data_dir=tmp_path))
    service = installer.LocalWorkerSetup(lifecycle, None)
    service._save()
    monkeypatch.setattr(installer, "STEP_TIMEOUT", 0.15)
    command = installer.LocalWorkerSetup._command

    async def run():
        with pytest.raises(TimeoutError):
            await command(service, [sys.executable, "-c", "import time; time.sleep(30)"], "fixture")
        assert service.process is None

    asyncio.run(run())


@pytest.mark.parametrize("part", ["local-environments", "local-environments/smolvla-cu126"])
def test_rejects_symlink_destinations(host, tmp_path, part):
    settings, calls = host
    with TestClient(create_app(settings)) as client:
        target = tmp_path / "foreign"
        target.mkdir()
        link = settings.data_dir / part
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        response = client.post("/api/v1/compute-settings/local/setup")
        assert response.status_code == 409 and "symbolic link" in response.text
        assert calls == [] and list(target.iterdir()) == []


def test_active_jobs_block_installation(host):
    from sqlalchemy import delete, insert
    from vla_platform.storage import jobs

    settings, calls = host
    app = create_app(settings)
    with TestClient(app) as client:

        async def change(add):
            async with app.state.execution.storage.engine.begin() as connection:
                await connection.execute(
                    insert(jobs).values(
                        id="queued-fixture", project_id="fixture", status="queued", record={}
                    )
                    if add
                    else delete(jobs).where(jobs.c.id == "queued-fixture")
                )

        client.portal.call(change, True)
        try:
            response = client.post("/api/v1/compute-settings/local/setup")
            assert response.status_code == 409 and "active jobs" in response.text
            assert calls == []
        finally:
            client.portal.call(change, False)


def test_managed_worker_cannot_be_admitted_during_installation(tmp_path):
    from vla_platform.lifecycle.service import Lifecycle

    directory = tmp_path / "local-environments"
    runtime = SimpleNamespace(
        provider="local", training_python=str(directory / "smolvla-cu126/bin/python")
    )
    lifecycle = SimpleNamespace(
        runtime=lambda ident: runtime,
        local_worker_setup=SimpleNamespace(
            state=SimpleNamespace(status="running"), directory=directory
        ),
    )
    with pytest.raises(ValueError, match="Wait for local setup"):
        asyncio.run(Lifecycle.validate(lifecycle, "fixture", SimpleNamespace(runtime_id="managed")))


def test_immediate_cancellation_is_terminal_even_before_task_starts(tmp_path):
    service = installer.LocalWorkerSetup(
        SimpleNamespace(settings=Settings(data_dir=tmp_path)), None
    )

    async def run():
        service.state.status = "running"
        service.task = asyncio.create_task(asyncio.sleep(30))
        result = await service.cancel()
        assert result.status == "cancelled" and result.finished_at

    asyncio.run(run())
