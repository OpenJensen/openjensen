"""Explicit discovery/registration uses server-owned candidates, never browser paths."""

import json
import os

import pytest
from fastapi.testclient import TestClient
from vla_platform import local_worker_discovery as discovery
from vla_platform.api import create_app
from vla_platform.settings import Settings


@pytest.fixture
def discovered_host(tmp_path, monkeypatch):
    root = tmp_path / "private-worker"
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("generated readiness fixture; not an executable worker")
    python.chmod(0o700)
    monkeypatch.setattr(discovery, "_worker_root", lambda: root)
    monkeypatch.setattr(discovery.platform, "system", lambda: "Linux")
    monkeypatch.setattr(discovery.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(discovery.platform, "node", lambda: "gpu-host")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    state = {"probes": [], "ready": True}

    def probe(argv, **kwargs):
        state["probes"].append(argv)
        assert argv[0] == str(python)
        assert argv[1] == "-I"
        assert "PRIVATE_TOKEN" not in kwargs["env"]
        value = {
            "schema_version": 1,
            "status": "ready" if state["ready"] else "unavailable",
            "reason": None if state["ready"] else "cuda_unavailable",
            "gpu": {
                "name": "Generated NVIDIA GPU",
                "memory_mib": 8192,
                "uuid": "GPU-aabbccdd-1122-3344",
                "index": 0,
            }
            if state["ready"]
            else None,
        }
        return json.dumps(value).encode()

    monkeypatch.setenv("PRIVATE_TOKEN", "private-secret")
    monkeypatch.setattr(discovery, "_run_bounded", probe)
    monkeypatch.setattr(discovery, "_hardware", lambda env: None)
    return Settings(data_dir=tmp_path / "workspace"), state


def test_check_is_explicit_and_does_not_register_enable_or_start_jobs(discovered_host):
    settings, state = discovered_host
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/v1/projects", json={"name": "Discovery fixture"}).json()
        original = client.get("/api/v1/compute-settings").json()
        client.get("/api/v1/policy-options")
        assert state["probes"] == []
        result = client.post("/api/v1/compute-settings/local/check")
        assert result.status_code == 200
        body = result.json()
        assert body["host"] == {"name": "gpu-host", "platform": "Linux", "architecture": "x86_64"}
        assert body["status"] == "ready"
        assert body["candidates"][0]["status"] == "ready"
        assert len(state["probes"]) == 1
        assert not (settings.data_dir / "local-workers.json").exists()
        assert client.get("/api/v1/compute-settings").json() == original
        assert client.get(f"/api/v1/projects/{project['id']}/jobs").json() == []
        assert all(
            token not in result.text
            for token in ("private-worker", "GPU-aabb", "private-secret", "/bin/python")
        )


def test_add_rechecks_persists_and_preserves_preferences(discovered_host):
    settings, state = discovered_host
    with TestClient(create_app(settings)) as client:
        client.put(
            "/api/v1/compute-settings", json={"local": {"enabled": False, "label": "My lab"}}
        )
        before = (settings.data_dir / "compute-settings.json").read_bytes()
        found = client.post("/api/v1/compute-settings/local/check").json()["candidates"][0]
        response = client.post(
            "/api/v1/compute-settings/local/workers", json={"candidate_id": found["id"]}
        )
        assert response.status_code == 200
        body = response.json()
        runtime = body["runtime"]
        assert len(state["probes"]) == 2
        assert runtime["training"] is True
        assert runtime["training_only"] is True
        assert runtime["engine_evaluation"] is False and runtime["run"] is False
        assert runtime["enabled"] is False
        assert body["compute"]["local"] == {"enabled": False, "label": "My lab"}
        assert body["discovery"]["candidates"][0]["status"] == "registered"
        assert body["discovery"]["candidates"][0]["runtime_id"] == runtime["id"]
        assert all(
            token not in response.text
            for token in ("private-worker", "GPU-aabb", "private-secret", "/bin/python")
        )
        assert (settings.data_dir / "compute-settings.json").read_bytes() == before
        saved = settings.data_dir / "local-workers.json"
        content = saved.read_bytes()
        repeated = client.post(
            "/api/v1/compute-settings/local/workers", json={"candidate_id": found["id"]}
        )
        assert repeated.status_code == 200
        assert repeated.json()["runtime"]["id"] == runtime["id"]
        assert saved.read_bytes() == content
        if os.name == "posix":
            assert saved.stat().st_mode & 0o077 == 0
    with TestClient(create_app(settings)) as client:
        local = [
            item
            for item in client.get("/api/v1/policy-options").json()["runtimes"]
            if item["provider"] == "local"
        ]
        assert len(local) == 1 and local[0]["id"] == runtime["id"]
        assert local[0]["enabled"] is False
        client.put("/api/v1/compute-settings", json={"local": {"enabled": True, "label": "My lab"}})
        assert (
            next(
                item
                for item in client.get("/api/v1/policy-options").json()["runtimes"]
                if item["id"] == runtime["id"]
            )["enabled"]
            is True
        )


def test_add_rejects_unscanned_unknown_and_caller_paths(discovered_host):
    settings, _ = discovered_host
    with TestClient(create_app(settings)) as client:
        assert (
            client.post(
                "/api/v1/compute-settings/local/workers",
                json={"candidate_id": "managed-local-unknown"},
            ).status_code
            == 409
        )
        found = client.post("/api/v1/compute-settings/local/check").json()["candidates"][0]
        for extra in (
            {"python": "/tmp/evil"},
            {"env": {"TOKEN": "secret"}},
            {"training_root": "/tmp/worker"},
        ):
            assert (
                client.post(
                    "/api/v1/compute-settings/local/workers",
                    json={"candidate_id": found["id"], **extra},
                ).status_code
                == 422
            )
        assert (
            client.post(
                "/api/v1/compute-settings/local/workers", json={"candidate_id": "../worker"}
            ).status_code
            == 422
        )
        assert not (settings.data_dir / "local-workers.json").exists()


def test_stale_worker_is_not_registered_and_errors_are_safe(discovered_host, monkeypatch):
    settings, state = discovered_host
    with TestClient(create_app(settings)) as client:
        found = client.post("/api/v1/compute-settings/local/check").json()["candidates"][0]
        state["ready"] = False
        response = client.post(
            "/api/v1/compute-settings/local/workers", json={"candidate_id": found["id"]}
        )
        assert response.status_code == 409
        assert not (settings.data_dir / "local-workers.json").exists()
        state["ready"] = True
        found = client.post("/api/v1/compute-settings/local/check").json()["candidates"][0]

        def failure(runtime):
            raise OSError("PRIVATE /secret/worker-config")

        monkeypatch.setattr(client.app.state.execution.lifecycle.local_workers, "register", failure)
        response = client.post(
            "/api/v1/compute-settings/local/workers", json={"candidate_id": found["id"]}
        )
        assert response.status_code == 500
        assert "PRIVATE" not in response.text and "/secret/" not in response.text
        assert client.get("/api/v1/compute-settings").json()["local"]["enabled"] is False
