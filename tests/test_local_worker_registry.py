"""Discovered trainers stay opt-in, durable, and isolated from engine capabilities."""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform.api import create_app
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog, Source, command
from vla_platform.local_worker_registry import LocalWorkerRegistry
from vla_platform.settings import Settings


def trainer(tmp_path, **updates):
    return Runtime.model_validate(
        {
            "id": "managed-local-smolvla-test",
            "label": "Local GPU",
            "device": "cuda",
            "training_only": True,
            "training_python": sys.executable,
            "training_root": str(tmp_path / "private-worker"),
            "gpu_name": "Test GPU",
            "gpu_memory_mib": 8192,
            "env": {"CUDA_VISIBLE_DEVICES": "GPU-test-1234"},
            **updates,
        }
    )


def test_registration_is_idempotent_persistent_and_preserves_operator_sources(tmp_path):
    source = Source(id="source", label="Source", path="/private/source", sha256="a" * 64)
    operator = RuntimeCatalog(sources=[source])
    registry = LocalWorkerRegistry(tmp_path, operator)
    registered = registry.register(trainer(tmp_path))
    assert registry.register(trainer(tmp_path, id="managed-local-another-id")) == registered
    assert len(registry.merge().runtimes) == 1
    assert LocalWorkerRegistry(tmp_path, operator).merge() == registry.merge()
    assert registry.merge().sources == operator.sources
    assert operator.runtimes == []
    if os.name == "posix":
        assert registry.path.stat().st_mode & 0o077 == 0
    result = registry.merge()
    result.runtimes[0].label = "Changed copy"
    assert registry.merge().runtimes[0].label == "Local GPU"


def test_operator_equivalent_is_reused_and_conflicts_cannot_override_it(tmp_path):
    operator_runtime = trainer(tmp_path, id="operator-gpu")
    operator = RuntimeCatalog(runtimes=[operator_runtime])
    registry = LocalWorkerRegistry(tmp_path, operator)
    assert registry.register(trainer(tmp_path)).id == "operator-gpu"
    assert not registry.path.exists()
    occupied = trainer(tmp_path, training_root=str(tmp_path / "another-worker"))
    registry = LocalWorkerRegistry(tmp_path, RuntimeCatalog(runtimes=[occupied]))
    with pytest.raises(ValueError, match="different worker"):
        registry.register(trainer(tmp_path))
    assert registry.merge().runtimes == [occupied]
    assert not registry.path.exists()


def test_distinct_virtualenvs_are_not_collapsed_to_their_shared_python(tmp_path):
    registry = LocalWorkerRegistry(tmp_path, RuntimeCatalog())
    for name in ("one", "two"):
        python = tmp_path / name / "bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        registry.register(
            trainer(tmp_path, id=f"managed-local-{name}", training_python=str(python))
        )
    assert len(registry.merge().runtimes) == 2


def test_restart_conflict_preserves_operator_and_exposes_issue(tmp_path):
    original = trainer(tmp_path)
    LocalWorkerRegistry(tmp_path, RuntimeCatalog()).register(original)
    operator = trainer(tmp_path, training_root=str(tmp_path / "operator-worker"))
    registry = LocalWorkerRegistry(tmp_path, RuntimeCatalog(runtimes=[operator]))
    assert registry.merge().runtimes == [operator]
    assert len(registry.issues) == 1
    assert "conflicts with an operator worker" in registry.issues[0]
    assert (
        json.loads(registry.path.read_text())["runtimes"][0]["training_root"]
        == original.training_root
    )


def test_atomic_save_failure_preserves_saved_and_active_workers(tmp_path, monkeypatch):
    registry = LocalWorkerRegistry(tmp_path, RuntimeCatalog())
    original = registry.register(trainer(tmp_path))
    saved = registry.path.read_text()

    def fail(*args):
        raise OSError("Disk unavailable")

    monkeypatch.setattr("vla_platform.local_worker_registry.os.replace", fail)
    with pytest.raises(OSError):
        registry.register(
            trainer(tmp_path, id="managed-local-other", env={"CUDA_VISIBLE_DEVICES": "1"})
        )
    assert registry.merge().runtimes == [original]
    assert registry.path.read_text() == saved
    assert not list(tmp_path.glob(".local-workers-*"))


@pytest.mark.parametrize(
    "change",
    [
        {"id": "operator-owned"},
        {"training_python": "untrusted-command"},
        {"training_root": "/tmp/../untrusted"},
        {"training_module": "untrusted.application"},
        {"env": {"PYTHONSTARTUP": "/private/secret"}},
        {"env": {"CUDA_VISIBLE_DEVICES": "0; arbitrary-command"}},
        {"python": "/other/python"},
        {"worker_root": "/other/worker"},
        {"training_only": False},
    ],
)
def test_unsafe_saved_workers_fail_closed_without_disclosing_values(tmp_path, change):
    payload = trainer(tmp_path).model_dump()
    payload.update(change)
    path = tmp_path / "local-workers.json"
    path.write_text(json.dumps({"version": 1, "runtimes": [payload]}))
    saved = path.read_text()
    registry = LocalWorkerRegistry(tmp_path, RuntimeCatalog())
    assert registry.merge().runtimes == []
    assert registry.issues and "private/secret" not in str(registry.issues)
    with pytest.raises(ValueError, match="could not be loaded"):
        registry.register(trainer(tmp_path))
    assert path.read_text() == saved


@pytest.mark.parametrize(
    "change",
    [
        {"device": "cpu"},
        {"provider": "gcp"},
        {"execution": "skypilot"},
        {"training_python": None},
        {"training_root": None},
        {"vendor": "/native/vendor"},
        {"build": "/native/build"},
        {"simulator_lane": "/simulator"},
        {"export_only": True},
        {"conversion_python": sys.executable},
        {"training_image": "some-image"},
    ],
)
def test_training_only_rejects_incompatible_capabilities(tmp_path, change):
    with pytest.raises(ValidationError):
        trainer(tmp_path, **change)


@pytest.mark.parametrize("operation", ["policy.export", "policy.run", "policy.quantize", None])
def test_command_rejects_other_operations_even_with_training_flag(tmp_path, operation):
    with pytest.raises(ValueError, match="fine-tuning only"):
        command(
            trainer(tmp_path),
            tmp_path / "request",
            tmp_path / "output",
            tmp_path,
            "job",
            training=True,
            operation=operation,
        )
    argv, cwd, env = command(
        trainer(tmp_path),
        tmp_path / "request",
        tmp_path / "output",
        tmp_path,
        "job",
        training=True,
        operation="policy.finetune",
    )
    assert argv[:3] == [sys.executable, "-m", "firebird_vla.application"]
    assert cwd == str(tmp_path / "private-worker")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test-1234"


def test_api_uses_registered_catalog_without_enabling_local_or_exposing_private_fields(tmp_path):
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.execution.lifecycle.local_workers.register(trainer(tmp_path))
        options = client.get("/api/v1/policy-options").json()
        runtime = options["runtimes"][0]
        assert runtime["training"] and runtime["training_only"]
        assert not any(
            runtime[key] for key in ("run", "engine_evaluation", "simulation", "enabled")
        )
        assert options["compute"]["local"]["enabled"] is False
        assert not any(
            value in json.dumps(options) for value in ("private-worker", "GPU-test-1234")
        )
        assert options["training_models"][0]["available"] is False
        client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
        assert client.get("/api/v1/policy-options").json()["training_models"][0]["available"]
        project = client.post("/api/v1/projects", json={"name": "Worker admission"}).json()["id"]
        for operation in ("policy.export", "policy.quantize", "policy.run", "policy.evaluate"):
            response = client.post(
                f"/api/v1/projects/{project}/policy-jobs",
                json={
                    "operation": operation,
                    "runtime_id": runtime["id"],
                    "artifact_id": "artifact",
                },
            )
            assert response.status_code == 422
            assert "fine-tuning only" in response.json()["detail"]
        assert client.get(f"/api/v1/projects/{project}/jobs").json() == []
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/policy-options").json()["runtimes"][0]["id"] == runtime["id"]
