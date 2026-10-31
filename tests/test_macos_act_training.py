"""Native Apple GPU admission and existing-runtime adoption without installing ML packages."""

import ast
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform import local_worker_discovery as discovery
from vla_platform import local_worker_setup as setup
from vla_platform import macos_act_runtime as mac
from vla_platform.api import create_app
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog, command
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.lifecycle.training_catalog import public_training_models
from vla_platform.local_worker_registry import LocalWorkerRegistry
from vla_platform.settings import Settings


def runtime(tmp_path, **changes):
    return Runtime(
        id="managed-local-act-mps-fixture",
        label="Apple GPU · ACT",
        device="mps",
        training_only=True,
        training_python=sys.executable,
        training_root=str(tmp_path / "worker"),
        training_module="jensen_vla.lerobot_application",
        training_model_ids=["act"],
        gpu_name="Apple GPU",
        gpu_memory_mib=16384,
        **changes,
    )


def test_mps_is_closed_to_native_act_and_has_no_container_or_environment_overrides(tmp_path):
    model = runtime(tmp_path)
    for change in (
        {"training_model_ids": ["smolvla"]},
        {"training_module": "custom.application"},
        {"training_only": False},
        {"env": {"PYTHONPATH": "/untrusted"}},
        {"training_image": "untrusted"},
        {"provider": "gcp"},
    ):
        with pytest.raises(ValidationError):
            Runtime.model_validate(model.model_dump() | change)
    registry = LocalWorkerRegistry(tmp_path / "workspace", RuntimeCatalog())
    registry.register(model)
    restored = LocalWorkerRegistry(tmp_path / "workspace", RuntimeCatalog()).merge()
    assert restored.runtimes[0].device == "mps"
    assert restored.public()["runtimes"][0]["device"] == "mps"
    models = {item["id"]: item for item in public_training_models(restored)}
    assert model.id in models["act"]["runtime_ids"]
    assert model.id not in models["smolvla"]["runtime_ids"]


def test_existing_interpreter_binding_is_private_and_does_not_install(tmp_path, monkeypatch):
    python = tmp_path / "existing/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture executable")
    python.chmod(0o700)
    monkeypatch.setattr(mac, "installed_python", lambda: python)
    assert mac.configure_existing(tmp_path / "data") == python
    assert mac.binding_python(tmp_path / "data") == python
    path = tmp_path / "data/local-environments" / mac.BINDING
    assert path.stat().st_mode & 0o077 == 0
    assert json.loads(path.read_text())["revision"] == mac.REVISION
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private owned"):
        mac.binding_python(tmp_path / "data")
    path.chmod(0o600)
    path.rename(path.with_suffix(".owned"))
    path.symlink_to(path.with_suffix(".owned"))
    with pytest.raises(OSError):
        mac.binding_python(tmp_path / "data")


def test_initial_mps_recipe_envelope_is_explicit(tmp_path):
    recipe = {
        "camera_keys": ["observation.images.front"],
        "batch_size": 1,
        "prediction_horizon": 32,
        "execution_horizon": 8,
    }
    features = {"observation.images.front": {"shape": [480, 640, 3]}}
    mac.validate_training_recipe(recipe, features)
    for change in (
        {"batch_size": 2},
        {"batch_size": True},
        {"prediction_horizon": 100},
        {"prediction_horizon": "32"},
        {"execution_horizon": 33},
        {"camera_keys": ["front", "wrist"]},
        {"gradient_accumulation_steps": 4},
    ):
        with pytest.raises(ValueError):
            mac.validate_training_recipe(recipe | change, features)
    with pytest.raises(ValueError, match="640 by 480"):
        mac.validate_training_recipe(
            recipe, {"observation.images.front": {"shape": [720, 1280, 3]}}
        )


def test_macos_discovery_and_setup_reuse_fixed_existing_env_without_install(tmp_path, monkeypatch):
    root = tmp_path / "worker"
    (root / "src/jensen_vla").mkdir(parents=True)
    (root / "src/jensen_vla/lerobot_application.py").write_text("# protocol fixture")
    (root / "src/jensen_vla/application.py").write_text("# package fixture")
    (root / "src/jensen_vla/__init__.py").write_text("")
    python = tmp_path / "existing/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("# existing interpreter fixture")
    python.chmod(0o700)
    monkeypatch.setattr(mac, "installed_python", lambda: python)
    monkeypatch.setattr(discovery, "_worker_root", lambda: root)
    monkeypatch.setattr(setup, "_worker_root", lambda: root)
    monkeypatch.setattr(discovery.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(discovery.platform, "machine", lambda: "arm64")
    calls = []

    def probe(arguments, *, timeout, env):
        assert arguments[:3] == [str(python), "-I", "-B"]
        assert arguments[3].endswith("macos_act_probe.py")
        assert "HF_TOKEN" not in env and env["HF_HUB_OFFLINE"] == "1"
        calls.append(arguments)
        return json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "reason": None,
                "gpu": {
                    "name": "Apple GPU",
                    "memory_mib": 16384,
                    "uuid": "GPU-APPLE-MPS",
                    "index": 0,
                },
            }
        ).encode()

    async def no_install(*args, **kwargs):
        pytest.fail("Adoption must never install packages")

    monkeypatch.setattr(discovery, "_run_bounded", probe)
    monkeypatch.setattr(setup.LocalWorkerSetup, "_command", no_install)
    settings = Settings(data_dir=tmp_path / "workspace")
    with TestClient(create_app(settings)) as client:
        checked = client.post("/api/v1/compute-settings/local/check").json()
        assert checked["candidates"][0]["setup_supported"]
        assert checked["candidates"][0]["training_model_ids"] == ["act"]
        assert calls == []
        adopted = client.post("/api/v1/compute-settings/local/setup")
        assert adopted.status_code == 202, adopted.text
        assert adopted.json()["status"] == "succeeded"
        checked = client.post("/api/v1/compute-settings/local/check").json()
        assert checked["status"] == "ready"
        assert checked["candidates"][0]["status"] == "registered"
        assert str(python) not in json.dumps(checked)
        options = client.get("/api/v1/policy-options").json()
        worker = next(
            item for item in options["runtimes"] if item["id"] == adopted.json()["runtime_id"]
        )
        assert worker["device"] == "mps" and worker["training_model_ids"] == ["act"]
    assert len(calls) >= 2


def test_probe_source_is_python312_and_has_no_model_creation_or_download():
    from vla_platform import macos_act_probe

    path = Path(macos_act_probe.__file__)
    tree = ast.parse(path.read_text(), feature_version=(3, 12))
    assert tree
    source = path.read_text()
    assert "commit_id" in source and mac.REVISION in source
    assert ".from_pretrained(" not in source and "snapshot_download" not in source


@pytest.mark.skipif(os.name != "posix", reason="Native process group ownership")
def test_mps_native_command_uses_existing_interpreter_and_owned_cancel(tmp_path):
    worker = tmp_path / "worker"
    package = worker / "src/jensen_vla"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "lerobot_application.py").write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "request=Path(sys.argv[1]); job=json.loads(request.read_text())\n"
        "assert job['runtime']['device']=='mps'\n"
        "(request.parent/'started').write_text(str(__import__('os').getpid()))\n"
        "time.sleep(60)\nPath(sys.argv[2]).write_text('late result must not publish')\n"
    )
    model = runtime(tmp_path)
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps({"runtime": model.model_dump()}))
    argv, cwd, env = command(
        model,
        request,
        result,
        tmp_path,
        "mps-fixture",
        training=True,
        operation="policy.finetune",
    )
    assert argv[0] == sys.executable and argv[1:3] == ["-m", "jensen_vla.lerobot_application"]
    assert "docker" not in argv and "--gpus" not in argv

    async def exercise():
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(5):
                while not (tmp_path / "started").exists():
                    await asyncio.sleep(0.01)
            await Lifecycle.__new__(Lifecycle).stop(process, None, grace_seconds=1)
            assert process.returncode is not None
            assert not result.exists()
            with pytest.raises(ProcessLookupError):
                os.kill(process.pid, 0)
        finally:
            if process.returncode is None:
                os.killpg(process.pid, 9)
                await process.wait()

    asyncio.run(exercise())
