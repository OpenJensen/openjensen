"""Install lifecycle and isolation without downloading dependencies or touching USB."""

import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.datasets.local_preview import reader_python
from vla_platform.module_installations import (
    InstallOperation,
    InstallRequest,
    ModuleInstallations,
    RuntimePackage,
)
from vla_platform.settings import Settings


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    for name in ("JENSEN_MODULES", "FIREBIRD_MODULES"):
        monkeypatch.delenv(name, raising=False)


def test_add_hide_survive_restart_and_keep_configuration(tmp_path):
    config = tmp_path / "robot-lab-connection.json"
    saved_pairing = json.dumps(
        {"url": "http://127.0.0.1:8765", "token": "a" * 32, "label": "USB computer"}
    )
    config.write_text(saved_pairing)
    config.chmod(0o600)
    for enabled in (True, False):
        with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
            response = client.put("/api/v1/modules/robot-lab", json={"enabled": enabled})
            assert response.status_code == 200
        with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
            robot = next(
                item for item in client.get("/api/v1/modules").json() if item["id"] == "robot-lab"
            )
            assert robot["installed"] is enabled
            assert robot["configured"] is True
        assert config.read_text() == saved_pairing


def test_operator_override_and_invalid_inputs_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("JENSEN_MODULES", "all")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert all(item["managed"] for item in client.get("/api/v1/modules").json())
        assert client.put("/api/v1/modules/teaching", json={"enabled": False}).status_code == 409
        assert client.put("/api/v1/modules/shell", json={"enabled": True}).status_code == 422
        assert (
            client.post(
                "/api/v1/module-installations",
                json={
                    "profile": "core",
                    "operation_id": "a" * 32,
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/module-installations",
                json={
                    "profile": "robot",
                    "operation_id": "a" * 32,
                    "argv": ["shell"],
                },
            ).status_code
            == 422
        )
    assert not (tmp_path / "workspace-modules.json").exists()


def ready(ident, installed=False):
    return RuntimePackage(
        id=ident,
        name=ident,
        ready=installed,
        installable=True,
        message="fixture",
        minimum_free_gib=11,
    )


def test_install_deduplicates_and_activates_only_after_verification(tmp_path, monkeypatch):
    import vla_platform.module_installations as module

    async def run():
        service = ModuleInstallations(tmp_path)
        monkeypatch.setattr(service, "inspect", lambda ident: ready(ident))
        monkeypatch.setattr(module.shutil, "which", lambda name: "/trusted/uv")
        monkeypatch.setattr(module, "check_profile", lambda *a, **k: {"ready": True})
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def command(argv, stage):
            calls.append(argv)
            entered.set()
            await release.wait()
            (service.target("datasets") / "bin").mkdir(parents=True, exist_ok=True)
            (service.target("datasets") / "bin/python").touch()

        monkeypatch.setattr(service, "_command", command)
        payload = InstallRequest(profile="datasets", operation_id="a" * 32)
        first = await service.start(payload)
        await entered.wait()
        assert reader_python(tmp_path) != service.target("datasets") / "bin/python"
        again = await service.start(payload)
        assert first.id == again.id
        with pytest.raises(ValueError, match="current module"):
            await service.start(InstallRequest(profile="robot", operation_id="b" * 32))
        release.set()
        await service.task
        assert service.operation.status == "succeeded"
        assert reader_python(tmp_path) == service.target("datasets") / "bin/python"
        assert len(calls) == 2
        assert all(command[0] == "/trusted/uv" for command in calls)
        assert all(str(service.target("datasets")) in " ".join(command) for command in calls)
        assert ModuleInstallations(tmp_path).operation.status == "succeeded"

    asyncio.run(run())


def test_failed_verification_keeps_partial_environment_inactive(tmp_path, monkeypatch):
    import vla_platform.module_installations as module

    async def run():
        service = ModuleInstallations(tmp_path)
        monkeypatch.setattr(service, "inspect", lambda ident: ready(ident))
        monkeypatch.setattr(module.shutil, "which", lambda name: "/trusted/uv")
        monkeypatch.setattr(
            module, "check_profile", lambda *a, **k: {"ready": False, "message": "Pins mismatch"}
        )

        async def command(argv, stage):
            (service.target("robot") / "bin").mkdir(parents=True, exist_ok=True)
            (service.target("robot") / "bin/python").touch()

        monkeypatch.setattr(service, "_command", command)
        await service.start(InstallRequest(profile="robot", operation_id="a" * 32))
        await service.task
        assert service.operation.status == "failed"
        assert service.operation.message == "Pins mismatch"
        assert not (service.target("robot") / ".openjensen-installed.json").exists()
        assert not (tmp_path / "workspace-modules.json").exists()
        assert (service.target("robot") / "bin/python").exists()

    asyncio.run(run())


def test_cancel_and_restart_report_the_actual_state(tmp_path, monkeypatch):
    async def run():
        service = ModuleInstallations(tmp_path)
        monkeypatch.setattr(service, "inspect", lambda ident: ready(ident))
        entered = asyncio.Event()

        async def command(*a):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(service, "_command", command)
        await service.start(InstallRequest(profile="datasets", operation_id="a" * 32))
        await entered.wait()
        await service.cancel()
        assert not service.pending
        assert service.operation.status == "cancelled"
        saved = json.loads(service.path.read_text())
        saved["status"] = "running"
        service.path.write_text(json.dumps(saved))
        recovered = ModuleInstallations(tmp_path)
        assert recovered.operation.status == "interrupted"
        assert recovered.operation.finished_at

    asyncio.run(run())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_timeout_retires_owned_installer(tmp_path, monkeypatch):
    import vla_platform.module_installations as module

    async def run():
        service = ModuleInstallations(tmp_path)
        service.operation = InstallOperation(
            id="a" * 32,
            profile="datasets",
            status="running",
            stage="test",
            message="test",
            started_at="fixture",
        )
        service._save()
        monkeypatch.setattr(module, "TIMEOUT", 0.2)
        pidfile = tmp_path / "pid"
        with pytest.raises(TimeoutError):
            await service._command(
                [
                    sys.executable,
                    "-c",
                    "import os,time,pathlib; pathlib.Path('"
                    + str(pidfile)
                    + "').write_text(str(os.getpid())); time.sleep(30)",
                ],
                "fixture",
            )
        pid = int(pidfile.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert service.process is None

    asyncio.run(run())


def test_invalid_optional_state_does_not_take_down_core(tmp_path):
    folder = tmp_path / "module-environments"
    folder.mkdir()
    (folder / "installation.json").write_text("broken JSON")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/api/v1/health").status_code == 200
        status = client.get("/api/v1/module-installations")
        assert status.status_code == 200
        assert "administrator" in status.json()["error"]
        assert (
            client.post(
                "/api/v1/module-installations",
                json={
                    "profile": "datasets",
                    "operation_id": "a" * 32,
                },
            ).status_code
            == 409
        )
    assert (folder / "installation.json").read_text() == "broken JSON"


def test_cloud_snapshots_work_after_adding_module(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.put("/api/v1/modules/cloud-runs", json={"enabled": True}).status_code == 200
        module = next(
            item for item in client.get("/api/v1/modules").json() if item["id"] == "cloud-runs"
        )
        assert module["installed"] and module["configured"]
        response = client.get("/api/v1/cloud-runs")
        assert response.status_code == 200
        assert (tmp_path / "cloud-runs").is_dir()


def test_paired_companion_is_installed_without_local_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = tmp_path / "robot-lab-connection.json"
    config.write_text(
        json.dumps({"url": "http://127.0.0.1:8765", "token": "a" * 32, "label": "USB computer"})
    )
    config.chmod(0o600)
    service = ModuleInstallations(tmp_path)
    package = service.inspect("robot")
    assert package.ready and not package.installable
    assert package.location == "companion" and package.host == "USB computer"
    assert package.connected is None

    async def run():
        async def disconnected():
            return {"connected": False}

        async def connected():
            return {"connected": True}

        for callback, expected in ((disconnected, False), (connected, True)):
            status = await service.status(SimpleNamespace(status=callback))
            robot = next(item for item in status.profiles if item.id == "robot")
            assert robot.ready and robot.connected is expected
            assert robot.location == "companion"
        with pytest.raises(ValueError, match="paired computer"):
            await service.start(InstallRequest(profile="robot", operation_id="a" * 32))
        assert not service.target("robot").exists()

    asyncio.run(run())


def test_navigation_checks_never_contact_companion(tmp_path, monkeypatch):
    from vla_platform.robot_lab_bridge import RobotLabBridge

    config = tmp_path / "robot-lab-connection.json"
    config.write_text(json.dumps({"url": "http://127.0.0.1:8765", "token": "a" * 32}))
    config.chmod(0o600)

    async def forbidden(self):
        raise AssertionError("Navigation contacted the companion")

    monkeypatch.setattr(RobotLabBridge, "status", forbidden)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        modules = client.get("/api/v1/modules").json()
        assert next(item for item in modules if item["id"] == "robot-lab")["installed"]


def test_installed_runtime_does_not_require_installer_tool(tmp_path, monkeypatch):
    import vla_platform.module_installations as module

    service = ModuleInstallations(tmp_path, root=tmp_path / "source")
    (service.root / "workers/robot_lab").mkdir(parents=True)
    target = service.target("robot")
    target.mkdir(parents=True)
    (target / ".openjensen-installed.json").write_text("{}")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        module,
        "check_profile",
        lambda *args: {
            "ready": False,
            "supported_platform": True,
            "python_matches": True,
            "package_versions_match": True,
            "external_tools": {"uv": False},
            "message": "uv missing",
        },
    )
    package = service.inspect("robot")
    assert package.ready and not package.installable
    assert package.location == "server"


def test_runtime_tools_still_required_without_installer(tmp_path, monkeypatch):
    import vla_platform.module_installations as module

    service = ModuleInstallations(tmp_path, root=tmp_path / "source")
    (service.root / "workers/_cpu_readers").mkdir(parents=True)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        module,
        "check_profile",
        lambda *args: {
            "ready": False,
            "supported_platform": True,
            "python_matches": True,
            "package_versions_match": True,
            "external_tools": {"uv": False, "ffmpeg": False, "ffprobe": True},
            "message": "Tools missing",
        },
    )
    assert not service.inspect("datasets").ready


def test_verified_runtime_available_before_optional_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(ModuleInstallations, "inspect", lambda self, ident: ready(ident, True))
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = {item["id"]: item for item in client.get("/api/v1/modules").json()}
        assert values["robot-lab"]["available"] and values["robot-lab"]["installed"]
        assert values["decision-lab"]["available"] and values["decision-lab"]["installed"]
        assert not values["robot-lab"]["configured"] and not values["decision-lab"]["configured"]


def test_status_refreshes_cached_local_runtime_readiness(tmp_path, monkeypatch):
    import vla_platform.module_installations as module

    service = ModuleInstallations(tmp_path, root=tmp_path / "source")
    for name in ("robot_lab", "_cpu_readers", "decision"):
        (service.root / "workers" / name).mkdir(parents=True)
    current = {"ready": True}
    monkeypatch.setattr(module.shutil, "which", lambda name: "/trusted/" + name)
    monkeypatch.setattr(
        module,
        "check_profile",
        lambda *args: {
            "ready": current["ready"],
            "external_tools": {"uv": True},
            "message": "fixture",
        },
    )
    # A legacy source environment can be inspected without a managed-install receipt.
    assert service.inspect("robot").ready
    current["ready"] = False
    assert service.inspect("robot").ready
    status = asyncio.run(service.status())
    assert not next(item for item in status.profiles if item.id == "robot").ready
