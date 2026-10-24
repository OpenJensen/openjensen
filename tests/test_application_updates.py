"""Update discovery, confirmation, admission, data-preserving activation and rollback."""

import asyncio
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from vla_platform import update_runner as runner
from vla_platform.api import create_app
from vla_platform.application_updates import ApplicationUpdates, UpdateError, installed_commit
from vla_platform.settings import Settings

OLD, NEW, LATER = "1" * 40, "2" * 40, "3" * 40


@pytest.fixture
def deployment(tmp_path):
    root = tmp_path / "deployment"
    for name in ("data", "run", "private", "releases/old", "repository", ".supervisor/bin", "node"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "current").symlink_to(root / "releases/old")
    (root / "releases/old/.openjensen-build.json").write_text(json.dumps({"commit": OLD}))
    for name in (".supervisor/bin/supervisorctl", "supervisord.conf", "uv", "pnpm"):
        (root / name).touch()
    (root / "data/.saved-config").write_bytes(b"private fixture configuration")
    (root / "data/workspace.sqlite3").write_bytes(b"old schema fixture")
    config = dict(
        root=str(root),
        repository=str(root / "repository"),
        supervisorctl=str(root / ".supervisor/bin/supervisorctl"),
        supervisor_config=str(root / "supervisord.conf"),
        service="firebird",
        port=8096,
        uv=str(root / "uv"),
        pnpm=str(root / "pnpm"),
        node_bin=str(root / "node"),
    )
    path = root / "private/updates.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    return root, config, path


def upstream(relation="ahead", *, target=NEW, release=True):
    def respond(request):
        if request.url.path.endswith("/commits/main"):
            return httpx.Response(200, json={"sha": target})
        if "/compare/" in request.url.path:
            return httpx.Response(404 if relation == "missing" else 200, json={"status": relation})
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(200 if release else 503, json={"tag_name": "v0.0.1"})
        raise AssertionError(str(request.url))

    return httpx.AsyncClient(transport=httpx.MockTransport(respond))


@pytest.mark.parametrize(
    "relation,state",
    [("ahead", "available"), ("behind", "ahead"), ("diverged", "diverged"), ("missing", "unknown")],
)
def test_compares_main_history_instead_of_package_numbers(deployment, relation, state):
    async def check():
        service = ApplicationUpdates(
            config_path=deployment[2], client=upstream(relation), commit=OLD
        )
        value = await service.check()
        assert value.state == state
        assert value.installed.version == "0.1.0"
        assert value.upstream_commit == NEW
        assert value.latest_release == "v0.0.1"
        assert value.automatic_update
        await service.close()

    asyncio.run(check())


def test_identical_commit_is_current_even_when_release_number_is_older(deployment):
    async def check():
        service = ApplicationUpdates(
            config_path=deployment[2], client=upstream(target=OLD), commit=OLD
        )
        assert (await service.check()).state == "current"
        await service.close()

    asyncio.run(check())


def test_release_outage_does_not_hide_main_update(deployment):
    async def check():
        service = ApplicationUpdates(
            config_path=deployment[2], client=upstream(release=False), commit=OLD
        )
        value = await service.check()
        assert value.state == "available" and value.latest_release is None
        await service.close()

    asyncio.run(check())


def test_upstream_outage_is_not_reported_as_up_to_date(deployment):
    async def check():
        def offline(request):
            raise httpx.ConnectError("fixture outage", request=request)

        service = ApplicationUpdates(
            config_path=deployment[2],
            client=httpx.AsyncClient(transport=httpx.MockTransport(offline)),
            commit=OLD,
        )
        assert (await service.check()).state == "error"
        await service.close()

    asyncio.run(check())


def test_discovery_cache_and_recorded_build_identity(tmp_path):
    calls = []

    async def check():
        def respond(request):
            calls.append(str(request.url))
            return httpx.Response(200, json={"sha": OLD, "tag_name": "v0.0.1"})

        service = ApplicationUpdates(
            client=httpx.AsyncClient(transport=httpx.MockTransport(respond)), commit=OLD
        )
        await service.check()
        count = len(calls)
        await service.check()
        assert len(calls) == count
        await service.close()

    asyncio.run(check())
    (tmp_path / ".openjensen-build.json").write_text(json.dumps({"commit": NEW}))
    assert installed_commit(tmp_path) == NEW


@pytest.mark.parametrize("invalid", ["public", "symlink"])
def test_update_configuration_is_private_and_not_a_symlink(deployment, invalid):
    root, _, path = deployment
    if invalid == "public":
        path.chmod(0o644)
    else:
        other = root / "private/actual.json"
        path.rename(other)
        path.symlink_to(other)
    service = ApplicationUpdates(config_path=path, commit=OLD)
    with pytest.raises(UpdateError):
        service.configuration()
    asyncio.run(service.close())


def app_client(tmp_path, deployment, monkeypatch, *, relation="ahead", target=NEW):
    service = ApplicationUpdates(
        config_path=deployment[2], client=upstream(relation, target=target), commit=OLD
    )
    monkeypatch.setattr("vla_platform.api.ApplicationUpdates", lambda **_kwargs: service)
    (deployment[0] / "data/workspace.sqlite3").unlink()
    app = create_app(Settings(data_dir=deployment[0] / "data"))
    return app, TestClient(app)


def test_exact_confirmation_and_no_automatic_apply(tmp_path, deployment, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    _, client = app_client(tmp_path, deployment, monkeypatch)
    with client:
        assert client.get("/api/v1/updates/check").json()["state"] == "available"
        assert not calls
        for payload in (
            {"target_commit": NEW},
            {"target_commit": NEW, "confirmation": "yes"},
            {"target_commit": NEW, "confirmation": "update-and-restart", "command": "evil"},
        ):
            assert client.post("/api/v1/updates/apply", json=payload).status_code == 422
        assert (
            client.post(
                "/api/v1/updates/apply",
                json={"target_commit": LATER, "confirmation": "update-and-restart"},
            ).status_code
            == 409
        )
        response = client.post(
            "/api/v1/updates/apply",
            json={"target_commit": NEW, "confirmation": "update-and-restart"},
        )
        assert response.status_code == 202
        assert len(calls) == 1 and calls[0][1]["start_new_session"]
        assert (
            client.post(
                "/api/v1/updates/apply",
                json={"target_commit": NEW, "confirmation": "update-and-restart"},
            ).status_code
            == 409
        )
        assert (
            client.post("/api/v1/projects", json={"name": "Not while updating"}).status_code == 503
        )
        assert client.get("/api/v1/projects").status_code == 200


@pytest.mark.parametrize("busy", ["jobs", "robot"])
def test_busy_server_cannot_start_update(tmp_path, deployment, monkeypatch, busy):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Must not spawn")
    )
    app, client = app_client(tmp_path, deployment, monkeypatch)
    with client:
        if busy == "jobs":
            app.state.execution.cloud_cleanup_pending = True
        else:
            app.state.robot_lab.config_path.touch()
            app.state.robot_lab.request = AsyncMock(
                return_value={"session": {"mode": "manual", "torque_enabled": True}}
            )
        response = client.post(
            "/api/v1/updates/apply",
            json={"target_commit": NEW, "confirmation": "update-and-restart"},
        )
        assert response.status_code == 409
        assert not app.state.application_updates.pending


def test_upstream_change_requires_new_confirmation(tmp_path, deployment, monkeypatch):
    _, client = app_client(tmp_path, deployment, monkeypatch, target=LATER)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Must not spawn")
    )
    with client:
        assert (
            client.post(
                "/api/v1/updates/apply",
                json={"target_commit": NEW, "confirmation": "update-and-restart"},
            ).status_code
            == 409
        )


@pytest.mark.parametrize("failure", [None, "migration", "startup", "start"])
def test_workspace_activation_and_full_rollback(deployment, monkeypatch, failure):
    root, config, _ = deployment
    release = root / "releases/new"
    release.mkdir()
    events = []
    monkeypatch.setattr(runner, "verify_target", lambda *_: None)
    monkeypatch.setattr(runner, "prepare_release", lambda *_: release)
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda *_: SimpleNamespace(free=1000 * 1024**3)
    )

    def control(_, action):
        events.append(action)
        if failure == "start" and action == "start" and events.count("start") == 1:
            raise RuntimeError("Start fixture failure")

    monkeypatch.setattr(runner, "service", control)

    class Ready:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b'{"idle":true}'

    monkeypatch.setattr(runner, "urlopen", lambda *_args, **_kwargs: Ready())

    def migrate(_, candidate):
        assert events == ["stop"]
        (candidate / "workspace.sqlite3").write_bytes(b"new schema fixture")
        if failure == "migration":
            raise RuntimeError("Migration fixture failure")

    monkeypatch.setattr(runner, "migrate_candidate", migrate)

    def healthy(_, commit):
        if failure == "startup" and commit == NEW:
            raise RuntimeError("Startup fixture failure")

    monkeypatch.setattr(runner, "health", healthy)
    ident = "a" * 32
    if failure:
        with pytest.raises(RuntimeError):
            runner.update(config, NEW, OLD, ident)
    else:
        runner.update(config, NEW, OLD, ident)
    backup = root / "private/application-updates" / ident / "backup"
    assert (backup / "workspace.sqlite3").read_bytes() == b"old schema fixture"
    assert (backup / ".saved-config").read_bytes() == b"private fixture configuration"
    assert (root / "data/.saved-config").read_bytes() == b"private fixture configuration"
    assert (root / "data/workspace.sqlite3").read_bytes() == (
        b"old schema fixture" if failure else b"new schema fixture"
    )
    assert (root / "current").resolve() == (root / "releases/old" if failure else release)
    state = json.loads((root / "run/application-update.json").read_text())
    assert state["phase"] == (
        "rolled_back" if failure in {"startup", "start"} else "failed" if failure else "complete"
    )
    assert events[-1] == "start"


@pytest.mark.parametrize("invalid", ["remote", "moved", "diverged"])
def test_runner_rechecks_canonical_remote_exact_tip_and_forward_ancestry(
    deployment, monkeypatch, invalid
):
    calls = []

    def git(_, *args):
        calls.append(args)
        if args[0] == "remote":
            return (
                "https://example.com/evil.git"
                if invalid == "remote"
                else "https://github.com/OpenJensen/openjensen.git"
            )
        if args[0] == "rev-parse":
            return LATER if invalid == "moved" else NEW
        if args[0] == "merge-base" and invalid == "diverged":
            raise subprocess.CalledProcessError(1, args)
        return ""

    monkeypatch.setattr(runner, "git", git)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        runner.verify_target(deployment[1], NEW, OLD)
    assert calls


def test_mismatched_workspace_cannot_enable_automatic_updates(deployment, tmp_path):
    service = ApplicationUpdates(config_path=deployment[2], commit=OLD, data_dir=tmp_path / "other")
    assert not service.status().automatic_update
    asyncio.run(service.close())


@pytest.mark.parametrize(
    "phase,expected", [("building", "failed"), ("restarting", "recovery_required")]
)
def test_interrupted_activation_stays_write_blocked(deployment, phase, expected):
    root, _, path = deployment
    runner.write_state(
        root / "run/application-update.json",
        dict(
            id="a" * 32, phase=phase, target_commit=NEW, message="Fixture", pid=999999999, created=0
        ),
    )
    service = ApplicationUpdates(config_path=path, commit=OLD)
    assert service.status().operation.phase == expected
    assert service.pending == (expected == "recovery_required")
    asyncio.run(service.close())


def test_supervisor_start_failure_can_stop_a_fatal_service_for_rollback(deployment, monkeypatch):
    def not_running(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["fixture stop"])

    monkeypatch.setattr(runner, "run", not_running)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="firebird FATAL fixture"),
    )
    runner.service(deployment[1], "stop")
