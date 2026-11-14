"""Update discovery, confirmation, admission, data-preserving activation and rollback."""

import asyncio
import json
import os
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from vla_platform import __version__
from vla_platform import update_runner as runner
from vla_platform.api import create_app
from vla_platform.application_updates import ApplicationUpdates, UpdateError, installed_commit
from vla_platform.settings import Settings

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Managed updates require POSIX Supervisor"
)

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
        assert value.installed.version == __version__
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


@pytest.mark.parametrize("failure", [None, "migration", "startup", "start", "records"])
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
    snapshots = iter([{"preserved"}, {"lost"} if failure == "records" else {"preserved"}])
    monkeypatch.setattr(runner, "project_ids", lambda _: next(snapshots))
    ident = "a" * 32
    if failure:
        with pytest.raises(RuntimeError):
            runner.update(config, NEW, OLD, ident)
    else:
        runner.update(config, NEW, OLD, ident)
    backup = root / "private/application-updates" / ident / "backup"
    assert (backup / "workspace.sqlite3").read_bytes() == b"old schema fixture"
    # Only an update that completed is marked; later updates may remove its copies.
    assert (backup.parent / "completed.json").exists() is (failure is None)
    assert (backup / ".saved-config").read_bytes() == b"private fixture configuration"
    assert (root / "data/.saved-config").read_bytes() == b"private fixture configuration"
    assert (root / "data/workspace.sqlite3").read_bytes() == (
        b"old schema fixture" if failure else b"new schema fixture"
    )
    assert (root / "current").resolve() == (root / "releases/old" if failure else release)
    state = json.loads((root / "run/application-update.json").read_text())
    assert state["phase"] == (
        "rolled_back"
        if failure in {"startup", "start", "records"}
        else "failed"
        if failure
        else "complete"
    )
    assert events[-1] == "start"


GIB = 1024**3


def sized_workspace(root, monkeypatch, *, free, shared):
    """A 30 GiB workspace (one sparse file) on a disk with ``free`` GiB left.

    The update is stopped right after its disk check, so nothing is copied.
    """
    with (root / "data/recordings.bin").open("wb") as stream:
        stream.truncate(30 * GIB)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda *_: SimpleNamespace(free=free * GIB))
    monkeypatch.setattr(runner, "clones", lambda directory: shared)
    reached = []

    def checked(*_):
        reached.append("disk check passed")
        raise RuntimeError("stop here")

    monkeypatch.setattr(runner, "verify_target", checked)
    return reached


def test_copies_that_share_their_data_need_no_room_of_their_own(deployment, monkeypatch):
    root, config, _ = deployment
    # 14 GiB is the application build plus headroom; the 60 GiB two full copies would
    # take is not asked for when a copy costs nothing until it is changed.
    reached = sized_workspace(root, monkeypatch, free=15, shared=True)
    with pytest.raises(RuntimeError, match="stop here"):
        runner.update(config, NEW, OLD, "a" * 32)
    assert reached == ["disk check passed"]


def test_real_copies_still_need_room_for_two_workspaces(deployment, monkeypatch):
    root, config, _ = deployment
    reached = sized_workspace(root, monkeypatch, free=73, shared=False)
    with pytest.raises(ValueError, match="Insufficient disk space"):
        runner.update(config, NEW, OLD, "a" * 32)
    assert reached == []
    # 10 GiB headroom + 4 GiB build + 2 x 30 GiB: one more gibibyte is enough.
    reached = sized_workspace(root, monkeypatch, free=75, shared=False)
    with pytest.raises(RuntimeError, match="stop here"):
        runner.update(config, NEW, OLD, "b" * 32)
    assert reached == ["disk check passed"]
    assert (root / "data/workspace.sqlite3").read_bytes() == b"old schema fixture"


def test_even_shared_copies_keep_the_headroom_for_the_build(deployment, monkeypatch):
    root, config, _ = deployment
    reached = sized_workspace(root, monkeypatch, free=13, shared=True)
    with pytest.raises(ValueError, match="Insufficient disk space"):
        runner.update(config, NEW, OLD, "a" * 32)
    assert reached == []


def test_the_copy_is_complete_and_independent_of_the_original(tmp_path):
    source = tmp_path / "workspace"
    (source / "jobs/run").mkdir(parents=True)
    (source / "jobs/run/model.bin").write_bytes(os.urandom(300_000))
    (source / "workspace.sqlite3").write_bytes(b"projects")
    (source / "secret.json").write_text("{}")
    (source / "secret.json").chmod(0o600)
    (source / "datasets").symlink_to("/somewhere/outside/the/workspace")
    copy = tmp_path / "copy"
    runner.copy_workspace(source, copy)
    assert (copy / "jobs/run/model.bin").read_bytes() == (
        source / "jobs/run/model.bin"
    ).read_bytes()
    assert (copy / "secret.json").stat().st_mode & 0o777 == 0o600
    # A link to data kept elsewhere is copied as a link, never followed.
    assert os.readlink(copy / "datasets") == "/somewhere/outside/the/workspace"
    # Each side can change without the other noticing, shared data or not.
    (copy / "workspace.sqlite3").write_bytes(b"migrated")
    with (source / "jobs/run/model.bin").open("r+b") as stream:
        stream.write(b"changed afterwards")
    assert (source / "workspace.sqlite3").read_bytes() == b"projects"
    assert not (copy / "jobs/run/model.bin").read_bytes().startswith(b"changed afterwards")


def test_asking_whether_copies_share_data_leaves_nothing_behind(tmp_path, monkeypatch):
    assert runner.clones(tmp_path) in {True, False}
    assert list(tmp_path.iterdir()) == []
    # A platform with no way to ask is answered with the cautious no.
    monkeypatch.setattr(runner.sys, "platform", "win32")
    assert runner.clones(tmp_path) is False
    assert list(tmp_path.iterdir()) == []
    # So is a folder that cannot be written to.
    assert runner.clones(tmp_path / "missing") is False


def test_only_copies_of_earlier_completed_updates_are_removed(deployment):
    root, config, _ = deployment
    records = root / "private/application-updates"

    def kept(ident, *, completed):
        folder = records / ident
        (folder / "backup").mkdir(parents=True)
        (folder / "backup/workspace.sqlite3").write_bytes(b"an earlier workspace")
        if completed:
            runner.write_state(folder / "completed.json", {"target_commit": OLD})
        return folder

    earlier, failed, newest = (
        kept("a" * 32, completed=True),
        kept("b" * 32, completed=False),
        kept("c" * 32, completed=True),
    )
    forged = kept("d" * 32, completed=False)
    (forged / "completed.json").write_text('{"target_commit": "not a commit"}')
    other = records / "notes-kept-by-the-operator"
    other.mkdir()
    runner.write_state(other / "completed.json", {"target_commit": OLD})
    runner.prune_backups(config, newest)
    assert not earlier.exists()
    # The newest copy, a failed update's evidence and anything not named by the updater stay.
    assert (newest / "backup/workspace.sqlite3").read_bytes() == b"an earlier workspace"
    assert (failed / "backup").is_dir() and (forged / "backup").is_dir() and other.is_dir()
    assert (root / "data/workspace.sqlite3").read_bytes() == b"old schema fixture"


def test_an_ordinary_installation_keeps_its_copies_beside_the_workspace(tmp_path):
    """Started with ``jensen serve``, the copies live next to the workspace, not in the cache."""
    root, workspace = tmp_path / "cache", tmp_path / "share/openjensen"
    workspace.mkdir(parents=True)
    (workspace / "workspace.sqlite3").write_bytes(b"live projects")
    config = {"kind": "native", "root": str(root), "workspace": str(workspace)}

    def kept(ident, *, completed):
        folder = workspace.parent / (".openjensen-update-" + ident)
        (folder / "previous-data").mkdir(parents=True)
        (folder / "previous-data/workspace.sqlite3").write_bytes(b"an earlier workspace")
        record = root / "private/application-updates" / ident
        record.mkdir(parents=True)
        runner.write_state(record / "workspace-backup.json", {"directory": str(folder)})
        if completed:
            runner.write_state(folder / "completed.json", {"target_commit": OLD})
        return folder, record

    (earlier, earlier_record), (failed, failed_record), (newest, newest_record) = (
        kept("a" * 32, completed=True),
        kept("b" * 32, completed=False),
        kept("c" * 32, completed=True),
    )
    neighbour = workspace.parent / "another-application"
    neighbour.mkdir()
    runner.prune_backups(config, newest)
    assert not earlier.exists() and not earlier_record.exists()
    assert failed.is_dir() and failed_record.is_dir()
    assert newest.is_dir() and newest_record.is_dir() and neighbour.is_dir()
    assert (workspace / "workspace.sqlite3").read_bytes() == b"live projects"


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


def test_replayed_confirmation_returns_original_operation(tmp_path, deployment, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: calls.append(True))
    _, client = app_client(tmp_path, deployment, monkeypatch)
    payload = dict(target_commit=NEW, confirmation="update-and-restart", operation_id="b" * 32)
    with client:
        first = client.post("/api/v1/updates/apply", json=payload)
        second = client.post("/api/v1/updates/apply", json=payload)
        assert first.status_code == second.status_code == 202
        assert first.json()["operation"]["id"] == second.json()["operation"]["id"] == "b" * 32
        assert calls == [True]
        payload["target_commit"] = LATER
        assert client.post("/api/v1/updates/apply", json=payload).status_code == 409


def test_timed_out_build_stops_its_owned_process_group(tmp_path):
    import socket
    import sys

    if os.name != "posix":
        pytest.skip("Managed updates require POSIX")
    ready = tmp_path / "ready.json"
    code = """import json,os,signal,socket,subprocess,sys,time
child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])
sock=socket.socket();sock.bind(("127.0.0.1",0));sock.listen()
open(sys.argv[1],"w").write(json.dumps({"port":sock.getsockname()[1],"pid":os.getpid()}))
def finish(*_):
    child.wait(timeout=3);sys.exit(0)
signal.signal(signal.SIGTERM,finish)
time.sleep(60)
"""
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run([sys.executable, "-c", code, str(ready)], timeout=1)
    value = json.loads(ready.read_text())
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", value["port"])) != 0
    with pytest.raises(ProcessLookupError):
        os.kill(value["pid"], 0)


def test_pnpm_modern_package_launcher_uses_node(deployment):
    root, config, _ = deployment
    package = root / "modern-pnpm"
    (package / "bin").mkdir(parents=True)
    (package / "package.json").write_text('{"name":"pnpm"}')
    (package / "pnpm").write_text('exec ./bin/pnpm.mjs "$@"\n')
    entry = package / "bin/pnpm.mjs"
    entry.write_text("// intentionally no executable shebang\n")
    config["pnpm"] = str(package / "pnpm")
    assert runner.pnpm_command(config) == [str(root / "node/node"), str(entry)]
    config["pnpm"] = str(entry)
    assert runner.pnpm_command(config) == [str(root / "node/node"), str(entry)]


def test_pnpm_corepack_and_executables_remain_supported(deployment):
    root, config, _ = deployment
    assert runner.pnpm_command(config) == [str(root / "pnpm")]
    config["pnpm"] = str(root / "corepack")
    assert runner.pnpm_command(config) == [str(root / "corepack"), "pnpm"]


def test_prepare_release_does_not_require_a_global_pnpm(deployment, monkeypatch):
    root, config, _ = deployment
    uv, pnpm = root / "uv", root / "pnpm"
    uv.write_text("#!/bin/sh\nexit 0\n")
    pnpm.write_text(
        '#!/bin/sh\nset -eu\ncase "$*" in\n'
        "'install --frozen-lockfile --prefer-offline') exit 0;;\n"
        "'--filter @openjensen/web build') mkdir -p apps/web/out/dashboard; "
        "printf '<html></html>' > apps/web/out/dashboard/index.html;;\n"
        "*) exit 42;;\nesac\n"
    )
    uv.chmod(0o700)
    pnpm.chmod(0o700)

    def archive(_config, *args):
        output = next(arg.removeprefix("--output=") for arg in args if arg.startswith("--output="))
        with tarfile.open(output, "w"):
            pass

    monkeypatch.setattr(runner, "git", archive)
    prepared = runner.prepare_release(config, NEW, "fixture")
    assert (prepared / "apps/web/out/dashboard/index.html").is_file()
    assert json.loads((prepared / ".openjensen-build.json").read_text())["commit"] == NEW
