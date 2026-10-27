"""Ordinary-install update launcher, real process replacement and rollback."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
from vla_platform import native_launch, update_runner
from vla_platform.application_updates import ApplicationUpdates

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Native launcher requires POSIX")
OLD, NEW = "1" * 40, "2" * 40

CHILD = """
import argparse,json,os
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--child",action="store_true")
parser.add_argument("--port",type=int)
args=parser.parse_args()
source=Path.cwd()
if (source/"fail-start").exists(): raise SystemExit(23)
commit=json.loads((source/".openjensen-build.json").read_text())["commit"]
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_): pass
    def do_GET(self):
        values={
            "/api/v1/updates":{"installed":{"commit":commit}},
            "/api/v1/projects":[{"id":"retained-project"}],
            "/api/v1/updates/readiness":{"idle":True},
        }
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(values[self.path]).encode())
class Server(HTTPServer): allow_reuse_address=True
Server(("127.0.0.1",args.port),Handler).serve_forever()
"""


def wait_commit(port, commit, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/v1/updates", timeout=0.2) as response:
                if json.load(response)["installed"]["commit"] == commit:
                    return
        except OSError, ValueError:
            pass
        time.sleep(0.05)
    raise RuntimeError("Fixture application did not start")


def source_at(path, commit):
    package = path / "packages/core/src/vla_platform"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "native_launch.py").write_text(CHILD)
    (path / ".openjensen-build.json").write_text(json.dumps({"commit": commit}))
    (path / ".venv/bin").mkdir(parents=True)
    (path / ".venv/bin/python").symlink_to(sys.executable)
    return path


@pytest.fixture
def ordinary_install(tmp_path):
    source = source_at(tmp_path / "source archive", OLD)
    workspace = tmp_path / "saved workspace"
    workspace.mkdir()
    (workspace / "workspace.sqlite3").write_bytes(b"old schema")
    (workspace / ".connection-config").write_text("preserved private settings")
    root = tmp_path / "private updates"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code = (
        "from pathlib import Path; from vla_platform.native_launch import Launcher; "
        "import sys; raise SystemExit(Launcher(Path(sys.argv[1]),Path(sys.argv[2]),"
        "int(sys.argv[3]),root=Path(sys.argv[4])).run())"
    )

    def start():
        return subprocess.Popen(
            [sys.executable, "-c", code, str(source), str(workspace), str(port), str(root)],
            env=dict(os.environ, PYTHONPATH=str(Path(native_launch.__file__).parents[1])),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    process = start()
    try:
        wait_commit(port, OLD)
        yield source, workspace, root, port, process, start
    finally:
        process.terminate()
        process.wait(timeout=8)
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize("fails", [False, True])
def test_source_archive_updates_and_recovers_without_supervisor_or_source_changes(
    ordinary_install, monkeypatch, fails
):
    source, workspace, root, port, process, start = ordinary_install
    config = update_runner.load_config(root / "run/native-launch.json")
    assert config["workspace"] == str(workspace)
    assert config["launcher_pid"] == process.pid
    target = source_at(root / "releases/new", NEW)
    if fails:
        (target / "fail-start").touch()
    monkeypatch.setattr(update_runner, "verify_target", lambda *_: None)
    monkeypatch.setattr(update_runner, "update_tools", lambda *_: None)
    monkeypatch.setattr(update_runner, "prepare_release", lambda *_: target)
    monkeypatch.setattr(
        update_runner.shutil, "disk_usage", lambda *_: type("Disk", (), {"free": 1000 * 1024**3})()
    )
    monkeypatch.setattr(
        update_runner, "health", lambda cfg, commit: wait_commit(cfg["port"], commit, timeout=1)
    )

    def migrate(_release, candidate):
        (candidate / "workspace.sqlite3").write_bytes(b"new schema")

    monkeypatch.setattr(update_runner, "migrate_candidate", migrate)
    operation = "a" * 32
    if fails:
        with pytest.raises(RuntimeError):
            update_runner.update(config, NEW, OLD, operation)
    else:
        update_runner.update(config, NEW, OLD, operation)
    wait_commit(port, OLD if fails else NEW)
    assert process.poll() is None
    assert (workspace / ".connection-config").read_text() == "preserved private settings"
    assert (workspace / "workspace.sqlite3").read_bytes() == (
        b"old schema" if fails else b"new schema"
    )
    assert json.loads((source / ".openjensen-build.json").read_text())["commit"] == OLD
    snapshots = workspace.parent / (".openjensen-update-" + operation)
    assert (snapshots / "backup/workspace.sqlite3").read_bytes() == b"old schema"
    assert json.loads(
        (root / "private/application-updates" / operation / "workspace-backup.json").read_text()
    )["directory"] == str(snapshots)
    assert json.loads((root / "run/application-update.json").read_text())["phase"] == (
        "rolled_back" if fails else "complete"
    )
    if not fails:
        process.terminate()
        process.wait(timeout=8)
        restarted = start()
        try:
            wait_commit(port, NEW)
            assert restarted.poll() is None
        finally:
            restarted.terminate()
            restarted.wait(timeout=8)
        # An explicit source edit takes precedence over the cached application update.
        original_module = source / "packages/core/src/vla_platform/native_launch.py"
        original_module.write_text(original_module.read_text() + "\n# local customization\n")
        customized = start()
        try:
            wait_commit(port, OLD)
            assert customized.poll() is None
        finally:
            customized.terminate()
            customized.wait(timeout=8)


def test_native_launch_configuration_matches_running_process_and_workspace(
    ordinary_install, monkeypatch
):
    source, workspace, root, _, process, _ = ordinary_install
    path = root / "run/native-launch.json"
    monkeypatch.setattr(os, "getppid", lambda: process.pid)
    config = native_launch.configuration(path, workspace, source)
    assert config["kind"] == "native"
    with pytest.raises(ValueError, match="different"):
        native_launch.configuration(path, workspace / "other", source)
    with pytest.raises(ValueError, match="different"):
        native_launch.configuration(path, workspace, source / "other")
    monkeypatch.setattr(os, "getppid", lambda: process.pid + 1)
    with pytest.raises(ValueError, match="no longer"):
        native_launch.configuration(path, workspace, source)


def test_source_archive_can_install_canonical_main_without_an_old_commit(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    calls = []

    def git(_config, *args):
        calls.append(args)
        if args[0] == "remote":
            return "https://github.com/OpenJensen/openjensen.git"
        if args[0] == "rev-parse":
            return NEW

    monkeypatch.setattr(update_runner, "git", git)
    update_runner.verify_target({"kind": "native", "repository": str(repository)}, NEW, "")
    assert not any(args[0] == "merge-base" for args in calls)
    with pytest.raises(ValueError, match="Invalid commit"):
        update_runner.verify_target({"repository": str(repository)}, NEW, "")


def test_private_native_state_and_no_recursive_workspace_backups(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    assert root not in native_launch.state_root(root).parents
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private"):
        native_launch.private_directory(public)
    path = tmp_path / "launch.json"
    path.write_text("{}")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        native_launch.read_private(path)
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        native_launch.read_private(link)


def test_archive_discovery_and_apply_use_the_automatic_launcher(ordinary_install, monkeypatch):
    import asyncio

    import httpx
    from vla_platform.application_updates import UpdateRequest

    source, workspace, root, _, process, _ = ordinary_install
    path = root / "run/native-launch.json"
    monkeypatch.setenv(native_launch.CONTEXT, str(path))
    monkeypatch.delenv("JENSEN_UPDATE_CONFIG", raising=False)
    monkeypatch.setattr(os, "getppid", lambda: process.pid)
    monkeypatch.setattr("vla_platform.application_updates.installed_commit", lambda: None)
    configure = native_launch.configuration
    monkeypatch.setattr(native_launch, "configuration", lambda p, w: configure(p, w, source))
    config = json.loads(path.read_text())
    config.update(uv=sys.executable, pnpm=sys.executable, node_bin=str(Path(sys.executable).parent))
    update_runner.write_state(path, config)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))

    def respond(request):
        if request.url.path.endswith("/commits/main"):
            return httpx.Response(200, json={"sha": NEW})
        return httpx.Response(200, json={"tag_name": "v0.1.4"})

    async def check():
        service = ApplicationUpdates(
            data_dir=workspace, client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        assert service.config_path is None
        status = await service.check()
        assert status.automatic_update and status.state == "available"
        assert status.setup_message is None
        await service.start(UpdateRequest(target_commit=NEW, confirmation="update-and-restart"))
        await service.close()

    asyncio.run(check())
    assert len(calls) == 1
    args = calls[0][0][0]
    assert args[args.index("--config") + 1] == str(path)
    assert args[args.index("--installed") + 1] == ""


@pytest.mark.parametrize("writable_cache", [True, False])
def test_normal_serve_exposes_updates_without_operator_setup(tmp_path, writable_cache):
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    state = tmp_path / "state"
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"JENSEN_UPDATE_CONFIG", native_launch.CONTEXT, "JENSEN_UPDATE_UNAVAILABLE"}
    }
    environment.update(
        JENSEN_DATA_DIR=str(workspace),
        XDG_STATE_HOME=str(state),
        PYTHONPATH=str(Path(native_launch.__file__).parents[1]),
    )
    if not writable_cache:
        import hashlib

        cache = (
            state
            / "openjensen/application-updates"
            / hashlib.sha256(os.fsencode(workspace)).hexdigest()[:24]
        )
        cache.mkdir(mode=0o755, parents=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "vla_platform.cli", "serve", "--port", str(port)],
        cwd=Path(native_launch.__file__).parents[4],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 12
        while True:
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/v1/updates", timeout=0.5) as response:
                    status = json.load(response)
                break
            except OSError:
                if time.monotonic() > deadline or process.poll() is not None:
                    pytest.fail("The normal application launcher did not start")
                time.sleep(0.1)
        assert status["automatic_update"] is writable_cache
        if writable_cache:
            assert status["setup_message"] is None
        else:
            assert "cannot write their private cache" in status["setup_message"]
        assert (workspace / "workspace.sqlite3").is_file()
    finally:
        process.terminate()
        process.wait(timeout=12)
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0
