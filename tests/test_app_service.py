"""Starting OpenJensen at login: the service it writes is one that Update can restart."""

import json
import os
import plistlib
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
from vla_platform import app_service
from vla_platform.application_updates import ApplicationUpdates

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Login services are macOS and Linux")
SOURCE = Path(app_service.__file__).resolve().parents[4]


class Recorder:
    def __init__(self):
        self.commands = []

    def __call__(self, args, **_):
        self.commands.append(list(args))
        return subprocess.CompletedProcess(args, 0)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def computer(tmp_path, monkeypatch):
    """A home folder, a workspace whose name needs quoting, and the update tools on PATH."""
    tools = tmp_path / "tool shims"
    tools.mkdir()
    for name in ("uv", "pnpm", "node", "git"):
        (tools / name).write_text("#!/bin/sh\n")
        (tools / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + "/usr/bin:/bin")
    workspace = tmp_path / "robot 100% $work"
    monkeypatch.setenv("JENSEN_DATA_DIR", str(workspace))
    monkeypatch.setenv("JENSEN_LOCAL_DATA_ROOT", str(tmp_path / "recordings"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    return home, workspace, tools


def test_macos_service_starts_a_server_that_can_install_updates(computer, tmp_path):
    home, workspace, tools = computer
    port = free_port()
    run = Recorder()

    result = app_service.install(port, platform="darwin", home=home, run=run, wait=0)

    path = home / "Library/LaunchAgents/com.openjensen.app.plist"
    assert result["service"] == str(path)
    assert result["workspace"] == str(workspace.resolve())
    assert result["missing_tools"] == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    service = plistlib.loads(path.read_bytes())
    assert service["Label"] == "com.openjensen.app"
    assert service["ProgramArguments"] == [
        str(Path(sys.executable).absolute()),
        "-m",
        "vla_platform.cli",
        "serve",
        "--port",
        str(port),
    ]
    assert service["RunAtLoad"] and service["KeepAlive"]
    values = service["EnvironmentVariables"]
    assert values["PATH"].split(os.pathsep)[0] == str(tools)
    assert values["JENSEN_DATA_DIR"] == str(workspace.resolve())
    assert values["JENSEN_LOCAL_DATA_ROOT"] == str((tmp_path / "recordings").resolve())
    assert values["JENSEN_KEEP_AWAKE"] == "1"
    assert run.commands == [["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)]]

    # Start exactly what launchd would start, and ask it the question the dialog asks.
    environment = dict(
        values,
        HOME=str(home),
        XDG_STATE_HOME=str(tmp_path / "state"),
        JENSEN_KEEP_AWAKE="0",
        PYTHONPATH=str(SOURCE / "packages/core/src"),
    )
    process = subprocess.Popen(
        service["ProgramArguments"],
        cwd=service["WorkingDirectory"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/v1/updates", timeout=0.5) as response:
                    status = json.load(response)
                break
            except OSError:
                if time.monotonic() > deadline or process.poll() is not None:
                    pytest.fail("The login service command did not start OpenJensen")
                time.sleep(0.1)
        assert status["automatic_update"] is True
        assert status["setup_message"] is None
        assert (workspace / "workspace.sqlite3").is_file()
    finally:
        process.terminate()
        process.wait(timeout=60)


def test_linux_unit_quotes_every_path_and_runs_the_launcher(computer, tmp_path, monkeypatch):
    home, workspace, _ = computer
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    run = Recorder()

    result = app_service.install(8123, platform="linux", home=home, run=run, wait=0)

    path = tmp_path / "config/systemd/user/openjensen.service"
    assert result["service"] == str(path)
    unit = path.read_text()
    python = str(Path(sys.executable).absolute()).replace("%", "%%").replace("$", "$$")
    assert f'ExecStart="{python}" "-m" "vla_platform.cli" "serve" "--port" "8123"' in unit
    data = str(workspace.resolve()).replace("%", "%%")
    assert f'Environment="JENSEN_DATA_DIR={data}"' in unit
    assert "WantedBy=default.target" in unit
    assert run.commands == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "openjensen.service"],
    ]


def test_reinstall_replaces_its_own_service_but_never_another_server(computer):
    home, _, _ = computer
    port = free_port()
    run = Recorder()
    app_service.install(port, platform="darwin", home=home, run=run, wait=0)
    app_service.install(port, platform="darwin", home=home, run=run, wait=0)
    assert run.commands[1] == ["launchctl", "bootout", f"gui/{os.getuid()}/com.openjensen.app"]
    assert run.commands[2][:2] == ["launchctl", "bootstrap"]

    other = home / "other"
    other.mkdir()
    run = Recorder()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        busy = server.getsockname()[1]
        with pytest.raises(ValueError, match=f"Port {busy} is in use"):
            app_service.install(busy, platform="darwin", home=other, run=run, wait=0)
    assert run.commands == []
    assert not (other / "Library/LaunchAgents/com.openjensen.app.plist").exists()


def test_remove_stops_only_this_service_and_keeps_the_workspace(computer):
    home, workspace, _ = computer
    run = Recorder()
    app_service.install(free_port(), platform="darwin", home=home, run=run, wait=0)
    run.commands.clear()

    assert app_service.remove(platform="darwin", home=home, run=run)["removed"] is True
    assert run.commands == [["launchctl", "bootout", f"gui/{os.getuid()}/com.openjensen.app"]]
    assert not (home / "Library/LaunchAgents/com.openjensen.app.plist").exists()
    assert workspace.is_dir()
    assert app_service.remove(platform="darwin", home=home, run=run)["removed"] is False


def test_a_server_started_by_hand_names_the_one_command_that_fixes_updates(monkeypatch):
    monkeypatch.delenv("JENSEN_UPDATE_CONFIG", raising=False)
    monkeypatch.delenv("JENSEN_UPDATE_UNAVAILABLE", raising=False)
    monkeypatch.delenv("JENSEN_NATIVE_LAUNCH", raising=False)
    status = ApplicationUpdates(commit="1" * 40).status()
    assert status.automatic_update is False
    assert "jensen app install" in status.setup_message
