"""Restart devices: restart this computer's robot service, never one reached through a tunnel.

launchctl, lsof, systemctl and ss are replaced by small scripts on PATH that record their
arguments, so the real subprocess calls run and no service on the test machine is touched.
"""

import json
import os
import plistlib
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vla_platform import robot_lab_api, robot_service_restart
from vla_platform.robot_lab_api import router
from vla_platform.robot_lab_bridge import RobotLabBridge

TOKEN = "private-companion-token-" + "a" * 32
AGENT = robot_service_restart.LAUNCH_AGENT
UNIT = robot_service_restart.SYSTEMD_UNIT


def tool(folder, name, body):
    path = folder / name
    path.write_text("#!/bin/sh\n" + f'echo "{name} $*" >> "$TOOL_LOG"\n' + body)
    path.chmod(0o755)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    folder = tmp_path / "bin"
    folder.mkdir()
    log = tmp_path / "calls.log"
    log.write_text("")
    monkeypatch.setenv("TOOL_LOG", str(log))
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    class Tools:
        def add(self, name, body=""):
            tool(folder, name, body)

        def calls(self):
            return log.read_text().splitlines()

    return Tools()


def launch_agent(port=18101):
    agents = Path(os.environ["HOME"]) / "Library/LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    arguments = ["/runtime/bin/python", "/source/server.py", "--host", "127.0.0.1"]
    arguments += ["--port", str(port), "--data-dir", "/data/robot lab"]
    (agents / f"{AGENT}.plist").write_bytes(
        plistlib.dumps({"Label": AGENT, "ProgramArguments": arguments})
    )


def test_mac_restarts_the_launch_agent_that_serves_the_connection(tools, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    launch_agent()
    tools.add("launchctl", 'case "$1" in print) echo "	pid = 4242";; esac\n')
    tools.add("lsof", "printf 'p4242\\nf4\\n'\n")  # real `-Fp` output: process, then fd
    robot_service_restart.restart(18101)
    target = f"gui/{os.getuid()}/{AGENT}"
    assert tools.calls() == [
        f"launchctl print {target}",
        "lsof -nP -iTCP:18101 -sTCP:LISTEN -Fp",
        f"launchctl kickstart -k {target}",
    ]


def test_mac_starts_an_unloaded_agent_when_nothing_listens(tools, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    launch_agent()
    tools.add("launchctl", 'case "$1" in print) exit 113;; esac\n')
    tools.add("lsof", "exit 1\n")
    robot_service_restart.restart(18101)
    calls = tools.calls()
    assert calls[2].startswith(f"launchctl bootstrap gui/{os.getuid()} ")
    assert calls[2].endswith(f"Library/LaunchAgents/{AGENT}.plist")
    assert calls[3] == f"launchctl kickstart -k gui/{os.getuid()}/{AGENT}"


@pytest.mark.parametrize(
    "listener, agent_port", [("p9999", 18101), ("p4242", 18102)], ids=["tunnel", "other-port"]
)
def test_mac_refuses_a_connection_served_by_another_process(
    tools, monkeypatch, listener, agent_port
):
    monkeypatch.setattr(sys, "platform", "darwin")
    launch_agent(agent_port)
    tools.add("launchctl", 'case "$1" in print) echo "	pid = 4242";; esac\n')
    tools.add("lsof", f"printf '{listener}\\nf4\\n'\n")
    with pytest.raises(robot_service_restart.RestartError, match="another computer") as error:
        robot_service_restart.restart(18101)
    assert error.value.status == 409
    assert not any("kickstart" in call for call in tools.calls())


def test_mac_without_the_service_installed_says_so(tools, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    tools.add("launchctl")
    with pytest.raises(robot_service_restart.RestartError, match="no robot connection") as error:
        robot_service_restart.restart(18101)
    assert error.value.status == 404
    assert tools.calls() == []


def systemd(tools, tmp_path, *, port=18101, active="active", listener=""):
    # The installed unit wraps the service in `sg dialout -c`, as systemd shows it.
    command = (
        "/usr/bin/sg dialout -c /opt/robot/.venv/bin/python /opt/robot/source/server.py "
        f"--host 127.0.0.1 --port {port} --token-file /opt/robot/private/token "
        "--data-dir /opt/robot/data"
    )
    (tmp_path / "show.txt").write_text(
        f"LoadState=loaded\nActiveState={active}\n"
        f"ExecStart={{ path=/usr/bin/sg ; argv[]={command} ; ignore_errors=no ; pid=1306735 }}\n"
    )
    (tmp_path / "ss.txt").write_text(listener)
    tools.add("systemctl", f'case "$2" in show) cat "{tmp_path}/show.txt";; esac\n')
    tools.add("ss", f'cat "{tmp_path}/ss.txt"\n')


# `ss -p` cannot see a process started through `sg`, so the socket line carries no owner.
LISTENING = "LISTEN 0      5      127.0.0.1:18101 0.0.0.0:*\n"


def test_linux_restarts_the_running_user_unit_that_serves_the_port(tools, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    systemd(tools, tmp_path, listener=LISTENING)
    robot_service_restart.restart(18101)
    assert tools.calls() == [
        f"systemctl --user show {UNIT} --property=LoadState,ActiveState,ExecStart",
        "ss -Hltn sport = :18101",
        f"systemctl --user restart {UNIT}",
    ]


def test_linux_starts_a_stopped_unit_when_nothing_listens(tools, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    systemd(tools, tmp_path, active="failed")
    robot_service_restart.restart(18101)
    assert tools.calls()[-1] == f"systemctl --user restart {UNIT}"


@pytest.mark.parametrize(
    "active, port", [("inactive", 18101), ("active", 18102)], ids=["tunnel", "other-port"]
)
def test_linux_refuses_a_port_the_unit_does_not_serve(tools, tmp_path, monkeypatch, active, port):
    monkeypatch.setattr(sys, "platform", "linux")
    systemd(tools, tmp_path, port=port, active=active, listener=LISTENING)
    with pytest.raises(robot_service_restart.RestartError, match="another computer"):
        robot_service_restart.restart(18101)
    assert not any("--user restart" in call for call in tools.calls())


def test_linux_without_the_unit_says_so(tools, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    tools.add("systemctl", 'echo "LoadState=not-found"\n')
    with pytest.raises(robot_service_restart.RestartError, match="no robot connection") as error:
        robot_service_restart.restart(18101)
    assert error.value.status == 404


def test_port_match_is_exact():
    assert robot_service_restart.serves_port("server.py --port 18101 --data-dir x", 18101)
    assert robot_service_restart.serves_port("server.py --port=18101", 18101)
    assert not robot_service_restart.serves_port("server.py --port 181010", 18101)
    assert not robot_service_restart.serves_port("server.py --port 18102", 18101)


class Companion:
    """A robot service whose boot id changes when the fake restart runs."""

    def __init__(self):
        self.boot = "first-boot"
        self.down = False

    def handler(self, request):
        if self.down:
            # The old process has exited; the next request reaches the new one.
            self.down, self.boot = False, "second-boot"
            raise httpx.ConnectError("restarting", request=request)
        assert request.headers["authorization"] == "Bearer " + TOKEN
        return httpx.Response(
            200,
            json={"ok": True, "session": {"mode": "idle"}, "diagnostics": {"boot_id": self.boot}},
        )


def browser(tmp_path, companion):
    connection = tmp_path / "robot-lab-connection.json"
    connection.write_text(json.dumps({"url": "http://127.0.0.1:18101", "token": TOKEN}))
    connection.chmod(0o600)
    app = FastAPI()
    app.include_router(router)
    app.state.robot_lab = RobotLabBridge(
        tmp_path, client=httpx.AsyncClient(transport=httpx.MockTransport(companion.handler))
    )
    return TestClient(app)


def test_restart_answers_with_the_new_service_status(tmp_path, monkeypatch):
    companion = Companion()
    ports = []

    def restart(port):
        ports.append(port)
        companion.down = True

    monkeypatch.setattr(robot_service_restart, "restart", restart)
    response = browser(tmp_path, companion).post("/api/v1/robot-lab/service/restart")
    assert response.status_code == 200
    assert response.json()["diagnostics"]["boot_id"] == "second-boot"
    assert response.json()["connected"] is True
    assert response.headers["cache-control"] == "no-store"
    assert ports == [18101]


def test_restart_waits_for_a_new_process_not_the_old_one_still_answering(tmp_path, monkeypatch):
    companion = Companion()
    monkeypatch.setattr(robot_service_restart, "restart", lambda port: None)
    monkeypatch.setattr(robot_lab_api, "RESTART_SECONDS", 0)
    response = browser(tmp_path, companion).post("/api/v1/robot-lab/service/restart")
    assert response.status_code == 503
    assert "did not come back" in response.json()["detail"]


def test_restart_refusal_reaches_the_page(tmp_path, monkeypatch):
    def refuse(port):
        raise robot_service_restart.RestartError(robot_service_restart.ELSEWHERE)

    monkeypatch.setattr(robot_service_restart, "restart", refuse)
    response = browser(tmp_path, Companion()).post("/api/v1/robot-lab/service/restart")
    assert response.status_code == 409
    assert response.json()["detail"] == robot_service_restart.ELSEWHERE


def test_restart_without_a_connection_explains_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_service_restart, "restart", lambda port: pytest.fail("restarted"))
    app = FastAPI()
    app.include_router(router)
    app.state.robot_lab = RobotLabBridge(tmp_path)
    response = TestClient(app).post("/api/v1/robot-lab/service/restart")
    assert response.status_code == 503
    assert "Connect the local Robot Lab companion" in response.json()["detail"]
