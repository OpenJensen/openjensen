"""Restart this computer's robot control service, the process that owns the USB devices.

A restart is how the service finds cameras and arms plugged in after it started, and how
a stuck camera reader or policy process is cleared. Asked to stop, the service switches
motor torque off, closes the loaded policy and finishes an open recording, so a restart
is safe whatever the arm is doing.

Only the service installed on this computer is restarted, and only when it is the one the
saved connection reaches: a loopback address can also be an SSH tunnel to another
computer, whose service has to be restarted there.
"""

import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

LAUNCH_AGENT = "com.openjensen.robot-lab"
SYSTEMD_UNIT = "openjensen-robot-lab.service"
ELSEWHERE = (
    "This workspace reaches a robot service on another computer. "
    "Restart it on the computer with the USB cables."
)
NOT_INSTALLED = (
    "This computer has no robot connection service to restart. "
    "Restart it on the computer with the USB cables."
)
FAILED = "The robot connection service could not be restarted."


class RestartError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def run(arguments, timeout=30):
    try:
        return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RestartError(FAILED, 503) from error


def serves_port(command_line, port):
    return re.search(rf"--port[ =]{port}(?!\d)", command_line) is not None


def restart(port):
    """Restart the local service listening on ``port``; raise RestartError otherwise."""
    if sys.platform == "darwin":
        restart_launch_agent(port)
    elif sys.platform.startswith("linux"):
        restart_systemd_unit(port)
    else:
        raise RestartError(NOT_INSTALLED, 404)


def restart_launch_agent(port):
    plist = Path.home() / "Library/LaunchAgents" / f"{LAUNCH_AGENT}.plist"
    try:
        arguments = plistlib.loads(plist.read_bytes())["ProgramArguments"]
        command_line = " ".join(str(argument) for argument in arguments)
    except OSError, ValueError, KeyError, TypeError, plistlib.InvalidFileException:
        raise RestartError(NOT_INSTALLED, 404) from None
    if not serves_port(command_line, port):
        raise RestartError(ELSEWHERE)
    domain = f"gui/{os.getuid()}"
    target = f"{domain}/{LAUNCH_AGENT}"
    loaded = run(["launchctl", "print", target])
    pid = None
    if loaded.returncode == 0:
        match = re.search(r"^\s*pid = (\d+)\s*$", loaded.stdout, re.MULTILINE)
        pid = int(match.group(1)) if match else None
    listener = run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"])
    # `-Fp` prints a `p<pid>` line per process, followed by `f<fd>` lines.
    listeners = {
        int(line[1:])
        for line in listener.stdout.splitlines()
        if line.startswith("p") and line[1:].isdigit()
    }
    if listeners and listeners != {pid}:
        raise RestartError(ELSEWHERE)
    if loaded.returncode != 0 and run(["launchctl", "bootstrap", domain, str(plist)]).returncode:
        raise RestartError(FAILED, 503)
    if run(["launchctl", "kickstart", "-k", target]).returncode:
        raise RestartError(FAILED, 503)


def restart_systemd_unit(port):
    shown = run(
        ["systemctl", "--user", "show", SYSTEMD_UNIT, "--property=LoadState,ActiveState,ExecStart"]
    )
    values = dict(line.split("=", 1) for line in shown.stdout.splitlines() if "=" in line)
    if shown.returncode or values.get("LoadState") != "loaded":
        raise RestartError(NOT_INSTALLED, 404)
    if not serves_port(values.get("ExecStart", ""), port):
        raise RestartError(ELSEWHERE)
    # The service runs under `sg dialout` for USB access, which hides its process from
    # `ss -p`. Only one process can listen on the port, so a running unit configured for
    # it is the listener; a listener while the unit is down belongs to something else.
    listening = run(["ss", "-Hltn", f"sport = :{port}"]).stdout.strip()
    if listening and values.get("ActiveState") != "active":
        raise RestartError(ELSEWHERE)
    if run(["systemctl", "--user", "restart", SYSTEMD_UNIT]).returncode:
        raise RestartError(FAILED, 503)
