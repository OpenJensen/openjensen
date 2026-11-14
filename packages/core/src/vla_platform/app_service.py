"""Start OpenJensen when this user logs in, through the launcher that owns updates.

A server that a service file starts directly (a web server command, a copied
release folder) cannot replace its own code, so Update has nothing to restart.
This writes the one service definition each platform provides for a user
application: a LaunchAgent on macOS and a systemd user unit on Linux. Both run
``jensen serve``, which keeps running while it swaps the application in and out.
"""

import json
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from vla_platform.settings import Settings

LABEL = "com.openjensen.app"
UNIT = "openjensen.service"
ROOT = Path(__file__).resolve().parents[4]
# Settings the launcher passes to every application it starts; see native_launch.serve.
PATHS = (
    ("local_root", "JENSEN_LOCAL_DATA_ROOT"),
    ("runtime_config", "JENSEN_RUNTIME_CONFIG"),
    ("cloud_runs_dir", "JENSEN_CLOUD_RUNS_DIR"),
    ("simulation_config", "JENSEN_SIMULATION_CONFIG"),
    ("recording_config", "JENSEN_RECORDING_CONFIG"),
    ("teaching_session_config", "JENSEN_TEACHING_SESSION_CONFIG"),
)
# Update builds the new version with these; a login service starts with an empty PATH.
TOOLS = (("uv", "uv"), ("pnpm", "pnpm"), ("corepack", "pnpm"), ("node", "Node.js"), ("git", "git"))


def service_file(platform=sys.platform, home=None):
    home = Path(home or Path.home())
    if platform == "darwin":
        return home / "Library/LaunchAgents" / (LABEL + ".plist")
    if platform.startswith("linux"):
        config = os.environ.get("XDG_CONFIG_HOME")
        return (Path(config) if config else home / ".config") / "systemd/user" / UNIT
    raise ValueError("Starting OpenJensen at login is available on macOS and Linux.")


def search_path(platform=sys.platform):
    """The directories of the update tools first, then the system's own."""
    found = []
    for name, _ in TOOLS:
        if location := shutil.which(name):
            # Keep shims and Homebrew links: the file they point to moves on upgrade.
            found.append(str(Path(location).absolute().parent))
    system = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    if platform == "darwin":
        system.insert(0, "/opt/homebrew/bin")
    return os.pathsep.join(dict.fromkeys([*found, *system]))


def missing_tools():
    names = {label for name, label in TOOLS if shutil.which(name)}
    return [label for label in ("uv", "pnpm", "Node.js", "git") if label not in names]


def environment(settings, platform=sys.platform):
    values = {
        "PATH": search_path(platform),
        "JENSEN_DATA_DIR": str(settings.data_dir),
        "PYTHONUNBUFFERED": "1",
    }
    for field, name in PATHS:
        if value := getattr(settings, field):
            values[name] = str(Path(value).absolute())
    if platform == "darwin":
        # Training downloads and robot runs stop when an idle Mac sleeps.
        values["JENSEN_KEEP_AWAKE"] = "1"
    return values


def command(port, python=None):
    # The environment's own interpreter: resolving its link would leave the environment.
    python = str(Path(python or sys.executable).absolute())
    return [python, "-m", "vla_platform.cli", "serve", "--port", str(port)]


def launch_agent(port, values, *, python=None, source=ROOT, home=None):
    logs = Path(home or Path.home()) / "Library/Logs/OpenJensen"
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": command(port, python),
            "WorkingDirectory": str(source),
            "EnvironmentVariables": values,
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": 10,
            # The launcher stops the application and its jobs before it exits.
            "ExitTimeOut": 60,
            "StandardOutPath": str(logs / "app.log"),
            "StandardErrorPath": str(logs / "app.log"),
        }
    )


def unit_value(value, *, command=False):
    value = str(value)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Service paths must not contain control characters.")
    value = value.replace("%", "%%")
    return json.dumps(value.replace("$", "$$") if command else value, ensure_ascii=False)


def user_unit(port, values, *, python=None, source=ROOT):
    working = str(source).replace("%", "%%")
    lines = [
        "[Unit]",
        "Description=OpenJensen",
        "After=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        "ExecStart=" + " ".join(unit_value(part, command=True) for part in command(port, python)),
        "WorkingDirectory=" + working,
        *("Environment=" + unit_value(key + "=" + value) for key, value in values.items()),
        "Restart=on-failure",
        "RestartSec=5",
        # The launcher stops the application and its jobs before it exits.
        "TimeoutStopSec=60",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ]
    return "\n".join(lines).encode()


def write_private(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(body)
    os.replace(temporary, path)


def port_free(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def wait_port_free(port, timeout):
    deadline = time.monotonic() + timeout
    while not port_free(port):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.5)
    return True


def stop(platform, run):
    """Stop this application's own service. Nothing else is touched."""
    if platform == "darwin":
        domain = "gui/" + str(os.getuid())
        run(["launchctl", "bootout", domain + "/" + LABEL], capture_output=True)
    else:
        run(["systemctl", "--user", "disable", "--now", UNIT], capture_output=True)


def start(platform, path, run):
    if platform == "darwin":
        run(["launchctl", "bootstrap", "gui/" + str(os.getuid()), str(path)], check=True)
    else:
        run(["systemctl", "--user", "daemon-reload"], check=True)
        run(["systemctl", "--user", "enable", "--now", UNIT], check=True)


def read_status(port, timeout):
    """What the started application says about updates, or None if it is still starting."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/v1/updates", timeout=3) as response:
                return json.load(response)
        except OSError, ValueError:
            if time.monotonic() > deadline:
                return None
            time.sleep(1)


def install(port, *, platform=sys.platform, home=None, run=subprocess.run, wait=120):
    path = service_file(platform, home)
    settings = Settings.from_env()
    values = environment(settings, platform)
    body = (
        launch_agent(port, values, home=home) if platform == "darwin" else user_unit(port, values)
    )
    # Replace an earlier copy of this service; a different server on the port stays.
    if path.exists():
        stop(platform, run)
        wait_port_free(port, 70)
    if not port_free(port):
        raise ValueError(
            f"Port {port} is in use by another server. Stop it first (for example the "
            "terminal running jensen serve), or choose another port with --port."
        )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if platform == "darwin":
        (Path(home or Path.home()) / "Library/Logs/OpenJensen").mkdir(parents=True, exist_ok=True)
    write_private(path, body)
    start(platform, path, run)
    status = read_status(port, wait) if wait else None
    return {
        "url": f"http://127.0.0.1:{port}",
        "service": str(path),
        "workspace": str(settings.data_dir),
        "started": status is not None,
        "updates": None if status is None else bool(status.get("automatic_update")),
        "updates_message": None if status is None else status.get("setup_message"),
        "missing_tools": missing_tools(),
    }


def remove(*, platform=sys.platform, home=None, run=subprocess.run):
    path = service_file(platform, home)
    if not path.exists():
        return {"removed": False, "service": str(path)}
    stop(platform, run)
    path.unlink()
    if platform != "darwin":
        run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    return {"removed": True, "service": str(path)}
