#!/usr/bin/env python3
"""Install the private local SO-101 companion without changing OpenJensen's ML workers."""

import argparse
import json
import os
import plistlib
import secrets
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = [
    "numpy==2.5.4",
    "opencv-python-headless==5.0.0.93",
    "pillow==13.0.0",
    "pyserial==3.5",
    "feetech-servo-sdk==1.0.0",
]
SOURCE_MODULES = (
    "dispatch.py",
    "server.py",
    "hardware.py",
    "lerobot_control.py",
    "motion.py",
    "camera.py",
    "calibration_draft.py",
    "dataset.py",
    "assembled.py",
    "identify.py",
    "verification.py",
    "assembled_session.py",
)
SUPPORT_FILES = ("LEROBOT-LICENSE", "LEROBOT-PROVENANCE.json")
VENDOR_FILES = (
    "__init__.py",
    "bus.py",
    "config.py",
    "framing.py",
    "registers.py",
    "LICENSE",
    "PROVENANCE.json",
)
SERVICE_LABELS = ("com.openjensen.robot-lab", "com.openjensen.robot-lab.tunnel")


def read_private_token(path):
    """Never follow a link or alter an unrelated existing credential file."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > 4096
        ):
            raise ValueError("The existing token must be an owned private regular file.")
        token = stream.read().decode("ascii").strip()
    if not 32 <= len(token) <= 512 or any(c.isspace() or not c.isprintable() for c in token):
        raise ValueError("The existing private token has an invalid format.")
    return token


def stop_launch_agents():
    """Stop only our LaunchAgents before replacing their loaded code or runtime."""
    domain = "gui/" + str(os.getuid())
    for label in SERVICE_LABELS:
        identity = domain + "/" + label
        result = subprocess.run(["launchctl", "print", identity], capture_output=True)
        if result.returncode == 0:
            subprocess.run(["launchctl", "bootout", identity], capture_output=True, check=True)


LINUX_SERVICES = ("openjensen-robot-lab.service", "openjensen-robot-lab-tunnel.service")


def linux_units_dir():
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd/user"


def default_data_dir():
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/OpenJensen/robot-lab"
    return (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "OpenJensen/robot-lab"
    )


def stop_linux_services(no_start):
    # A first --no-start install can prepare units on a host without a user bus.
    # Reinstallation must be able to stop the existing managed services first.
    systemctl = shutil.which("systemctl")
    if systemctl:
        result = subprocess.run([systemctl, "--user", "show-environment"], capture_output=True)
        if result.returncode == 0:
            for unit in LINUX_SERVICES:
                loaded = subprocess.run(
                    [systemctl, "--user", "show", "--property=LoadState", "--value", unit],
                    capture_output=True,
                    text=True,
                )
                if loaded.stdout.strip() == "not-found":
                    continue
                if loaded.returncode:
                    raise ValueError(
                        f"Cannot inspect managed service {unit}; check systemctl --user."
                    )
                subprocess.run([systemctl, "--user", "stop", unit], check=True)
            return systemctl
    if no_start and not any((linux_units_dir() / unit).exists() for unit in LINUX_SERVICES):
        return None
    raise ValueError(
        "A running systemd user manager is required. Log in with a user session, "
        "or use --no-start for a first install and run the companion manually."
    )


def unit_quote(value, *, command=False):
    value = str(value)
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Service paths must not contain control characters")
    value = value.replace("%", "%%")
    if command:
        return json.dumps(value.replace("$", "$$"), ensure_ascii=False)
    # WorkingDirectory= consumes a whole scalar path, not an ExecStart argument.
    # Quotes here become literal filename characters on supported systemd versions.
    return value


def install_linux_services(python, source, token_file, data, logs, ssh_host, no_start, systemctl):
    units = linux_units_dir()
    units.mkdir(parents=True, exist_ok=True)
    companion = [
        str(python),
        str(source / "server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        "18101",
        "--token-file",
        str(token_file),
        "--data-dir",
        str(data),
    ]
    services = [(LINUX_SERVICES[0], companion)]
    if ssh_host:
        services.append(
            (
                LINUX_SERVICES[1],
                [
                    "/usr/bin/ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                    "-N",
                    "-R",
                    "127.0.0.1:18101:127.0.0.1:18101",
                    "-L",
                    "127.0.0.1:18096:127.0.0.1:8096",
                    ssh_host,
                ],
            )
        )
    elif (units / LINUX_SERVICES[1]).exists():
        # A removed SSH option must not resurrect an old tunnel after login.
        subprocess.run([systemctl, "--user", "disable", LINUX_SERVICES[1]], check=True)
        (units / LINUX_SERVICES[1]).unlink()
    for name, arguments in services:
        content = "\n".join(
            [
                "[Unit]",
                "Description=OpenJensen Robot Lab"
                + (" SSH bridge" if name == LINUX_SERVICES[1] else " companion"),
                "StartLimitIntervalSec=0",
                "",
                "[Service]",
                "Type=exec",
                "ExecStart=" + " ".join(unit_quote(arg, command=True) for arg in arguments),
                "WorkingDirectory=" + unit_quote(source),
                "Restart=on-failure",
                "RestartSec=10",
                "TimeoutStopSec=20",
                "UMask=0077",
                "Environment=PYTHONUNBUFFERED=1",
                'Environment="PATH=/usr/local/bin:/usr/bin:/bin"',
                "StandardOutput=journal",
                "StandardError=journal",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
        target = units / name
        if target.is_symlink():
            raise ValueError("Managed unit files must not be symbolic links")
        target.write_text(content)
        target.chmod(0o600)
    if systemctl:
        subprocess.run([systemctl, "--user", "daemon-reload"], check=True)
        if not no_start:
            subprocess.run(
                [systemctl, "--user", "enable", "--now", *[name for name, _ in services]],
                check=True,
            )
    return [name for name, _ in services]


def install_mac_services(python, source, token_file, data, logs, ssh_host, no_start):
    agents = Path.home() / "Library/LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    service = {
        "Label": "com.openjensen.robot-lab",
        "ProgramArguments": [
            str(python),
            str(source / "server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            "18101",
            "--token-file",
            str(token_file),
            "--data-dir",
            str(data),
        ],
        "WorkingDirectory": str(source),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED": "1",
        },
        "StandardOutPath": str(logs / "companion.log"),
        "StandardErrorPath": str(logs / "companion-error.log"),
    }
    services = [service]
    if ssh_host:
        services.append(
            {
                "Label": "com.openjensen.robot-lab.tunnel",
                "ProgramArguments": [
                    "/usr/bin/ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                    "-N",
                    "-R",
                    "127.0.0.1:18101:127.0.0.1:18101",
                    "-L",
                    "127.0.0.1:18096:127.0.0.1:8096",
                    ssh_host,
                ],
                "RunAtLoad": True,
                "KeepAlive": True,
                "ThrottleInterval": 10,
                "StandardOutPath": str(logs / "tunnel.log"),
                "StandardErrorPath": str(logs / "tunnel-error.log"),
            }
        )
    for service in services:
        target = agents / (service["Label"] + ".plist")
        with target.open("wb") as stream:
            plistlib.dump(service, stream)
        target.chmod(0o600)
        if not no_start:
            domain = "gui/" + str(os.getuid())
            subprocess.run(["launchctl", "bootout", domain, str(target)], capture_output=True)
            subprocess.run(["launchctl", "bootstrap", domain, str(target)], check=True)
    return [service["Label"] for service in services]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="workers/robot_lab directory")
    parser.add_argument(
        "--prefix", type=Path, default=Path.home() / ".local/share/jensen/robot-lab"
    )
    parser.add_argument("--python", default="3.14.7")
    parser.add_argument("--runtime", type=Path, help="Reuse a Robot Lab profile environment")
    parser.add_argument(
        "--offline", action="store_true", help="Reuse cached wheels; never download"
    )
    parser.add_argument(
        "--arrow-site-packages", type=Path, help="Reuse an existing compatible PyArrow"
    )
    parser.add_argument(
        "--ssh-host", help="Private Xbox SSH alias for automatic local/reverse tunnel"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Private captures and recovery data directory"
    )
    parser.add_argument(
        "--no-start", action="store_true", help="Prepare services without starting them"
    )
    args = parser.parse_args()
    if sys.platform not in {"darwin", "linux"}:
        parser.error("Robot Lab supports macOS and Linux.")
    disk_parent = args.prefix.expanduser().resolve()
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    if shutil.disk_usage(disk_parent).free < 10 * 1024**3:
        parser.error(
            "Keep at least 10 GiB free space before installing, including offline installs."
        )
    required = [
        *(Path(name) for name in (*SOURCE_MODULES, *SUPPORT_FILES)),
        *(Path("_vendor/servotools") / name for name in VENDOR_FILES),
    ]
    if (
        any(
            not (args.source / name).is_file() or (args.source / name).is_symlink()
            for name in required
        )
        or (args.source / "_vendor").is_symlink()
        or (args.source / "_vendor/servotools").is_symlink()
    ):
        parser.error(
            "--source must contain all companion modules, LeRobot license/provenance "
            "and licensed _vendor/servotools source"
        )
    if args.ssh_host and (
        args.ssh_host.startswith("-")
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._@"
            for c in args.ssh_host
        )
    ):
        parser.error("--ssh-host must be a plain SSH host alias")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("Install uv from https://docs.astral.sh/uv/ first.")
    prefix = args.prefix.expanduser().resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    private = prefix / "private"
    if private.is_symlink():
        parser.error("The private directory must not be a symlink.")
    private.mkdir(mode=0o700, exist_ok=True)
    if private.stat().st_uid != os.getuid():
        parser.error("The private directory must be owned by the current user.")
    private.chmod(0o700)
    token_file = private / "companion-token"
    if token_file.exists() or token_file.is_symlink():
        try:
            read_private_token(token_file)
        except (OSError, ValueError, UnicodeDecodeError) as error:
            parser.error(
                "The existing token must be an owned private regular file with a valid token "
                f"({type(error).__name__})."
            )
    else:
        descriptor = os.open(
            token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_urlsafe(48) + "\n")
    logs = prefix / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    logs.chmod(0o700)
    runtime = args.runtime.expanduser().resolve() if args.runtime else prefix / "runtime"
    runtime_parent = runtime
    while not runtime_parent.exists():
        runtime_parent = runtime_parent.parent
    if shutil.disk_usage(runtime_parent).free < 10 * 1024**3:
        parser.error("Keep at least 10 GiB free space on the selected runtime disk.")
    # A stopped companion releases its serial devices and camera. Unknown SSH
    # processes and listeners are never killed or reconfigured by this installer.
    systemctl = None
    if sys.platform == "darwin":
        stop_launch_agents()
    else:
        try:
            systemctl = stop_linux_services(args.no_start)
        except ValueError as error:
            parser.error(str(error))
    if not (runtime / "bin/python").exists():
        subprocess.run(
            [
                uv,
                "venv",
                *(["--offline"] if args.offline else []),
                "--python",
                args.python,
                str(runtime),
            ],
            check=True,
        )
    python = runtime / "bin/python"
    command = [uv, "pip", "install", "--python", str(python)]
    if args.offline:
        command.append("--offline")
    requirements = list(REQUIREMENTS)
    if args.arrow_site_packages is None:
        requirements.append("pyarrow==25.0.1")
    subprocess.run(command + requirements, check=True)
    if args.arrow_site_packages is not None:
        arrow = args.arrow_site_packages.expanduser().resolve()
        if not (arrow / "pyarrow").is_dir() or "\n" in str(arrow):
            parser.error("The supplied site-packages does not contain PyArrow.")
        site = Path(
            subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        (site / "openjensen_arrow.pth").write_text(str(arrow) + "\n")
    subprocess.run(
        [str(python), "-c", "import cv2,numpy,serial,scservo_sdk,pyarrow,PIL"], check=True
    )
    source = prefix / "source"
    if source.is_symlink():
        raise ValueError("Installed source must not be a symbolic link")
    source.mkdir(exist_ok=True)
    for name in (*required, Path("API.md")):
        current = args.source / name
        if current.is_file() and not current.is_symlink():
            target = source / name
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
                raise ValueError("Installed source must not contain symbolic links")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, target)
    data = args.data_dir.expanduser().resolve() if args.data_dir else default_data_dir()
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    data.chmod(0o700)
    if sys.platform == "darwin":
        services = install_mac_services(
            python, source, token_file, data, logs, args.ssh_host, args.no_start
        )
    else:
        services = install_linux_services(
            python, source, token_file, data, logs, args.ssh_host, args.no_start, systemctl
        )
    receipt = {
        "python": str(python),
        "source": str(source),
        "data": str(data),
        "token_file": str(token_file),
        "port": 18101,
        "services": services,
        "platform": sys.platform,
        "arrow_site_packages": str(args.arrow_site_packages) if args.arrow_site_packages else None,
        "offline": args.offline,
    }
    (private / "installation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (private / "installation.json").chmod(0o600)
    print("Robot Lab installed. Keep the private token file private.")
    print("Companion: http://127.0.0.1:18101; workspace: http://127.0.0.1:18096/robot-lab/")


if __name__ == "__main__":
    main()
