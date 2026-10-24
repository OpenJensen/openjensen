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
    "motion.py",
    "camera.py",
    "dataset.py",
    "assembled.py",
    "identify.py",
    "verification.py",
    "assembled_session.py",
)
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


def stop_managed_services():
    """Stop only our LaunchAgents before replacing their loaded code or runtime."""
    domain = "gui/" + str(os.getuid())
    for label in SERVICE_LABELS:
        identity = domain + "/" + label
        result = subprocess.run(["launchctl", "print", identity], capture_output=True)
        if result.returncode == 0:
            subprocess.run(["launchctl", "bootout", identity], capture_output=True, check=True)


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
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This installer is for macOS; run server.py directly on Linux.")
    disk_parent = args.prefix.expanduser().resolve()
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    if shutil.disk_usage(disk_parent).free < 10 * 1024**3:
        parser.error(
            "Keep at least 10 GiB free space before installing, including offline installs."
        )
    required = [
        *(Path(name) for name in SOURCE_MODULES),
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
            "--source must contain all companion modules and the licensed _vendor/servotools source"
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
        except OSError, ValueError, UnicodeDecodeError:
            parser.error(
                "The existing token must be an owned private regular file with a valid token."
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
    stop_managed_services()
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
    data = Path.home() / "Library/Application Support/OpenJensen/robot-lab"
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    data.chmod(0o700)
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
    if args.ssh_host:
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
                    args.ssh_host,
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
        if not args.no_start:
            domain = "gui/" + str(os.getuid())
            subprocess.run(["launchctl", "bootout", domain, str(target)], capture_output=True)
            subprocess.run(["launchctl", "bootstrap", domain, str(target)], check=True)
    receipt = {
        "python": str(python),
        "source": str(source),
        "data": str(data),
        "token_file": str(token_file),
        "port": 18101,
        "services": [s["Label"] for s in services],
        "arrow_site_packages": str(args.arrow_site_packages) if args.arrow_site_packages else None,
        "offline": args.offline,
    }
    (private / "installation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (private / "installation.json").chmod(0o600)
    print("Robot Lab installed. Keep the private token file private.")
    print("Companion: http://127.0.0.1:18101; workspace: http://127.0.0.1:18096/robot-lab/")


if __name__ == "__main__":
    main()
