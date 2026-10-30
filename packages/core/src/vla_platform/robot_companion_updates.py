"""Reconcile an attested local Linux robot service with the installed application.

Pairing with a loopback URL is not proof of locality: it may be an SSH tunnel.
Only an owned user service, matching live process arguments and matching private
pairing credential permit replacement. Existing installer receipts must also
match. Runtime, device data and credentials stay at their existing paths. No
dependency installation or motor command is issued.
"""

import configparser
import contextvars
import hmac
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

UNIT = "openjensen-robot-lab.service"
DEADLINE = contextvars.ContextVar("robot_service_update_deadline", default=None)
HELP = (
    "Robot control service update could not be verified automatically. "
    "Run deploy/install_robot_lab.py on the USB computer "
    "with its existing --prefix, --runtime and --data-dir, then restart OpenJensen. "
    "The application update does not replace an external or unverified service."
)


def result(state, message, *, managed=False):
    return {"state": state, "message": message, "managed": managed}


def private_bytes(path, limit=65536, *, private=True):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & (0o077 if private else 0o022)
            or info.st_size > limit
        ):
            raise ValueError("Local service metadata is not a private owned file")
        return stream.read()


def command(*args, timeout=10):
    return subprocess.check_output(
        args, text=True, timeout=remaining(timeout), stderr=subprocess.DEVNULL
    )


def remaining(maximum):
    deadline = DEADLINE.get()
    if deadline is None:
        return maximum
    budget = deadline - time.monotonic()
    if budget <= 0:
        raise RuntimeError("The robot service update exceeded its time budget")
    return min(maximum, budget)


def process_matches(pid, arguments, *, descendants=False, proc_root=Path("/proc")):
    pending = [(pid, 0)]
    visited = set()
    while pending and len(visited) < 16:
        current, depth = pending.pop(0)
        if current in visited or current < 1:
            continue
        visited.add(current)
        process = proc_root / str(current)
        if process.stat().st_uid != os.getuid():
            raise ValueError("The robot process is not owned by this user")
        running = process.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        if running == arguments:
            return True
        if descendants and depth < 3:
            children = process.joinpath("task", str(current), "children").read_text().split()
            pending.extend((int(child), depth + 1) for child in children)
    return False


def service_info():
    output = command(
        "systemctl",
        "--user",
        "show",
        UNIT,
        "--property=FragmentPath,MainPID,ActiveState,DropInPaths,WorkingDirectory",
    )
    value = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if value.get("ActiveState") != "active":
        raise ValueError("The robot service is inactive")
    unit = Path(value["FragmentPath"])
    units = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd/user"
    if unit != units / UNIT or unit.is_symlink():
        raise ValueError("The robot service file is not an owned local unit")
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    body = private_bytes(unit, private=False)
    parser.read_string(body.decode())
    evidence = [(str(unit), body)]
    for filename in shlex.split(value.get("DropInPaths", "")):
        override = Path(filename)
        if override.parent != units / (UNIT + ".d") or override.is_symlink():
            raise ValueError("The robot service override is not a local managed file")
        body = private_bytes(override, private=False)
        patch = configparser.ConfigParser(interpolation=None, strict=False)
        patch.read_string(body.decode())
        if patch.sections() != ["Service"] or set(patch["Service"]) - {
            "execstart",
            "workingdirectory",
        }:
            raise ValueError("The robot service has unsupported overrides")
        parser.read_string(body.decode())
        evidence.append((str(override), body))
    section = parser["Service"]
    if any(key.startswith("exec") and key != "execstart" for key in section):
        raise ValueError("The robot service has unverified execution hooks")
    arguments = shlex.split(section["ExecStart"])
    # The installer quotes literal systemd '%' and '$' escapes. Do not expand
    # arbitrary service specifiers or environment-dependent launch commands.
    for argument in arguments:
        if "%" in argument.replace("%%", "") or "$" in argument.replace("$$", ""):
            raise ValueError("The robot service uses unsupported command expansion")
    arguments = [argument.replace("%%", "%").replace("$$", "$") for argument in arguments]
    wrapped = len(arguments) == 4 and arguments[:3] == ["/usr/bin/sg", "dialout", "-c"]
    launch_arguments = arguments
    if wrapped:
        if any(character in arguments[3] for character in ";|&><`$\n\r"):
            raise ValueError("The robot group wrapper contains shell operations")
        arguments = shlex.split(arguments[3])
    pid = int(value["MainPID"])
    if not process_matches(pid, arguments, descendants=wrapped):
        raise ValueError("The running robot process does not match the managed service")
    return {
        "arguments": arguments,
        "unit": str(unit),
        "launch_arguments": launch_arguments,
        "unit_evidence": evidence,
        "working_directory": value.get("WorkingDirectory"),
    }


def attest(pairing):
    if sys.platform != "linux" or pairing.get("url") != "http://127.0.0.1:18101":
        raise ValueError("Only the verified local Linux service can update automatically")
    info = service_info()
    arguments = info["arguments"]
    if len(arguments) != 10 or arguments[2::2] != [
        "--host",
        "--port",
        "--token-file",
        "--data-dir",
    ]:
        raise ValueError("The robot service has unrecognized launch arguments")
    python, script, host, port, token_path, data = (
        arguments[0],
        arguments[1],
        arguments[3],
        arguments[5],
        arguments[7],
        arguments[9],
    )
    source = Path(script).parent
    if host != "127.0.0.1" or port != "18101" or Path(script).name != "server.py":
        raise ValueError("The robot service endpoint is not the paired endpoint")
    if info["working_directory"] != str(source):
        raise ValueError("The robot service working directory is unverified")
    token_file = Path(token_path)
    receipt_path = token_file.parent / "installation.json"
    expected = {"python": python, "source": str(source), "token_file": token_path, "data": data}
    try:
        receipt_body = private_bytes(receipt_path)
    except FileNotFoundError:
        # Older local deployments predate installer receipts. The exact owned
        # systemd unit, live process tree and matching private pairing credential
        # above/below attest the same service without guessing installation paths.
        receipt_body = None
    if receipt_body is not None:
        receipt = json.loads(receipt_body)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("The installer receipt does not match the running service")
        if receipt.get("platform") != "linux" or UNIT not in receipt.get("services", []):
            raise ValueError("The service is not registered by the local installer")
    token = private_bytes(token_file, 4096).decode().strip()
    if not isinstance(pairing.get("token"), str) or not hmac.compare_digest(
        token, pairing["token"]
    ):
        raise ValueError("The paired credential does not match the local robot service")
    if (
        not source.is_absolute()
        or source.is_symlink()
        or not source.is_dir()
        or source.stat().st_uid != os.getuid()
        or source.stat().st_mode & 0o022
    ):
        raise ValueError("The installed robot sources are not safely replaceable")
    for path in (Path(python), Path(data), token_file, receipt_path):
        if not path.is_absolute() or path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Runtime, recordings and credentials must be outside robot sources")
    if not Path(python).is_file() or not Path(data).is_dir():
        raise ValueError("The existing runtime or robot data directory is unavailable")
    return {**info, **expected, "receipt_path": receipt_path, "receipt_body": receipt_body}


def code_files(source):
    """Reject mixed code/data trees rather than moving unrelated user files."""
    files = {}
    for path in source.rglob("*"):
        remaining(1)
        if path.is_symlink():
            raise ValueError("Robot source tree contains symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts and path.suffix == ".pyc":
            continue
        allowed = (
            path.suffix == ".py"
            or str(relative)
            in {"API.md", "requirements-hardware.txt", "LEROBOT-LICENSE", "LEROBOT-PROVENANCE.json"}
            or str(relative) in {"_vendor/servotools/LICENSE", "_vendor/servotools/PROVENANCE.json"}
        )
        if not allowed or path.stat().st_size > 2 * 1024**2:
            raise ValueError("Robot source directory also contains unverified files")
        files[str(relative)] = path.read_bytes()
    if not {"server.py", "hardware.py", "assembled.py", "diagnostics.py"} <= files.keys():
        raise ValueError("Robot source tree is incomplete")
    if sum(map(len, files.values())) > 16 * 1024**2:
        raise ValueError("Robot source tree is too large for a source-only update")
    return files


def status(pairing):
    request = Request(
        pairing["url"] + "/v1/status", headers={"Authorization": "Bearer " + pairing["token"]}
    )
    with urlopen(request, timeout=remaining(3)) as response:
        return json.loads(response.read(4 * 1024**2))


def idle(pairing):
    value = status(pairing)
    if value.get("version") != "1" or value.get("backend") != "feetech-sdk":
        raise ValueError("The paired endpoint is not the supported robot control service")
    session = value["session"]
    return (
        session["mode"] == "idle"
        and session["torque_enabled"] is False
        and not value.get("recording", {}).get("active", False)
        and not value.get("recording", {}).get("episode_active", False)
        and not session.get("centering", {}).get("active", False)
        and session.get("rest_position", {}).get("phase", "idle") in {"idle", "saved"}
        and value.get("assembled", {}).get("phase", "idle")
        in {"idle", "complete", "error", "stopped", "cancelled"}
        and not value.get("assembled", {}).get("motion_active", False)
        and value.get("verification", {}).get("phase", "idle")
        in {"idle", "done", "error", "stopped", "cancelled"}
        and not value.get("verification", {}).get("motion_active", False)
    )


def control(action):
    command("systemctl", "--user", action, UNIT, timeout=25)


def wait_healthy(pairing, expected):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        try:
            if attest(pairing)["arguments"] == expected["arguments"] and idle(pairing):
                return
        except OSError, ValueError, KeyError, subprocess.SubprocessError:
            pass
        time.sleep(0.25)
    raise RuntimeError("The updated robot service did not become healthy")


def reconcile(source, pairing_path, *, allow_update=False):
    # The previous release's updater may only wait 60 seconds for startup.
    # Reserve 25 seconds for preparation/activation and 20 for rollback.
    token = DEADLINE.set(time.monotonic() + 25)
    try:
        return _reconcile(source, pairing_path, allow_update=allow_update)
    finally:
        DEADLINE.reset(token)


def _reconcile(source, pairing_path, *, allow_update=False):
    if not pairing_path.exists():
        return None
    try:
        pairing = json.loads(private_bytes(pairing_path))
        local = attest(pairing)
        installed = Path(local["source"])
        desired = code_files(source)
        previous = code_files(installed)
        if desired == previous:
            return result(
                "current", "Robot control service matches this application.", managed=True
            )
        if not allow_update:
            return result("needs_update", HELP, managed=True)
        if not idle(pairing):
            return result(
                "needs_update",
                "Robot control service needs update. Stop Robot Lab and restart "
                "OpenJensen to update it.",
                managed=True,
            )
        # Serialize with another OpenJensen workspace trying to update this same
        # service, then re-attest after acquiring ownership.
        import fcntl

        lock_path = local["receipt_path"].with_name("application-update.lock")
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if attest(pairing) != local or not idle(pairing):
                raise ValueError("The local robot service changed during update preparation")
            return replace_sources(desired, installed, local, pairing)
    except OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError:
        return result("needs_update", HELP)


def replace_sources(desired, installed, local, pairing):
    operation = uuid.uuid4().hex
    candidate = installed.with_name(installed.name + ".candidate-" + operation)
    backup = installed.with_name(installed.name + ".rollback-" + operation)
    candidate.mkdir(mode=0o700)
    try:
        for name, body in desired.items():
            path = candidate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        requirements = []
        for line in desired.get("requirements-hardware.txt", b"").decode().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                pieces = line.strip().split("==")
                if len(pieces) != 2:
                    raise ValueError("Robot dependencies require a manual installer update")
                requirements.append(pieces)
        if not requirements:
            raise ValueError("Robot dependency requirements are unavailable")
        # Verify every pinned runtime dependency, including lazily imported
        # camera/recording packages, before stopping the service. No pip runs.
        command(
            local["python"],
            "-B",
            "-c",
            "import importlib.metadata as m,json,sys; "
            "assert all(m.version(n)==v for n,v in json.loads(sys.argv[2])); "
            "sys.path.insert(0, sys.argv[1]); import server",
            str(candidate),
            json.dumps(requirements),
            timeout=20,
        )
        if attest(pairing) != local or not idle(pairing):
            raise ValueError("The local robot service changed before source activation")
    except Exception:
        shutil.rmtree(candidate)  # This operation's private, newly created code only.
        raise
    stopped = switched = False
    try:
        # A timeout can happen after systemd accepted Stop. Recovery must restart
        # and verify the previous service even if no source rename happened.
        stopped = True
        control("stop")
        os.rename(installed, backup)
        try:
            os.rename(candidate, installed)
        except OSError:
            os.rename(backup, installed)
            raise
        switched = True
        control("start")
        wait_healthy(pairing, local)
        return result(
            "updated",
            "Robot control service updated and verified with this application.",
            managed=True,
        )
    except Exception:
        DEADLINE.set(time.monotonic() + 20)
        try:
            if switched:
                control("stop")
                failed = installed.with_name(installed.name + ".failed-" + operation)
                os.rename(installed, failed)
                os.rename(backup, installed)
            if stopped:
                control("start")
                wait_healthy(pairing, local)
            if candidate.exists():
                shutil.rmtree(candidate)
        except Exception:
            return result(
                "needs_update",
                "Robot control service needs recovery. Its previous sources are retained; "
                "restore the saved service before using Robot Lab.",
                managed=True,
            )
        return result(
            "needs_update",
            "Robot control service update failed and was rolled back. The previous "
            "service and saved data are retained.",
            managed=True,
        )
