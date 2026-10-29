"""Host-owned optional dependencies. Never install into Core or control hardware.

Browser requests choose an allowlisted profile only. A page closing does not
cancel installation; service shutdown cancels its whole installer process group.
"""

import asyncio
import json
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vla_platform.environment import getenv
from vla_platform.install_profiles import (
    GIB,
    check_profile,
    install_plan,
    interpreter,
    profile,
)
from vla_platform.robot_lab_bridge import RobotLabBridge, RobotLabError

InstallableProfile = Literal["robot", "datasets", "decision"]
ROOT = Path(__file__).resolve().parents[4]
TIMEOUT = 3600
LOG_LIMIT = 2 * 1024**2
RUNTIME_MODULES = {"datasets": "datasets", "robot-lab": "robot", "decision-lab": "decision"}


class InstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: InstallableProfile
    operation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    activate_module: bool = False


class InstallOperation(BaseModel):
    id: str
    profile: InstallableProfile
    status: Literal["running", "succeeded", "failed", "cancelled", "interrupted"]
    stage: str
    message: str
    started_at: str
    finished_at: str | None = None
    pid: int | None = None
    module: Literal["datasets", "robot-lab", "decision-lab"] | None = None


class RuntimePackage(BaseModel):
    id: InstallableProfile
    name: str
    ready: bool
    installable: bool
    message: str
    minimum_free_gib: int
    location: Literal["server", "companion"] = "server"
    host: str | None = None
    connected: bool | None = None
    managed: bool = False


class ModuleInstallationStatus(BaseModel):
    host: str
    profiles: list[RuntimePackage]
    operation: InstallOperation | None = None
    error: str | None = None


def now():
    return datetime.now(UTC).isoformat()


def require_runtime_unused(target):
    """Jobs are checked by the caller; also protect independently running companions."""
    result = subprocess.run(
        ["ps", "-ax", "-o", "command="], capture_output=True, text=True, timeout=5, check=True
    )
    if str(target) in result.stdout:
        raise ValueError("This runtime is in use by a process. Stop it before uninstalling.")


class ModuleInstallations:
    def __init__(self, data_dir: Path, root: Path = ROOT):
        self.data_dir, self.root = data_dir, root
        self.directory = data_dir / "module-environments"
        self.path = self.directory / "installation.json"
        self.operation = None
        self.task = self.process = None
        self.lock = asyncio.Lock()
        self.cached = {}
        self.problem = None
        try:
            self._load()
        except OSError, ValueError:
            self.problem = "Saved module installation state needs administrator attention."
            self.operation = None

    def _load(self):
        if self.directory.is_symlink():
            raise ValueError("The module directory must not be a symbolic link.")
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size > 16384:
                raise ValueError("Module installation state needs administrator attention.")
            self.operation = InstallOperation.model_validate_json(self.path.read_text())
            if self.operation.status == "running" and not self._previous_process_alive():
                self.operation.status = "interrupted"
                self.operation.message = "Server restarted during installation. Retry to continue."
                self.operation.finished_at = now()
                self._save()

    def _previous_process_alive(self):
        if not self.operation or not self.operation.pid:
            return False
        try:
            os.kill(self.operation.pid, 0)
            return True
        except ProcessLookupError:
            return False

    @property
    def pending(self):
        return (self.task is not None and not self.task.done()) or (
            self.operation is not None
            and self.operation.status == "running"
            and self._previous_process_alive()
        )

    def target(self, ident):
        return self.directory / ident

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ValueError("The module environment directory must not be a symbolic link.")
        descriptor, temporary = tempfile.mkstemp(prefix=".installation-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(self.operation.model_dump_json(indent=2) + "\n")
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def inspect(self, ident):
        if ident == "robot":
            config_path = Path(
                os.environ.get(
                    "JENSEN_ROBOT_LAB_CONFIG", self.data_dir / "robot-lab-connection.json"
                )
            )
            try:
                pairing = RobotLabBridge.read_config(config_path)
            except RobotLabError:
                pass
            else:
                return RuntimePackage(
                    id="robot",
                    name="Robot Lab",
                    ready=True,
                    installable=False,
                    message="Installed on the paired computer. Open Robot Lab to use it.",
                    minimum_free_gib=11,
                    location="companion",
                    host=pairing["label"],
                )
        cached = self.cached.get(ident)
        if cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        value = profile(ident)
        # Existing environments are inspected and reused, never rewritten.
        targets = [self.target(ident), self.root / value.directory / ".venv"]
        if ident == "datasets":
            targets.insert(1, self.data_dir / "local-environments/dataset-reader")
        explicit = getenv("JENSEN_CPU_READER_PYTHON") if ident == "datasets" else None
        if explicit:
            targets = [Path(explicit).parent.parent]
        elif ident == "decision" and getenv("JENSEN_DECISION_PYTHON"):
            targets.insert(0, Path(getenv("JENSEN_DECISION_PYTHON")).parent.parent)
        checked = None
        for target in targets:
            state = check_profile(value, self.root, target)
            # uv installs packages; its absence does not remove an installed runtime.
            if (
                state.get("supported_platform")
                and state.get("python_matches")
                and state.get("package_versions_match")
                and all(found for name, found in state["external_tools"].items() if name != "uv")
            ):
                state["ready"] = True
                state["message"] = "Installed package metadata and runtime tools are available."
            if (
                target == self.target(ident)
                and not (target / ".openjensen-installed.json").is_file()
            ):
                state["ready"] = False
            if state["ready"]:
                checked = state
                break
            checked = checked or state
        uv = shutil.which("uv")
        tools = checked["external_tools"]
        missing = [name for name, found in tools.items() if not found]
        runtime_missing = [name for name in missing if name != "uv"]
        installable = platform.system() in value.platforms and os.name == "posix" and bool(uv)
        message = checked["message"]
        if platform.system() not in value.platforms or os.name != "posix":
            message = "Automatic installation requires " + " or ".join(value.platforms) + "."
        elif missing:
            installable = False
            if not checked["ready"]:
                message = "Install on the server first: " + ", ".join(missing) + "."
        if checked["ready"] and runtime_missing:
            checked["ready"] = False
        if explicit and not checked["ready"]:
            installable = False
            message = "The server's configured dataset reader needs administrator attention."
        if not (self.root / value.directory).is_dir():
            installable = False
            message = "This server does not include the module sources."
        result = RuntimePackage(
            id=ident,
            name=value.name,
            ready=checked["ready"],
            installable=installable,
            message=message,
            minimum_free_gib=10 + value.estimated_install_gib,
            host=platform.node(),
            managed=checked["ready"] and target == self.target(ident),
        )
        self.cached[ident] = (time.monotonic(), result)
        return result

    async def status(self, robot_lab=None):
        async with self.lock:
            if self.problem:
                return ModuleInstallationStatus(
                    host=platform.node(), profiles=[], error=self.problem
                )
            if self.operation and self.operation.status == "running" and not self.pending:
                self.operation.status = "interrupted"
                self.operation.message = "Installation was interrupted. Retry to continue."
                self.operation.finished_at = now()
                self.operation.pid = None
                self._save()
            # This explicit status check refreshes runtime metadata; sidebar reads reuse it.
            self.cached.clear()
            values = await asyncio.to_thread(
                lambda: [self.inspect(ident) for ident in ("datasets", "robot", "decision")]
            )
            companion = next((item for item in values if item.location == "companion"), None)
            if companion is not None and robot_lab is not None:
                try:
                    async with asyncio.timeout(4):
                        health = await robot_lab.status()
                    companion.connected = health.get("connected") is True
                except TimeoutError, RobotLabError:
                    companion.connected = False
                companion.message = (
                    "Installed on the paired computer and connected."
                    if companion.connected
                    else "Installed on the paired computer. Reconnect it to use Robot Lab."
                )
            return ModuleInstallationStatus(
                host=platform.node(),
                profiles=values,
                operation=self.operation,
            )

    async def start(self, payload: InstallRequest):
        async with self.lock:
            if self.problem:
                raise ValueError(self.problem)
            if self.operation and payload.operation_id == self.operation.id:
                if payload.profile != self.operation.profile or (
                    payload.activate_module != (self.operation.module is not None)
                ):
                    raise ValueError("This request already belongs to another module.")
                return self.operation.model_copy(deep=True)
            if self.pending:
                raise ValueError("Wait for the current module installation to finish.")
            if payload.activate_module:
                from vla_platform.modules import _editable, read_registry

                _editable()
                if read_registry(self.data_dir) is None:
                    raise ValueError("The module registry has not been initialized.")
            value = profile(payload.profile)
            self.cached.pop(value.id, None)
            state = await asyncio.to_thread(self.inspect, value.id)
            if state.location == "companion" and not payload.activate_module:
                raise ValueError(
                    "Robot Lab is installed on the paired computer. Open or reconnect it there."
                )
            if not state.ready and not state.installable:
                raise ValueError(state.message)
            if (
                not state.ready
                and shutil.disk_usage(self.data_dir).free < state.minimum_free_gib * GIB
            ):
                raise ValueError(
                    f"Free at least {state.minimum_free_gib} GiB on this server first."
                )
            target = self.target(value.id)
            if self.directory.is_symlink() or target.is_symlink():
                raise ValueError("Managed module environments must not be symbolic links.")
            self.operation = InstallOperation(
                id=payload.operation_id,
                profile=value.id,
                status="running",
                stage="Preparing",
                message="Downloading pinned dependencies from upstream package indexes on "
                + platform.node()
                + ".",
                started_at=now(),
                module=next(
                    (key for key, value in RUNTIME_MODULES.items() if value == payload.profile),
                    None,
                )
                if payload.activate_module
                else None,
            )
            self._save()
            self.task = asyncio.create_task(self._install(value, target, state.ready))
            return self.operation.model_copy(deep=True)

    async def _stop(self, process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def _command(self, argv, stage):
        self.operation.stage = stage
        self._save()
        env = {
            key: os.environ[key] for key in ("HOME", "PATH", "LANG", "TMPDIR") if key in os.environ
        }
        env.update(
            UV_NO_CONFIG="1",
            UV_DEFAULT_INDEX="https://pypi.org/simple",
            UV_LINK_MODE="copy",
            UV_NO_CACHE="1",
            PYTHONDONTWRITEBYTECODE="1",
            UV_CONCURRENT_BUILDS="1",
            UV_CONCURRENT_INSTALLS="1",
        )
        descriptor = os.open(
            self.directory / "install.log",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.root,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            os.close(descriptor)
            raise
        self.process = process
        try:
            with os.fdopen(descriptor, "ab") as stream:
                self.operation.pid = process.pid
                self._save()
                written = stream.tell()
                async with asyncio.timeout(TIMEOUT):
                    while chunk := await process.stdout.read(4096):
                        available = max(0, LOG_LIMIT - written)
                        stream.write(chunk[:available])
                        written += min(len(chunk), available)
                    if await process.wait():
                        raise ValueError(
                            "Dependency installation failed. "
                            "Retry or inspect the server install log."
                        )
        finally:
            await self._stop(process)
            self.process = None
            self.operation.pid = None
            self._save()

    async def _install(self, value, target, already_ready):
        try:
            if not already_ready:
                uv = shutil.which("uv")
                if not interpreter(target).is_file():
                    await self._command(
                        [uv, "venv", "--python", value.python, str(target)],
                        "Creating Python environment",
                    )
                for command in install_plan(value, self.root, target):
                    await self._command([uv, *command["argv"][1:]], "Installing " + value.name)
                self.operation.stage = "Checking dependencies"
                self._save()
                state = await asyncio.to_thread(check_profile, value, self.root, target)
                if not state["ready"]:
                    raise ValueError(state["message"])
                (target / ".openjensen-installed.json").write_text(
                    json.dumps({"profile": value.id})
                )
            if self.operation.module:
                from vla_platform.modules import set_module_installed

                await asyncio.to_thread(
                    set_module_installed, self.data_dir, self.operation.module, True
                )
            # Only verified dependencies can activate a requested feature.
            self.operation.status = "succeeded"
            self.operation.message = (
                "Dependencies installed. Pair the companion on the USB computer in Robot Lab."
                if value.id == "robot"
                else "Installed. Configure model files and accept the license before scoring."
                if value.id == "decision"
                else "Dataset readers installed and available to this server."
            )
        except asyncio.CancelledError:
            self.operation.status = "cancelled"
            self.operation.message = (
                "Installation stopped; downloaded dependencies are kept for retry."
            )
        except (OSError, ValueError, TimeoutError) as exc:
            self.operation.status = "failed"
            self.operation.message = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Installation failed. Retry or inspect the server install log."
            )
        finally:
            self.operation.finished_at = now()
            self.cached.pop(value.id, None)
            self._save()

    def remove(self, ident):
        """Remove only this manager's environment, never a companion or a shared runtime."""
        if self.pending:
            raise ValueError("Finish or cancel the current installation first.")
        target = self.target(ident)
        if self.directory.is_symlink() or target.is_symlink():
            raise ValueError("Managed module environments must not be symbolic links.")
        if not target.exists():
            return False
        receipt = target / ".openjensen-installed.json"
        owned = False
        if receipt.is_file() and not receipt.is_symlink():
            try:
                owned = json.loads(receipt.read_text()).get("profile") == ident
            except OSError, ValueError, AttributeError:
                pass
        # A failed/cancelled installation also owns its partial environment.
        owned |= self.operation is not None and self.operation.profile == ident
        if not owned:
            raise ValueError(
                "No installation receipt was found. This environment needs administrator review."
            )
        require_runtime_unused(target)
        shutil.rmtree(target)
        self.cached.pop(ident, None)
        return True

    async def cancel(self):
        async with self.lock:
            if self.pending and (self.task is None or self.task.done()):
                raise ValueError(
                    "The previous server's installer is still running. Wait for it to exit."
                )
            if self.pending:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    self.operation.status = "cancelled"
                    self.operation.message = "Installation stopped before it started."
                    self.operation.finished_at = now()
                    self._save()
            return self.operation

    async def close(self):
        if self.task is not None and not self.task.done():
            await self.cancel()
