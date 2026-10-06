"""Explicit, resumable dependency setup on the app host; never starts a policy job."""

import asyncio
import os
import platform
import shutil
import signal
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from sqlalchemy import select

from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.local_worker_discovery import _environment, _hardware, _worker_root
from vla_platform.storage import jobs

REQUIRED_FREE_GIB = 24  # Installer/cache space plus the repository's 10 GiB reserve.
STEP_TIMEOUT = 3600
LOG_LIMIT = 8 * 1024**2


class LocalSetupState(StrictRecord):
    id: str | None = None
    status: Literal["idle", "running", "succeeded", "failed", "cancelled", "interrupted"] = "idle"
    stage: str = ""
    message: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    runtime_id: str | None = None


def now():
    return datetime.now(UTC).isoformat()


class LocalWorkerSetup:
    def __init__(self, lifecycle, discovery):
        self.lifecycle = lifecycle
        self.discovery = discovery
        self.directory = lifecycle.settings.data_dir / "local-environments"
        self.path = self.directory / "setup.json"
        self.state = LocalSetupState()
        self.task = None
        self.process = None
        self.lock = asyncio.Lock()
        if self.path.is_file() and not self.path.is_symlink() and self.path.stat().st_size < 16384:
            try:
                self.state = LocalSetupState.model_validate_json(self.path.read_bytes())
                if self.state.status == "running":
                    self.state.status = "interrupted"
                    self.state.message = (
                        "Setup was interrupted. Retry to continue the installation."
                    )
                    self.state.finished_at = now()
                    self._save()
            except OSError, ValueError:
                self.state = LocalSetupState()

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ValueError("The local installation directory must not be a symbolic link.")
        fd, temporary = tempfile.mkstemp(prefix=".setup-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(self.state.model_dump_json(indent=2) + "\n")
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def public(self):
        return self.state.model_copy(deep=True)

    async def start(self):
        async with self.lock:
            if self.task and not self.task.done():
                return self.public()
            root = _worker_root()
            if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
                raise ValueError("Automatic NVIDIA setup requires Linux x86-64.")
            if root is None or not (root / "requirements-smolvla-linux.txt").is_file():
                raise ValueError("The bundled local worker is unavailable on this app host.")
            hardware = await asyncio.to_thread(_hardware, _environment())
            if hardware is None:
                raise ValueError("No visible NVIDIA GPU was found. Check this machine first.")
            uv = shutil.which("uv")
            if uv is None:
                raise ValueError("Install uv on the app host to enable automatic setup.")
            if (
                shutil.disk_usage(self.lifecycle.settings.data_dir).free
                < REQUIRED_FREE_GIB * 1024**3
            ):
                raise ValueError(
                    f"Free at least {REQUIRED_FREE_GIB} GB on the app host before setup."
                )
            if self.directory.is_symlink():
                raise ValueError("The local installation directory must not be a symbolic link.")
            destination = self.directory / "smolvla-cu126"
            if destination.is_symlink():
                raise ValueError("The managed worker environment must not be a symbolic link.")
            # A repeated browser request must not install over a running trainer.
            async with self.lifecycle.execution.storage.engine.connect() as connection:
                active = (
                    await connection.execute(
                        select(jobs.c.id).where(jobs.c.status.in_(["queued", "running"])).limit(1)
                    )
                ).first()
            if active:
                raise ValueError(
                    "Wait for active jobs to finish before installing local dependencies."
                )
            found = await self.discovery.check()
            ready = next(
                (item for item in found.candidates if item.status in {"ready", "registered"}), None
            )
            if ready:
                runtime = await self.discovery.add(ready.id)
                self.state = LocalSetupState(
                    status="succeeded",
                    runtime_id=runtime.id,
                    message="The local SmolVLA worker is already installed and added.",
                    finished_at=now(),
                )
                self._save()
                return self.public()
            self.state = LocalSetupState(
                id=str(uuid4()), status="running", stage="Preparing setup", started_at=now()
            )
            self._save()
            self.task = asyncio.create_task(self._install(root, Path(uv), destination))
            return self.public()

    async def _stop(self, process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        # Also retire descendants of an exited installer parent.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _command(self, arguments, stage):
        self.state.stage = stage
        self._save()
        env = {
            key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR") if key in os.environ
        }
        env.update(
            UV_NO_CONFIG="1",
            UV_DEFAULT_INDEX="https://pypi.org/simple",
            UV_LINK_MODE="copy",
            PYTHONDONTWRITEBYTECODE="1",
        )
        descriptor = os.open(
            self.directory / "install.log",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
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
            async with asyncio.timeout(STEP_TIMEOUT):
                with os.fdopen(descriptor, "ab") as stream:
                    written = stream.tell()
                    while chunk := await process.stdout.read(4096):
                        available = max(0, LOG_LIMIT - written)
                        stream.write(chunk[:available])
                        written += min(len(chunk), available)
                    if await process.wait():
                        raise ValueError(
                            "Installation failed. Retry setup to reuse partial downloads."
                        )
        finally:
            await self._stop(process)
            self.process = None

    async def _install(self, root, uv, destination):
        try:
            if not (destination / "bin/python").is_file():
                await self._command(
                    [str(uv), "venv", "--python", "3.11", str(destination)],
                    "Installing Python environment",
                )
            await self._command(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--python",
                    str(destination / "bin/python"),
                    "-r",
                    str(root / "requirements-smolvla-linux.txt"),
                ],
                "Installing SmolVLA dependencies",
            )
            self.state.stage = "Checking GPU and worker"
            self._save()
            found = await self.discovery.check()
            candidate = next(
                (item for item in found.candidates if item.status in {"ready", "registered"}), None
            )
            if candidate is None:
                raise ValueError(
                    next(
                        (item.reason for item in found.candidates if item.reason),
                        "The installed worker did not pass its readiness check.",
                    )
                )
            runtime = await self.discovery.add(candidate.id)
            self.state.runtime_id = runtime.id
            self.state.status = "succeeded"
            self.state.message = (
                "SmolVLA is installed and added. Enable local runs when you are ready."
            )
        except asyncio.CancelledError:
            if self.state.status != "cancelled":
                self.state.status = "interrupted"
                self.state.message = "Setup was interrupted. Retry to continue the installation."
        except OSError, ValueError, TimeoutError:
            self.state.status = "failed"
            self.state.message = (
                "Setup did not finish. Check this machine for details, then retry setup."
            )
        finally:
            self.state.finished_at = now()
            self._save()

    async def cancel(self):
        async with self.lock:
            if self.task and not self.task.done():
                self.state.status = "cancelled"
                self.state.message = (
                    "Setup cancelled. Installed dependencies are retained for a retry."
                )
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass  # The task may not have started yet.
                self.state.finished_at = now()
                self._save()
            return self.public()

    async def close(self):
        if self.task and not self.task.done():
            if not self.task.cancelling():
                self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            if self.state.status == "running":
                self.state.status = "interrupted"
                self.state.message = "Setup was interrupted. Retry to continue the installation."
                self.state.finished_at = now()
                self._save()
