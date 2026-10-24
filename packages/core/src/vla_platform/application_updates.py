"""Version discovery and confirmed, host-owned updates from the canonical main branch."""

import asyncio
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from vla_platform import __version__, update_runner

UPSTREAM = "https://api.github.com/repos/OpenJensen/openjensen"
ROOT = Path(__file__).resolve().parents[4]
ACTIVE_PHASES = {"queued", "building", "backing_up", "validating", "restarting"}
SHA = re.compile(r"^[0-9a-f]{40}$")


class UpdateError(ValueError):
    def __init__(self, message: str, status: int = 409):
        self.status = status
        super().__init__(message)


class UpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    confirmation: Literal["update-and-restart"]
    operation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class UpdateOperation(BaseModel):
    id: str
    phase: str
    target_commit: str
    message: str


class ApplicationVersion(BaseModel):
    version: str
    commit: str | None
    host: str


class UpdateStatus(BaseModel):
    installed: ApplicationVersion
    channel: Literal["main"] = "main"
    upstream_commit: str | None = None
    latest_release: str | None = None
    state: Literal["unchecked", "current", "available", "ahead", "diverged", "unknown", "error"]
    checked_at: str | None = None
    message: str
    automatic_update: bool
    setup_message: str | None = None
    operation: UpdateOperation | None = None


def installed_commit(root: Path = ROOT) -> str | None:
    for name in (".openjensen-build.json", ".openjensen-release.json"):
        try:
            value = json.loads((root / name).read_text())["commit"]
            if isinstance(value, str) and SHA.fullmatch(value):
                return value
        except OSError, ValueError, KeyError, TypeError:
            pass
    if (root / ".git").exists():
        try:
            value = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=3
            ).strip()
            return value if SHA.fullmatch(value) else None
        except OSError, subprocess.SubprocessError:
            pass
    return None


class ApplicationUpdates:
    def __init__(self, *, config_path: Path | None = None, client=None, commit=None, data_dir=None):
        # Capture this at process startup; a checkout moving later cannot relabel running code.
        self.installed = ApplicationVersion(
            version=__version__, commit=commit or installed_commit(), host=platform.node()
        )
        configured = os.environ.get("JENSEN_UPDATE_CONFIG")
        self.config_path = config_path or (Path(configured) if configured else None)
        self.data_dir = Path(data_dir).resolve() if data_dir is not None else None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(6, connect=3),
            follow_redirects=False,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "OpenJensen-updates"},
        )
        self.lock = asyncio.Lock()
        self.reserving = False
        self.checked = 0.0
        self.value = {"state": "unchecked", "message": "Updates have not been checked."}

    async def close(self):
        await self.client.aclose()

    def configuration(self):
        if self.config_path is None:
            raise UpdateError("Automatic updates are not configured on this server.")
        if os.name != "posix":
            raise UpdateError("Automatic updates require a managed macOS/Linux server.")
        try:
            info = self.config_path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_size > 16384
            ):
                raise ValueError("The update configuration must be a private, owned file.")
            config = update_runner.load_config(self.config_path)
            root = Path(config["root"])
            if self.data_dir is not None and self.data_dir != (root / "data").resolve():
                raise ValueError("The configured update workspace does not match this server.")
            if installed_commit((root / "current").resolve()) != self.installed.commit:
                raise ValueError("The configured release does not match this server.")
            return config
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise UpdateError(
                "The server update configuration needs administrator attention."
            ) from exc

    def operation(self):
        try:
            config = self.configuration()
            path = Path(config["root"]) / "run/application-update.json"
            value = json.loads(path.read_text())
            if value["phase"] in ACTIVE_PHASES:
                pid = value.get("pid")
                alive = False
                if isinstance(pid, int):
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except ProcessLookupError:
                        pass
                elif time.time() - value.get("created", 0) < 20:
                    alive = True
                if not alive:
                    value.update(
                        phase="recovery_required"
                        if value["phase"] in {"backing_up", "validating", "restarting"}
                        else "failed",
                        message="Update interrupted. Inspect the server update log.",
                    )
            return UpdateOperation.model_validate(value)
        except UpdateError, OSError, ValueError, KeyError, TypeError:
            return None

    @property
    def pending(self):
        operation = self.operation()
        return (
            self.reserving
            or operation is not None
            and operation.phase in ACTIVE_PHASES | {"recovery_required"}
        )

    def status(self):
        try:
            self.configuration()
            supported, setup = True, None
        except UpdateError as exc:
            supported, setup = False, str(exc)
        return UpdateStatus(
            installed=self.installed,
            automatic_update=supported,
            setup_message=setup,
            operation=self.operation(),
            **self.value,
        )

    async def check(self, *, force=False):
        async with self.lock:
            if not force and self.checked and time.monotonic() - self.checked < 600:
                return self.status()
            from datetime import UTC, datetime

            checked_at = datetime.now(UTC).isoformat()
            try:
                response = await self.client.get(UPSTREAM + "/commits/main")
                response.raise_for_status()
                target = response.json()["sha"]
                if not isinstance(target, str) or not SHA.fullmatch(target):
                    raise ValueError("Invalid upstream commit")
                commit = self.installed.commit
                state, message = (
                    "unknown",
                    "This build has no recorded commit; automatic update is unavailable.",
                )
                if target == commit:
                    state, message = "current", "Up to date with upstream main."
                elif commit:
                    response = await self.client.get(UPSTREAM + f"/compare/{commit}...{target}")
                    if response.status_code == 404:
                        state, message = (
                            "unknown",
                            "This development build cannot be compared with upstream main.",
                        )
                    else:
                        response.raise_for_status()
                        relation = response.json()["status"]
                        state, message = {
                            "ahead": ("available", "An update from upstream main is available."),
                            "behind": (
                                "ahead",
                                "This development build is ahead of upstream main.",
                            ),
                            "diverged": (
                                "diverged",
                                "This development build differs from upstream main; "
                                "update manually.",
                            ),
                            "identical": ("current", "Up to date with upstream main."),
                        }[relation]
                release = self.value.get("latest_release")
                try:
                    response = await self.client.get(UPSTREAM + "/releases/latest")
                    response.raise_for_status()
                    tag = response.json()["tag_name"]
                    if isinstance(tag, str) and re.fullmatch(
                        r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[\w.-]+)?", tag
                    ):
                        release = tag
                except httpx.HTTPError, ValueError, KeyError, TypeError:
                    pass  # Release metadata is separate from the main update channel.
                self.value = dict(
                    state=state,
                    message=message,
                    upstream_commit=target,
                    latest_release=release,
                    checked_at=checked_at,
                )
            except httpx.HTTPError, ValueError, KeyError, TypeError:
                self.value = dict(
                    state="error",
                    message="Could not check upstream main. Try again later.",
                    checked_at=checked_at,
                    latest_release=self.value.get("latest_release"),
                )
            self.checked = time.monotonic()
            return self.status()

    async def start(self, payload: UpdateRequest):
        from uuid import uuid4

        config = self.configuration()
        async with self.lock:
            operation = self.operation()
            if operation and operation.phase in ACTIVE_PHASES | {"recovery_required"}:
                raise UpdateError("An update is already in progress.")
            if (
                self.value.get("state") != "available"
                or self.value.get("upstream_commit") != payload.target_commit
            ):
                raise UpdateError(
                    "The available version changed. Check for updates and confirm again."
                )
            state = dict(
                id=payload.operation_id or uuid4().hex,
                phase="queued",
                target_commit=payload.target_commit,
                message="Preparing update…",
                created=time.time(),
                pid=None,
            )
            path = Path(config["root"]) / "run/application-update.json"
            update_runner.write_state(path, state)
            log = path.with_name("application-update.log")
            try:
                with log.open("ab") as output:
                    os.chmod(log, 0o600)
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(update_runner.__file__)),
                            "--config",
                            str(self.config_path),
                            "--target",
                            payload.target_commit,
                            "--installed",
                            self.installed.commit,
                            "--operation",
                            state["id"],
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=output,
                        start_new_session=True,
                        close_fds=True,
                    )
            except OSError as exc:
                state.update(phase="failed", message="Could not start the server updater.")
                update_runner.write_state(path, state)
                raise UpdateError(state["message"], 503) from exc
            return self.status()
