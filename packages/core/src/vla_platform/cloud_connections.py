"""Verify operator-managed cloud credentials without provisioning resources.

The workspace stores provider selection only. Tokens, keys, and CLI auth files
remain managed by the provider CLIs, outside Firebird.
"""

import asyncio
import json
import os
import shutil
import signal
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Provider = Literal["gcp"]
ConnectionStatus = Literal["disconnected", "unverified", "connected", "setup_required", "error"]


class GcpConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(pattern=r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
    region: str = Field(min_length=3, max_length=64, pattern=r"^[a-z][a-z0-9-]+[0-9]$")


ConnectionConfig = GcpConnectionConfig


class CloudIdentity(BaseModel):
    account: str | None = None


class CloudConnection(BaseModel):
    provider: Provider
    name: str
    status: ConnectionStatus = "disconnected"
    config: ConnectionConfig | None = None
    identity: CloudIdentity | None = None
    checked_at: str | None = None
    message: str | None = None
    setup_commands: list[str] = Field(default_factory=list)


class CloudConnectionsResponse(BaseModel):
    providers: list[CloudConnection]


class _CheckFailed(Exception):
    def __init__(self, status: ConnectionStatus, message: str):
        super().__init__(message)
        self.status = status


async def run_cloud_cli(executable: str, args: list[str]) -> object:
    """Run a fixed, read-only provider command, with no shell or login prompts."""
    path = shutil.which(executable)
    if path is None:
        raise _CheckFailed("setup_required", "Install Google Cloud CLI on this machine.")
    env = {
        **os.environ,
        "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
        "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK": "1",
    }
    try:
        process = await asyncio.create_subprocess_exec(
            path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise _CheckFailed("setup_required", f"Could not start {executable}.") from exc
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=15)
    except (TimeoutError, asyncio.CancelledError) as exc:
        try:
            if os.name == "posix":
                # A credential helper may outlive an already-exited CLI parent.
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        # Credential helpers may inherit the pipes; cleanup must also be bounded.
        try:
            await asyncio.wait_for(process.communicate(), timeout=3)
        except TimeoutError:
            pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise _CheckFailed("error", "Connection check timed out. Try again.") from exc
    if process.returncode:
        # CLI diagnostics can include credentials or private paths; never expose them.
        raise _CheckFailed("setup_required", "Check your CLI sign-in and access, then retry.")
    try:
        return json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _CheckFailed("error", "The provider CLI returned an invalid response.") from exc


def _identity_text(value: object, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise _CheckFailed("error", "The provider CLI returned an invalid identity.")
    if any(not character.isprintable() for character in value):
        raise _CheckFailed("error", "The provider CLI returned an invalid identity.")
    return value


class CloudConnections:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "cloud-connections.json"
        self._configs: dict[Provider, ConnectionConfig] = {}
        self._states: dict[Provider, CloudConnection] = {}
        self._locks: dict[Provider, asyncio.Lock] = {
            "gcp": asyncio.Lock(),
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            document = json.loads(self.path.read_text())
            if not isinstance(document, dict) or document.get("version") != 1:
                raise ValueError("Unsupported cloud connection file")
            configs = document["providers"]
            if not isinstance(configs, dict):
                raise ValueError("Invalid cloud connection file")
            if "gcp" in configs:
                self._configs["gcp"] = GcpConnectionConfig.model_validate(configs["gcp"])
        except OSError, ValueError, KeyError, ValidationError:
            self._configs.clear()
            for provider in ("gcp",):
                state = self._base(provider)
                state.status = "error"
                state.message = "Saved cloud settings could not be read. Reconnect the provider."
                self._states[provider] = state

    def _save(self, configs: dict[Provider, ConnectionConfig]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "providers": {p: c.model_dump() for p, c in configs.items()}}
        handle, temporary = tempfile.mkstemp(prefix=".cloud-connections-", dir=self.path.parent)
        try:
            with os.fdopen(handle, "w") as file:
                json.dump(payload, file, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _base(self, provider: Provider) -> CloudConnection:
        config = self._configs.get(provider)
        return CloudConnection(
            provider=provider,
            name="Google Cloud",
            status="unverified" if config else "disconnected",
            config=config,
            setup_commands=["gcloud auth login"],
        )

    def list(self) -> CloudConnectionsResponse:
        return CloudConnectionsResponse(
            providers=[
                self._states.get(provider, self._base(provider)).model_copy(deep=True)
                for provider in ("gcp",)
            ]
        )

    async def _verify(self, provider: Provider, config: ConnectionConfig) -> CloudIdentity:
        if provider == "gcp" and isinstance(config, GcpConnectionConfig):
            accounts = await run_cloud_cli(
                "gcloud",
                ["auth", "list", "--filter=status:ACTIVE", "--format=json(account)", "--quiet"],
            )
            if not isinstance(accounts, list) or not accounts or not isinstance(accounts[0], dict):
                raise _CheckFailed("setup_required", "Sign in to Google Cloud on this machine.")
            account = _identity_text(accounts[0].get("account"), maximum=256)
            project = await run_cloud_cli(
                "gcloud", ["projects", "describe", config.project_id, "--format=json", "--quiet"]
            )
            if (
                not isinstance(project, dict)
                or project.get("projectId") != config.project_id
                or project.get("lifecycleState") != "ACTIVE"
            ):
                raise _CheckFailed(
                    "setup_required", "Select an active Google Cloud project you can access."
                )
            return CloudIdentity(account=account)
        raise ValueError("Configuration does not match the provider")

    async def _connect(
        self, provider: Provider, config: ConnectionConfig, *, record_failure: bool = False
    ) -> CloudConnection:
        if provider != "gcp" or not isinstance(config, GcpConnectionConfig):
            raise ValueError("Configuration does not match the provider")
        state = self._base(provider)
        state.config = config
        state.checked_at = datetime.now(UTC).isoformat()
        try:
            state.identity = await self._verify(provider, config)
            configs = {**self._configs, provider: config}
            self._save(configs)
            self._configs = configs
            state.status = "connected"
        except _CheckFailed as exc:
            state.status = exc.status
            state.message = str(exc)
            state.identity = None
        except OSError:
            state.status = "error"
            state.message = "Could not save the cloud connection. Check workspace permissions."
            state.identity = None
        if state.status == "connected" or record_failure:
            self._states[provider] = state
        return state.model_copy(deep=True)

    async def connect(self, provider: Provider, config: ConnectionConfig) -> CloudConnection:
        async with self._locks[provider]:
            return await self._connect(provider, config)

    async def recheck(self, provider: Provider) -> CloudConnection:
        async with self._locks[provider]:
            config = self._configs.get(provider)
            if config is None:
                return self._states.get(provider, self._base(provider)).model_copy(deep=True)
            return await self._connect(provider, config, record_failure=True)

    async def disconnect(self, provider: Provider) -> CloudConnection:
        async with self._locks[provider]:
            configs = {key: value for key, value in self._configs.items() if key != provider}
            self._save(configs)
            self._configs = configs
            self._states.pop(provider, None)
            return self._base(provider)
