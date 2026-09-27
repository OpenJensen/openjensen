"""Durable compute preferences; executable worker configuration stays operator-owned."""

import asyncio
import json
import os
import re
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from vla_platform import cloud_compute_catalog as cloud_catalog
from vla_platform.cloud_compute_catalog import CloudGpuOption, SetupCheckError
from vla_platform.cloud_connections import CloudConnections, GcpConnectionConfig
from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.lifecycle.native_profiles import NATIVE_PROFILES
from vla_platform.lifecycle.runtime import PublicRuntime, Runtime, RuntimeCatalog


class LocalComputeSettings(StrictRecord):
    enabled: bool = Field(default=False, strict=True)
    label: str = Field(default="Local machine", min_length=1, max_length=100)

    @field_validator("label")
    @classmethod
    def printable_label(cls, value: str) -> str:
        if any(not character.isprintable() for character in value):
            raise ValueError("Local compute label must contain printable characters only")
        return value


class GcpComputeSettings(StrictRecord):
    enabled: bool = Field(default=True, strict=True)
    default_gpu: str = "A100"
    disk_size_gb: int = Field(default=200, ge=100, le=2000, strict=True)
    idle_minutes: int = Field(default=10, ge=1, le=60, strict=True)

    @field_validator("default_gpu")
    @classmethod
    def supported_gpu(cls, value: str) -> str:
        gpu = cloud_catalog.GCP_GPU_BY_ID.get(value)
        if gpu is None or gpu.unsupported_reason:
            raise ValueError("Choose a supported Google Cloud GPU")
        return value


class ComputePreferences(StrictRecord):
    local: LocalComputeSettings = Field(default_factory=LocalComputeSettings)
    gcp: GcpComputeSettings = Field(default_factory=GcpComputeSettings)


class ComputeSettingsUpdate(StrictRecord):
    local: LocalComputeSettings | None = None
    gcp: GcpComputeSettings | None = None

    @model_validator(mode="after")
    def at_least_one_setting(self):
        if self.local is None and self.gcp is None:
            raise ValueError("Provide local or Google Cloud compute settings")
        return self


class GcpComputeStatus(StrictRecord):
    status: Literal["unchecked", "ready", "setup_required", "error"] = "unchecked"
    configured: bool = False
    skypilot_installed: bool = False
    project_id: str | None = None
    region: str | None = None
    workspace: str | None = None
    sky_api_endpoint: str | None = None
    checked_at: str | None = None
    message: str
    setup_commands: list[str] = Field(default_factory=list)


class ComputeSettingsResponse(ComputePreferences):
    runtimes: list[PublicRuntime]
    gcp_status: GcpComputeStatus
    gpu_options: list[CloudGpuOption]


class _SavedPreferences(ComputePreferences):
    version: Literal[1] = 1


class ComputeSettings:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "compute-settings.json"
        self._lock = threading.Lock()
        self._check_lock = asyncio.Lock()
        self._launch_prepare_lock = asyncio.Lock()
        self._gcp_status: GcpComputeStatus | None = None
        self._gcp_checked_config: GcpConnectionConfig | None = None
        self._gcp_checked_stamp: tuple[int, int, int, int, int, str | None] | None = None
        self._gcp_revision = 0
        self._gcp_offerings: set[str] = set()
        self._preferences = ComputePreferences()
        if self.path.exists():
            # Do not silently re-enable a disabled machine if its settings are corrupt.
            document = json.loads(self.path.read_text())
            migrate_gpu = (
                isinstance(document, dict)
                and isinstance(document.get("gcp"), dict)
                and document["gcp"].get("default_gpu") == "A100-80GB"
            )
            if migrate_gpu:
                document["gcp"]["default_gpu"] = "A100"
            saved = _SavedPreferences.model_validate(document)
            self._preferences = ComputePreferences(local=saved.local, gcp=saved.gcp)
            if migrate_gpu:
                self.update(self._preferences)
        self._restore_failure()

    def preferences(self) -> ComputePreferences:
        with self._lock:
            return self._preferences.model_copy(deep=True)

    def enabled(self, runtime: Runtime) -> bool:
        if runtime.execution == "skypilot":
            gpu_id = runtime.id.removeprefix(cloud_catalog.RUNTIME_PREFIX)
            return any(option.id == gpu_id and option.launchable for option in self.gpu_options())
        return runtime.provider != "local" or self.preferences().local.enabled

    def require_enabled(self, runtime: Runtime) -> None:
        if not self.enabled(runtime):
            if runtime.execution == "skypilot":
                gpu_id = runtime.id.removeprefix(cloud_catalog.RUNTIME_PREFIX)
                option = next(
                    (option for option in self.gpu_options() if option.id == gpu_id), None
                )
                raise ValueError(
                    option.unavailable_reason if option else "Unknown Google Cloud GPU target"
                )
            raise ValueError("Local runs are disabled. Enable local compute in Settings.")

    def available_catalog(self, catalog: RuntimeCatalog) -> RuntimeCatalog:
        return catalog.model_copy(
            update={
                "runtimes": [
                    runtime
                    for runtime in [*catalog.runtimes, *self.cloud_runtimes()]
                    if self.enabled(runtime)
                ]
            }
        )

    def public_catalog(self, catalog: RuntimeCatalog) -> dict:
        local = self.preferences().local
        public = catalog.public(local_enabled=local.enabled, local_label=local.label)
        cloud_options = {option.id: option for option in self.gpu_options()}
        for runtime in self.cloud_runtimes():
            option = cloud_options[runtime.accelerator]
            value = RuntimeCatalog(runtimes=[runtime]).public()["runtimes"][0]
            value.update(
                enabled=option.launchable,
                launchable=option.launchable,
                needs_preparation=option.needs_preparation,
                unavailable_reason=option.unavailable_reason if not option.launchable else None,
            )
            public["runtimes"].append(value)
        return public

    def public(self, catalog: RuntimeCatalog) -> ComputeSettingsResponse:
        preferences = self.preferences()
        return ComputeSettingsResponse(
            **preferences.model_dump(),
            runtimes=self.public_catalog(catalog)["runtimes"],
            gcp_status=self.gcp_status(),
            gpu_options=self.gpu_options(),
        )

    def _gcp_config(self) -> GcpConnectionConfig | None:
        connections = CloudConnections(self.path.parent).list()
        connection = next((item for item in connections.providers if item.provider == "gcp"), None)
        return connection.config if connection else None

    def _gcp_connection_stamp(self) -> tuple[int, int, int, int, int, str | None] | None:
        try:
            stat = (self.path.parent / "cloud-connections.json").stat()
            return (
                stat.st_dev,
                stat.st_ino,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_size,
                CloudConnections(self.path.parent).generation,
            )
        except OSError:
            return None

    def gcp_status(self) -> GcpComputeStatus:
        config = self._gcp_config()
        installed = cloud_catalog.sky_executable() is not None
        commands = ['uv tool install --with pip "skypilot[gcp]"'] if not installed else []
        if (
            config
            and (installed or self._gcp_status and self._gcp_status.status != "ready")
            and config == self._gcp_checked_config
            and self._gcp_connection_stamp() == self._gcp_checked_stamp
            and self._gcp_status
        ):
            return self._gcp_status.model_copy(deep=True)
        # Once a prerequisite disappears or changes, its prior verification is
        # revoked. A recreated file can reuse inode/time values on overlayfs.
        self._gcp_status = None
        self._gcp_checked_config = None
        self._gcp_checked_stamp = None
        self._gcp_offerings.clear()
        return GcpComputeStatus(
            configured=config is not None,
            skypilot_installed=installed,
            project_id=config.project_id if config else None,
            region=config.region if config else None,
            workspace=cloud_catalog.sky_workspace_name(config.project_id) if config else None,
            message=(
                "Connect Google Cloud in Settings first."
                if not config
                else "SkyPilot is missing on this server. Install its Google Cloud support."
                if not installed
                else "Your cloud GPU will be prepared automatically when training starts."
            ),
            setup_commands=commands,
        )

    def gpu_options(self) -> list[CloudGpuOption]:
        settings = self.preferences().gcp
        status = self.gcp_status()
        result = []
        for gpu in cloud_catalog.GCP_GPUS:
            launchable = bool(status.configured and settings.enabled and not gpu.unsupported_reason)
            reason = gpu.unsupported_reason
            if not reason and not settings.enabled:
                reason = "Enable Google Cloud training in Settings."
            elif not reason and status.status != "ready":
                reason = status.message
            elif not reason and gpu.id not in self._gcp_offerings:
                reason = f"SkyPilot does not offer this single-GPU machine in {status.region}."
            result.append(
                CloudGpuOption(
                    id=gpu.id,
                    label=gpu.label,
                    accelerator=gpu.id,
                    gpu_memory_mib=gpu.memory_mib,
                    instance_type=gpu.instance_type,
                    supported=gpu.unsupported_reason is None,
                    available=reason is None,
                    launchable=launchable,
                    needs_preparation=launchable and reason is not None,
                    unavailable_reason=reason,
                )
            )
        return result

    def runtime(self, runtime_id: str) -> Runtime | None:
        if not runtime_id.startswith(cloud_catalog.RUNTIME_PREFIX):
            return None
        gpu = cloud_catalog.GCP_GPU_BY_ID.get(runtime_id.removeprefix(cloud_catalog.RUNTIME_PREFIX))
        if gpu is None:
            return None
        config = self._gcp_config()
        return Runtime(
            id=runtime_id,
            label=gpu.label,
            execution="skypilot",
            provider="gcp",
            region=config.region if config else None,
            accelerator=gpu.id,
            gpu_count=1,
            device="cuda",
            gpu_name=gpu.label,
            gpu_memory_mib=gpu.memory_mib,
            # Controlled bundle paths are interpreted by the SkyPilot runner.
            # command() rejects these targets so they cannot execute locally.
            worker_root="workers/vla_cpp",
            training_python="python",
            training_root="workers/smolvla_qlora",
            training_model_ids=["smolvla", *NATIVE_PROFILES, "psi0"],
            vendor="skypilot",
            build="skypilot",
        )

    def cloud_runtimes(self) -> list[Runtime]:
        if self._gcp_config() is None:
            return []
        return [
            self.runtime(cloud_catalog.RUNTIME_PREFIX + gpu.id) for gpu in cloud_catalog.GCP_GPUS
        ]

    def cloud_target(self, runtime_id: str) -> dict:
        runtime = self.runtime(runtime_id)
        if runtime is None:
            raise ValueError("Unknown Google Cloud GPU target")
        self.require_enabled(runtime)
        option = next(option for option in self.gpu_options() if option.id == runtime.accelerator)
        if not option.available:
            raise ValueError(option.unavailable_reason or "The selected GPU is not prepared yet.")
        config = self._gcp_config()
        if config is None:
            raise ValueError("Connect Google Cloud in Settings first.")
        settings = self.preferences().gcp
        gpu = cloud_catalog.GCP_GPU_BY_ID[runtime.accelerator]
        return {
            "project_id": config.project_id,
            "workspace": cloud_catalog.sky_workspace_name(config.project_id),
            "sky_api_endpoint": self.gcp_status().sky_api_endpoint,
            "region": config.region,
            "accelerator": gpu.id,
            "gpu_count": 1,
            "instance_type": gpu.instance_type,
            "disk_size_gb": settings.disk_size_gb,
            "idle_minutes": settings.idle_minutes,
        }

    async def plan_cloud_target(self, runtime_id: str) -> dict:
        """Capture a server-owned selection quickly, without preparing or provisioning."""
        runtime = self.runtime(runtime_id)
        if runtime is None:
            raise ValueError("Unknown Google Cloud GPU target")
        self.require_enabled(runtime)
        config = self._gcp_config()
        if config is None:
            raise ValueError("Connect Google Cloud in Settings first.")
        sky = cloud_catalog.sky_executable()
        if not sky:
            raise ValueError(
                "SkyPilot is missing on the application server. Install its Google Cloud support."
            )
        endpoint = await cloud_catalog.configured_sky_endpoint(sky)
        if self._gcp_config() != config:
            raise ValueError(
                "Google Cloud settings changed. Start again with the current selection."
            )
        settings = self.preferences().gcp
        gpu = cloud_catalog.GCP_GPU_BY_ID[runtime.accelerator]
        return {
            "project_id": config.project_id,
            "workspace": cloud_catalog.sky_workspace_name(config.project_id),
            "sky_api_endpoint": endpoint,
            "region": config.region,
            "accelerator": gpu.id,
            "gpu_count": 1,
            "instance_type": gpu.instance_type,
            "disk_size_gb": settings.disk_size_gb,
            "idle_minutes": settings.idle_minutes,
        }

    async def ensure_cloud_ready(self, runtime_id: str, target: dict) -> dict:
        """Prepare exactly the queued selection after Start; never change billing targets."""
        async with self._launch_prepare_lock:
            runtime = self.runtime(runtime_id)
            if runtime is None:
                raise ValueError("Unknown Google Cloud GPU target")
            self.require_enabled(runtime)
            config = GcpConnectionConfig(project_id=target["project_id"], region=target["region"])
            gpu = cloud_catalog.GCP_GPU_BY_ID[runtime.accelerator]
            if (
                self._gcp_config() != config
                or target.get("workspace") != cloud_catalog.sky_workspace_name(config.project_id)
                or target.get("accelerator") != gpu.id
                or target.get("instance_type") != gpu.instance_type
                or target.get("gpu_count") != 1
            ):
                raise ValueError(
                    "Google Cloud settings differ from this queued run. "
                    "Start again with the current selection."
                )
            endpoint = cloud_catalog.safe_sky_endpoint(target["sky_api_endpoint"])
            status = self.gcp_status()
            if status.status != "ready" or status.sky_api_endpoint != endpoint:
                status = await self.prepare_gcp(sky_api_endpoint=endpoint, expected_config=config)
            if status.status != "ready":
                raise ValueError(status.message)
            self.require_enabled(runtime)
            option = next(option for option in self.gpu_options() if option.id == gpu.id)
            if not option.available:
                raise ValueError(
                    option.unavailable_reason or "The selected GPU is not offered in this region."
                )
            if self._gcp_config() != config:
                raise ValueError(
                    "Google Cloud settings changed while preparing this run. Start again."
                )
            return dict(target)

    async def check_gcp(
        self,
        *,
        sky_api_endpoint: str | None = None,
        expected_config: GcpConnectionConfig | None = None,
    ) -> GcpComputeStatus:
        async with self._check_lock:
            config = self._gcp_config()
            connection_stamp = self._gcp_connection_stamp()
            revision = self._gcp_revision
            state = self.gcp_status()
            state.status = "setup_required"
            state.checked_at = datetime.now(UTC).isoformat()
            state.setup_commands = []
            self._gcp_offerings = set()
            sky = cloud_catalog.sky_executable()
            gcloud = shutil.which("gcloud")
            try:
                if config is None:
                    raise SetupCheckError("Connect Google Cloud in Settings first.")
                if expected_config is not None and config != expected_config:
                    raise SetupCheckError(
                        "Google Cloud settings changed. Start again with the current selection."
                    )
                if not sky:
                    state.setup_commands = ['uv tool install --with pip "skypilot[gcp]"']
                    raise SetupCheckError("Install SkyPilot with Google Cloud support first.")
                python = cloud_catalog.sky_python(sky)
                if not python:
                    state.setup_commands = ['uv tool install --force --with pip "skypilot[gcp]"']
                    raise SetupCheckError(
                        "Could not locate SkyPilot's Python environment. Reinstall SkyPilot."
                    )
                if not gcloud:
                    raise SetupCheckError("Install Google Cloud CLI and sign in first.")
                try:
                    await cloud_catalog.run_readonly(
                        python,
                        [
                            "-c",
                            "import google.auth; import googleapiclient.discovery; "
                            "import google.cloud.storage",
                        ],
                        timeout=10,
                    )
                except SetupCheckError:
                    state.setup_commands = ['uv tool install --force --with pip "skypilot[gcp]"']
                    raise SetupCheckError(
                        "Install SkyPilot's Google Cloud dependencies, then check again."
                    ) from None
                try:
                    token = await cloud_catalog.run_readonly(
                        gcloud,
                        ["auth", "application-default", "print-access-token", "--quiet"],
                        timeout=20,
                    )
                    if len(token) > 16384 or not re.fullmatch(
                        rb"[A-Za-z0-9._~+/=-]+", token.strip()
                    ):
                        raise SetupCheckError("Invalid application credentials")
                    del token
                except SetupCheckError:
                    state.setup_commands = ["gcloud auth application-default login"]
                    raise SetupCheckError(
                        "Google Cloud application credentials are missing or expired. "
                        "Run gcloud auth application-default login."
                    ) from None
                sky_target = await cloud_catalog.verify_sky_target(
                    sky, config.project_id, endpoint=sky_api_endpoint
                )
                state.workspace = sky_target["workspace"]
                state.sky_api_endpoint = sky_target["sky_api_endpoint"]
                try:
                    services = json.loads(
                        await cloud_catalog.run_readonly(
                            gcloud,
                            [
                                "services",
                                "list",
                                "--enabled",
                                "--project",
                                config.project_id,
                                "--format=json(config.name)",
                                "--quiet",
                            ],
                            timeout=20,
                        )
                    )
                    names = {
                        entry.get("config", {}).get("name")
                        for entry in services
                        if isinstance(entry, dict)
                    }
                    if not cloud_catalog.REQUIRED_GCP_SERVICES.issubset(names):
                        raise ValueError("Missing GCP APIs")
                except SetupCheckError, ValueError, TypeError, AttributeError:
                    state.setup_commands = [f"sky check --workspace {state.workspace} gcp"]
                    raise SetupCheckError(
                        "Required Google Cloud APIs are not enabled or visible. "
                        "Complete sky check gcp for this project, then retry."
                    ) from None
                catalog = await cloud_catalog.run_readonly(
                    sky,
                    [
                        "gpus",
                        "list",
                        "--all",
                        "--infra",
                        f"gcp/{config.region}",
                        "--output",
                        "json",
                    ],
                    timeout=40,
                    env_overrides={"SKYPILOT_API_SERVER_ENDPOINT": state.sky_api_endpoint},
                )
                self._gcp_offerings = cloud_catalog.regional_gpu_offerings(catalog, config.region)
                if not self._gcp_offerings:
                    raise SetupCheckError(
                        "No supported single-GPU GCP machines are listed in this region. "
                        "Choose another region."
                    )
                if (
                    self._gcp_config() != config
                    or self._gcp_connection_stamp() != connection_stamp
                    or self._gcp_revision != revision
                ):
                    raise SetupCheckError(
                        "Google Cloud settings changed during the check. "
                        "Check the new settings again."
                    )
                state.status = "ready"
                state.message = (
                    "SkyPilot setup verified. GPU quota, capacity, and billing "
                    "are checked when you launch."
                )
            except SetupCheckError as exc:
                state.message = str(exc)
            self._gcp_checked_config = config
            self._gcp_checked_stamp = connection_stamp
            self._gcp_status = state
            self._persist_failure(state, config, connection_stamp)
            return state.model_copy(deep=True)

    async def prepare_gcp(
        self,
        *,
        sky_api_endpoint: str | None = None,
        expected_config: GcpConnectionConfig | None = None,
    ) -> GcpComputeStatus:
        """Prepare a project-pinned workspace as part of the requested training run."""
        async with self._check_lock:
            config = self._gcp_config()
            state = self.gcp_status()
            state.status = "setup_required"
            state.checked_at = datetime.now(UTC).isoformat()
            state.setup_commands = []
            self._gcp_offerings = set()
            try:
                if config is None:
                    raise SetupCheckError("Connect Google Cloud in Settings first.")
                if expected_config is not None and config != expected_config:
                    raise SetupCheckError(
                        "Google Cloud settings changed. Start again with the current selection."
                    )
                sky = cloud_catalog.sky_executable()
                if not sky:
                    state.setup_commands = ['uv tool install --with pip "skypilot[gcp]"']
                    raise SetupCheckError("Install SkyPilot with Google Cloud support first.")
                sky_target = await cloud_catalog.prepare_sky_target(
                    sky, config.project_id, endpoint=sky_api_endpoint
                )
                if self._gcp_config() != config:
                    raise SetupCheckError(
                        "Google Cloud settings changed while preparing. Check the new settings."
                    )
            except SetupCheckError as exc:
                state.message = str(exc)
                self._gcp_checked_config = config
                self._gcp_checked_stamp = self._gcp_connection_stamp()
                self._gcp_status = state
                self._persist_failure(state, config, self._gcp_checked_stamp)
                return state.model_copy(deep=True)
        return await self.check_gcp(
            sky_api_endpoint=sky_target["sky_api_endpoint"], expected_config=config
        )

    def invalidate_cloud_check(self) -> None:
        """Revoke cached and in-flight checks when an operator changes connection state."""
        self._gcp_revision += 1
        self._gcp_status = None
        self._gcp_checked_config = None
        self._gcp_checked_stamp = None
        self._gcp_offerings.clear()

    def enable_cloud(self) -> None:
        preferences = self.preferences()
        preferences.gcp.enabled = True
        self.update(ComputeSettingsUpdate(gcp=preferences.gcp))
        self.invalidate_cloud_check()
        (self.path.parent / "compute-last-failure.json").unlink(missing_ok=True)

    def _persist_failure(self, state, config, stamp) -> None:
        path = self.path.parent / "compute-last-failure.json"
        if state.status == "ready":
            path.unlink(missing_ok=True)
            return
        if config is None:
            return
        payload = {
            "version": 1,
            "config": config.model_dump(),
            "connection_stamp": list(stamp) if stamp else None,
            "status": state.model_dump(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".compute-failure-", dir=self.path.parent)
        try:
            with os.fdopen(handle, "w") as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _restore_failure(self) -> None:
        path = self.path.parent / "compute-last-failure.json"
        try:
            if not path.exists() or path.stat().st_size > 65536:
                return
            saved = json.loads(path.read_text())
            config = GcpConnectionConfig.model_validate(saved["config"])
            state = GcpComputeStatus.model_validate(saved["status"])
            stamp = tuple(saved["connection_stamp"] or ())
            if (
                saved.get("version") != 1
                or state.status == "ready"
                or config != self._gcp_config()
                or stamp != self._gcp_connection_stamp()
            ):
                return
            self._gcp_checked_config = config
            self._gcp_checked_stamp = stamp
            self._gcp_status = state
        except OSError, ValueError, KeyError, TypeError:
            return

    def update(self, preferences: ComputePreferences | ComputeSettingsUpdate) -> None:
        with self._lock:
            if isinstance(preferences, ComputeSettingsUpdate):
                preferences = ComputePreferences(
                    local=preferences.local or self._preferences.local,
                    gcp=preferences.gcp or self._preferences.gcp,
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = _SavedPreferences(**preferences.model_dump()).model_dump()
            handle, temporary = tempfile.mkstemp(prefix=".compute-settings-", dir=self.path.parent)
            try:
                with os.fdopen(handle, "w") as file:
                    json.dump(payload, file, indent=2)
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
                self._preferences = preferences.model_copy(deep=True)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
