"""Declarative compute destinations independent of workload operation recipes.

The registry stores no commands, environment variables or credentials. Provider
sign-in and API-server connections remain operator-owned. Reading or validating a
target never allocates compute.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from filelock import FileLock
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from vla_platform.compute.recipes import PORTABLE_RECIPE_IDS, recipe_catalog

TargetId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{0,62}$")]
Reference = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")]
Infrastructure = Literal["local", "ssh", "gcp", "runpod", "nebius", "slurm"]
POLICY_OPERATIONS = frozenset(
    {
        "policy.finetune",
        "policy.distill",
        "policy.quantize",
        "policy.export",
        "policy.import",
        "policy.evaluate",
        "policy.run",
        "policy.workflow",
    }
)
REMOTE_OPERATIONS = frozenset(
    {"policy.finetune", "policy.quantize", "policy.evaluate", "policy.run"}
)
MAX_RECORD_BYTES = 1024 * 1024
MAX_TARGETS = 128


class TargetError(ValueError):
    def __init__(self, message, status=422):
        self.status = status
        super().__init__(message)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class TargetResources(Strict):
    cpus: int = Field(default=2, ge=1, le=1024, strict=True)
    memory_gib: int = Field(default=4, ge=1, le=65536, strict=True)
    accelerator: Reference | None = None
    gpu_count: int = Field(default=0, ge=0, le=1, strict=True)
    disk_gib: int = Field(default=100, ge=10, le=65536, strict=True)
    timeout_seconds: int = Field(default=7200, ge=30, le=604800, strict=True)
    concurrency: int = Field(default=1, ge=1, le=128, strict=True)

    @model_validator(mode="after")
    def paired_gpu(self):
        if bool(self.accelerator) != bool(self.gpu_count):
            raise ValueError("Select an accelerator and GPU count together")
        return self


class ComputeTarget(Strict):
    id: TargetId
    label: str = Field(min_length=1, max_length=100)
    adapter: Literal["local", "skypilot"] = "local"
    infrastructure: Infrastructure = "local"
    enabled: bool = Field(default=False, strict=True)
    runtime_id: Reference
    connection_ref: Reference | None = None
    location: Reference | None = None
    partition: Reference | None = None
    account: Reference | None = None
    qos: Reference | None = None
    resources: TargetResources = Field(default_factory=TargetResources)
    operations: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("label")
    @classmethod
    def readable(cls, value):
        if any(not character.isprintable() for character in value):
            raise ValueError("Target labels must contain printable characters")
        return value

    @field_validator("operations")
    @classmethod
    def known_operations(cls, value):
        if len(set(value)) != len(value) or not set(value).issubset(POLICY_OPERATIONS):
            raise ValueError("Choose unique registered workload operations")
        return value

    @model_validator(mode="after")
    def matching_adapter(self):
        if (self.adapter == "local") != (self.infrastructure == "local"):
            raise ValueError("Local destinations require the local adapter")
        if self.adapter == "local" and any(
            (self.connection_ref, self.location, self.partition, self.account, self.qos)
        ):
            raise ValueError("Local destinations do not use remote connection settings")
        if self.infrastructure != "slurm" and any((self.partition, self.account, self.qos)):
            raise ValueError("Partition, account and QoS apply only to Slurm")
        if self.infrastructure in {"ssh", "slurm"} and not self.location:
            raise ValueError("Choose the registered node pool or cluster")
        return self


class TargetSnapshot(Strict):
    definition: ComputeTarget
    configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    connection_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class PublicComputeTarget(Strict):
    definition: ComputeTarget
    operations: list[str]
    status: Literal["disabled", "setup_required", "configured", "error"]
    message: str
    launchable: bool
    checked_at: str | None = None
    sha256: str


class TargetRuntimeOption(Strict):
    id: str
    label: str
    operations: list[str]
    device: Literal["cpu", "cuda"] = "cpu"
    accelerator: str | None = None
    remote_only: bool = False


class ComputeTargetList(Strict):
    schema_version: Literal[1] = 1
    targets: list[PublicComputeTarget]
    runtime_options: list[TargetRuntimeOption]
    connection_refs: list[str]


class _Registry(Strict):
    schema_version: Literal[1] = 1
    revision: int = Field(default=0, ge=0, strict=True)
    targets: list[ComputeTarget] = Field(default_factory=list, max_length=MAX_TARGETS)
    checks: dict[str, dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique(self):
        if len({target.id for target in self.targets}) != len(self.targets):
            raise ValueError("Duplicate target identity")
        return self


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_operations(runtime, *, remote=False):
    if runtime is None:
        return []
    specialized = any(
        (
            runtime.native_quantization_only,
            runtime.native_replay_only,
            runtime.native_distillation_only,
            runtime.export_only,
        )
    )
    if remote and specialized:
        return []
    operations = set()
    if runtime.training_python and runtime.training_root:
        operations.add("policy.finetune")
    if runtime.training_only:
        return sorted(operations)
    if runtime.native_distillation_python and not remote:
        operations.add("policy.distill")
    if runtime.native_quantization_python and not remote:
        operations.add("policy.quantize")
    if runtime.native_replay_python and not remote:
        operations.add("policy.run")
    if runtime.act_export_python and not remote:
        operations.add("policy.export")
    if not specialized:
        # The regular worker implements conversion and inference protocols.
        if runtime.worker_root or runtime.conversion_python or runtime.conversion_image:
            operations.update(
                {"policy.import", "policy.quantize", "policy.export", "policy.workflow"}
            )
        if runtime.worker_root or runtime.evaluation_python or runtime.evaluation_image:
            operations.update({"policy.evaluate", "policy.run"})
    return sorted(operations.intersection(REMOTE_OPERATIONS) if remote else operations)


class ComputeTargets:
    def __init__(self, data_dir):
        self.directory = Path(data_dir)
        self.path = self.directory / "compute-targets.json"
        self.connection_path = self.directory / "compute-connections.json"
        self.lock = FileLock(self.directory / "compute-targets.lock")
        self._check_lock = asyncio.Lock()

    def _read(self):
        if not self.path.exists():
            return _Registry()
        if self.path.is_symlink() or self.path.stat().st_size > MAX_RECORD_BYTES:
            raise TargetError("Saved compute targets need administrator attention", 503)
        try:
            return _Registry.model_validate_json(self.path.read_bytes())
        except OSError, ValueError:
            raise TargetError("Saved compute targets need administrator attention", 503) from None

    def _write(self, value):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise TargetError("Saved compute targets need administrator attention", 503)
        raw = value.model_dump_json(indent=2).encode() + bytes([10])
        if len(raw) > MAX_RECORD_BYTES:
            raise TargetError("Compute target registry exceeds its supported size")
        fd, name = tempfile.mkstemp(prefix=".compute-targets-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def _connections(self):
        if not self.connection_path.exists():
            return {}
        try:
            if (
                self.connection_path.is_symlink()
                or self.connection_path.stat().st_size > MAX_RECORD_BYTES
            ):
                raise ValueError
            value = json.loads(self.connection_path.read_bytes())
            if value.get("schema_version") != 1 or not isinstance(value.get("connections"), dict):
                raise ValueError
            if len(value["connections"]) > MAX_TARGETS or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name)
                for name in value["connections"]
            ):
                raise ValueError
            return value["connections"]
        except OSError, ValueError, TypeError:
            raise TargetError(
                "Compute connection references need administrator attention", 503
            ) from None

    def connection(self, target):
        if not target.connection_ref:
            return {}
        value = self._connections().get(target.connection_ref)
        if not isinstance(value, dict):
            raise TargetError("The selected compute connection reference is not configured")
        if set(value) - {"sky_api_endpoint", "workspace", "project_id"}:
            raise TargetError("The compute connection reference contains unsupported fields")
        endpoint = value.get("sky_api_endpoint")
        try:
            parsed = urlsplit(endpoint) if isinstance(endpoint, str) else None
        except ValueError:
            raise TargetError("The compute connection reference has an invalid endpoint") from None
        if (
            not parsed
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in endpoint)
        ):
            raise TargetError("The compute connection reference needs an API server endpoint")
        try:
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError
        except ValueError:
            raise TargetError("The compute connection reference has an invalid endpoint") from None
        workspace = value.get("workspace", "default")
        if not isinstance(workspace, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", workspace):
            raise TargetError("The compute connection workspace is invalid")
        project = value.get("project_id")
        if project is not None and (
            not isinstance(project, str)
            or not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project)
        ):
            raise TargetError("The compute connection billing project is invalid")
        resolved = {**value, "sky_api_endpoint": endpoint.rstrip("/")}
        # A missing workspace is resolved by the selected infrastructure at admission.
        # In particular, cloud billing projects can require a project-specific workspace.
        if "workspace" in value:
            resolved["workspace"] = workspace
        return resolved

    def get(self, ident):
        value = next((target for target in self._read().targets if target.id == ident), None)
        if value is None:
            raise TargetError("Compute target not found", 404)
        return value

    def save(self, target):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock:
            current = self._read()
            existing = [item for item in current.targets if item.id != target.id]
            if len(existing) >= MAX_TARGETS:
                raise TargetError("This workspace has reached its compute target limit")
            checks = {key: value for key, value in current.checks.items() if key != target.id}
            self._write(
                _Registry(revision=current.revision + 1, targets=[*existing, target], checks=checks)
            )
        return target

    def remove(self, ident):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock:
            current = self._read()
            if not any(target.id == ident for target in current.targets):
                raise TargetError("Compute target not found", 404)
            self._write(
                _Registry(
                    revision=current.revision + 1,
                    targets=[t for t in current.targets if t.id != ident],
                    checks={k: v for k, v in current.checks.items() if k != ident},
                )
            )

    def _compatible(self, target, catalog):
        catalog = recipe_catalog(catalog)
        runtime = catalog.runtime(target.runtime_id)
        if runtime is None:
            return [], "Register the selected worker runtime before using this target."
        if runtime.id in PORTABLE_RECIPE_IDS:
            if target.adapter != "skypilot":
                return [], "Portable worker recipes require a remote compute target."
        elif runtime.execution != "native":
            return [], "Select an installed worker recipe for this compute target."
        operations = runtime_operations(runtime, remote=target.adapter == "skypilot")
        if target.operations:
            if not set(target.operations).issubset(operations):
                return [], "The selected runtime does not support every allowed operation."
            operations = sorted(set(operations).intersection(target.operations))
        if not operations:
            return [], "This runtime has no compatible operations on the selected destination."
        if target.adapter == "local" and runtime.provider != "local":
            return [], "Local targets require a local worker runtime."
        if runtime.device == "cuda" and target.resources.gpu_count != 1:
            return [], "The selected runtime requires one GPU."
        if target.adapter == "skypilot" and target.resources.gpu_count != 1:
            return [], "The portable remote worker currently requires one GPU."
        try:
            self.connection(target)
        except TargetError as exc:
            return [], str(exc)
        return operations, None

    def fingerprint(self, target):
        connection = self.connection(target)
        return digest(
            {
                "definition": target.model_dump(mode="json"),
                "connection_sha256": digest(connection) if connection else None,
            }
        )

    def public(self, target, catalog):
        operations, problem = self._compatible(target, catalog)
        try:
            fingerprint = self.fingerprint(target)
        except TargetError:
            fingerprint = digest(target.model_dump(mode="json"))
        check = self._read().checks.get(target.id, {})
        if check.get("sha256") != fingerprint:
            check = {}
        if not target.enabled:
            status = "disabled"
            detail = problem or (check.get("message") if check.get("status") == "error" else None)
            message = f"Disabled. {detail}" if detail else "Enable this target to allow jobs."
        elif problem:
            status, message = "setup_required", problem
        elif target.adapter == "skypilot" and not self.sky_executable():
            status, message = (
                "setup_required",
                "Install SkyPilot with support for this infrastructure.",
            )
        elif check.get("status") == "error":
            status, message = "error", check.get("message", "The last connection check failed.")
        else:
            status, message = (
                "configured",
                (
                    (
                        "Configuration is valid. Provider capacity and worker readiness "
                        "are checked when a job starts."
                    )
                    if target.adapter == "skypilot"
                    else (
                        "Configuration is valid. The job validates worker readiness "
                        "before execution."
                    )
                ),
            )
            if check.get("status") == "configured":
                message = check.get("message", message)
        return PublicComputeTarget(
            definition=target,
            operations=operations,
            status=status,
            message=message,
            launchable=status == "configured",
            checked_at=check.get("checked_at"),
            sha256=fingerprint,
        )

    def list(self, catalog):
        catalog = recipe_catalog(catalog)
        saved = self._read()
        return ComputeTargetList(
            targets=[self.public(target, catalog) for target in saved.targets],
            runtime_options=[
                TargetRuntimeOption(
                    id=runtime.id,
                    label=runtime.label,
                    operations=runtime_operations(
                        runtime, remote=runtime.id in PORTABLE_RECIPE_IDS
                    ),
                    device=runtime.device,
                    accelerator=getattr(runtime, "accelerator", None),
                    remote_only=runtime.id in PORTABLE_RECIPE_IDS,
                )
                for runtime in catalog.runtimes
                if runtime.execution == "native" or runtime.id in PORTABLE_RECIPE_IDS
            ],
            connection_refs=sorted(self._connections()),
        )

    def snapshot(self, ident, operation, catalog, *, require_checked=False):
        target = self.get(ident)
        public = self.public(target, catalog)
        if not public.launchable:
            raise TargetError(public.message, 409)
        if operation not in public.operations:
            raise TargetError("The compute target does not support this operation", 409)
        if require_checked and not public.checked_at:
            raise TargetError("Check this compute target before submitting work", 409)
        connection = self.connection(target)
        return TargetSnapshot(
            definition=target,
            configuration_sha256=public.sha256,
            connection_sha256=digest(connection) if connection else None,
        )

    @staticmethod
    def sky_executable():
        found = shutil.which("sky")
        fallback = Path.home() / ".local/bin/sky"
        return found or (
            str(fallback) if fallback.is_file() and os.access(fallback, os.X_OK) else None
        )

    async def check(self, ident, catalog):
        async with self._check_lock:
            return await self._check(ident, catalog)

    async def _check(self, ident, catalog):
        target = self.get(ident)
        operations, problem = self._compatible(target, catalog)
        if problem:
            status, message = "error", problem
        elif target.adapter == "local":
            status, message = (
                "configured",
                (
                    "The registered local worker supports this target's operations. "
                    "Execution validates its environment."
                ),
            )
        else:
            executable = self.sky_executable()
            if not executable:
                status, message = "error", "Install SkyPilot with support for this infrastructure."
            else:
                status, message = await self._check_provider(executable, target)
        try:
            fingerprint = self.fingerprint(target)
        except TargetError:
            fingerprint = digest(target.model_dump(mode="json"))
        checked = {
            "sha256": fingerprint,
            "status": status,
            "message": message,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock:
            current = self._read()
            if next((item for item in current.targets if item.id == ident), None) == target:
                self._write(
                    current.model_copy(update={"checks": {**current.checks, ident: checked}})
                )
        return self.public(self.get(ident), catalog)

    async def _check_provider(self, executable, target):
        process = None
        reads = []
        try:
            # Fixed argv only; this verifies access and never provisions a worker.
            arguments = [executable, "check", target.infrastructure, "--output", "json"]
            connection = self.connection(target)
            env = dict(os.environ)
            if connection.get("sky_api_endpoint"):
                env["SKYPILOT_API_SERVER_ENDPOINT"] = connection["sky_api_endpoint"]
            if connection.get("workspace"):
                arguments.extend(["--workspace", connection["workspace"]])
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *arguments,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    start_new_session=os.name == "posix",
                )
            )
            interrupted = False
            while not spawn.done():
                try:
                    await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    interrupted = True
            process = spawn.result()
            if interrupted:
                raise asyncio.CancelledError

            async def bounded(stream):
                value = bytearray()
                while chunk := await stream.read(8192):
                    value.extend(chunk)
                    if len(value) > MAX_RECORD_BYTES:
                        raise ValueError("Provider check exceeded its response limit")
                return bytes(value)

            reads = [
                asyncio.create_task(bounded(process.stdout)),
                asyncio.create_task(bounded(process.stderr)),
            ]
            stdout, _ = await asyncio.wait_for(asyncio.gather(*reads), 45)
            await asyncio.wait_for(process.wait(), 5)
            if process.returncode:
                return "error", (
                    "Provider access could not be verified. "
                    "Check the configured CLI credentials and connection."
                )
            value = json.loads(stdout)
            if not self._enabled_infrastructure(value, target.infrastructure):
                return (
                    "error",
                    "Provider access is unavailable. Review its SkyPilot configuration and retry.",
                )
            return "configured", (
                "Provider access verified. Specific capacity and worker requirements "
                "are checked when a job starts."
            )
        except asyncio.CancelledError:
            raise
        except OSError, ValueError, TimeoutError:
            return "error", (
                "Provider check did not return a valid bounded response. "
                "Review the connection and retry."
            )
        finally:

            async def cleanup():
                for task in reads:
                    if not task.done():
                        task.cancel()
                if reads:
                    await asyncio.gather(*reads, return_exceptions=True)
                if process is not None:
                    try:
                        # Credential helpers may survive their CLI parent.
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        elif process.returncode is None:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), 5)
                    except TimeoutError:
                        pass

            owned = asyncio.create_task(cleanup())
            while not owned.done():
                try:
                    await asyncio.shield(owned)
                except asyncio.CancelledError:
                    continue
            owned.result()

    @staticmethod
    def _enabled_infrastructure(value, infrastructure):
        if isinstance(value, dict):
            # Accept the CLI enabled-cloud list and explicit enabled-status records.
            for key, entry in value.items():
                if str(key).lower() == infrastructure:
                    return entry is True or isinstance(entry, dict) and entry.get("enabled") is True
            enabled = value.get("enabled_clouds", value.get("enabled_infra", []))
            if isinstance(enabled, list) and infrastructure in [
                str(item).lower() for item in enabled
            ]:
                return True
            if (
                str(value.get("name", value.get("infra", ""))).lower() == infrastructure
                and value.get("enabled") is True
            ):
                return True
            return any(
                ComputeTargets._enabled_infrastructure(entry, infrastructure)
                for entry in value.values()
                if isinstance(entry, (dict, list))
            )
        if isinstance(value, list):
            return infrastructure in [
                str(entry).lower() for entry in value if isinstance(entry, str)
            ] or any(
                ComputeTargets._enabled_infrastructure(entry, infrastructure) for entry in value
            )
        return False
