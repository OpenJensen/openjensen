"""Discover only the bundled, installed SmolVLA worker on this application host."""

import asyncio
import csv
import hashlib
import io
import json
import os
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.lifecycle.runtime import Runtime

PROBE_TIMEOUT = 30
HARDWARE_TIMEOUT = 5
MAX_OUTPUT = 64 * 1024
CANDIDATE_TTL = 300
REASONS = {
    "missing_dependencies": "Install the pinned SmolVLA worker dependencies.",
    "incompatible_dependencies": "The installed worker dependencies do not match SmolVLA's pins.",
    "incompatible_environment": "The SmolVLA worker needs its own Python 3.11 or 3.12 environment.",
    "cuda_unavailable": "CUDA is unavailable to the installed worker.",
    "unsupported_gpu": "SmolVLA requires an NVIDIA GPU with compute capability 7.0 or newer.",
    "dependency_import_failed": "The installed worker could not load its training dependencies.",
    "probe_failed": "The installed worker could not complete its readiness check.",
}


class LocalWorkerHost(StrictRecord):
    name: str
    platform: str
    architecture: str


class LocalWorkerCandidate(StrictRecord):
    id: str
    label: str
    gpu_name: str | None = None
    gpu_memory_mib: int | None = None
    training_model_ids: list[str] = Field(default_factory=lambda: ["smolvla"])
    status: Literal["ready", "registered", "setup_required"]
    runtime_id: str | None = None
    reason: str | None = None


class LocalWorkerDiscovery(StrictRecord):
    host: LocalWorkerHost
    checked_at: str
    status: Literal["ready", "unavailable", "error"]
    message: str | None = None
    candidates: list[LocalWorkerCandidate] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class _ProbeError(Exception):
    pass


@dataclass(frozen=True)
class _ReadyCandidate:
    runtime: Runtime
    visibility: str | None
    device_order: str | None
    checked_at: float


def _environment():
    # Deliberately exclude credentials, PYTHONPATH and arbitrary package indexes.
    names = (
        "PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
    )
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        WANDB_MODE="disabled",
    )
    return env


def _run_bounded(argv, *, timeout, env):
    """Bound time and captured bytes; discard stderr rather than publish private diagnostics."""
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _ProbeError("The local worker check timed out.")
                if not selector.select(remaining):
                    raise _ProbeError("The local worker check timed out.")
                chunk = os.read(process.stdout.fileno(), min(8192, MAX_OUTPUT + 1 - len(output)))
                if not chunk:
                    selector.unregister(process.stdout)
                    break
                output.extend(chunk)
                if len(output) > MAX_OUTPUT:
                    raise _ProbeError("The local worker returned an invalid readiness response.")
        remaining = deadline - time.monotonic()
        if remaining <= 0 or process.wait(timeout=remaining):
            raise _ProbeError("The local worker check did not finish successfully.")
        return bytes(output)
    except subprocess.TimeoutExpired as error:
        raise _ProbeError("The local worker check timed out.") from error
    finally:
        # Imports must not leave background children behind, even when the
        # probe parent exited successfully and closed its stdout pipe.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


def _worker_root():
    if getattr(sys, "frozen", False):
        return None
    source = Path(__file__).resolve()
    if len(source.parents) < 5:
        return None
    repository = source.parents[4]
    if repository / "packages/core/src/vla_platform/local_worker_discovery.py" != source:
        return None
    root = repository / "workers/smolvla_qlora"
    if not (repository / "pyproject.toml").is_file() or not all(
        (root / name).is_file() for name in ("pyproject.toml", "src/firebird_vla/application.py")
    ):
        return None
    return root


def _safe_text(value, maximum=200):
    return (
        isinstance(value, str)
        and 0 < len(value) <= maximum
        and all(character.isprintable() for character in value)
    )


def _hardware(env):
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        raw = _run_bounded(
            [
                executable,
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=HARDWARE_TIMEOUT,
            env=env,
        )
        visibility = env.get("CUDA_VISIBLE_DEVICES")
        allowed = visibility.split(",") if visibility is not None else None
        devices = []
        for row in csv.reader(io.StringIO(raw.decode("utf-8"))):
            if len(row) != 4:
                return None
            index, uuid, name, memory = (value.strip() for value in row)
            if not index.isdigit() or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
                return None
            if not _safe_text(name) or not memory.isdigit() or not 0 < int(memory) < 2**24:
                return None
            devices.append((index, uuid, name, int(memory)))
        if allowed is None:
            return devices[0] if devices else None
        for selected in allowed:
            matches = [row for row in devices if row[0] == selected or row[1] == selected]
            if matches:
                return matches[0]
        return None
    except OSError, UnicodeError, _ProbeError:
        return None


def _parse_probe(raw):
    def unique_pairs(items):
        value = {}
        for key, entry in items:
            if key in value:
                raise ValueError("Duplicate readiness field")
            value[key] = entry
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "status",
            "reason",
            "gpu",
        }:
            raise ValueError
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError
        if value["status"] == "unavailable":
            if value["reason"] not in REASONS or value["gpu"] is not None:
                raise ValueError
            return value
        if value["status"] != "ready" or value["reason"] is not None:
            raise ValueError
        gpu = value["gpu"]
        if not isinstance(gpu, dict) or set(gpu) != {"name", "memory_mib", "uuid", "index"}:
            raise ValueError
        if (
            not _safe_text(gpu["name"])
            or type(gpu["memory_mib"]) is not int
            or not 0 < gpu["memory_mib"] < 2**24
            or type(gpu["index"]) is not int
            or gpu["index"] != 0
            or (
                gpu["uuid"] is not None and not re.fullmatch(r"GPU-[A-Za-z0-9-]{1,80}", gpu["uuid"])
            )
        ):
            raise ValueError
        return value
    except (TypeError, ValueError, KeyError, UnicodeError) as error:
        raise _ProbeError("The local worker returned an invalid readiness response.") from error


class LocalWorkerDiscoveryService:
    def __init__(self, lifecycle):
        self.lifecycle = lifecycle
        self._lock = asyncio.Lock()
        self._candidates = {}
        self.last_result: LocalWorkerDiscovery | None = None

    def _same_device(self, existing, runtime):
        process_order = os.environ.get("CUDA_DEVICE_ORDER")
        same_order = existing.env.get("CUDA_DEVICE_ORDER", process_order) == process_order
        if "CUDA_VISIBLE_DEVICES" not in existing.env:
            # An operator worker without an override sees exactly the process
            # subset just probed; the trainer also selects its first device.
            return same_order
        target = runtime.env.get("CUDA_VISIBLE_DEVICES")
        if existing.env["CUDA_VISIBLE_DEVICES"] != target:
            return False
        # A UUID identifies the same physical GPU regardless of device order.
        return target.startswith("GPU-") or same_order

    def _equivalent(self, runtime):
        return next(
            (
                existing
                for existing in self.lifecycle.catalog.runtimes
                if existing.execution == "native"
                and existing.provider == "local"
                and existing.device == runtime.device
                and existing.training_image == runtime.training_image
                and existing.training_python == runtime.training_python
                and existing.training_root == runtime.training_root
                and existing.training_module == runtime.training_module
                and sorted(existing.training_model_ids) == sorted(runtime.training_model_ids)
                and self._same_device(existing, runtime)
            ),
            None,
        )

    def _discover(self):
        host = LocalWorkerHost(
            name=platform.node()[:200] or "This machine",
            platform=platform.system(),
            architecture=platform.machine(),
        )
        candidate = LocalWorkerCandidate(
            id="local-smolvla", label="SmolVLA training", status="setup_required"
        )
        response = LocalWorkerDiscovery(
            host=host,
            checked_at=datetime.now(UTC).isoformat(),
            status="unavailable",
            candidates=[candidate],
            issues=list(self.lifecycle.local_workers.issues),
        )
        ready = {}
        if (host.platform, host.architecture) not in {("Linux", "x86_64"), ("Linux", "AMD64")}:
            candidate.reason = (
                "Local SmolVLA training requires x86-64 Linux and an NVIDIA CUDA GPU."
            )
            return response, ready
        root = _worker_root()
        env = _environment()
        if root is None:
            candidate.reason = "The bundled SmolVLA worker sources are not installed on this host."
        else:
            python = root / ".venv/bin/python"
            if env.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
                candidate.reason = "CUDA devices are hidden by this host's visible GPU selection."
            elif not python.is_file() or not os.access(python, os.X_OK):
                candidate.reason = "Install the local SmolVLA training environment."
            else:
                try:
                    result = _parse_probe(
                        _run_bounded(
                            [
                                str(python),
                                "-I",
                                str(Path(__file__).with_name("local_worker_probe.py")),
                                str(root),
                            ],
                            timeout=PROBE_TIMEOUT,
                            env=env,
                        )
                    )
                    if result["status"] == "unavailable":
                        candidate.reason = REASONS[result["reason"]]
                    elif result["gpu"]["uuid"] is None:
                        candidate.gpu_name = result["gpu"]["name"]
                        candidate.gpu_memory_mib = result["gpu"]["memory_mib"]
                        candidate.reason = (
                            "The worker could not verify a stable GPU identity. "
                            "Update or configure the worker manually."
                        )
                    else:
                        gpu = result["gpu"]
                        # Pin the verified first visible GPU by UUID. Numeric ordinals
                        # can target another physical device after a host-order change.
                        visible = env.get("CUDA_VISIBLE_DEVICES")
                        target = gpu["uuid"]
                        identity = hashlib.sha256(
                            (str(root) + "\0" + str(python) + "\0" + target).encode()
                        ).hexdigest()[:16]
                        runtime = Runtime(
                            id="managed-local-smolvla-" + identity,
                            label=(gpu["name"] + " · SmolVLA")[:200],
                            provider="local",
                            device="cuda",
                            training_only=True,
                            training_python=str(python),
                            training_root=str(root),
                            training_model_ids=["smolvla"],
                            gpu_name=gpu["name"],
                            gpu_memory_mib=gpu["memory_mib"],
                            env={"CUDA_VISIBLE_DEVICES": target},
                        )
                        existing = self._equivalent(runtime)
                        candidate.id = runtime.id
                        candidate.gpu_name = runtime.gpu_name
                        candidate.gpu_memory_mib = runtime.gpu_memory_mib
                        candidate.status = "registered" if existing else "ready"
                        candidate.runtime_id = existing.id if existing else None
                        response.status = "ready"
                        response.message = "First visible GPU checked for SmolVLA training."
                        ready[candidate.id] = _ReadyCandidate(
                            runtime, visible, env.get("CUDA_DEVICE_ORDER"), time.monotonic()
                        )
                except (OSError, _ProbeError) as error:
                    response.status = "error"
                    candidate.reason = (
                        str(error)
                        if isinstance(error, _ProbeError)
                        else "The local worker could not be started."
                    )
        if candidate.status == "setup_required":
            hardware = _hardware(env)
            if hardware:
                candidate.gpu_name, candidate.gpu_memory_mib = hardware[2:]
        return response, ready

    async def _scan(self):
        # Keep the serialization lock until the bounded child finishes, even if
        # the requesting browser disconnects while its thread is running.
        task = asyncio.create_task(asyncio.to_thread(self._discover))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def check(self) -> LocalWorkerDiscovery:
        async with self._lock:
            result, candidates = await self._scan()
            self._candidates = candidates
            self.last_result = result
            return result

    async def add(self, candidate_id: str) -> Runtime:
        async with self._lock:
            previous = self._candidates.get(candidate_id)
            if previous is None or time.monotonic() - previous.checked_at > CANDIDATE_TTL:
                raise ValueError("Check this machine again before adding a worker.")
            if previous.visibility != os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ) or previous.device_order != os.environ.get("CUDA_DEVICE_ORDER"):
                self._candidates = {}
                raise ValueError("The visible GPU selection changed. Check this machine again.")
            result, candidates = await self._scan()
            self._candidates = candidates
            self.last_result = result
            current = candidates.get(candidate_id)
            if current is None or current.runtime != previous.runtime:
                raise ValueError(
                    "The local worker changed or is unavailable. Check this machine again."
                )
            registered = self._equivalent(current.runtime)
            if registered is None:
                registered = self.lifecycle.local_workers.register(current.runtime)
            for candidate in result.candidates:
                if candidate.id == candidate_id:
                    candidate.status = "registered"
                    candidate.runtime_id = registered.id
            return registered
