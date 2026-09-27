"""Persist server-discovered trainers separately from operator-owned runtimes."""

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Literal

from pydantic import Field

from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog

_MANAGED_FIELDS = {
    "id",
    "label",
    "device",
    "training_only",
    "training_python",
    "training_root",
    "training_module",
    "training_model_ids",
    "gpu_name",
    "gpu_memory_mib",
    "env",
}
_TRAINING_MODULES = {"firebird_vla.application", "firebird_vla.lerobot_application"}
_CUDA_TARGETS = re.compile(
    r"(?:\d+|(?:GPU|MIG)-[A-Za-z0-9/-]+)(?:,(?:\d+|(?:GPU|MIG)-[A-Za-z0-9/-]+))*"
)


class _SavedWorkers(StrictRecord):
    version: Literal[1]
    runtimes: list[Runtime] = Field(default_factory=list, max_length=32)


def _validate_managed(runtime: Runtime) -> Runtime:
    # Revalidation matters: model_copy(update=...) does not run Pydantic validators.
    try:
        runtime = Runtime.model_validate(runtime.model_dump())
    except ValueError:
        raise ValueError("Only discovered local training workers can be registered") from None
    if (
        not runtime.id.startswith("managed-local-")
        or len(runtime.id) > 100
        or not runtime.training_only
        or runtime.training_module not in _TRAINING_MODULES
        or set(runtime.model_dump(exclude_defaults=True)) - _MANAGED_FIELDS
        or set(runtime.env) != {"CUDA_VISIBLE_DEVICES"}
        or len(runtime.env["CUDA_VISIBLE_DEVICES"]) > 1024
        or not _CUDA_TARGETS.fullmatch(runtime.env["CUDA_VISIBLE_DEVICES"])
        or len(runtime.label) > 200
        or not runtime.label
        or not runtime.label.isprintable()
    ):
        raise ValueError("Only discovered local training workers can be registered")
    for value in (runtime.training_python, runtime.training_root):
        if (
            not value
            or len(value) > 4096
            or not value.isprintable()
            or not Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise ValueError("Discovered worker paths must be absolute server-owned paths")
    return runtime


def _identity(runtime: Runtime) -> tuple:
    return (
        runtime.execution,
        runtime.provider,
        runtime.device,
        runtime.training_module,
        runtime.training_image,
        tuple(sorted(runtime.training_model_ids)),
        os.path.normpath(runtime.training_python or ""),
        os.path.normpath(runtime.training_root or ""),
        runtime.env.get("CUDA_VISIBLE_DEVICES"),
    )


class LocalWorkerRegistry:
    def __init__(self, data_dir: Path, operator_catalog: RuntimeCatalog):
        self.path = data_dir / "local-workers.json"
        self._operator = operator_catalog.model_copy(deep=True)
        self._lock = threading.Lock()
        self._runtimes: list[Runtime] = []
        self._load_error: str | None = None
        try:
            if self.path.exists():
                if self.path.is_symlink() or self.path.stat().st_size > 131072:
                    raise ValueError("Unsafe registry")
                saved = _SavedWorkers.model_validate_json(self.path.read_text())
                entries = [_validate_managed(runtime) for runtime in saved.runtimes]
                if len({runtime.id for runtime in entries}) != len(entries):
                    raise ValueError("Duplicate registered worker IDs")
                if len({_identity(runtime) for runtime in entries}) != len(entries):
                    raise ValueError("Duplicate registered workers")
                self._runtimes = entries
        except OSError, ValueError:
            self._load_error = (
                "Saved local workers could not be loaded. Repair local-workers.json "
                "on the application server, then restart."
            )

    def _conflicts(self) -> list[str]:
        return [
            f"Saved local worker {runtime.id} conflicts with an operator worker and is unavailable."
            for runtime in self._runtimes
            if (operator := self._operator.runtime(runtime.id)) is not None
            and _identity(operator) != _identity(runtime)
        ]

    @property
    def issues(self) -> list[str]:
        with self._lock:
            return [self._load_error] if self._load_error else self._conflicts()

    def merge(self) -> RuntimeCatalog:
        with self._lock:
            identities = {_identity(runtime) for runtime in self._operator.runtimes}
            entries = [
                runtime
                for runtime in self._runtimes
                if self._operator.runtime(runtime.id) is None
                and _identity(runtime) not in identities
            ]
            return RuntimeCatalog(
                runtimes=[*self._operator.runtimes, *entries], sources=self._operator.sources
            ).model_copy(deep=True)

    def register(self, runtime: Runtime) -> Runtime:
        runtime = _validate_managed(runtime)
        with self._lock:
            if self._load_error:
                raise ValueError(self._load_error)
            existing = [*self._operator.runtimes, *self._runtimes]
            if any(
                item.id == runtime.id and _identity(item) != _identity(runtime) for item in existing
            ):
                raise ValueError("A different worker already uses this local worker ID")
            equivalent = next(
                (item for item in existing if _identity(item) == _identity(runtime)), None
            )
            if equivalent:
                return equivalent.model_copy(deep=True)
            if len(self._runtimes) >= 32:
                raise ValueError("The local worker registry is full")
            updated = [*self._runtimes, runtime]
            payload = json.dumps(_SavedWorkers(version=1, runtimes=updated).model_dump(), indent=2)
            if len(payload.encode("utf-8")) + 1 > 131072:
                raise ValueError("The local worker registry is full")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".local-workers-", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    stream.write(payload)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
                self._runtimes = updated
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return runtime.model_copy(deep=True)
