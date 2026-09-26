"""Operator-owned runtime configuration. HTTP requests cannot supply commands."""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from vla_platform.lifecycle.contracts import StrictRecord


class Runtime(StrictRecord):
    id: str = Field(pattern=r"^[\w-]+$")
    label: str
    python: str = "python"
    worker_root: str
    conversion_python: str | None = None
    conversion_image: str | None = None
    training_python: str | None = None
    training_root: str | None = None
    vendor: str
    conversion_vendor: str | None = None
    build: str
    device: Literal["cpu", "cuda"] = "cpu"
    image: str | None = None
    training_image: str | None = None
    mounts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    simulator_lane: str | None = None
    tokenizer: str | None = None


class Source(StrictRecord):
    id: str = Field(pattern=r"^[\w-]+$")
    label: str
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    task: Literal["libero_object", "unverified"] = "unverified"
    action_dim: int = Field(default=7, ge=1, le=32)
    provenance: dict[str, Any] = Field(default_factory=dict)


class RuntimeCatalog(StrictRecord):
    runtimes: list[Runtime] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None):
        value = cls.model_validate_json(path.read_text()) if path else cls()
        for entries in (value.runtimes, value.sources):
            if len({x.id for x in entries}) != len(entries):
                raise ValueError("Duplicate runtime/source IDs")
        return value

    def runtime(self, name: str):
        return next((x for x in self.runtimes if x.id == name), None)

    def source(self, name: str):
        return next((x for x in self.sources if x.id == name), None)

    def public(self):
        return {
            "runtimes": [
                {
                    "id": x.id,
                    "label": x.label,
                    "device": x.device,
                    "training": bool(x.training_python and x.training_root),
                    "simulation": bool(x.simulator_lane),
                }
                for x in self.runtimes
            ],
            "sources": [{"id": x.id, "label": x.label, "task": x.task} for x in self.sources],
        }


def command(
    runtime: Runtime,
    request: Path,
    output: Path,
    data_dir: Path,
    container_name: str,
    training: bool = False,
    conversion: bool = False,
):
    root = runtime.training_root if training else runtime.worker_root
    python = runtime.training_python if training else runtime.python
    if not root or not python:
        raise ValueError("The selected runtime has no training environment")
    module = "firebird_vla.application" if training else "policykit.application"
    image = runtime.training_image if training else runtime.image
    if conversion and runtime.conversion_python:
        python = runtime.conversion_python
        image = runtime.conversion_image
    argv = [python, "-m", module, str(request), str(output)]
    env = {**os.environ, **runtime.env, "PYTHONUNBUFFERED": "1"}
    paths = [root, str(Path(root) / "src")]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if not image:
        return argv, root, env
    docker = ["docker", "run", "--rm", "--init", "--name", container_name]
    if runtime.device == "cuda" or training:
        docker += ["--gpus", "all"]
    for path in sorted({root, str(data_dir), *runtime.mounts}):
        docker += ["--mount", f"type=bind,source={path},target={path}"]
    for key, value in {
        **runtime.env,
        "PYTHONPATH": ":".join(paths),
        "PYTHONUNBUFFERED": "1",
    }.items():
        docker += ["--env", f"{key}={value}"]
    docker += ["--workdir", root, "--entrypoint", python, image, *argv[1:]]
    return docker, None, os.environ.copy()
