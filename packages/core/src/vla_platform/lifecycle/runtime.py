"""Operator-owned runtime configuration. HTTP requests cannot supply commands."""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from vla_platform.lifecycle.contracts import StrictRecord
from vla_platform.lifecycle.native_profiles import NATIVE_PROFILES
from vla_platform.lifecycle.training_catalog import TRAINING_MODEL_BY_ID


class Runtime(StrictRecord):
    id: str = Field(pattern=r"^[\w-]+$")
    label: str
    execution: Literal["native", "skypilot"] = "native"
    accelerator: str | None = None
    gpu_count: int = Field(default=1, ge=1, le=1, strict=True)
    # Native workers use provider as hosting metadata. SkyPilot targets are
    # generated from compute settings and dispatch through the cloud runner.
    provider: Literal["local", "gcp"] = "local"
    region: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[\w-]+$")
    python: str = "python"
    worker_root: str
    conversion_python: str | None = None
    conversion_image: str | None = None
    evaluation_python: str | None = None
    evaluation_image: str | None = None
    training_python: str | None = None
    training_root: str | None = None
    # Separate pinned CPU consumer; never a caller-selected module or cloud action.
    act_export_python: str | None = Field(default=None, min_length=1)
    act_export_root: str | None = Field(default=None, min_length=1)
    # An operator-installed module must implement the native application protocol.
    # API callers may select a registered model, never its executable/module.
    training_module: str = Field(
        default="firebird_vla.application", pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$"
    )
    training_model_ids: list[str] = Field(default_factory=lambda: ["smolvla"], min_length=1)
    vendor: str
    conversion_vendor: str | None = None
    build: str
    device: Literal["cpu", "cuda"] = "cpu"
    # Optional operator-declared specifications, not live device discovery.
    gpu_name: str | None = Field(default=None, min_length=1, max_length=200)
    gpu_memory_mib: int | None = Field(default=None, gt=0, strict=True)
    image: str | None = None
    training_image: str | None = None
    mounts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    simulator_lane: str | None = None
    tokenizer: str | None = None

    @model_validator(mode="after")
    def registered_training_models(self):
        if bool(self.act_export_python) != bool(self.act_export_root):
            raise ValueError("ACT export interpreter and worker root must be configured together")
        if self.act_export_python and (self.execution != "native" or self.provider != "local"):
            raise ValueError("ACT inference export currently requires a local CPU worker")
        if len(self.training_model_ids) != len(set(self.training_model_ids)):
            raise ValueError("Training model IDs must be unique")
        if any(model_id not in TRAINING_MODEL_BY_ID for model_id in self.training_model_ids):
            raise ValueError("Training model is not registered")
        if (
            self.training_module == "firebird_vla.application"
            and self.execution != "skypilot"
            and self.training_model_ids != ["smolvla"]
        ):
            raise ValueError("The bundled training module supports only SmolVLA")
        if self.training_module == "firebird_vla.lerobot_application" and any(
            key not in NATIVE_PROFILES for key in self.training_model_ids
        ):
            raise ValueError("The native LeRobot module supports registered native policies only")
        if any(TRAINING_MODEL_BY_ID[key].model_revision is None for key in self.training_model_ids):
            raise ValueError("Training requires a pinned model checkpoint")
        return self


class Source(StrictRecord):
    id: str = Field(pattern=r"^[\w-]+$")
    label: str
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    task: Literal["libero_object", "libero_spatial", "unverified"] = "unverified"
    evaluation_bundle: str | None = None
    evaluation_bundle_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    action_dim: int = Field(default=7, ge=1, le=32)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def pinned_evaluation_bundle(self):
        if bool(self.evaluation_bundle) != bool(self.evaluation_bundle_sha256):
            raise ValueError("Evaluation bundle path and SHA256 must be supplied together")
        if self.task == "libero_spatial" and not self.evaluation_bundle:
            raise ValueError("Spatial source requires a pinned local evaluation bundle")
        return self


class PublicRuntime(StrictRecord):
    id: str
    label: str
    execution: Literal["native", "skypilot"] = "native"
    accelerator: str | None = None
    unavailable_reason: str | None = None
    launchable: bool = True
    needs_preparation: bool = False
    provider: Literal["local", "gcp"]
    provider_label: str
    region: str | None
    enabled: bool
    device: Literal["cpu", "cuda"]
    training: bool
    act_export: bool = False
    training_model_ids: list[str]
    simulation: bool
    engine_evaluation: bool = False
    run: bool = False
    gpu_name: str | None
    gpu_memory_mib: int | None
    training_gpu_count: int | None


class RuntimeCatalog(StrictRecord):
    runtimes: list[Runtime] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None):
        value = cls.model_validate_json(path.read_text()) if path else cls()
        if any(
            runtime.execution != "native" or runtime.id.startswith("skypilot-gcp-")
            for runtime in value.runtimes
        ):
            raise ValueError("SkyPilot runtime IDs and execution are managed by compute settings")
        for entries in (value.runtimes, value.sources):
            if len({x.id for x in entries}) != len(entries):
                raise ValueError("Duplicate runtime/source IDs")
        return value

    def runtime(self, name: str):
        return next((x for x in self.runtimes if x.id == name), None)

    def source(self, name: str):
        return next((x for x in self.sources if x.id == name), None)

    def public(self, *, local_enabled: bool = True, local_label: str = "Local machine"):
        provider_labels = {"local": local_label, "gcp": "Google Cloud"}
        return {
            "runtimes": [
                {
                    "id": x.id,
                    "label": x.label,
                    "execution": x.execution,
                    "accelerator": x.accelerator,
                    "unavailable_reason": None,
                    "launchable": local_enabled if x.provider == "local" else True,
                    "needs_preparation": False,
                    "provider": x.provider,
                    "provider_label": provider_labels[x.provider],
                    "region": x.region,
                    "enabled": local_enabled if x.provider == "local" else True,
                    "device": x.device,
                    "training": bool(x.training_python and x.training_root),
                    "act_export": bool(x.act_export_python and x.act_export_root),
                    "training_model_ids": (
                        x.training_model_ids if x.training_python and x.training_root else []
                    ),
                    "simulation": bool(x.simulator_lane),
                    "engine_evaluation": True,
                    "run": True,
                    "gpu_name": x.gpu_name,
                    "gpu_memory_mib": x.gpu_memory_mib,
                    # The current native trainer uses one device per run. This
                    # is a recipe constraint, not an available-device count.
                    "training_gpu_count": (
                        1 if x.device == "cuda" and x.training_python and x.training_root else None
                    ),
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
    evaluation: bool = False,
    act_export: bool = False,
):
    if runtime.execution != "native":
        raise ValueError("SkyPilot targets must execute through the cloud runner")
    if act_export:
        if (
            runtime.provider != "local"
            or not runtime.act_export_root
            or not runtime.act_export_python
        ):
            raise ValueError("This runtime has no local ACT inference export environment")
        root = runtime.act_export_root
        env = {**os.environ, **runtime.env, "PYTHONUNBUFFERED": "1"}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
        env["PYTHONPATH"] = os.pathsep.join([root, str(Path(root) / "src")])
        argv = [
            runtime.act_export_python,
            "-m",
            "firebird_act.application",
            str(request),
            str(output),
        ]
        return argv, root, env
    root = runtime.training_root if training else runtime.worker_root
    python = runtime.training_python if training else runtime.python
    if not root or not python:
        raise ValueError("The selected runtime has no training environment")
    module = runtime.training_module if training else "policykit.application"
    image = runtime.training_image if training else runtime.image
    if conversion and runtime.conversion_python:
        python = runtime.conversion_python
        image = runtime.conversion_image
    if evaluation and (runtime.evaluation_python or runtime.evaluation_image):
        python = runtime.evaluation_python or python
        image = runtime.evaluation_image
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
