"""Lifecycle stages use the application's existing job owner and persistence."""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import signal
import tarfile
import tempfile
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from vla_platform.compute_settings import ComputeSettings
from vla_platform.contracts import TERMINAL, DatasetProfile, Job, now
from vla_platform.lifecycle import telemetry
from vla_platform.lifecycle.contracts import (
    CloudExecutionTarget,
    JobEvent,
    LifecycleResult,
    PolicyArtifact,
    PolicyRequest,
    Precision,
)
from vla_platform.lifecycle.runtime import RuntimeCatalog, command
from vla_platform.lifecycle.training_catalog import training_model_for_recipe

logger = logging.getLogger(__name__)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_training_inputs(recipe: dict) -> None:
    """Validate shared form fields before starting a worker or allocating a GPU."""
    positive = {
        "steps",
        "batch_size",
        "gradient_accumulation_steps",
        "save_every",
        "eval_every",
        "eval_batches",
        "log_every",
        "lora_rank",
        "lora_alpha",
        "chunk_size",
    }
    for name in positive | {"seed", "warmup_steps", "num_workers"}:
        if name not in recipe:
            continue
        value = recipe[name]
        if type(value) is not int or value < (1 if name in positive else 0):
            raise ValueError(
                f"{name} must be a {'positive' if name in positive else 'nonnegative'} integer"
            )
    if recipe.get("seed", 42) > 2147483647:
        raise ValueError("seed must be between 0 and 2147483647")
    if recipe.get("warmup_steps", 50) >= recipe.get("steps", 20000):
        raise ValueError("Warmup steps must be less than the total training steps")
    for name in (
        "learning_rate",
        "validation_fraction",
        "weight_decay",
        "max_grad_norm",
        "lora_dropout",
    ):
        if name not in recipe:
            continue
        value = recipe[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if name in {"learning_rate", "max_grad_norm"} and value <= 0:
            raise ValueError(f"{name} must be positive")
        if name == "weight_decay" and value < 0:
            raise ValueError("weight_decay must be nonnegative")
        if name == "validation_fraction" and not 0 < value < 1:
            raise ValueError("validation_fraction must be between 0 and 1, exclusive")
        if name == "lora_dropout" and not 0 <= value < 1:
            raise ValueError("lora_dropout must be at least 0 and less than 1")


def validate_cloud_recipe(recipe: dict) -> dict:
    """Reject invalid form inputs before accepting a cloud job or preparing services."""
    if {"method", "output_dir", "dataset_id", "dataset_revision"}.intersection(recipe):
        raise ValueError("The application manages the method, output folder, and dataset")
    recipe = dict(recipe)
    for name, default in (
        ("steps", 20000),
        ("batch_size", 64),
        ("gradient_accumulation_steps", 1),
        (
            "save_every",
            max(1, math.ceil(recipe.get("steps", 20000) / 5))
            if type(recipe.get("steps", 20000)) is int
            else 4000,
        ),
    ):
        value = recipe.setdefault(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    warmup = recipe.setdefault("warmup_steps", min(50, recipe["steps"] - 1))
    if type(warmup) is not int or not 0 <= warmup < recipe["steps"]:
        raise ValueError("Warmup steps must be less than the total training steps")
    rate = recipe.get("learning_rate", 0.0001)
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be a positive number")
    if math.ceil(recipe["steps"] / recipe["save_every"]) > 10000:
        raise ValueError("Increase the checkpoint interval to keep this run within 10,000 saves")
    validate_training_inputs(recipe)
    return recipe


def reject_nonfinite(value: str):
    raise ValueError("Worker response contains a non-finite number: " + value)


def finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        reject_nonfinite(value)
    return parsed


def complete_measurement(report: dict, episodes: int) -> bool:
    """Quality decisions require complete, finite evidence from every stage."""
    if type(report.get("complete_episodes")) is not int or report["complete_episodes"] != episodes:
        return False
    if not isinstance(report.get("runtime"), dict) or not report["runtime"]:
        return False
    coverage = report.get("memory_coverage")
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        return False
    for key in ("success_rate", "p95_ms", "peak_device_mib"):
        value = report.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            return False
        if key == "success_rate":
            if not 0 <= value <= 1:
                return False
        elif value <= 0:
            return False
    return True


def validate_bundle(directory: Path, job_dir: Path):
    directory = directory.resolve()
    if not directory.is_relative_to(job_dir.resolve()) or not directory.is_dir():
        raise ValueError("Worker artifact must stay in its application job directory")
    manifest_path = directory / "manifest.json"
    if (directory / "remote.json").is_file():
        from vla_platform.lifecycle.cloud_storage import read_descriptor

        if any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError("Artifact symlinks are not permitted")
        pointer, manifest = read_descriptor(directory)
        return manifest, pointer["manifest_sha256"], pointer["file_bytes"]
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Invalid artifact manifest")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty artifact manifest")
    actual = set()
    total = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Artifact symlinks are not permitted")
        if path.is_file() and path != manifest_path:
            name = path.relative_to(directory).as_posix()
            actual.add(name)
            total += path.stat().st_size
            if digest(path) != files.get(name):
                raise ValueError(f"Artifact hash mismatch: {name}")
    if actual != set(files):
        raise ValueError("Artifact inventory differs from manifest")
    return manifest, digest(manifest_path), total


def validate_training_inventory(manifest: dict, step: int):
    """Verify each bundled trainer's complete resume inventory without importing ML."""
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or type(manifest.get("step")) is not int
        or manifest["step"] != step
        or step <= 0
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("Incomplete training checkpoint manifest")
    backend = manifest.get("training_backend", "smolvla")
    inventories = {
        "smolvla": {
            "recipe.json",
            "stats.json",
            "splits.json",
            "training.pt",
            "probe.safetensors",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "policy/config.json",
        },
        "lerobot": {
            "recipe.json",
            "probe-batch.pt",
            "probe-action.pt",
            "pretrained_model/config.json",
            "pretrained_model/model.safetensors",
            "pretrained_model/train_config.json",
            "training_state/optimizer_state.safetensors",
            "training_state/optimizer_param_groups.json",
            "training_state/rng_state.safetensors",
            "training_state/training_step.json",
        },
        "psi0": {
            "recipe.json",
            "stats.json",
            "splits.json",
            "training.pt",
            "probe.safetensors",
            "probe-input.safetensors",
            "action_header.safetensors",
        },
    }
    if (
        not isinstance(backend, str)
        or backend not in inventories
        or not inventories[backend].issubset(manifest["files"])
    ):
        raise ValueError("Incomplete training checkpoint inventory")


def validate_archive(path: Path, files: dict[str, str], manifest_sha256: str):
    """Check the delivered bytes, including inventory, without extracting the archive."""
    expected = {"policy/" + name: sha for name, sha in files.items()}
    expected["policy/manifest.json"] = manifest_sha256
    directories = {"policy"}
    for name in expected:
        directories.update(
            str(parent) for parent in PurePosixPath(name).parents if str(parent) != "."
        )
    seen = set()
    with tarfile.open(path, "r:", stream=True) as archive:
        for member in archive:
            if member.name in seen:
                raise ValueError("Duplicate archive member")
            seen.add(member.name)
            if member.isdir() and member.name in directories:
                continue
            if not member.isfile() or member.name not in expected:
                raise ValueError("Unexpected archive member")
            with archive.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[member.name]:
                    raise ValueError("Archive member hash mismatch")
        if set(expected) != seen - directories:
            raise ValueError("Archive inventory differs from manifest")
        # tarfile also accepts archives missing the closing blocks. A published
        # cache entry must be complete, with no payload hidden after the end marker.
        archive.fileobj.seek(archive.offset)
        if archive.fileobj.read(1024) != bytes(1024):
            raise ValueError("Incomplete archive terminator")
        while block := archive.fileobj.read(1024 * 1024):
            if any(block):
                raise ValueError("Unexpected data after archive terminator")


class Lifecycle:
    def __init__(self, execution):
        self.execution = execution
        self.settings = execution.settings
        self.catalog = RuntimeCatalog.load(self.settings.runtime_config)
        self.compute = ComputeSettings(self.settings.data_dir)
        self.event_lock = threading.Lock()
        self.export_lock = threading.Lock()

    def runtime(self, runtime_id: str):
        return self.compute.runtime(runtime_id) or self.catalog.runtime(runtime_id)

    def _cloud_checkpoint_artifact(self, job, name, label=None):
        if not isinstance(name, str) or not re.fullmatch(r"checkpoint-\d{6,}", name):
            raise ValueError("Invalid checkpoint identity")
        step = int(name.removeprefix("checkpoint-"))
        stage = "operation" if job.kind == "policy.finetune" else "training"
        directory = self.settings.data_dir / "jobs" / job.id / stage
        path = directory / "cloud-checkpoints" / name
        if path.is_symlink() or not (path / "remote.json").is_file():
            raise ValueError("Cloud checkpoint descriptor is missing or unsafe")
        manifest, sha, total = validate_bundle(path, directory)
        metadata = manifest.get("metadata", {})
        if type(metadata.get("step")) is not int or metadata["step"] != step:
            raise ValueError("Cloud checkpoint descriptor step differs from its identity")
        return PolicyArtifact(
            id=f"{job.id}:{name}",
            project_id=job.project_id,
            job_id=job.id,
            label=label
            if isinstance(label, str) and label
            else f"{metadata.get('architecture', 'Policy')} · step {step}",
            format="training_checkpoint",
            path=path.relative_to(self.settings.data_dir).as_posix(),
            manifest_sha256=sha,
            file_bytes=total,
            metadata=metadata,
        )

    async def artifacts(self, project_id: str):
        artifacts = []
        for job in await self.execution.list(project_id):
            if isinstance(job.result, LifecycleResult):
                artifacts.extend(job.result.artifacts)
            stage = "operation" if job.kind == "policy.finetune" else "training"
            directory = self.settings.data_dir / "jobs" / job.id / stage
            index_path = directory / "cloud-checkpoints.json"
            if not index_path.is_file() or index_path.stat().st_size > 8 * 1024**2:
                continue
            try:
                index = json.loads(index_path.read_text())
                entries = index.get("checkpoints", [])
                if not isinstance(entries, list):
                    raise ValueError("Invalid checkpoint index")
            except (OSError, ValueError, AttributeError) as exc:
                logger.warning("Skipping invalid checkpoint index %s: %s", index_path, exc)
                continue
            for item in entries:
                try:
                    if not isinstance(item, dict):
                        raise ValueError("Invalid checkpoint index entry")
                    artifacts.append(
                        self._cloud_checkpoint_artifact(job, item.get("name"), item.get("label"))
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("Skipping unavailable checkpoint in %s: %s", index_path, exc)
        return artifacts

    async def artifact(self, project_id: str, artifact_id: str):
        value = next((x for x in await self.artifacts(project_id) if x.id == artifact_id), None)
        if value is not None:
            return value
        # Descriptor publication can precede the advisory index. Explicit valid
        # steps remain addressable even if a later index write was interrupted.
        job_id, separator, name = artifact_id.rpartition(":")
        if separator and re.fullmatch(r"checkpoint-\d{6,}", name):
            job = await self.execution.get(job_id)
            if (
                job
                and job.project_id == project_id
                and job.kind in {"policy.finetune", "policy.workflow"}
            ):
                try:
                    return self._cloud_checkpoint_artifact(job, name)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise ValueError("Selected checkpoint is unavailable or corrupt") from exc
        raise ValueError("Artifact does not exist in this project")

    async def resume_checkpoint(self, project_id, job_id):
        job = await self.execution.get(job_id)
        if (
            not job
            or job.project_id != project_id
            or job.status not in {"failed", "cancelled", "interrupted"}
            or job.kind not in {"policy.finetune", "policy.workflow"}
        ):
            raise ValueError("Resume requires an interrupted training job in this project")
        stage = "operation" if job.kind == "policy.finetune" else "training"
        directory = self.settings.data_dir / "jobs" / job.id / stage / "training"

        def discover():
            job_dir = self.settings.data_dir / "jobs" / job.id
            if directory.is_symlink() or not directory.resolve().is_relative_to(job_dir.resolve()):
                raise ValueError("Checkpoint directory escapes its run")
            # latest.json is advisory. A kill after bundle publication can leave
            # it missing, stale or truncated; never follow paths stored in it.
            candidates = []
            if directory.is_dir():
                for path in directory.iterdir():
                    if (
                        re.fullmatch(r"checkpoint-\d{6,}", path.name)
                        and not path.is_symlink()
                        and path.is_dir()
                    ):
                        candidates.append((int(path.name.removeprefix("checkpoint-")), path))
            for step, checkpoint in sorted(candidates, reverse=True):
                try:
                    if (checkpoint / "remote-checkpoint.json").is_file():
                        # Derive the descriptor location from the run and step, never
                        # follow the advisory path inside remote-checkpoint.json.
                        descriptor = directory.parent / "cloud-checkpoints" / checkpoint.name
                        if descriptor.is_symlink() or any(
                            path.is_symlink() for path in checkpoint.iterdir()
                        ):
                            raise ValueError("Checkpoint metadata must not be a symlink")
                        bundle, _, _ = validate_bundle(descriptor, directory.parent)
                        manifest_path = checkpoint / "manifest.json"
                        if (
                            bundle.get("metadata", {}).get("step") != step
                            or digest(manifest_path)
                            != bundle["files"].get("checkpoint/manifest.json")
                            or digest(checkpoint / "recipe.json")
                            != bundle["files"].get("checkpoint/recipe.json")
                        ):
                            raise ValueError(
                                "Cloud checkpoint metadata differs from its descriptor"
                            )
                        manifest = json.loads(manifest_path.read_text())
                        if not isinstance(manifest, dict) or not isinstance(
                            manifest.get("files"), dict
                        ):
                            raise ValueError("Incomplete cloud training checkpoint manifest")
                        if any(
                            bundle["files"].get("checkpoint/" + name) != sha
                            for name, sha in manifest.get("files", {}).items()
                        ):
                            raise ValueError(
                                "Cloud checkpoint inventory differs from its descriptor"
                            )
                    else:
                        manifest, _, _ = validate_bundle(checkpoint, directory)
                    validate_training_inventory(manifest, step)
                    # Recipe compatibility and optimizer/RNG deserialization are
                    # worker gates; the Torch-free core verifies bundle integrity.
                    return checkpoint
                except (OSError, ValueError) as exc:
                    logger.warning("Skipping incomplete checkpoint %s: %s", checkpoint, exc)
            raise ValueError("This run has no completed checkpoint to resume")

        return await asyncio.to_thread(discover)

    async def download(self, project_id, artifact_id):
        artifact = await self.artifact(project_id, artifact_id)
        directory = self.settings.data_dir / artifact.path
        if digest(directory / "manifest.json") != artifact.manifest_sha256:
            raise ValueError("Registered artifact manifest changed")
        if (directory / "remote.json").is_file():
            from vla_platform.lifecycle.cloud_download import open_download
            from vla_platform.lifecycle.cloud_storage import read_descriptor

            await asyncio.to_thread(read_descriptor, directory)
            return await open_download(directory, artifact.id.replace(":", "-") + ".tar")
        cache = self.settings.data_dir / "exports" / artifact.id.replace(":", "-")

        def archive():
            # The lock survives cancellation of an awaiting HTTP request. Each
            # repaired archive gets a new name: Windows readers can hold an old
            # generation open without blocking publication of its replacement.
            with self.export_lock:
                manifest, sha, _ = validate_bundle(directory, directory.parent)
                if sha != artifact.manifest_sha256:
                    raise ValueError("Registered artifact manifest changed")
                cache.mkdir(parents=True, exist_ok=True)
                for candidate in cache.glob("*.tar"):
                    try:
                        validate_archive(candidate, manifest["files"], sha)
                        return candidate
                    except OSError, ValueError, tarfile.TarError:
                        logger.warning("Discarding corrupt export cache %s", candidate)
                        try:
                            candidate.unlink()
                        except OSError:
                            # In particular, an existing Windows response may
                            # still hold this file. Never overwrite it in place.
                            pass
                with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    with tarfile.open(temporary, "w") as tar:
                        tar.add(directory, arcname="policy", recursive=False)
                        for name in ["manifest.json", *sorted(manifest["files"])]:
                            tar.add(directory / name, arcname="policy/" + name, recursive=False)
                    # Detect source changes between bundle validation and archiving.
                    validate_archive(temporary, manifest["files"], sha)
                    # Windows _commit requires a writable descriptor, even
                    # after tarfile has closed its writer.
                    with temporary.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    destination = temporary.with_suffix(".tar")
                    temporary.replace(destination)
                    return destination
                finally:
                    temporary.unlink(missing_ok=True)

        return await asyncio.to_thread(archive)

    def _saved_recipe(self, path, expected_sha=None):
        if path.is_symlink() or not path.resolve().is_relative_to(self.settings.data_dir.resolve()):
            raise ValueError("Unsafe checkpoint recipe path")
        if not path.is_file():
            return None
        if path.stat().st_size > 1024**2:
            raise ValueError("Checkpoint recipe is too large")
        content = path.read_bytes()
        if expected_sha is not None and hashlib.sha256(content).hexdigest() != expected_sha:
            raise ValueError("Checkpoint recipe differs from its registered manifest")
        recipe = json.loads(content)
        if not isinstance(recipe, dict):
            raise ValueError("Invalid checkpoint recipe")
        return recipe

    def _artifact_recipe(self, artifact, original):
        directory = self.settings.data_dir / artifact.path
        if not (directory / "remote.json").is_file():
            return self._saved_recipe(directory / "checkpoint" / "recipe.json")
        manifest, sha, _ = validate_bundle(directory, self.settings.data_dir / "jobs" / original.id)
        if sha != artifact.manifest_sha256:
            raise ValueError("Registered artifact manifest changed")
        expected = manifest["files"].get("checkpoint/recipe.json")
        if expected is None:
            return None
        stage = "operation" if original.kind == "policy.finetune" else "training"
        stage_dir = self.settings.data_dir / "jobs" / original.id / stage
        name = artifact.id.rpartition(":")[2]
        candidates = (
            [stage_dir / "training" / name / "recipe.json"]
            if re.fullmatch(r"checkpoint-\d{6,}", name)
            else [stage_dir / "recipe.json", stage_dir / "training" / "recipe.json"]
        )
        for path in candidates:
            recipe = self._saved_recipe(path, expected)
            if recipe is not None:
                return recipe
        return None

    async def resumed_training_contract(self, project_id: str, request: PolicyRequest):
        """Use the selected checkpoint's recipe, then persisted lineage, then metadata."""
        current, fallback = request, {}
        for _ in range(32):
            metadata, recipe = {}, None
            if current.resume_job_id:
                checkpoint = await self.resume_checkpoint(project_id, current.resume_job_id)
                original = await self.execution.get(current.resume_job_id)
                recipe = self._saved_recipe(checkpoint / "recipe.json")
            elif current.artifact_id:
                artifact = await self.artifact(project_id, current.artifact_id)
                if artifact.format != "training_checkpoint":
                    raise ValueError("Resume requires a native training checkpoint")
                metadata = artifact.metadata
                original = await self.execution.get(artifact.job_id)
                if original:
                    recipe = self._artifact_recipe(artifact, original)
            else:
                raise ValueError("The original training recipe is unavailable")
            if (
                not original
                or original.project_id != project_id
                or not isinstance(original.request, PolicyRequest)
            ):
                raise ValueError("The original training recipe is unavailable")
            base = metadata.get("base_model")
            if isinstance(base, dict):
                for key, value in {
                    "model_id": base.get("repository"),
                    "model_revision": base.get("revision"),
                    "method": metadata.get("method", original.request.training_method),
                    "camera_keys": metadata.get("camera_keys"),
                }.items():
                    if value is not None:
                        fallback.setdefault(key, value)
            if recipe is None:
                if original.request.training is not None:
                    recipe = original.request.training
                elif original.request.resume_job_id or original.request.artifact_id:
                    current = original.request
                    continue
                else:
                    # Legacy default SmolVLA jobs may have no explicit recipe.
                    recipe = {}
            recipe = {**fallback, **recipe}
            return (
                recipe,
                recipe.get("method", original.request.training_method),
                original.request.dataset_job_id,
            )
        raise ValueError("Checkpoint lineage is too deep")

    async def validate(self, project_id: str, request: PolicyRequest):
        runtime = self.runtime(request.runtime_id)
        if runtime is None:
            raise ValueError("Runtime is not configured on this application host")
        self.compute.require_enabled(runtime)
        cloud_target = None
        if runtime.execution == "skypilot":
            if request.operation not in {"policy.finetune", "policy.quantize"}:
                raise ValueError("SkyPilot cloud GPUs support fine-tuning and quantization")
        if request.source_id and self.catalog.source(request.source_id) is None:
            raise ValueError("Policy source is not configured")
        if request.evaluation.suite == "libero_spatial" and request.operation in {
            "policy.workflow",
            "policy.evaluate",
            "policy.run",
        }:
            if runtime.device != "cuda" or not runtime.evaluation_python:
                raise ValueError(
                    "Spatial execution requires a prepared CUDA evaluation environment"
                )
            if request.training is not None:
                raise ValueError(
                    "Spatial training admission is not implemented; select a pinned Spatial source"
                )
            if (
                request.source_id
                and self.catalog.source(request.source_id).task != "libero_spatial"
            ):
                raise ValueError("Spatial evaluation requires a Spatial policy source")
            if request.artifact_id:
                artifact = await self.artifact(project_id, request.artifact_id)
                if artifact.metadata.get("task") != "libero_spatial":
                    raise ValueError("Spatial evaluation requires a Spatial policy artifact")
        if request.resume_job_id:
            if request.artifact_id:
                raise ValueError("Choose either a checkpoint artifact or an interrupted job")
            await self.resume_checkpoint(project_id, request.resume_job_id)
        if request.artifact_id:
            artifact = await self.artifact(project_id, request.artifact_id)
            if request.operation == "policy.quantize":
                if artifact.format not in {"training_checkpoint", "native_checkpoint", "gguf"}:
                    raise ValueError("Choose a trained checkpoint or floating GGUF to quantize")
                if artifact.metadata.get("architecture", "smolvla") != "smolvla":
                    raise ValueError("GGUF quantization currently supports SmolVLA checkpoints")
                if artifact.format == "gguf" and artifact.metadata.get("precision") != "float":
                    raise ValueError("Choose the floating source checkpoint, not a packed artifact")
        if request.operation == "policy.finetune" or request.training is not None:
            resuming = bool(request.resume_job_id or request.artifact_id)
            if resuming:
                recipe, method, dataset_job_id = await self.resumed_training_contract(
                    project_id, request
                )
                if request.training is not None:
                    changed = [
                        key for key, value in request.training.items() if recipe.get(key) != value
                    ]
                    if changed:
                        raise ValueError(
                            "Resume must preserve the saved recipe; cannot override "
                            + ", ".join(sorted(changed))
                        )
                    # Dispatch must derive its model from the checkpoint, never
                    # from a partial caller recipe that could select another worker.
                    request.training = None
                request.training_method = method
                request.dataset_job_id = dataset_job_id
            else:
                recipe = request.training or {}
            if request.training_method not in {"lora", "qlora", "full"}:
                raise ValueError("Training method is not registered")
            if not runtime.training_python or not runtime.training_root:
                raise ValueError("This runtime has no training environment")
            model = training_model_for_recipe(recipe)
            if not model.model_revision:
                raise ValueError("This model needs a pinned checkpoint before training")
            if model.id not in runtime.training_model_ids:
                raise ValueError(f"This runtime has no training adapter for {model.label}")
            revision = recipe.get("model_revision", model.model_revision)
            if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
                raise ValueError("Training requires an immutable model checkpoint revision")
            if not resuming and revision != model.model_revision:
                raise ValueError("Training must use the model catalog's pinned checkpoint")
            if (
                model.minimum_gpu_memory_gb
                and runtime.gpu_memory_mib
                and runtime.gpu_memory_mib < model.minimum_gpu_memory_gb * 1024
            ):
                raise ValueError(
                    f"{model.label} requires a GPU with at least "
                    f"{model.minimum_gpu_memory_gb} GB of memory"
                )
            if model.backend in {"lerobot", "psi0"}:
                if (recipe or {}).get("gradient_accumulation_steps", 1) != 1:
                    raise ValueError("Native policy training currently requires accumulation of 1")
                cameras = (recipe or {}).get("camera_keys") or [(recipe or {}).get("camera_key")]
                if (
                    model.required_cameras
                    and len([key for key in cameras if key]) != model.required_cameras
                ):
                    raise ValueError(
                        f"{model.label} requires {model.required_cameras} selected camera(s)"
                    )
            if request.training_method not in model.methods:
                raise ValueError("Training method is not supported by this model")
            if not resuming:
                if runtime.execution == "skypilot":
                    recipe = validate_cloud_recipe(recipe)
                else:
                    validate_training_inputs(recipe)
                # New jobs preserve an explicit pinned model contract. Resumes
                # keep their checkpoint-owned recipe and never substitute defaults.
                request.training = {
                    **recipe,
                    "model_id": model.model_id,
                    "model_revision": model.model_revision,
                }
                if model.checkpoint_subdirectory:
                    request.training["checkpoint_subdirectory"] = model.checkpoint_subdirectory
            dataset = await self.execution.get(request.dataset_job_id)
            if (
                not dataset
                or dataset.project_id != project_id
                or dataset.status != "succeeded"
                or not isinstance(dataset.result, DatasetProfile)
            ):
                raise ValueError("Training requires a successful dataset intake in this project")
            if model.backend == "lerobot":
                from vla_platform.lifecycle.native_profiles import (
                    native_profile_for_recipe,
                    validate_native_dataset,
                )

                selected_cameras = recipe.get("camera_keys") or [recipe.get("camera_key")]
                validate_native_dataset(
                    native_profile_for_recipe(recipe), dataset.result.features, selected_cameras
                )
            if model.backend == "lerobot" and dataset.result.format != "lerobot_v3":
                raise ValueError("Native LeRobot training currently requires a LeRobot v3 dataset")
            if model.backend == "psi0" and dataset.result.format != "lerobot_v2":
                raise ValueError("Psi-Zero currently requires a LeRobot v2.1 dataset")
            if dataset.result.source == "local":
                if model.backend != "lerobot" or dataset.result.snapshot is None:
                    raise ValueError(
                        "Local training requires a verified v3 snapshot and a native LeRobot model"
                    )
                from vla_platform.datasets.snapshots import resolve_snapshot, training_split

                snapshot_path = await asyncio.to_thread(
                    resolve_snapshot,
                    self.settings.data_dir / "dataset-snapshots",
                    dataset.result.snapshot.model_dump(),
                )
                await asyncio.to_thread(
                    training_split,
                    snapshot_path,
                    dataset.result.snapshot.manifest_sha256,
                    recipe.get("validation_fraction", 0.2),
                    recipe.get("seed", 42),
                )
        if (
            request.operation in {"policy.evaluate", "policy.run", "policy.workflow"}
            and request.evaluation.mode == "libero"
            and not runtime.simulator_lane
        ):
            raise ValueError("LIBERO is not configured for this runtime")
        if runtime.execution == "skypilot":
            cloud_target = CloudExecutionTarget.model_validate(
                await self.compute.plan_cloud_target(runtime.id)
            )
        return cloud_target

    def _read_events(self, path: Path):
        records: deque[JobEvent] = deque(maxlen=200)
        offset, terminated, recovery = 0, True, None
        if not path.is_file():
            return records, offset, terminated, recovery
        with path.open("rb") as stream:
            while line := stream.readline(16385):
                if len(line) > 16384:
                    raise ValueError(f"Oversized event record at byte {offset}")
                try:
                    record = JobEvent.model_validate_json(line)
                except ValidationError as exc:
                    if line.endswith(b"\n") or any(
                        error["type"] != "json_invalid" for error in exc.errors()
                    ):
                        raise ValueError(f"Corrupt event record at byte {offset}") from exc
                    # Only an invalid unterminated final record can be a torn
                    # write. Complete records and interior corruption fail closed.
                    recovery = JobEvent(
                        sequence=records[-1].sequence + 1 if records else 1,
                        stage="recovery",
                        message=(
                            "An incomplete final event was ignored after an interrupted write."
                        ),
                        timestamp=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                    )
                    records.append(recovery)
                    logger.warning("Incomplete final event record in %s at byte %s", path, offset)
                    break
                if record.sequence <= (records[-1].sequence if records else 0):
                    raise ValueError(f"Non-increasing event sequence at byte {offset}")
                records.append(record)
                offset += len(line)
                terminated = line.endswith(b"\n")
        return records, offset, terminated, recovery

    def events(self, job_id: str, after: int = 0):
        path = self.settings.data_dir / "jobs" / job_id / "events.jsonl"
        with self.event_lock:
            records, _, _, _ = self._read_events(path)
            return [
                record.model_copy(
                    update={
                        "message": telemetry.sanitize_text(record.message),
                        "data": telemetry.public_value(record.data),
                    }
                )
                for record in records
                if record.sequence > after
            ]

    async def event(self, job: Job, stage: str, message: str, data: dict | None = None):
        directory = self.settings.data_dir / "jobs" / job.id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "events.jsonl"
        with self.event_lock:
            prior, offset, terminated, recovery = self._read_events(path)
            record = JobEvent(
                sequence=prior[-1].sequence + 1 if prior else 1,
                stage=stage,
                message=telemetry.sanitize_text(message)[:2000],
                timestamp=now(),
                data=telemetry.public_value(data or {}),
            )
            encoded = record.model_dump_json().encode("utf-8") + b"\n"
            if len(encoded) > 16384:
                # Full recipe/environment/split evidence is persisted separately.
                # Keep progress usable without creating a record our reader rejects.
                compact = {
                    key: value
                    for key, value in record.data.items()
                    if key not in {"recipe", "environment", "splits"}
                }
                compact["event_data_truncated"] = True
                record = record.model_copy(update={"data": compact})
                encoded = record.model_dump_json().encode("utf-8") + b"\n"
                if len(encoded) > 16384:
                    record = record.model_copy(update={"data": {"event_data_truncated": True}})
                    encoded = record.model_dump_json().encode("utf-8") + b"\n"
            with path.open("r+b" if path.exists() else "w+b") as stream:
                stream.seek(offset)
                if recovery:
                    stream.truncate()
                    stream.write(recovery.model_dump_json().encode("utf-8") + b"\n")
                elif not terminated:
                    stream.write(b"\n")
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        async with self.execution.lock:
            current = await self.execution.get(job.id)
            if current and current.status not in TERMINAL:
                current.stage = stage
                await self.execution.save(current)

    async def training_observation(self, job: Job, line: str):
        observation = telemetry.parse_observation(line)
        if observation is None:
            return False
        directory = self.settings.data_dir / "jobs" / job.id
        directory.mkdir(parents=True, exist_ok=True)
        phase = observation.get("phase") or (
            "checkpoint" if observation.get("checkpoint_saved") else "training"
        )
        message = observation.get("message") or (
            f"Checkpoint {observation.get('step')} saved"
            if observation.get("checkpoint_saved")
            else f"Optimizer step {observation.get('step')}"
        )
        observation["phase"] = phase
        if "step" in observation and any(key in observation for key in telemetry.METRICS):
            with (directory / "training-metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"timestamp": now(), **observation}, allow_nan=False) + "\n"
                )
        if observation.get("recipe") or observation.get("environment") or observation.get("splits"):
            value = await telemetry.reproducibility(self, job)
            if observation.get("recipe"):
                value["recipe"] = observation["recipe"]
                value["recipe_source"] = "worker_resolved"
            for key in ("environment", "splits"):
                if observation.get(key):
                    value.setdefault("evidence", {})[key] = observation[key]
            telemetry.write_json(directory / "reproducibility.json", value)
        await self.event(job, phase, message, observation)
        return True

    async def publish(self, job: Job, result: LifecycleResult):
        async with self.execution.lock:
            current = await self.execution.get(job.id)
            if current and current.status not in TERMINAL:
                current.result = result
                await self.execution.save(current)
            else:
                raise asyncio.CancelledError

    async def stop(self, process, container_name: str | None):
        if container_name:
            cleanup = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(cleanup.wait(), 15)
            except TimeoutError:
                cleanup.kill()
                await cleanup.wait()
        if process.returncode is None or os.name == "posix":
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
            else:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                await process.wait()
            if os.name == "posix":
                # A leader can exit on SIGTERM while a native descendant ignores it.
                # Reap the remaining group even when waiting for the leader succeeded.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    async def native(
        self,
        job,
        operation,
        artifact,
        output,
        result,
        *,
        final=False,
        precision=None,
        evaluation_backend="cpp",
        register_artifact=True,
    ):
        request = job.request
        runtime = self.runtime(request.runtime_id)
        if runtime is None:
            raise ValueError("Runtime is not configured on this application host")
        self.compute.require_enabled(runtime)
        stage_dir = self.settings.data_dir / "jobs" / job.id / output
        if job.compute_target is not None and runtime.execution != "skypilot":
            raise ValueError("Cloud run cannot be redirected to a local worker")
        stage_dir.mkdir(parents=True, exist_ok=False)
        training = operation in {"policy.finetune", "policy.export"}
        payload = {
            "schema_version": 1,
            "operation": operation,
            "job_id": job.id,
            "runtime": runtime.model_dump(),
            "output_dir": str(stage_dir.resolve()),
            "parameters": request.model_dump(),
            "final": final,
            "evaluation_backend": evaluation_backend,
            "artifact": artifact.model_dump() if artifact else None,
            "source": None,
        }
        if artifact:
            artifact_dir = self.settings.data_dir / artifact.path
            manifest = artifact_dir / "manifest.json"
            if digest(manifest) != artifact.manifest_sha256:
                raise ValueError("Registered artifact manifest changed")
            payload["artifact"]["path"] = str(artifact_dir.resolve())
        if request.resume_job_id:
            checkpoint = await self.resume_checkpoint(job.project_id, request.resume_job_id)
            if (checkpoint / "remote-checkpoint.json").is_file():
                if runtime.execution != "skypilot":
                    raise ValueError("Resume this cloud checkpoint on a Google Cloud GPU")
                cloud_artifact = await self.artifact(
                    job.project_id, f"{request.resume_job_id}:{checkpoint.name}"
                )
                artifact = cloud_artifact
                payload["artifact"] = cloud_artifact.model_dump()
                payload["artifact"]["path"] = str(
                    (self.settings.data_dir / cloud_artifact.path).resolve()
                )
            else:
                payload["resume_checkpoint"] = str(checkpoint.resolve())
        payload["prior_reports"] = result.reports
        if request.source_id:
            payload["source"] = self.catalog.source(request.source_id).model_dump()
        chosen_precision = precision or request.precision or Precision()
        payload["parameters"]["precision"] = chosen_precision.model_dump()
        if training and operation == "policy.finetune":
            dataset = await self.execution.get(request.dataset_job_id)
            payload["dataset"] = dataset.result.model_dump()
            if dataset.result.source == "local":
                from vla_platform.datasets.snapshots import resolve_snapshot, stage_snapshot

                descriptor = dataset.result.snapshot.model_dump()
                source = await asyncio.to_thread(
                    resolve_snapshot, self.settings.data_dir / "dataset-snapshots", descriptor
                )
                staged = await asyncio.to_thread(
                    stage_snapshot,
                    source,
                    stage_dir / "dataset-snapshot",
                    descriptor["manifest_sha256"],
                )
                payload["dataset_snapshot"] = {
                    "path": str(staged.resolve()),
                    "id": descriptor["id"],
                    "manifest_sha256": descriptor["manifest_sha256"],
                }
        request_path, result_path = stage_dir / "request.json", stage_dir / "result.json"
        request_path.write_text(json.dumps(payload, allow_nan=False))
        if runtime.execution == "skypilot":
            from vla_platform.lifecycle import sky_runner

            if job.compute_target is None:
                raise ValueError("Cloud run has no saved compute selection; submit a new job")
            target = job.compute_target.model_dump()
            self.compute.require_enabled(runtime)
            await self.event(job, "preparing", "Preparing your GPU…")
            await self.compute.ensure_cloud_ready(runtime.id, target)
            current = await self.execution.get(job.id)
            if current is None or current.status in TERMINAL:
                raise asyncio.CancelledError
            await self.event(
                job,
                "preparing",
                f"Launching {target['accelerator']} on Google Cloud via SkyPilot "
                f"({target['project_id']}, {target['region']})",
            )

            cloud_stage = "preparing"

            async def progress(message):
                nonlocal cloud_stage
                if message.startswith("_quantization:"):
                    detail = json.loads(message[len("_quantization:") :])
                    cloud_stage = detail.get("phase", "quantizing")
                    await self.event(
                        job, cloud_stage, detail.get("message", "Quantizing checkpoint")
                    )
                    return
                if message.startswith("_telemetry:"):
                    line = message[len("_telemetry:") :]
                    observed = telemetry.parse_observation(line)
                    if observed:
                        cloud_stage = observed.get("phase") or (
                            "checkpoint" if observed.get("checkpoint_saved") else "training"
                        )
                    await self.training_observation(job, line)
                    return
                if message.startswith("Optimizer step"):
                    cloud_stage = "training"
                elif message.startswith("Downloading training"):
                    cloud_stage = "saving"
                elif message.startswith("Stopping Google Cloud"):
                    cloud_stage = "finishing"
                await self.event(job, cloud_stage, message)

            from vla_platform.huggingface_connection import HuggingFaceConnection

            hf_token = HuggingFaceConnection(self.settings.data_dir).token()
            secret_options = {"hf_token": hf_token} if hf_token else {}
            returncode = await sky_runner.run(
                payload, stage_dir, target, progress, **secret_options
            )
        else:
            container_name = "firebird-" + job.id + "-" + output
            argv, cwd, env = command(
                runtime,
                request_path.resolve(),
                result_path.resolve(),
                self.settings.data_dir.resolve(),
                container_name,
                training,
                operation in {"policy.import", "policy.quantize"},
                operation in {"policy.evaluate", "policy.run"},
            )
            image = runtime.training_image if training else runtime.image
            if operation in {"policy.import", "policy.quantize"} and runtime.conversion_python:
                image = runtime.conversion_image
            if operation in {"policy.evaluate", "policy.run"} and (
                runtime.evaluation_python or runtime.evaluation_image
            ):
                image = runtime.evaluation_image
            await self.event(job, output, f"Starting {operation} on {runtime.label}")
            # Settings may change while a queued run waits, or during an earlier
            # workflow stage. Check again immediately before starting each worker.
            self.compute.require_enabled(runtime)
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            written = 0
            pending = ""
            try:
                with (stage_dir / "worker.log").open("wb") as log:
                    while chunk := await process.stdout.read(8192):
                        if written < 5 * 1024 * 1024:
                            log.write(chunk)
                            log.flush()
                            written += len(chunk)
                        pending += chunk.decode("utf-8", errors="replace")
                        lines = pending.split("\n")
                        pending = lines.pop()[-16384:]
                        for line in lines:
                            if operation == "policy.finetune":
                                await self.training_observation(job, line)
                                continue
                            try:
                                progress = json.loads(line)
                            except ValueError, TypeError:
                                continue
                            if isinstance(progress, dict) and type(progress.get("step")) is int:
                                await self.event(job, output, f"Optimizer step {progress['step']}")
                await process.wait()
            finally:
                await self.stop(process, container_name if image else None)
            returncode = process.returncode
        if not result_path.exists() or result_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Worker did not produce a bounded result")
        response = json.loads(
            result_path.read_text(),
            parse_constant=reject_nonfinite,
            parse_float=finite_json_float,
        )
        if response.get("schema_version") != 1 or response.get("job_id") != job.id:
            raise ValueError("Worker response identity mismatch")
        if returncode or response.get("error"):
            raise RuntimeError(str(response.get("error", "Native worker failed"))[:2000])
        report = {
            **response.get("report", {}),
            "stage": output,
            "operation": operation,
            "artifact_id": artifact.id if artifact else None,
            "final": final,
        }
        result.reports.append(report)
        created = None
        if response.get("artifact"):
            info = response["artifact"]
            path = Path(info["path"])
            manifest, sha, total = await asyncio.to_thread(validate_bundle, path, stage_dir)
            created = PolicyArtifact(
                id=f"{job.id}:{output}",
                project_id=job.project_id,
                job_id=job.id,
                label=info["label"],
                format=info["format"],
                path=path.resolve().relative_to(self.settings.data_dir.resolve()).as_posix(),
                manifest_sha256=sha,
                file_bytes=total,
                parent_ids=[artifact.id] if artifact else [],
                metadata=manifest.get("metadata", {}),
            )
            if register_artifact:
                result.artifacts.append(created)
        await self.publish(job, result)
        await self.event(job, output, f"Completed {operation}")
        return created, report

    async def run(self, job: Job):
        request = job.request
        result = LifecycleResult()
        artifact = (
            await self.artifact(job.project_id, request.artifact_id)
            if request.artifact_id
            else None
        )
        if request.operation != "policy.workflow":
            # Native runtimes can use distinct training and conversion environments.
            # Cloud workers perform these steps together beside the remote checkpoint.
            if (
                request.operation == "policy.quantize"
                and self.runtime(request.runtime_id).execution != "skypilot"
            ):
                if artifact.format == "training_checkpoint":
                    artifact, _ = await self.native(
                        job, "policy.export", artifact, "training-export", result
                    )
                if artifact.format == "native_checkpoint":
                    artifact, _ = await self.native(
                        job, "policy.import", artifact, "floating-conversion", result
                    )
            _, report = await self.native(job, request.operation, artifact, "operation", result)
            if report.get("scope") == "engine_diagnostics":
                result.decision = "diagnostics_only"
            await self.publish(job, result)
            return result
        if request.training is not None:
            artifact, _ = await self.native(job, "policy.finetune", artifact, "training", result)
        elif request.source_id:
            artifact, _ = await self.native(job, "policy.import", None, "import", result)
        if artifact.format == "training_checkpoint":
            artifact, _ = await self.native(
                job, "policy.export", artifact, "training-export", result
            )
        if artifact.format == "native_checkpoint":
            artifact, _ = await self.native(
                job, "policy.import", artifact, "floating-conversion", result
            )
        if artifact.format != "gguf" or artifact.metadata.get("precision") != "float":
            raise ValueError("Workflow baseline must be a floating GGUF")
        baseline = artifact
        if request.evaluation.suite == "libero_spatial":
            from .spatial import run_spatial

            return await run_spatial(self, job, result, baseline)
        _, reference = await self.native(job, "policy.evaluate", baseline, "baseline", result)
        measured = [(baseline, reference)]
        for index, precision in enumerate(request.candidates):
            try:
                candidate, _ = await self.native(
                    job,
                    "policy.quantize",
                    baseline,
                    f"quantize-{index}",
                    result,
                    precision=precision,
                )
                _, measurement = await self.native(
                    job, "policy.evaluate", candidate, f"evaluate-{index}", result
                )
                if measurement.get("runtime") != reference.get("runtime"):
                    raise ValueError("Candidate runtime differs from the measured reference")
                measured.append((candidate, measurement))
            except Exception as exc:
                result.reports.append(
                    {"stage": f"candidate-{index}", "status": "failed", "error": str(exc)[:2000]}
                )
                await self.event(job, f"candidate-{index}", f"Candidate failed: {exc}")
                await self.publish(job, result)
        if len(measured) < 2:
            result.decision = "diagnostics_only"
            await self.event(
                job,
                "selection",
                "Insufficient comparison evidence: at least two runnable configurations required",
            )
            await self.publish(job, result)
            return result
        if request.evaluation.mode == "engine" or request.limits is None:
            result.decision = "diagnostics_only"
            await self.event(
                job,
                "selection",
                "Measurements complete; no automatic promotion without task-quality constraints",
            )
            await self.publish(job, result)
            return result
        limits = request.limits
        eligible = []
        if not complete_measurement(reference, len(request.evaluation.initial_states)):
            result.decision = "no_feasible_candidate"
            await self.publish(job, result)
            return result
        for candidate, measurement in measured:
            if (
                complete_measurement(measurement, len(request.evaluation.initial_states))
                and measurement["success_rate"] >= limits.min_success_rate
                and measurement["success_rate"]
                >= reference["success_rate"] - limits.max_success_drop
                and measurement["p95_ms"] <= limits.max_p95_ms
                and measurement.get("peak_device_mib") is not None
                and measurement["peak_device_mib"] <= limits.max_peak_device_mib
            ):
                eligible.append((candidate, measurement))
        if not eligible:
            result.decision = "no_feasible_candidate"
            await self.event(
                job, "selection", "No tested candidate meets all requested constraints"
            )
        else:
            chosen, _ = min(eligible, key=lambda item: item[0].metadata["weight_bytes"])
            await self.event(
                job, "selection", f"Frozen candidate: {chosen.label}; checking unused states"
            )
            _, final_reference = await self.native(
                job, "policy.evaluate", baseline, "final-reference", result, final=True
            )
            _, final = await self.native(
                job, "policy.evaluate", chosen, "final-evaluation", result, final=True
            )
            if (
                not complete_measurement(final_reference, len(request.evaluation.final_states))
                or not complete_measurement(final, len(request.evaluation.final_states))
                or final.get("runtime") != reference.get("runtime")
                or final_reference.get("runtime") != reference.get("runtime")
                or final.get("complete_episodes") != len(request.evaluation.final_states)
                or final["success_rate"] < final_reference["success_rate"] - limits.max_success_drop
                or final.get("success_rate", -1) < limits.min_success_rate
                or final["p95_ms"] > limits.max_p95_ms
                or final.get("peak_device_mib") is None
                or final["peak_device_mib"] > limits.max_peak_device_mib
            ):
                result.decision = "no_feasible_candidate"
                await self.event(
                    job,
                    "final-evaluation",
                    "Frozen candidate failed final acceptance; no replacement selected",
                )
            else:
                package, reloaded = await self.native(
                    job, "policy.run", chosen, "package-and-reload", result, final=True
                )
                if (
                    not complete_measurement(reloaded, len(request.evaluation.final_states))
                    or reloaded.get("runtime") != reference.get("runtime")
                    or reloaded.get("complete_episodes") != len(request.evaluation.final_states)
                    or reloaded.get("success_rate", -1)
                    < final_reference["success_rate"] - limits.max_success_drop
                    or reloaded.get("success_rate", -1) < limits.min_success_rate
                    or reloaded.get("p95_ms", float("inf")) > limits.max_p95_ms
                    or reloaded.get("peak_device_mib") is None
                    or reloaded["peak_device_mib"] > limits.max_peak_device_mib
                ):
                    result.decision = "no_feasible_candidate"
                    await self.event(job, "package-and-reload", "Package failed final acceptance")
                else:
                    result.selected_artifact_id = package.id
                    result.decision = "validated"
        await self.publish(job, result)
        return result
