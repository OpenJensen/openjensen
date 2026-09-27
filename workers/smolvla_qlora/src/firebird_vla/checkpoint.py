"""Atomic, hashed adapter bundles. Full-precision base weights stay in the pinned Hub cache."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


def write_json(path, value):
    """Replace JSON atomically; a stopped writer leaves the previous version readable."""
    path = Path(path)
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}-", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sync_directory(path):
    # Windows does not expose directory handles to os.open; file fsync still applies there.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(directory):
    if Path(directory).is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    directory = Path(directory).resolve()
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Checkpoint files must not be symlinks")
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Checkpoint manifest is missing or invalid: {directory}") from exc
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise ValueError("Unsupported or empty checkpoint manifest")
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Checkpoint file inventory differs from manifest")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or not path.is_relative_to(directory)
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError(f"Missing, unsafe or corrupt checkpoint file: {name}")
    return manifest


def verify_training_checkpoint(directory):
    """Validate the complete native training inventory without importing Torch."""
    manifest = verify_bundle(directory)
    required = {
        "recipe.json",
        "stats.json",
        "splits.json",
        "training.pt",
        "probe.safetensors",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "policy/config.json",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Incomplete training checkpoint inventory")
    step = manifest.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint manifest must contain a positive optimizer step")
    name = Path(directory).name
    if re.fullmatch(r"checkpoint-\d{6,}", name) and int(name.split("-")[1]) != step:
        raise ValueError("Checkpoint directory and manifest optimizer steps differ")
    return manifest


def resolve_checkpoint(path):
    """Select the latest verified checkpoint; latest.json is only an advisory index.

    A rename may have committed a bundle before the pointer was published. Ignore
    incomplete staging directories and damaged candidates, but never silently fall
    back when an explicitly selected checkpoint fails validation.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    if (path / "manifest.json").exists() or re.fullmatch(r"checkpoint-\d{6,}", path.name):
        verify_training_checkpoint(path)
        return path
    candidates = sorted(
        (
            candidate
            for candidate in path.iterdir()
            if re.fullmatch(r"checkpoint-\d{6,}", candidate.name)
        ),
        key=lambda candidate: int(candidate.name.split("-")[1]),
        reverse=True,
    )
    failures = []
    for candidate in candidates:
        try:
            verify_training_checkpoint(candidate)
        except (OSError, ValueError) as exc:
            failures.append(f"{candidate.name}: {exc}")
            continue
        return candidate
    detail = "; ".join(failures) if failures else "no completed checkpoint directories"
    raise ValueError(f"No complete, integrity-verified training checkpoint in {path}: {detail}")


def commit_checkpoint_directory(staging, destination):
    """Durably expose a finished private directory; the caller validates its inventory."""
    staging, destination = Path(staging), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
    # Flush contents and directory entries before making this bundle discoverable.
    for path in staging.rglob("*"):
        if path.is_file():
            # Windows _commit requires write access for flushing file data.
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True):
        _sync_directory(path)
    _sync_directory(staging)
    os.rename(staging, destination)
    _sync_directory(destination.parent)
    return destination


def publish_checkpoint(staging, destination):
    """Commit a complete bundle before atomically advancing its advisory pointer."""
    staging, destination = Path(staging), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
    manifest = verify_training_checkpoint(staging)
    if destination.name != f"checkpoint-{manifest['step']:06d}":
        raise ValueError("Checkpoint destination and manifest optimizer steps differ")
    commit_checkpoint_directory(staging, destination)
    write_json(
        destination.parent / "latest.json",
        {"checkpoint": destination.name, "step": manifest["step"]},
    )
    return destination


def validate_resume_cursor(step, consumed, accumulation, scaled):
    """Scaled FP16 can consume batches without completing an optimizer update."""
    if (
        type(step) is not int
        or type(consumed) is not int
        or step < 0
        or consumed < step * accumulation
        or consumed % accumulation
        or (not scaled and consumed != step * accumulation)
    ):
        raise ValueError("Checkpoint batch cursor is inconsistent with optimizer step")


def validate_resume_state(state, manifest, cfg, *, scaled=False):
    """Reject a hashed but inconsistent optimizer cursor before restoring state."""
    if not isinstance(state, dict):
        raise ValueError("Checkpoint training state must be a mapping")
    step, consumed = state.get("step"), state.get("consumed_batches")
    if type(step) is not int or step < 1 or step != manifest["step"]:
        raise ValueError("Checkpoint training state and manifest optimizer steps differ")
    validate_resume_cursor(step, consumed, cfg.gradient_accumulation_steps, scaled)
    if step >= cfg.steps:
        raise ValueError("Checkpoint already completed this recipe")


def validate_resume_recipe(directory, cfg):
    from .config import TrainConfig

    previous = TrainConfig.load(Path(directory) / "recipe.json")
    if not cfg.resume_matches(previous):
        raise ValueError("Resume requires the original recipe except output_dir")


def save_checkpoint(
    destination,
    policy,
    policy_config,
    cfg,
    stats,
    splits,
    quantized,
    optimizer,
    scheduler,
    step,
    consumed_batches,
    probe_action,
    *,
    scaler=None,
    compute_dtype=None,
):
    import random
    from importlib.metadata import version

    import torch
    from safetensors.torch import save_file

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        policy.save_pretrained(
            staging / "adapter", safe_serialization=True, save_embedding_layers=False
        )
        policy_config.save_pretrained(staging / "policy")
        write_json(staging / "recipe.json", cfg.to_dict())
        write_json(staging / "stats.json", stats)
        write_json(staging / "splits.json", splits)
        save_file({"action": probe_action.contiguous()}, str(staging / "probe.safetensors"))
        training_state = {
            "step": step,
            "consumed_batches": consumed_batches,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(0),
            "python_rng": random.getstate(),
        }
        if compute_dtype is not None:
            if compute_dtype not in {"float16", "bfloat16"}:
                raise ValueError("Unsupported checkpoint compute dtype")
            training_state["compute_dtype"] = compute_dtype
        if scaler is not None and scaler.is_enabled():
            training_state["scaler"] = scaler.state_dict()
        torch.save(training_state, staging / "training.pt")
        files = {
            p.relative_to(staging).as_posix(): sha256(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "step": step,
                "files": files,
                "representation": (
                    "pinned-base+nf4-double-quant-"
                    + ("fp16" if compute_dtype == "float16" else "bf16")
                    + "-storage+peft-adapter"
                    if cfg.method == "qlora"
                    else "pinned-base+peft-adapter"
                ),
                "quantized_modules": quantized,
                "versions": {
                    name: version(name)
                    for name in (
                        "torch",
                        "lerobot",
                        "transformers",
                        "peft",
                        "bitsandbytes",
                        "safetensors",
                        "datasets",
                        "accelerate",
                        "huggingface-hub",
                    )
                },
                "task_success": None,
                "reload_verified": False,
                "compute_dtype": compute_dtype or "bfloat16",
            },
        )
        publish_checkpoint(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if os.getenv("FIREBIRD_CHECKPOINT_EXPORT_ROOT"):
        from .snapshots import publish_checkpoint as publish_snapshot

        publish_snapshot(destination, Path(os.environ["FIREBIRD_CHECKPOINT_EXPORT_ROOT"]))
    return destination


def load_for_inference(directory):
    """Return (policy, preprocessor, postprocessor) for raw batched LeRobot observations."""
    from .config import TrainConfig
    from .model import build_policy, make_processors, require_runtime, runtime_compute_dtype

    require_runtime()
    directory = Path(directory)
    manifest = verify_bundle(directory)
    # FP16-trained NF4 bases must still be packed from FP16 on newer GPUs;
    # silently taking the newer GPU's BF16 default changes the trained policy.
    selected_dtype = runtime_compute_dtype(manifest.get("compute_dtype", "bfloat16"))
    cfg = TrainConfig.load(directory / "recipe.json")
    policy, config, quantized = build_policy(
        cfg,
        policy_config_dir=directory / "policy",
        adapter_dir=directory / "adapter",
        trainable=False,
        compute_dtype=selected_dtype,
    )
    if quantized != manifest["quantized_modules"]:
        raise ValueError("Reloaded quantization layout differs from the saved checkpoint")
    pre, post = make_processors(config, json.loads((directory / "stats.json").read_text()))
    policy.eval()
    return policy, pre, post
