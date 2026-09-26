"""Atomic, hashed adapter bundles. Full-precision base weights stay in the pinned Hub cache."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty checkpoint manifest")
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Checkpoint file inventory differs from manifest")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing, unsafe or corrupt checkpoint file: {name}")
    return manifest


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
        torch.save(
            {
                "step": step,
                "consumed_batches": consumed_batches,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(0),
                "python_rng": random.getstate(),
            },
            staging / "training.pt",
        )
        files = {
            str(p.relative_to(staging)): sha256(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "step": step,
                "files": files,
                "representation": "pinned-base+nf4-double-quant-bf16-storage+peft-adapter",
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
            },
        )
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_for_inference(directory):
    """Return (policy, preprocessor, postprocessor) for raw batched LeRobot observations."""
    from .config import TrainConfig
    from .model import build_policy, make_processors, require_runtime

    require_runtime()
    directory = Path(directory)
    manifest = verify_bundle(directory)
    cfg = TrainConfig.load(directory / "recipe.json")
    policy, config, quantized = build_policy(
        cfg,
        policy_config_dir=directory / "policy",
        adapter_dir=directory / "adapter",
        trainable=False,
    )
    if quantized != manifest["quantized_modules"]:
        raise ValueError("Reloaded quantization layout differs from the saved checkpoint")
    pre, post = make_processors(config, json.loads((directory / "stats.json").read_text()))
    policy.eval()
    return policy, pre, post
