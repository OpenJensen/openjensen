"""Train the released Psi-Zero action expert on a pinned LeRobot v2 dataset.

Uses the upstream model, data transforms, collator and flow-matching loss. The
Firebird loop adds disjoint episodes, train-only statistics, resumable optimizer
state, remote checkpoint publication and machine-readable metrics. VLM weights
remain frozen and are referenced by their immutable Hub revision in checkpoints.
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .checkpoint import sha256, verify_bundle, write_json
from .config import batch_indices, split_episodes
from .data import vector_stats
from .psi_profile import (
    PSI_EXPERT_DIRECTORY,
    PSI_MODEL_ID,
    PSI_MODEL_REVISION,
    PSI_REVISION,
    PSI_VLM_DIRECTORY,
    launch_config,
)
from .telemetry import environment_report, report


def require_runtime():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Psi-Zero needs its isolated Python 3.11 environment")
    root = Path(
        os.environ.get("FIREBIRD_PSI_ROOT") or os.environ.get("PSI_ROOT", "/opt/firebird/Psi0")
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PSI_REVISION:
        raise RuntimeError("Psi-Zero source checkout is not the registered immutable revision")
    from importlib.metadata import version

    # Official CUDA/Torch-specific wheels carry a PEP 440 local build suffix.
    if version("flash_attn").split("+", 1)[0] != "2.7.4.post1":
        raise RuntimeError("Psi-Zero requires flash_attn==2.7.4.post1")
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Psi-Zero requires a CUDA GPU with bfloat16 support")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The Firebird Psi-Zero adapter currently uses one visible GPU")


def model_snapshot():
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            PSI_MODEL_ID,
            revision=PSI_MODEL_REVISION,
            allow_patterns=[PSI_VLM_DIRECTORY + "/*", PSI_EXPERT_DIRECTORY + "/*"],
        )
    )


def build_trainer(recipe, model_root, stats_path, output):
    from psi.config.train.finetune_real_psi0_config import DynamicLaunchConfig
    from psi.trainers.finetune import FinetuneTrainer

    cfg = DynamicLaunchConfig.model_validate(
        launch_config(
            recipe,
            dataset_root=output,
            model_root=model_root,
            stats_path=stats_path,
            output=output,
        )
    )
    trainer = FinetuneTrainer(cfg, "cuda:0")
    trainer.init_models()
    trainer.model.to("cuda:0")
    # Every trainable tensor must be captured by our action-expert checkpoint.
    if any(
        parameter.requires_grad and not name.startswith("action_header.")
        for name, parameter in trainer.model.named_parameters()
    ):
        raise RuntimeError(
            "Upstream Psi-Zero introduced trainable weights outside the saved action expert"
        )
    return trainer


def load_data(recipe, previous_splits=None):
    from huggingface_hub import snapshot_download
    from psi.data.lerobot.compat import LeRobotDataset, LeRobotDatasetMetadata

    root = Path(
        snapshot_download(
            recipe["dataset_id"], repo_type="dataset", revision=recipe["dataset_revision"]
        )
    )
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError("The Psi-Zero upstream loader requires LeRobot v2.0 or v2.1 data")
    meta = LeRobotDatasetMetadata(
        recipe["dataset_id"], root=root, revision=recipe["dataset_revision"]
    )
    for key, dimension in (
        ("action", recipe["action_dim"]),
        (recipe["state_key"], recipe["state_dim"]),
    ):
        if meta.features.get(key, {}).get("shape") != [dimension]:
            raise ValueError(f"Dataset {key} differs from the inspected shape")
    ids = (
        list(meta.episodes)
        if isinstance(meta.episodes, dict)
        else [int(item["episode_index"]) for item in meta.episodes]
    )
    splits = split_episodes(ids, recipe["validation_fraction"], recipe["seed"])
    if previous_splits is not None and splits != previous_splits:
        raise ValueError("Resume requires the same disjoint episode split")
    # Single current images and state; only actions need a temporal chunk.
    kwargs = dict(
        repo_id=recipe["dataset_id"],
        root=root,
        revision=recipe["dataset_revision"],
        delta_timestamps={"action": [index / meta.fps for index in range(30)]},
        video_backend="pyav",
    )
    train = LeRobotDataset(**kwargs, episodes=splits["train"])
    validation = LeRobotDataset(**kwargs, episodes=splits["validation"])
    stats = {
        key: vector_stats((row[key] for row in train.hf_dataset.select_columns([key])), dimension)
        for key, dimension in (
            ("action", recipe["action_dim"]),
            (recipe["state_key"], recipe["state_dim"]),
        )
    }
    return train, validation, stats, splits


def prepare_batch(dataset, indices, trainer, recipe, collator):
    import numpy as np
    import torch

    samples = []
    for index in indices:
        raw = dict(dataset[index])
        if not isinstance(raw.get("task"), str) or not raw["task"].strip():
            raise ValueError("Every Psi-Zero sample requires a non-empty task instruction")
        if raw[recipe["state_key"]].ndim == 1:
            raw[recipe["state_key"]] = raw[recipe["state_key"]][None, :]
        item = trainer.data_cfg.transform(raw, vlm_processor=trainer.vlm_processor, no_aug=True)
        padding = raw.get("action_is_pad")
        if padding is None:
            raise ValueError("Psi-Zero needs the dataset's action chunk padding mask")
        item["actions_mask"] = (
            np.asarray(item["actions_mask"], dtype=bool) & ~np.asarray(padding, dtype=bool)[:, None]
        )
        samples.append(item)
    batch = collator(samples)
    # Logging-only raw images and strings need not occupy device memory.
    return {
        key: value.to("cuda:0")
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
        and key
        in {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "states",
            "actions",
            "actions_mask",
        }
    }


def predict(trainer, batch):
    import torch

    trainer.model.eval()
    with (
        torch.random.fork_rng(devices=[0]),
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        return trainer.inference(trainer.model, batch).float().cpu()


def save_checkpoint(
    output, trainer, recipe, stats, splits, optimizer, scheduler, step, consumed, probe
):
    import torch
    from safetensors.torch import save_file

    destination = output / f"checkpoint-{step:06d}"
    staging = Path(tempfile.mkdtemp(prefix=".psi-checkpoint-", dir=output))
    try:
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in trainer.model.action_header.state_dict().items()
            },
            str(staging / "action_header.safetensors"),
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "consumed_batches": consumed,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(0),
            },
            staging / "training.pt",
        )
        write_json(staging / "recipe.json", recipe)
        write_json(staging / "stats.json", stats)
        write_json(staging / "splits.json", splits)
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in probe.items()},
            str(staging / "probe-input.safetensors"),
        )
        save_file(
            {"action": predict(trainer, probe).contiguous()}, str(staging / "probe.safetensors")
        )
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "step": step,
                "representation": "pinned-psi-vlm+trained-action-expert",
                "training_backend": "psi0",
                "files": {
                    path.name: sha256(path) for path in sorted(staging.iterdir()) if path.is_file()
                },
                "task_success": None,
                "reload_verified": False,
            },
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    export_root = os.getenv("FIREBIRD_CHECKPOINT_EXPORT_ROOT")
    if export_root:
        from .snapshots import publish_checkpoint

        publish_checkpoint(destination, Path(export_root))
    write_json(output / "latest.json", {"step": step, "checkpoint": destination.name})
    report(output, "checkpoint", f"Checkpoint {step} saved", step=step, total_steps=recipe["steps"])


def train(recipe, output, resume=None):
    require_runtime()
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers.optimization import get_scheduler
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

    output.mkdir(parents=True, exist_ok=True)
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    torch.backends.cudnn.benchmark = False
    report(
        output,
        "preparing",
        "Downloading pinned Psi-Zero weights and dataset on the cloud GPU",
        step=0,
        total_steps=recipe["steps"],
    )
    previous_splits = json.loads((resume / "splits.json").read_text()) if resume else None
    train_data, val_data, stats, splits = load_data(recipe, previous_splits)
    if resume:
        saved_stats = json.loads((resume / "stats.json").read_text())
        if stats != saved_stats:
            raise ValueError("Training normalization changed from the checkpoint")
    write_json(output / "stats.json", stats)
    write_json(output / "splits.json", splits)
    write_json(output / "recipe.json", recipe)
    write_json(output / "environment.json", environment_report(torch, "bfloat16"))
    trainer = build_trainer(recipe, model_snapshot(), output / "stats.json", output)
    optimizer = trainer.create_optimizers()
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=recipe["warmup_steps"],
        num_training_steps=recipe["steps"],
    )
    collator = PaddedCollatorForTogether(
        model_max_length=4096, pad_token_id=trainer.tokenizer.pad_token_id
    )
    step = consumed = 0
    if resume:
        verify_bundle(resume)
        trainer.model.action_header.load_state_dict(
            load_file(str(resume / "action_header.safetensors")), strict=True
        )
        state = torch.load(resume / "training.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, consumed = state["step"], state["consumed_batches"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], 0)
        if step >= recipe["steps"]:
            raise ValueError("This checkpoint has already completed its recipe")
    probe = prepare_batch(val_data, [0], trainer, recipe, collator)
    probe = {key: value for key, value in probe.items() if key not in {"actions", "actions_mask"}}
    batches_per_epoch = math.ceil(len(train_data) / recipe["batch_size"])
    current_epoch, epoch_batches = None, []
    started = time.monotonic()
    while step < recipe["steps"]:
        trainer.model.train()
        trainer.model.vlm_model.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(recipe["gradient_accumulation_steps"]):
            epoch, position = divmod(consumed, batches_per_epoch)
            if epoch != current_epoch:
                epoch_batches = batch_indices(
                    len(train_data), recipe["batch_size"], recipe["seed"], epoch
                )
                current_epoch = epoch
            indices = epoch_batches[position]
            batch = prepare_batch(train_data, indices, trainer, recipe, collator)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = trainer.forward_and_loss(trainer.model, batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Psi-Zero produced a non-finite training loss")
            (loss / recipe["gradient_accumulation_steps"]).backward()
            losses.append(float(loss.detach()))
            consumed += 1
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in trainer.model.parameters() if parameter.requires_grad],
            recipe["max_grad_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        step += 1
        metrics = {
            "step": step,
            "train_loss": sum(losses) / len(losses),
            "learning_rate": scheduler.get_last_lr()[0],
            "grad_norm": float(norm),
            "elapsed_seconds": time.monotonic() - started,
        }
        if step % recipe["eval_every"] == 0 or step == recipe["steps"]:
            trainer.model.eval()
            values = []
            with torch.random.fork_rng(devices=[0]), torch.no_grad():
                torch.manual_seed(recipe["seed"] + 1)
                torch.cuda.manual_seed_all(recipe["seed"] + 1)
                for indices in batch_indices(
                    len(val_data), recipe["batch_size"], recipe["seed"], 0
                )[: recipe["eval_batches"]]:
                    batch = prepare_batch(val_data, indices, trainer, recipe, collator)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        value = float(trainer.forward_and_loss(trainer.model, batch)["loss"])
                    if not math.isfinite(value):
                        raise RuntimeError("Psi-Zero produced a non-finite validation loss")
                    values.append(value)
            metrics["validation_loss"] = sum(values) / len(values)
        record = report(
            output,
            "training",
            f"Optimizer step {step} completed",
            total_steps=recipe["steps"],
            **metrics,
        )
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if step % recipe["save_every"] == 0 or step == recipe["steps"]:
            save_checkpoint(
                output, trainer, recipe, stats, splits, optimizer, scheduler, step, consumed, probe
            )
    report(
        output,
        "completed",
        "Psi-Zero action-expert training completed",
        step=step,
        total_steps=recipe["steps"],
    )


def main():
    recipe_path, output, *resumes = sys.argv[1:]
    train(
        json.loads(Path(recipe_path).read_text()),
        Path(output),
        Path(resumes[0]) if resumes else None,
    )


if __name__ == "__main__":
    main()
