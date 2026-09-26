"""Run the upstream native trainer with real Firebird telemetry and checkpoints."""

import copy
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .checkpoint import sha256, write_json
from .lerobot_application import cli_arguments
from .native_profiles import LEROBOT_REVISION, native_profile_for_recipe
from .telemetry import emit, environment_report


def compatible_model_snapshot(source, destination, policy_type):
    """Apply explicit upstream config migrations without changing cached weights."""
    config = json.loads((source / "config.json").read_text())
    if policy_type != "eo1" or "use_language_recipe" not in config:
        return source
    enabled = config.pop("use_language_recipe")
    if not enabled:
        config["recipe"] = None
    destination.mkdir()
    for path in source.iterdir():
        if path.name != "config.json":
            (destination / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    write_json(destination / "config.json", config)
    return destination


def metric_value(value):
    value = getattr(value, "val", value)
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("Native trainer emitted a non-finite metric")
    return value


def copy_checkpoint_file(source, destination):
    # Both trees live on the same worker disk; immutable hardlinks avoid another
    # full model/optimizer allocation. Cross-filesystem deployments copy safely.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def prune_native_working_checkpoints(current):
    """Keep two upstream working copies; published Firebird artifacts own history."""
    candidates = sorted(
        (
            path
            for path in current.parent.iterdir()
            if path.is_dir() and not path.is_symlink() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    for path in candidates[:-2]:
        if path != current:
            shutil.rmtree(path)


def commit_checkpoint(source, destination, recipe, *, step):
    """Commit a complete upstream snapshot, including optimizer state, atomically."""
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination.name}")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Native checkpoint contains a symlink")
    staging = Path(tempfile.mkdtemp(prefix=".native-checkpoint-", dir=destination.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, copy_function=copy_checkpoint_file)
        if any(path.is_symlink() for path in staging.rglob("*")):
            raise ValueError("Native checkpoint contains a symlink")
        write_json(staging / "recipe.json", recipe)
        files = {
            path.relative_to(staging).as_posix(): sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "step": step,
                "files": files,
                "representation": "lerobot-native-full",
                "training_backend": "lerobot",
                "policy_type": recipe["policy_type"],
                "upstream_revision": LEROBOT_REVISION,
                "reload_verified": False,
                "task_success": None,
            },
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if os.getenv("FIREBIRD_CHECKPOINT_EXPORT_ROOT"):
        from .snapshots import publish_checkpoint

        publish_checkpoint(destination, Path(os.environ["FIREBIRD_CHECKPOINT_EXPORT_ROOT"]))
    write_json(destination.parent / "latest.json", {"checkpoint": destination.name, "step": step})
    return destination


def apply_optimizer_overrides(cfg, recipe):
    """Keep upstream parameter groups while honoring explicit shared form controls."""
    if cfg.resume:
        return
    for source, target in (
        ("learning_rate", "lr"),
        ("weight_decay", "weight_decay"),
        ("max_grad_norm", "grad_clip_norm"),
    ):
        if source in recipe:
            setattr(cfg.optimizer, target, recipe[source])
    if "warmup_steps" in recipe:
        warmup = recipe["warmup_steps"]
        if cfg.scheduler is None and warmup:
            from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig

            cfg.scheduler = ConstantWithWarmupSchedulerConfig(num_warmup_steps=warmup)
        elif cfg.scheduler is not None:
            cfg.scheduler.num_warmup_steps = warmup
            if hasattr(cfg.scheduler, "num_decay_steps"):
                cfg.scheduler.num_decay_steps = cfg.steps


def probe(policy, raw_batch, preprocessor, postprocessor, seed):
    import torch
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    was_training = policy.training
    policy.eval()
    policy.reset()
    precision_flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cameras = [key for key in policy.config.image_features if key in raw_batch]
    try:
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            processed = _preprocess_dataset_batch(
                copy.deepcopy(raw_batch), cameras, {}, preprocessor
            )
            action = postprocessor(policy.select_action(processed)).detach().float().cpu()
        if not torch.isfinite(action).all():
            raise FloatingPointError("Native checkpoint probe produced non-finite actions")
        return action
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        ) = precision_flags
        policy.reset()
        policy.train(was_training)


class ValidationLog(logging.Handler):
    def __init__(self, record):
        super().__init__()
        self.record = record

    def emit(self, record):
        match = re.search(r"step (\d+): eval_loss=([-+\d.eE]+)", record.getMessage())
        if match:
            self.record(int(match[1]), validation_loss=metric_value(match[2]))


def main():
    recipe_path, operation = map(Path, sys.argv[1:3])
    resume = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    recipe = json.loads(recipe_path.read_text())
    profile = native_profile_for_recipe(recipe)
    if profile is None or recipe.get("upstream_revision") != LEROBOT_REVISION:
        raise ValueError("Unregistered native worker recipe or upstream revision")
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.scripts import lerobot_train as trainer
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from torch.utils.data import default_collate

    if not torch.cuda.is_available():
        raise RuntimeError("Native policy training requires a CUDA GPU")
    training = operation / "training"
    training.mkdir()
    started = time.monotonic()
    write_json(training / "recipe.json", recipe)
    write_json(
        training / "environment.json",
        {
            **environment_report(torch, "policy_default"),
            "upstream_revision": LEROBOT_REVISION,
            "training_backend": "lerobot",
            "initialization": profile["initialization"],
        },
    )
    emit(
        "preparing",
        "Downloading the pinned dataset on the GPU worker",
        step=0,
        total_steps=recipe["steps"],
        recipe=recipe,
    )
    dataset_root = snapshot_download(
        recipe["dataset_id"], repo_type="dataset", revision=recipe["dataset_revision"]
    )
    meta = LeRobotDatasetMetadata(
        recipe["dataset_id"], root=dataset_root, revision=recipe["dataset_revision"]
    )
    features = {
        key: {"type": value.type.value, "shape": list(value.shape)}
        for key, value in dataset_to_policy_features(meta.features).items()
    }
    model_root = None
    if profile["initialization"] != "scratch":
        emit(
            "preparing",
            "Downloading the pinned model on the GPU worker",
            step=0,
            total_steps=recipe["steps"],
        )
        download_options = {}
        if profile.get("checkpoint_subdirectory"):
            download_options["allow_patterns"] = [profile["checkpoint_subdirectory"] + "/**"]
        model_root = Path(
            snapshot_download(
                profile["model_id"], revision=profile["model_revision"], **download_options
            )
        )
        if profile.get("checkpoint_subdirectory"):
            model_root /= profile["checkpoint_subdirectory"]
        if profile["initialization"] == "pretrained":
            model_root = compatible_model_snapshot(
                model_root, operation / "resolved-model", profile["policy_type"]
            )
    arguments = cli_arguments(
        recipe,
        profile,
        output=operation / "native-output",
        dataset_root=dataset_root,
        model_root=model_root,
        features=features,
    )
    if resume:
        arguments = [arg for arg in arguments if not arg.startswith(("--policy.",))]
        arguments += [
            "--resume=true",
            "--config_path=" + str(resume / "pretrained_model/train_config.json"),
        ]
    write_json(operation / "native-arguments.json", arguments)
    state = {
        "step": json.loads((resume / "manifest.json").read_text())["step"] if resume else 0,
        "validation": None,
    }

    def record(step, **metrics):
        measurement = {
            "step": step,
            "total_steps": recipe["steps"],
            "elapsed_seconds": time.monotonic() - started,
            **metrics,
        }
        with (training / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(measurement, allow_nan=False) + "\n")
        emit(
            "validating" if "validation_loss" in metrics else "training",
            "Held-out imitation loss"
            if "validation_loss" in metrics
            else "Optimizing native policy",
            **measurement,
        )

    original_data = trainer.make_train_eval_datasets

    def make_data(cfg):
        datasets = original_data(cfg)
        state["validation"] = datasets[1]
        if state["validation"] is None or len(state["validation"]) == 0:
            raise ValueError("The selected dataset has no held-out evaluation frames")
        splits = {"train": list(datasets[0].episodes), "validation": list(datasets[1].episodes)}
        if set(splits["train"]) & set(splits["validation"]):
            raise ValueError("Training and held-out episodes overlap")
        write_json(training / "splits.json", splits)
        emit(
            "preparing",
            "Dataset split prepared; loading native policy",
            step=0,
            total_steps=recipe["steps"],
            splits=splits,
        )
        return datasets

    original_optimizer = trainer.make_optimizer_and_scheduler

    def make_optimizer(cfg, policy):
        apply_optimizer_overrides(cfg, recipe)
        write_json(training / "resolved-config.json", cfg.to_dict())
        write_json(
            training / "environment.json",
            {
                **environment_report(torch, getattr(cfg.policy, "dtype", "policy_default")),
                "upstream_revision": LEROBOT_REVISION,
                "training_backend": "lerobot",
                "initialization": profile["initialization"],
            },
        )
        return original_optimizer(cfg, policy)

    trainer.make_optimizer_and_scheduler = make_optimizer
    original_update = trainer.update_policy

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        accelerator = kwargs["accelerator"]
        if accelerator.optimizer_step_was_skipped:
            raise FloatingPointError(
                "Native optimizer skipped an update due to non-finite gradients"
            )
        state["step"] += 1
        if state["step"] == 1 or state["step"] % recipe["log_every"] == 0:
            metrics = result[0]
            record(
                state["step"],
                train_loss=metric_value(metrics.loss),
                learning_rate=metric_value(metrics.lr),
                grad_norm=metric_value(metrics.grad_norm),
            )
        return result

    original_save = trainer.save_checkpoint

    def save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        step = kwargs["step"]
        state["step"] = step
        directory = Path(kwargs["checkpoint_dir"])
        raw_batch = default_collate([state["validation"][0]])
        raw_batch = {
            key: value
            for key, value in raw_batch.items()
            if not key.startswith("observation.images.") or key in recipe["camera_keys"]
        }
        policy = kwargs["accelerator"].unwrap_model(kwargs["policy"])
        action = probe(
            policy, raw_batch, kwargs["preprocessor"], kwargs["postprocessor"], recipe["seed"]
        )
        torch.save(raw_batch, directory / "probe-batch.pt")
        torch.save(action, directory / "probe-action.pt")
        destination = commit_checkpoint(
            directory, training / f"checkpoint-{step:06d}", recipe, step=step
        )
        emit(
            "checkpoint",
            "Native checkpoint committed",
            step=step,
            total_steps=recipe["steps"],
            checkpoint=destination.name,
        )
        return result

    trainer.make_train_eval_datasets = make_data
    trainer.update_policy = update
    trainer.save_checkpoint = save
    original_last_checkpoint = trainer.update_last_checkpoint

    def update_last_checkpoint(directory):
        original_last_checkpoint(directory)
        prune_native_working_checkpoints(Path(directory))

    trainer.update_last_checkpoint = update_last_checkpoint
    original_logging = trainer.init_logging

    def init_logging(*args, **kwargs):
        original_logging(*args, **kwargs)
        logging.getLogger().addHandler(ValidationLog(record))

    trainer.init_logging = init_logging
    sys.argv = ["lerobot-train", *arguments]
    trainer.main()
    emit(
        "verifying",
        "Native optimizer steps finished; checkpoint reload pending",
        step=state["step"],
        total_steps=recipe["steps"],
        reload_verified=False,
    )


if __name__ == "__main__":
    main()
