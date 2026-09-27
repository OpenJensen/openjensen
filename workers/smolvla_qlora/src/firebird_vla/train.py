"""Run an isolated native SmolVLA QLoRA worker with an adapter-only checkpoint contract."""

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

from .accumulation import (
    CONTRACT_FILE,
    STATE_FILE,
    loss_weight,
    progress_record,
    read_record,
    resolve_contract,
    validate_progress,
)
from .checkpoint import (
    resolve_checkpoint,
    save_checkpoint,
    validate_resume_recipe,
    validate_resume_state,
    verify_training_checkpoint,
    write_json,
)
from .checkpoint import validate_resume_cursor as validate_resume_cursor
from .config import TrainConfig, batch_indices
from .telemetry import environment_report, report


def lr_multiplier(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def optimizer_step(parameters, optimizer, scaler, max_grad_norm):
    """Unscale once per accumulated batch and never update weights on overflow."""
    import torch

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_grad_norm, error_if_nonfinite=not scaler.is_enabled()
    )
    if not torch.isfinite(grad_norm):
        # A finite-gradient norm itself can overflow; explicitly skip even if
        # GradScaler's element-wise checks did not detect that case.
        scaler.update(new_scale=scaler.get_scale() * 0.5)
        return grad_norm, False
    scaler.step(optimizer)
    scaler.update()
    return grad_norm, True


def evaluate(policy, loader, preprocessor, batches, seed, camera_keys=None, compute_dtype=None):
    import torch

    from .data import prepare_batch
    from .model import runtime_compute_dtype

    compute_dtype = compute_dtype or runtime_compute_dtype()
    policy.eval()
    total, count = 0.0, 0
    try:
        # Fixed flow-matching noise makes repeated validation comparable; preserve training RNG.
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            for index, batch in enumerate(loader):
                if index >= batches:
                    break
                with torch.autocast("cuda", dtype=compute_dtype):
                    loss, _ = policy(prepare_batch(batch, preprocessor, camera_keys))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation loss")
                n = len(batch["action"])
                total += loss.item() * n
                count += n
    finally:
        policy.train()
    if not count:
        raise ValueError("Validation split has no frames")
    return {"validation_loss": total / count, "validation_frames": count}


def train(cfg, resume=None):
    import torch
    from torch.utils.data import DataLoader, default_collate

    from .data import load_data, prepare_batch
    from .model import build_policy, make_processors, predict, require_runtime
    from .temporal import check_dataset_temporal, resolved_temporal

    compute_dtype = require_runtime()
    dtype_name = "float16" if compute_dtype == torch.float16 else "bfloat16"
    scaler = torch.amp.GradScaler("cuda", enabled=compute_dtype == torch.float16)
    output = Path(cfg.output_dir)
    # A resumed job writes into a NEW run directory, preserving its source evidence.
    output.mkdir(parents=True, exist_ok=False)
    step = 0
    started = time.monotonic()
    write_json(output / "recipe.json", cfg.to_dict())
    report(
        output,
        "preparing",
        "Training recipe resolved",
        step=0,
        total_steps=cfg.steps,
        recipe=cfg.to_dict(),
    )
    try:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        environment = environment_report(torch, dtype_name)
        write_json(output / "environment.json", environment)
        report(
            output,
            "preparing",
            "Loading pinned dataset and computing training statistics",
            step=0,
            total_steps=cfg.steps,
            environment=environment,
        )
        resume_path = resolve_checkpoint(resume) if resume else None
        manifest, saved_splits = None, None
        if resume_path:
            manifest = verify_training_checkpoint(resume_path)
            if manifest.get("compute_dtype", "bfloat16") != dtype_name:
                raise ValueError("Resume requires the checkpoint's original GPU compute precision")
            validate_resume_recipe(resume_path, cfg)
            saved_splits = json.loads((resume_path / "splits.json").read_text())
        optimization = resolve_contract(
            cfg.batch_size, cfg.gradient_accumulation_steps, resume=resume_path
        )
        if (
            resume_path
            and (resume_path / CONTRACT_FILE).exists() != (resume_path / STATE_FILE).exists()
        ):
            raise ValueError("Incomplete optimization checkpoint")
        train_set, validation_set, stats, splits = load_data(cfg, saved_splits)
        write_json(output / CONTRACT_FILE, optimization)
        write_json(output / "splits.json", splits)
        write_json(output / "stats.json", stats)
        report(
            output,
            "preparing",
            "Loading pinned base model and preparing adapters",
            step=0,
            total_steps=cfg.steps,
            splits=splits,
            train_frames=len(train_set),
            validation_frames=len(validation_set),
        )
        if resume_path and stats != json.loads((resume_path / "stats.json").read_text()):
            raise ValueError("Training normalization changed since checkpoint")
        policy, policy_config, quantized = build_policy(
            cfg,
            features=train_set.features,
            policy_config_dir=resume_path / "policy" if resume_path else None,
            adapter_dir=resume_path / "adapter" if resume_path else None,
        )
        temporal = resolved_temporal(policy_config, train_set.meta.fps, "smolvla", cfg.to_dict())
        for dataset in (train_set, validation_set):
            check_dataset_temporal(temporal, dataset)
        if resume_path and (resume_path / "temporal-contract.json").exists():
            if json.loads((resume_path / "temporal-contract.json").read_text()) != temporal:
                raise ValueError("Resume temporal contract differs from the saved checkpoint")
        write_json(output / "temporal-contract.json", temporal)
        if manifest and quantized != manifest["quantized_modules"]:
            raise ValueError("Quantization layout changed since checkpoint")
        preprocessor, postprocessor = make_processors(policy_config, stats)
        parameters = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: lr_multiplier(s, cfg.warmup_steps, cfg.steps)
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        probe_batch = prepare_batch(
            default_collate([validation_set[0]]), preprocessor, cfg.selected_camera_keys
        )
        consumed = 0
        if resume_path:
            state = torch.load(resume_path / "training.pt", map_location="cpu", weights_only=True)
            validate_resume_state(state, manifest, cfg, scaled=scaler.is_enabled())
            if state.get("compute_dtype", "bfloat16") != dtype_name:
                raise ValueError("Resume requires the checkpoint's original GPU compute precision")
            if scaler.is_enabled():
                if not state.get("scaler"):
                    raise ValueError("FP16 checkpoint is missing gradient scaler state")
                scaler.load_state_dict(state["scaler"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, consumed = state["step"], state["consumed_batches"]
            if (resume_path / STATE_FILE).exists():
                validate_progress(
                    read_record(resume_path / STATE_FILE),
                    optimization,
                    frames=len(train_set),
                    updates=step,
                    consumed=consumed,
                    skipped=consumed // cfg.gradient_accumulation_steps - step,
                )
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], device=0)
            random.setstate(state["python_rng"])
        write_json(output / "recipe.json", cfg.to_dict())
        write_json(output / "splits.json", splits)
        write_json(output / "stats.json", stats)
        write_json(
            output / "model_report.json",
            {
                "quantized_modules": quantized,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "gpu": torch.cuda.get_device_name(0),
                "compute_dtype": dtype_name,
                "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
                "optimization_contract": optimization,
                "note": "Last batch of each epoch may be smaller. No task-success measurement.",
            },
        )
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        report(
            output,
            "training",
            "Optimizing policy on demonstration batches",
            step=step,
            total_steps=cfg.steps,
            elapsed_seconds=time.monotonic() - started,
        )
        epoch_length = math.ceil(len(train_set) / cfg.batch_size)
        batch_iterator = None
        train_loss = 0.0
        consecutive_overflows = 0
        while step < cfg.steps:
            epoch, offset = divmod(consumed, epoch_length)
            if batch_iterator is None or offset == 0:
                sampler = batch_indices(len(train_set), cfg.batch_size, cfg.seed, epoch)[offset:]
                loader = DataLoader(
                    train_set,
                    batch_sampler=sampler,
                    num_workers=cfg.num_workers,
                    generator=torch.Generator().manual_seed(cfg.seed + epoch),
                )
                batch_iterator = iter(loader)
            batch = prepare_batch(next(batch_iterator), preprocessor, cfg.selected_camera_keys)
            with torch.autocast("cuda", dtype=compute_dtype):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss before optimizer step {step + 1}"
                )
            weight = loss_weight(optimization, consumed, len(train_set), len(batch["action"]))
            if optimization["loss_weighting"] == "equal_microbatch_mean_v1":
                # Keep historical checkpoint arithmetic, including division, unchanged.
                scaled_loss = loss / cfg.gradient_accumulation_steps
                loss_value = loss.detach().item() / cfg.gradient_accumulation_steps
            else:
                scaled_loss = loss if weight == 1.0 else loss * weight
                loss_value = loss.detach().item() * weight
            scaler.scale(scaled_loss).backward()
            train_loss += loss_value
            consumed += 1
            if consumed % cfg.gradient_accumulation_steps:
                continue
            if step == 0:
                if not any(
                    p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                    for name, p in policy.named_parameters()
                    if "lora_" in name
                ):
                    raise RuntimeError("No nonzero LoRA gradients on the first optimizer step")
            grad_norm, updated = optimizer_step(parameters, optimizer, scaler, cfg.max_grad_norm)
            optimizer.zero_grad(set_to_none=True)
            if not updated:
                train_loss = 0.0
                consecutive_overflows += 1
                if consecutive_overflows >= 32:
                    raise FloatingPointError(
                        "FP16 gradients remained non-finite after 32 scale reductions"
                    )
                continue
            consecutive_overflows = 0
            scheduler.step()
            step += 1
            record = {
                "step": step,
                "total_steps": cfg.steps,
                "phase": "training",
                "train_loss": train_loss,
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "skipped_optimizer_steps": consumed // cfg.gradient_accumulation_steps - step,
                "optimization_state": progress_record(
                    optimization,
                    frames=len(train_set),
                    updates=step,
                    consumed=consumed,
                    skipped=consumed // cfg.gradient_accumulation_steps - step,
                ),
                "compute_dtype": dtype_name,
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            train_loss = 0.0
            if step % cfg.eval_every == 0 or step == cfg.steps:
                report(
                    output,
                    "validation",
                    "Measuring loss on held-out episodes",
                    step=step,
                    total_steps=cfg.steps,
                    elapsed_seconds=time.monotonic() - started,
                )
                record.update(
                    evaluate(
                        policy,
                        validation_loader,
                        preprocessor,
                        cfg.eval_batches,
                        cfg.seed + 1,
                        cfg.selected_camera_keys,
                        compute_dtype,
                    )
                )
            record["elapsed_seconds"] = time.monotonic() - started
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if (
                step == 1
                or step % cfg.log_every == 0
                or step % cfg.eval_every == 0
                or step == cfg.steps
            ):
                report(
                    output,
                    "training",
                    "Optimizing policy on demonstration batches",
                    **{key: value for key, value in record.items() if key != "phase"},
                )
            if step % cfg.save_every == 0 or step == cfg.steps:
                report(
                    output,
                    "checkpoint",
                    "Saving model, optimizer and random states",
                    step=step,
                    total_steps=cfg.steps,
                )
                probe = predict(policy, probe_batch, postprocessor, cfg.seed)
                saved = save_checkpoint(
                    output / f"checkpoint-{step:06d}",
                    policy,
                    policy_config,
                    cfg,
                    stats,
                    splits,
                    quantized,
                    optimizer,
                    scheduler,
                    step,
                    consumed,
                    probe,
                    scaler=scaler,
                    compute_dtype=dtype_name,
                    temporal_contract=temporal,
                    optimization_contract=optimization,
                    optimization_state=record["optimization_state"],
                )
                print(json.dumps({"checkpoint_saved": saved.name, "step": step}), flush=True)
                if step < cfg.steps:
                    report(
                        output,
                        "training",
                        "Optimizing policy on demonstration batches",
                        step=step,
                        total_steps=cfg.steps,
                    )
            elif step % cfg.eval_every == 0:
                report(
                    output,
                    "training",
                    "Optimizing policy on demonstration batches",
                    step=step,
                    total_steps=cfg.steps,
                )
        # Application success requires its separate fresh-process reload check.
        report(
            output,
            "verifying",
            "Optimizer steps finished; checkpoint reload is pending",
            step=step,
            total_steps=cfg.steps,
            elapsed_seconds=time.monotonic() - started,
            optimizer_completed=True,
            reload_verified=False,
            task_success=None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except BaseException as error:
        report(
            output,
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            str(error),
            step=step,
            total_steps=cfg.steps,
            error=str(error),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smolvla_qlora.json")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--resume", help="Checkpoint or interrupted run directory to continue into a new output"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Two optimizer steps and one eval batch"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate/print recipe without ML imports"
    )
    args = parser.parse_args()
    cfg = TrainConfig.load(args.config)
    if args.smoke:
        if args.resume:
            parser.error("--smoke and --resume cannot be combined")
        cfg = replace(
            cfg,
            steps=2,
            warmup_steps=0,
            gradient_accumulation_steps=1,
            eval_every=2,
            eval_batches=1,
            save_every=2,
            log_every=1,
            output_dir=cfg.output_dir + "-smoke",
        )
    if args.output_dir:
        cfg = replace(cfg, output_dir=args.output_dir)
    cfg.validate()
    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    try:
        train(cfg, args.resume)
    except ImportError as error:
        raise SystemExit(f"Training dependency unavailable: {error}. Install .[smolvla] on CUDA.")


if __name__ == "__main__":
    main()
