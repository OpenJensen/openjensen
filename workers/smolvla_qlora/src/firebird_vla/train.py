"""Run an isolated native SmolVLA QLoRA worker with an adapter-only checkpoint contract."""

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

from .checkpoint import save_checkpoint, verify_bundle, write_json
from .config import TrainConfig, batch_indices


def lr_multiplier(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(policy, loader, preprocessor, batches, seed):
    import torch

    from .data import prepare_batch

    policy.eval()
    total, count = 0.0, 0
    try:
        # Fixed flow-matching noise makes repeated validation comparable; preserve training RNG.
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            for index, batch in enumerate(loader):
                if index >= batches:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = policy(prepare_batch(batch, preprocessor))
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

    require_runtime()
    output = Path(cfg.output_dir)
    # A resumed job writes into a NEW run directory, preserving its source evidence.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"state": "running", "step": 0})
    step = 0
    started = time.monotonic()
    try:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        resume_path = Path(resume) if resume else None
        manifest, saved_splits = None, None
        if resume_path:
            manifest = verify_bundle(resume_path)
            previous = TrainConfig.load(resume_path / "recipe.json")
            if replace(previous, output_dir=cfg.output_dir) != cfg:
                raise ValueError("Resume requires the original recipe except output_dir")
            saved_splits = json.loads((resume_path / "splits.json").read_text())
        train_set, validation_set, stats, splits = load_data(cfg, saved_splits)
        if resume_path and stats != json.loads((resume_path / "stats.json").read_text()):
            raise ValueError("Training normalization changed since checkpoint")
        policy, policy_config, quantized = build_policy(
            cfg,
            features=train_set.features,
            policy_config_dir=resume_path / "policy" if resume_path else None,
            adapter_dir=resume_path / "adapter" if resume_path else None,
        )
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
        probe_batch = prepare_batch(default_collate([validation_set[0]]), preprocessor)
        consumed = 0
        if resume_path:
            state = torch.load(resume_path / "training.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, consumed = state["step"], state["consumed_batches"]
            if consumed != step * cfg.gradient_accumulation_steps:
                raise ValueError("Checkpoint batch cursor is inconsistent with optimizer step")
            if step >= cfg.steps:
                raise ValueError("Checkpoint already completed this recipe")
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
                "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
                "note": "Last batch of each epoch may be smaller. No task-success measurement.",
            },
        )
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_length = math.ceil(len(train_set) / cfg.batch_size)
        batch_iterator = None
        train_loss = 0.0
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
            batch = prepare_batch(next(batch_iterator), preprocessor)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss before optimizer step {step + 1}"
                )
            (loss / cfg.gradient_accumulation_steps).backward()
            train_loss += loss.detach().item() / cfg.gradient_accumulation_steps
            consumed += 1
            if consumed % cfg.gradient_accumulation_steps:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, cfg.max_grad_norm, error_if_nonfinite=True
            )
            if step == 0:
                if not any(
                    p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                    for name, p in policy.named_parameters()
                    if "lora_" in name
                ):
                    raise RuntimeError("No nonzero LoRA gradients on the first optimizer step")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {
                "step": step,
                "train_loss": train_loss,
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            train_loss = 0.0
            if step % cfg.eval_every == 0 or step == cfg.steps:
                record.update(
                    evaluate(
                        policy, validation_loader, preprocessor, cfg.eval_batches, cfg.seed + 1
                    )
                )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if step % cfg.log_every == 0 or step == cfg.steps:
                print(json.dumps(record), flush=True)
            if step % cfg.save_every == 0 or step == cfg.steps:
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
                )
                write_json(output / "latest.json", {"checkpoint": saved.name, "step": step})
        write_json(
            output / "status.json",
            {
                "state": "completed",
                "step": step,
                "reload_verified": False,
                "task_success": None,
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        )
    except BaseException as error:
        write_json(
            output / "status.json",
            {
                "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "step": step,
                "error": str(error),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smolvla_qlora.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Checkpoint to continue into a new output directory")
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
