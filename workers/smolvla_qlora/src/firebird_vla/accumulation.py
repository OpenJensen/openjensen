"""Versioned optimizer-update semantics, independent of the optional ML runtime.

Native LeRobot's loop counter counts microbatches. Firebird's public counter
counts applied optimizer updates. Only complete windows may be checkpointed.
"""

import json
from pathlib import Path

CONTRACT_FILE = "optimization-contract.json"
STATE_FILE = "optimization-state.json"
SAMPLE_WEIGHTING = "sample_mean_v1"
LEGACY_WEIGHTING = "equal_microbatch_mean_v1"


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def accumulation_steps(recipe, family):
    count = positive(recipe.get("gradient_accumulation_steps", 1), "gradient_accumulation_steps")
    if count != 1 and family not in {"act", "smolvla"}:
        raise ValueError(f"{family} currently requires gradient_accumulation_steps=1")
    return count


def make_contract(batch_size, accumulation, *, weighting=SAMPLE_WEIGHTING):
    positive(batch_size, "batch_size")
    positive(accumulation, "gradient_accumulation_steps")
    if weighting not in {SAMPLE_WEIGHTING, LEGACY_WEIGHTING}:
        raise ValueError("Unknown optimization loss weighting")
    return {
        "schema_version": 1,
        "step_unit": "optimizer_updates",
        "scheduler_unit": "optimizer_updates",
        "checkpoint_boundary": "complete_window",
        "loss_weighting": weighting,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "world_size": 1,
        "nominal_effective_batch_size": batch_size * accumulation,
    }


def exact_json(left, right):
    """JSON equality without accepting bool/int or int/float substitutions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_json(a, b) for a, b in zip(left, right))
    return left == right


def read_record(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Optimization record exceeds 64 KiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Optimization record must be an object")
    return value


def resolve_contract(batch_size, accumulation, *, resume=None, native=False):
    """Old Smol checkpoints retain equal-microbatch weighting; old native means A=1."""
    expected = make_contract(batch_size, accumulation)
    if resume is None:
        return expected
    path = Path(resume) / CONTRACT_FILE
    if not path.exists():
        if native and accumulation != 1:
            raise ValueError("Accumulated native resume requires its optimization contract")
        return make_contract(batch_size, accumulation, weighting=LEGACY_WEIGHTING)
    saved = read_record(path)
    weighting = saved.get("loss_weighting")
    expected = make_contract(batch_size, accumulation, weighting=weighting)
    if not exact_json(saved, expected):
        raise ValueError("Checkpoint optimization contract differs from the recipe")
    return saved


def examples_consumed(microbatches, frames, batch_size):
    positive(frames, "training frames")
    positive(batch_size, "batch_size")
    if type(microbatches) is not int or microbatches < 0:
        raise ValueError("Invalid consumed microbatch cursor")
    per_epoch = (frames + batch_size - 1) // batch_size
    epoch, offset = divmod(microbatches, per_epoch)
    return epoch * frames + min(offset * batch_size, frames)


def window_counts(consumed, frames, batch_size, accumulation):
    """Count examples without prefetching images or buffering an accumulation window."""
    positive(accumulation, "gradient_accumulation_steps")
    start = consumed - consumed % accumulation
    examples = examples_consumed(consumed + 1, frames, batch_size) - examples_consumed(
        consumed, frames, batch_size
    )
    total = examples_consumed(start + accumulation, frames, batch_size) - examples_consumed(
        start, frames, batch_size
    )
    return examples, total


def loss_weight(contract, consumed, frames, observed_batch_size):
    expected, total = window_counts(
        consumed, frames, contract["batch_size"], contract["gradient_accumulation_steps"]
    )
    if type(observed_batch_size) is not int or observed_batch_size != expected:
        raise ValueError("Observed microbatch size differs from the saved sampler contract")
    if contract["gradient_accumulation_steps"] == 1:
        return 1.0  # Preserve the old no-accumulation arithmetic exactly.
    if contract["loss_weighting"] == LEGACY_WEIGHTING:
        return 1.0 / contract["gradient_accumulation_steps"]
    return expected / total


def progress_record(contract, *, frames, updates, consumed, skipped=0):
    for value, name in (
        (updates, "optimizer updates"),
        (consumed, "microbatches"),
        (skipped, "skips"),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {name} cursor")
    count = contract["gradient_accumulation_steps"]
    if consumed % count or consumed != (updates + skipped) * count:
        raise ValueError("Optimization checkpoint is not a complete update window")
    per_epoch = (positive(frames, "training frames") + contract["batch_size"] - 1) // contract[
        "batch_size"
    ]
    epoch, offset = divmod(consumed, per_epoch)
    return {
        "schema_version": 1,
        "completed_optimizer_updates": updates,
        "consumed_microbatches": consumed,
        "consumed_examples": examples_consumed(consumed, frames, contract["batch_size"]),
        "skipped_update_windows": skipped,
        "window_microbatches": 0,
        "training_frames": frames,
        "sampler_epoch": epoch,
        "sampler_batch_offset": offset,
    }


def validate_progress(saved, contract, *, frames, updates, consumed, skipped=0):
    expected = progress_record(
        contract, frames=frames, updates=updates, consumed=consumed, skipped=skipped
    )
    if not exact_json(saved, expected):
        raise ValueError("Checkpoint optimization/data cursors are inconsistent")
    return expected


def native_loop_values(recipe, family):
    count = accumulation_steps(recipe, family)
    # Legacy A=1 values are unchanged. All user-facing frequencies count updates.
    return {
        "steps": positive(recipe["steps"], "steps") * count,
        "save_freq": positive(recipe["save_every"], "save_every") * count,
        "eval_steps": positive(recipe["eval_every"], "eval_every") * count,
        "log_freq": positive(recipe["log_every"], "log_every") * count,
        "accelerator.gradient_accumulation.steps": count,
    }


def bind_resumed_sampler_epoch(sampler, *, consumed, frames, batch_size):
    """Translate Accelerate's fresh iterator epoch into the saved native epoch.

    LeRobot restores the sample offset before accelerator.prepare; DataLoaderShard
    then calls set_epoch(0). Keep that wrapper-local counter relative to the saved
    epoch, without changing LeRobot's permutation or within-epoch cursor.
    """
    per_epoch = (positive(frames, "training frames") + batch_size - 1) // positive(
        batch_size, "batch_size"
    )
    epoch, batch_offset = divmod(consumed, per_epoch)
    expected = {"epoch": epoch, "start_index": batch_offset * batch_size}
    if not exact_json(sampler.state_dict(), expected):
        raise ValueError("Native sampler does not match the saved microbatch cursor")
    original_set_epoch = sampler.set_epoch

    def set_epoch(wrapper_epoch):
        if type(wrapper_epoch) is not int or wrapper_epoch < 0:
            raise ValueError("Invalid wrapper sampler epoch")
        original_set_epoch(epoch + wrapper_epoch)

    sampler.set_epoch = set_epoch


class NativeAccumulation:
    """ACT-only microbatch adapter for the pinned LeRobot training loop.

    Retains LeRobot/Accelerate forward/backward and optimizer wrappers. No batch
    window is retained. Non-finite/skipped native updates fail rather than consume
    the requested update budget or publish an unachieved final checkpoint.
    """

    def __init__(self, contract, frames, *, updates=0, consumed=0):
        self.contract = contract
        self.frames = positive(frames, "training frames")
        progress_record(contract, frames=frames, updates=updates, consumed=consumed)
        self.updates = updates
        self.consumed = consumed
        self.window_loss = 0.0

    def record(self):
        return progress_record(
            self.contract, frames=self.frames, updates=self.updates, consumed=self.consumed
        )

    def update(
        self,
        metrics,
        policy,
        batch,
        optimizer,
        grad_clip_norm,
        *,
        accelerator,
        lr_scheduler=None,
        lock=None,
        sample_weighter=None,
    ):
        import torch

        if accelerator.num_processes != 1 or sample_weighter is not None or lock is not None:
            raise ValueError("Accumulated ACT training requires the single-process adapter")
        if accelerator.gradient_accumulation_steps != self.contract["gradient_accumulation_steps"]:
            raise ValueError("Accelerator accumulation differs from the optimization contract")
        if accelerator.gradient_state.sync_with_dataloader:
            raise ValueError("Accumulated ACT must not flush partial windows at an epoch boundary")
        weight = loss_weight(self.contract, self.consumed, self.frames, len(batch["action"]))
        policy.train()
        with accelerator.accumulate(policy):
            with accelerator.autocast():
                loss, output = policy(batch)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite native training loss")
            # Accelerator.backward divides by A. Undo that factor before applying
            # each microbatch's sample weight, including a short epoch-tail batch.
            accelerator.backward(loss * (weight * self.contract["gradient_accumulation_steps"]))
            self.window_loss += float(loss.detach()) * weight
            self.consumed += 1
            final = self.consumed % self.contract["gradient_accumulation_steps"] == 0
            if bool(accelerator.sync_gradients) != final:
                raise ValueError("Accelerator synchronization differs from the update cursor")
            grad_norm = None
            if final:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite accumulated native gradients")
            optimizer.step()  # AcceleratedOptimizer is a no-op before the sync boundary.
            optimizer.zero_grad()
            if final:
                if accelerator.optimizer_step_was_skipped:
                    raise FloatingPointError("Native optimizer skipped an accumulated update")
                if lr_scheduler is not None:
                    lr_scheduler.step()
                self.updates += 1
                metrics.grad_norm = float(grad_norm)
                metrics.loss = self.window_loss
                self.window_loss = 0.0
            else:
                metrics.loss = float(loss.detach())
            metrics.lr = optimizer.param_groups[0]["lr"]
        return metrics, output


def checkpoint_optimization(directory, *, native=False):
    """Validate hashed optimization evidence; dataset length is rechecked on resume."""
    directory = Path(directory)
    recipe = read_record(directory / "recipe.json")
    count = accumulation_steps(recipe, recipe.get("policy_type", "smolvla"))
    paths = [directory / CONTRACT_FILE, directory / STATE_FILE]
    present = [path.exists() for path in paths]
    if not any(present):
        if native and count != 1:
            raise ValueError("Accumulated native checkpoint lacks optimization evidence")
        return None
    if not all(present):
        raise ValueError("Incomplete optimization checkpoint")
    contract = resolve_contract(recipe["batch_size"], count, resume=directory, native=native)
    state = read_record(paths[1])
    manifest = read_record(directory / "manifest.json")
    validate_progress(
        state,
        contract,
        frames=state.get("training_frames"),
        updates=manifest.get("step"),
        consumed=state.get("consumed_microbatches"),
        skipped=state.get("skipped_update_windows"),
    )
    if native:
        metadata = read_record(directory / "training_state/training_step.json")
        expected = {
            "step": state["consumed_microbatches"],
            "batch_size": contract["batch_size"],
            "grad_accum_steps": count,
            "dp_world_size": 1,
        }
        if state["skipped_update_windows"] or not exact_json(
            {key: metadata.get(key) for key in expected}, expected
        ):
            raise ValueError("Native training metadata differs from optimization evidence")
    return {"contract": contract, "state": state}
