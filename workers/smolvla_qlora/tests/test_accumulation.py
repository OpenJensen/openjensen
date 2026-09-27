"""Optimizer semantics and cursor admission without optional model dependencies."""

import json

import pytest

from firebird_vla.accumulation import (
    CONTRACT_FILE,
    LEGACY_WEIGHTING,
    STATE_FILE,
    accumulation_steps,
    checkpoint_optimization,
    examples_consumed,
    loss_weight,
    make_contract,
    native_loop_values,
    progress_record,
    resolve_contract,
    validate_progress,
    window_counts,
)


@pytest.mark.parametrize("value", [True, False, 1.0, 0, -1, "2", None])
def test_invalid_accumulation_is_never_coerced(value):
    with pytest.raises(ValueError, match="positive integer"):
        accumulation_steps({"gradient_accumulation_steps": value}, "act")


def test_only_act_and_smol_admit_accumulation():
    for family in ("act", "smolvla"):
        assert accumulation_steps({"gradient_accumulation_steps": 3}, family) == 3
    for family in ("diffusion", "psi0", "vqbet", "unknown"):
        with pytest.raises(ValueError, match="requires gradient_accumulation_steps=1"):
            accumulation_steps({"gradient_accumulation_steps": 3}, family)
        assert accumulation_steps({}, family) == 1


@pytest.mark.parametrize("frames,batch,accumulation", [(5, 4, 3), (1, 4, 8), (8, 2, 3), (9, 3, 2)])
def test_short_batch_windows_match_actual_epoch_sequence(frames, batch, accumulation):
    # Independent finite sequence includes short tails and windows crossing epochs.
    epoch = [len(range(start, min(start + batch, frames))) for start in range(0, frames, batch)]
    sizes = epoch * (accumulation * 3 + 1)
    contract = make_contract(batch, accumulation)
    for start in range(0, accumulation * 3, accumulation):
        total = sum(sizes[start : start + accumulation])
        weights = []
        for cursor in range(start, start + accumulation):
            assert window_counts(cursor, frames, batch, accumulation) == (sizes[cursor], total)
            weights.append(loss_weight(contract, cursor, frames, sizes[cursor]))
        assert sum(weights) == pytest.approx(1)
        assert examples_consumed(start + accumulation, frames, batch) == sum(
            sizes[: start + accumulation]
        )


def test_short_batch_sample_weighting_is_distinct_from_legacy_objective():
    new = make_contract(4, 2)
    old = make_contract(4, 2, weighting=LEGACY_WEIGHTING)
    # Four observations with loss 1 followed by one with loss 9: sample mean is 2.6.
    assert sum(
        loss * loss_weight(new, i, 5, n) for i, (loss, n) in enumerate([(1, 4), (9, 1)])
    ) == pytest.approx(2.6)
    assert (
        sum(loss * loss_weight(old, i, 5, n) for i, (loss, n) in enumerate([(1, 4), (9, 1)])) == 5
    )
    with pytest.raises(ValueError, match="microbatch size"):
        loss_weight(new, 1, 5, 4)


def test_loop_translation_preserves_optimizer_update_cadences():
    recipe = dict(steps=7, save_every=2, eval_every=3, log_every=1, gradient_accumulation_steps=4)
    assert native_loop_values(recipe, "act") == {
        "steps": 28,
        "save_freq": 8,
        "eval_steps": 12,
        "log_freq": 4,
        "accelerator.gradient_accumulation.steps": 4,
    }
    recipe["gradient_accumulation_steps"] = 1
    assert native_loop_values(recipe, "act")["steps"] == 7


def test_legacy_resume_retains_old_weighting_and_native_a1_only(tmp_path):
    before = list(tmp_path.iterdir())
    assert resolve_contract(4, 2, resume=tmp_path)["loss_weighting"] == LEGACY_WEIGHTING
    assert (
        resolve_contract(4, 1, resume=tmp_path, native=True)["loss_weighting"] == LEGACY_WEIGHTING
    )
    with pytest.raises(ValueError, match="requires its optimization contract"):
        resolve_contract(4, 2, resume=tmp_path, native=True)
    assert list(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("world_size", 1.0),
        ("batch_size", 8),
        ("loss_weighting", "other"),
        ("step_unit", "microbatches"),
    ],
)
def test_contract_mismatch_is_rejected_before_resume(tmp_path, key, value):
    contract = make_contract(4, 2)
    contract[key] = value
    (tmp_path / CONTRACT_FILE).write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        resolve_contract(4, 2, resume=tmp_path)


def test_progress_retains_skipped_windows_and_tail_data_cursor():
    contract = make_contract(4, 2)
    state = progress_record(contract, frames=5, updates=2, consumed=6, skipped=1)
    assert state["consumed_examples"] == 15
    assert state["sampler_epoch"] == 3
    assert state["sampler_batch_offset"] == 0
    validate_progress(state, contract, frames=5, updates=2, consumed=6, skipped=1)
    for key, value in [
        ("completed_optimizer_updates", True),
        ("consumed_examples", 16),
        ("training_frames", 6),
        ("window_microbatches", 1),
    ]:
        with pytest.raises(ValueError, match="inconsistent"):
            validate_progress(
                dict(state, **{key: value}), contract, frames=5, updates=2, consumed=6, skipped=1
            )
    with pytest.raises(ValueError, match="complete update window"):
        progress_record(contract, frames=5, updates=2, consumed=5)


def native_checkpoint(root):
    contract = make_contract(4, 2)
    state = progress_record(contract, frames=5, updates=2, consumed=4)
    records = {
        "recipe.json": {"policy_type": "act", "batch_size": 4, "gradient_accumulation_steps": 2},
        "manifest.json": {"step": 2},
        CONTRACT_FILE: contract,
        STATE_FILE: state,
        "training_state/training_step.json": {
            "step": 4,
            "batch_size": 4,
            "grad_accum_steps": 2,
            "dp_world_size": 1,
        },
    }
    for name, value in records.items():
        path = root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(value))
    return records


def test_native_checkpoint_cross_checks_actual_upstream_microbatch_count(tmp_path):
    native_checkpoint(tmp_path)
    evidence = checkpoint_optimization(tmp_path, native=True)
    assert evidence["state"]["completed_optimizer_updates"] == 2
    assert evidence["state"]["consumed_microbatches"] == 4
    path = tmp_path / "training_state/training_step.json"
    before = json.loads(path.read_text())
    for key, value in [
        ("step", 2),
        ("batch_size", 8),
        ("grad_accum_steps", 1),
        ("dp_world_size", True),
    ]:
        path.write_text(json.dumps(dict(before, **{key: value})))
        with pytest.raises(ValueError, match="metadata differs"):
            checkpoint_optimization(tmp_path, native=True)
    path.write_text(json.dumps(before))
    (tmp_path / STATE_FILE).unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint_optimization(tmp_path, native=True)


def test_oversize_optimization_record_refused(tmp_path):
    (tmp_path / CONTRACT_FILE).write_bytes(b" " * 65537)
    with pytest.raises(ValueError, match="64 KiB"):
        resolve_contract(4, 2, resume=tmp_path)


def test_validation_log_converts_only_complete_microstep_boundaries():
    import logging

    from firebird_vla.lerobot_train import ValidationLog

    records = []
    handler = ValidationLog(lambda step, **kw: records.append((step, kw)), accumulation=4)
    handler.emit(logging.makeLogRecord({"msg": "step 8: eval_loss=0.5"}))
    assert records == [(2, {"validation_loss": 0.5})]
    with pytest.raises(ValueError, match="incomplete"):
        handler.emit(logging.makeLogRecord({"msg": "step 7: eval_loss=0.5"}))
