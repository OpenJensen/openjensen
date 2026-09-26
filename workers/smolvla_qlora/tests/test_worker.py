import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from firebird_vla.checkpoint import sha256, verify_bundle, write_json
from firebird_vla.config import TrainConfig, batch_indices, split_episodes
from firebird_vla.data import vector_stats
from firebird_vla.train import lr_multiplier


@pytest.mark.parametrize(
    "updates",
    [
        {"model_revision": "main"},
        {"dataset_revision": "v3.0"},
        {"batch_size": 0},
        {"steps": True},
        {"lora_rank": -1},
        {"num_workers": -1},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"lora_dropout": 1},
        {"validation_fraction": 0},
        {"warmup_steps": 1000},
        {"backbone_id": ""},
    ],
)
def test_invalid_recipe_rejected(updates):
    with pytest.raises(ValueError):
        replace(TrainConfig(), **updates).validate()


def test_typo_does_not_silently_use_default():
    with pytest.raises(ValueError, match="Unknown"):
        TrainConfig.from_dict({"lora_rnak": 8})


def test_split_is_disjoint_complete_and_independent_of_input_order():
    ids = list(range(30))
    split = split_episodes(ids, 0.2, 42)
    assert split == split_episodes(reversed(ids), 0.2, 42)
    assert len(split["train"]) == 24
    assert len(split["validation"]) == 6
    assert set(split["train"]).isdisjoint(split["validation"])
    assert sorted(split["train"] + split["validation"]) == ids
    with pytest.raises(ValueError):
        split_episodes([1, 1], 0.2, 42)


def test_training_statistics_exclude_held_out_outliers():
    split = split_episodes(range(5), 0.2, 42)
    frames = {i: [[float(i), 7.0]] for i in split["train"]}
    for i in split["validation"]:
        frames[i] = [[1e20, -1e20]]
    stats = vector_stats((v for i in split["train"] for v in frames[i]), 2)
    values = split["train"]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    assert stats["mean"] == pytest.approx([mean, 7.0])
    assert stats["std"] == pytest.approx([variance**0.5, 1e-6])
    assert stats["count"] == [4]


@pytest.mark.parametrize("rows", [[], [[1]], [[1, float("inf")]], [[float("nan"), 1]]])
def test_bad_data_fails_before_training(rows):
    with pytest.raises(ValueError):
        vector_stats(rows, 2)


def test_resume_batch_order_at_every_possible_checkpoint():
    # A non-divisible epoch length exercises both middle-of-epoch and epoch-boundary resumes.
    size, batch_size = 11, 3
    full = [batch for epoch in range(4) for batch in batch_indices(size, batch_size, 42, epoch)]
    epoch_length = 4
    for consumed in range(len(full)):
        epoch, offset = divmod(consumed, epoch_length)
        resumed = batch_indices(size, batch_size, 42, epoch)[offset:]
        resumed += [
            batch for e in range(epoch + 1, 4) for batch in batch_indices(size, batch_size, 42, e)
        ]
        assert resumed == full[consumed:]
    assert sorted(sum(full[:epoch_length], [])) == list(range(size))


def test_schedule_warms_up_and_reaches_floor():
    assert lr_multiplier(0, 10, 100) == pytest.approx(0.1)
    assert lr_multiplier(10, 10, 100) == pytest.approx(1)
    assert lr_multiplier(100, 10, 100) == pytest.approx(0.1)
    assert lr_multiplier(0, 0, 2) == pytest.approx(1)


def make_bundle(path):
    (path / "adapter").mkdir()
    weights = path / "adapter" / "weights.bin"
    weights.write_bytes(b"test fixture, not a model")
    write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "files": {"adapter/weights.bin": sha256(weights)},
        },
    )
    return weights


@pytest.mark.parametrize("mutation", ["corrupt", "missing", "extra", "traversal", "symlink"])
def test_checkpoint_integrity(tmp_path, mutation):
    weights = make_bundle(tmp_path)
    assert verify_bundle(tmp_path)["schema_version"] == 1
    if mutation == "corrupt":
        weights.write_bytes(b"changed")
    elif mutation == "missing":
        weights.unlink()
    elif mutation == "extra":
        (tmp_path / "unexpected.txt").write_text("unexpected")
    elif mutation == "traversal":
        write_json(
            tmp_path / "manifest.json",
            {
                "schema_version": 1,
                "files": {"../outside": "0" * 64},
            },
        )
    else:
        weights.unlink()
        weights.symlink_to(Path(__file__).resolve())
    with pytest.raises(ValueError):
        verify_bundle(tmp_path)


def test_cli_dry_run_needs_no_ml_stack_or_download(tmp_path):
    config = tmp_path / "recipe.json"
    write_json(config, TrainConfig().to_dict())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "firebird_vla.train",
            "--config",
            str(config),
            "--smoke",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = json.loads(result.stdout)
    assert resolved["steps"] == 2
    assert resolved["warmup_steps"] == 0
    assert not (tmp_path / "outputs").exists()
