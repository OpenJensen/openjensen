"""Temporal admission, loader timing, and checkpoint identity without CUDA."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lerobot_bridge import request

from firebird_vla.config import TrainConfig
from firebird_vla.lerobot_application import cli_arguments, resolve_recipe
from firebird_vla.model import configure_policy_temporal
from firebird_vla.temporal import (
    action_timestamps,
    check_dataset_temporal,
    resolved_temporal,
    validate_temporal,
)


@pytest.mark.parametrize("family", ["diffusion", "multi_task_dit", "vqbet"])
def test_previously_ignored_chunk_rejected_at_recipe_and_argument_boundaries(family):
    with pytest.raises(ValueError, match="does not support chunk_size"):
        resolve_recipe(request(family, chunk_size=13))
    recipe, profile = resolve_recipe(request(family))
    recipe["chunk_size"] = 13
    with pytest.raises(ValueError, match="does not support chunk_size"):
        cli_arguments(recipe, profile, output="out", dataset_root="data")
    # Historical checkpoints can contain the old ignored field; saved native
    # train_config owns resume and no temporal override is constructed.
    args = cli_arguments(recipe, profile, output="out", dataset_root="data", resume=True)
    assert not any("chunk_size" in arg or "n_action_steps" in arg for arg in args)


@pytest.mark.parametrize(
    "recipe",
    [
        {"prediction_horizon": True},
        {"execution_horizon": 1.0},
        {"prediction_horizon": 0},
        {"prediction_horizon": 1025},
        {"prediction_horizon": None},
        {"prediction_horizon": 2, "execution_horizon": 3},
        {"chunk_size": 50, "execution_horizon": 2},
        {"observation_history": 2},
        {"frame_stride": 2},
    ],
)
def test_strict_invalid_temporal_requests(recipe):
    for family in ("act", "smolvla"):
        with pytest.raises(ValueError):
            validate_temporal(recipe, family)


@pytest.mark.parametrize(
    "family", ["diffusion", "multi_task_dit", "vqbet", "pi05", "psi0", "openvla"]
)
def test_unverified_split_controls_rejected(family):
    with pytest.raises(ValueError, match="not verified"):
        validate_temporal({"execution_horizon": 2}, family)


def test_act_arguments_apply_separate_horizons_and_preserve_defaults():
    recipe, profile = resolve_recipe(request(prediction_horizon=8, execution_horizon=3))
    args = cli_arguments(recipe, profile, output="out", dataset_root="data")
    assert "--policy.chunk_size=8" in args
    assert "--policy.n_action_steps=3" in args
    assert validate_temporal({}, "act")["prediction_horizon"] == 100
    assert validate_temporal({"chunk_size": 7}, "act")["execution_horizon"] == 7
    with pytest.raises(ValueError, match="chunk_size=30"):
        validate_temporal({"chunk_size": 31}, "psi0")


def test_smol_recipe_roundtrip_legacy_and_exact_resume(tmp_path):
    old = TrainConfig.from_dict({"chunk_size": 9})
    assert old.temporal["prediction_horizon"] == old.temporal["execution_horizon"] == 9
    assert not set(old.to_dict()) & {"prediction_horizon", "execution_horizon"}
    cfg = TrainConfig.from_dict({"prediction_horizon": 8, "execution_horizon": 3})
    assert "chunk_size" not in cfg.to_dict()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(cfg.to_dict()))
    restored = TrainConfig.load(path)
    assert restored.resume_matches(cfg)
    assert not replace(cfg, execution_horizon=2).resume_matches(restored)
    assert old.resume_matches(TrainConfig.from_dict(old.to_dict()))
    config = SimpleNamespace(chunk_size=50, n_action_steps=50, n_obs_steps=1)
    configure_policy_temporal(cfg, config)
    assert (config.chunk_size, config.n_action_steps) == (8, 3)
    configure_policy_temporal(cfg, config, from_checkpoint=True)
    config.chunk_size = 7
    with pytest.raises(ValueError, match="Checkpoint temporal"):
        configure_policy_temporal(cfg, config, from_checkpoint=True)
    assert config.chunk_size == 7  # Refusal does not rewrite saved architecture.


def test_resolved_record_uses_real_config_and_split_timestamps():
    config = SimpleNamespace(
        chunk_size=8,
        n_action_steps=3,
        n_obs_steps=1,
        action_delta_indices=list(range(8)),
        observation_delta_indices=None,
    )
    recipe = {"prediction_horizon": 8, "execution_horizon": 3}
    record = resolved_temporal(config, 20, "act", recipe)
    assert record["action_delta_timestamps"] == [i / 20 for i in range(8)]
    dataset = SimpleNamespace(
        meta=SimpleNamespace(fps=20),
        features={"action": {}},
        delta_timestamps={"action": [i / 20 for i in range(8)]},
    )
    check_dataset_temporal(record, dataset)
    dataset.delta_timestamps["action"] = [0.0]
    with pytest.raises(ValueError, match="timestamps"):
        check_dataset_temporal(record, dataset)
    config.chunk_size = 7
    config.action_delta_indices = list(range(7))
    with pytest.raises(ValueError, match="differs from the recipe"):
        resolved_temporal(config, 20, "act", recipe)


@pytest.mark.parametrize("fps", [True, 0, -1, float("nan"), float("inf"), "20"])
def test_invalid_fps_rejected(fps):
    with pytest.raises(ValueError, match="FPS"):
        action_timestamps(8, fps)


def test_core_worker_temporal_rules_are_identical():
    root = Path(__file__).parents[3]
    assert (root / "packages/core/src/vla_platform/lifecycle/temporal.py").read_bytes() == (
        root / "workers/smolvla_qlora/src/firebird_vla/temporal.py"
    ).read_bytes()


def test_smol_loader_uses_prediction_horizon_not_execution_horizon(monkeypatch):
    import sys
    from types import ModuleType

    from firebird_vla.data import load_data

    calls = []
    features = {
        "action": {"shape": [2]},
        "observation.state": {"shape": [2]},
        "observation.images.front": {"dtype": "image"},
    }
    metadata = SimpleNamespace(
        features=features, fps=20, episodes=[{"episode_index": i} for i in range(3)]
    )

    class Dataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.hf_dataset = self

        def select_columns(self, keys):
            return [{keys[0]: [0.0, 1.0]}]

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: "/generated-fixture-only"
    native = ModuleType("lerobot.datasets.lerobot_dataset")
    native.LeRobotDataset = Dataset
    native.LeRobotDatasetMetadata = lambda *args, **kwargs: metadata
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", native)
    cfg = TrainConfig.from_dict({"prediction_horizon": 8, "execution_horizon": 3})
    load_data(cfg)
    assert len(calls) == 2
    assert all(call["delta_timestamps"] == {"action": [i / 20 for i in range(8)]} for call in calls)
    assert set(calls[0]["episodes"]).isdisjoint(calls[1]["episodes"])
