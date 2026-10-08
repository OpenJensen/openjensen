"""Native adapter contracts independent of optional CUDA packages."""

import json
from pathlib import Path

import pytest

from firebird_vla.checkpoint import verify_bundle
from firebird_vla.lerobot_application import cli_arguments, resolve_recipe
from firebird_vla.lerobot_train import commit_checkpoint, metric_value
from firebird_vla.native_profiles import LEROBOT_REVISION, NATIVE_PROFILES, native_requirement


def require_native_upstream():
    from importlib.metadata import PackageNotFoundError, version

    try:
        compatible = version("lerobot") == "0.6.2"
    except PackageNotFoundError:
        compatible = False
    if not compatible:
        pytest.skip("Requires the isolated pinned native LeRobot environment")


def request(model="act", **overrides):
    profile = NATIVE_PROFILES[model]
    return {
        "parameters": {
            "training_method": "full",
            "training": {
                "model_id": profile["model_id"],
                "model_revision": profile["model_revision"],
                "steps": 2,
                "camera_keys": ["observation.images.front"],
                **overrides,
            },
        },
        "dataset": {
            "source": "huggingface",
            "repo_id": "fixture/data",
            "revision": "a" * 40,
            "features": {
                "observation.images.front": {"dtype": "video", "shape": [32, 32, 3]},
                "observation.state": {"dtype": "float32", "shape": [6]},
                "action": {"dtype": "float32", "shape": [6]},
            },
        },
    }


def test_native_smolvla_supervised_training_uses_expert_weights_without_peft():
    from firebird_vla.native_profiles import SMOLVLA_NATIVE_PROFILE, native_profile_for_recipe

    job = request()
    job["parameters"]["training"].update(
        model_id=SMOLVLA_NATIVE_PROFILE["model_id"],
        model_revision=SMOLVLA_NATIVE_PROFILE["model_revision"],
    )
    recipe, profile = resolve_recipe(job)
    assert recipe["method"] == "full" and recipe["policy_type"] == "smolvla"
    args = cli_arguments(
        recipe,
        profile,
        output="out",
        dataset_root="dataset",
        model_root="pinned-model",
        features=job["dataset"]["features"],
    )
    assert "--policy.train_expert_only=true" in args
    assert "--policy.freeze_vision_encoder=true" in args
    assert not any("lora" in argument for argument in args)
    assert native_profile_for_recipe(job["parameters"]["training"], "lora") is None


def test_native_recipe_rejects_unknown_arguments_and_moving_models():
    with pytest.raises(ValueError, match="Unsupported native"):
        resolve_recipe(request(command="arbitrary command"))
    with pytest.raises(ValueError, match="immutable model"):
        resolve_recipe(request(model_revision="main"))
    with pytest.raises(ValueError, match="accumulation"):
        resolve_recipe(request("diffusion", gradient_accumulation_steps=2))


def test_camera_constraints_apply_before_worker_allocations():
    with pytest.raises(ValueError, match="requires 2 camera"):
        resolve_recipe(request("vla_jepa"))
    with pytest.raises(ValueError, match="not a dataset"):
        resolve_recipe(request(camera_keys=["observation.images.absent"]))


def test_scratch_baselines_do_not_fetch_fictitious_pretrained_weights():
    recipe, profile = resolve_recipe(request())
    arguments = cli_arguments(recipe, profile, output="out", dataset_root="pinned-data")
    assert "--policy.type=act" in arguments
    assert not any(argument.startswith("--policy.path=") for argument in arguments)
    assert "--dataset.revision=" + "a" * 40 in arguments
    assert "--dataset.root=pinned-data" in arguments
    assert "--dataset.eval_split=0.2" in arguments
    assert recipe["upstream_revision"] == LEROBOT_REVISION


def test_pretrained_profile_uses_the_resolved_worker_local_snapshot():
    recipe, profile = resolve_recipe(request("pi05"))
    arguments = cli_arguments(
        recipe, profile, output="out", dataset_root="pinned-data", model_root="cloud-cache/snapshot"
    )
    assert "--policy.path=cloud-cache/snapshot" in arguments
    assert "--policy.gradient_checkpointing=true" in arguments
    assert native_requirement(profile).endswith("@" + LEROBOT_REVISION)


def test_raw_groot_profile_passes_constructor_path_instead_of_hub_config():
    recipe, profile = resolve_recipe(request("gr00t_n17"))
    arguments = cli_arguments(
        recipe, profile, output="out", dataset_root="data", model_root="groot"
    )
    assert "--policy.type=groot" in arguments
    assert "--policy.base_model_path=groot" in arguments
    assert "--policy.path=groot" not in arguments


def test_complete_checkpoint_is_atomic_hashed_and_resumable(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREBIRD_CHECKPOINT_EXPORT_ROOT", raising=False)
    source = tmp_path / "native"
    (source / "pretrained_model").mkdir(parents=True)
    (source / "training_state").mkdir()
    (source / "pretrained_model/model.safetensors").write_bytes(b"weights")
    (source / "training_state/optimizer.safetensors").write_bytes(b"optimizer")
    training = tmp_path / "training"
    training.mkdir()
    recipe, _ = resolve_recipe(request())
    destination = commit_checkpoint(source, training / "checkpoint-000002", recipe, step=2)
    manifest = verify_bundle(destination)
    assert manifest["step"] == 2
    assert "training_state/optimizer.safetensors" in manifest["files"]
    assert json.loads((training / "latest.json").read_text())["step"] == 2
    with pytest.raises(FileExistsError):
        commit_checkpoint(source, destination, recipe, step=2)
    (destination / "pretrained_model/model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        verify_bundle(destination)


def test_nonfinite_metrics_cannot_be_published():
    with pytest.raises(FloatingPointError):
        metric_value(float("nan"))


def test_operator_and_worker_profile_copies_stay_identical():
    root = Path(__file__).parents[3]
    assert (root / "packages/core/src/vla_platform/lifecycle/native_profiles.py").read_bytes() == (
        root / "workers/smolvla_qlora/src/firebird_vla/native_profiles.py"
    ).read_bytes()


def test_eo1_checkpoint_config_migration_preserves_original_and_weights(tmp_path):
    from firebird_vla.lerobot_train import compatible_model_snapshot

    original = tmp_path / "snapshot"
    original.mkdir()
    config = {"type": "eo1", "use_language_recipe": False, "chunk_size": 16}
    (original / "config.json").write_text(json.dumps(config))
    (original / "model.safetensors").write_bytes(b"immutable-weights")
    resolved = compatible_model_snapshot(original, tmp_path / "resolved", "eo1")
    migrated = json.loads((resolved / "config.json").read_text())
    assert migrated["recipe"] is None
    assert "use_language_recipe" not in migrated
    assert json.loads((original / "config.json").read_text()) == config
    assert (resolved / "model.safetensors").read_bytes() == b"immutable-weights"


def test_explicit_optimizer_settings_survive_upstream_policy_presets():
    from types import SimpleNamespace

    from firebird_vla.lerobot_train import apply_optimizer_overrides

    cfg = SimpleNamespace(
        resume=False,
        steps=100,
        optimizer=SimpleNamespace(lr=0.1),
        scheduler=SimpleNamespace(num_warmup_steps=20, num_decay_steps=1000),
    )
    apply_optimizer_overrides(
        cfg, {"learning_rate": 0.001, "max_grad_norm": 0.5, "warmup_steps": 5}
    )
    assert cfg.optimizer.lr == 0.001
    assert cfg.optimizer.grad_clip_norm == 0.5
    assert cfg.scheduler.num_warmup_steps == 5
    assert cfg.scheduler.num_decay_steps == 100
    cfg.resume = True
    apply_optimizer_overrides(cfg, {"learning_rate": 0.2, "warmup_steps": 99})
    assert cfg.optimizer.lr == 0.001
    assert cfg.scheduler.num_warmup_steps == 5


def test_native_working_retention_preserves_latest_two_and_other_directories(tmp_path):
    from firebird_vla.lerobot_train import prune_native_working_checkpoints

    for name in ("000001", "000002", "000003", "notes"):
        (tmp_path / name).mkdir()
    (tmp_path / "last").symlink_to(tmp_path / "000003", target_is_directory=True)
    prune_native_working_checkpoints(tmp_path / "000003")
    assert not (tmp_path / "000001").exists()
    assert {path.name for path in tmp_path.iterdir()} == {"000002", "000003", "last", "notes"}


def test_probe_handles_pusht_camera_and_restores_training_precision(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    require_native_upstream()
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot.scripts.lerobot_train")
    from firebird_vla.lerobot_train import probe

    class Policy:
        training = True
        config = SimpleNamespace(image_features={"observation.image": object()})

        def eval(self):
            self.training = False

        def train(self, value):
            self.training = value

        def reset(self):
            pass

        def select_action(self, batch):
            assert batch["observation.image"].dtype == torch.float32
            assert batch["observation.image"].max().item() == 1
            assert not torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.deterministic
            return torch.tensor([[0.1, 0.2]])

    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: nullcontext())
    before = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    raw = {"observation.image": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)}
    policy = Policy()
    actual = probe(policy, raw, lambda batch: batch, lambda action: action, 42)
    assert actual.shape == (1, 2)
    assert raw["observation.image"].dtype == torch.uint8
    assert policy.training
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    ) == before


@pytest.mark.parametrize("policy_name,maximum", [("pi0", 32), ("evo1", 24), ("vla_jepa", None)])
def test_upstream_factory_derives_selected_robot_action_dimensions(
    monkeypatch, policy_name, maximum
):
    from types import SimpleNamespace

    require_native_upstream()
    torch = pytest.importorskip("torch")
    factory = pytest.importorskip("lerobot.policies.factory")
    from lerobot.configs.types import FeatureType, PolicyFeature

    images = {"observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))}
    inputs = {**images, "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    overrides = {}
    if policy_name == "vla_jepa":
        overrides = {"pre_snap_gripper_action": False, "binarize_gripper_action": False}
    cfg = factory.make_policy_config(
        policy_name,
        device="cpu",
        pretrained_path="fixture-pinned-model",
        input_features=inputs,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **overrides,
    )

    class LoaderFixture(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            config.validate_features()
            self.config = config

        @classmethod
        def from_pretrained(cls, **kwargs):
            return cls(**kwargs)

    monkeypatch.setattr(factory, "get_policy_class", lambda name: LoaderFixture)
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": [32, 32, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [6]},
    }
    policy = factory.make_policy(cfg, ds_meta=SimpleNamespace(features=features, stats={}))
    assert tuple(policy.config.action_feature.shape) == (6,)
    assert tuple(policy.config.robot_state_feature.shape) == (6,)
    if maximum is not None:
        assert policy.config.max_action_dim == maximum
    else:
        assert policy.config.action_dim == policy.config.state_dim == 6


def test_native_dimension_bounds_fail_before_cloud_worker_allocation():
    job = request("evo1")
    job["dataset"]["features"]["action"]["shape"] = [25]
    with pytest.raises(ValueError, match="at most 24 action"):
        resolve_recipe(job)
    assert NATIVE_PROFILES["vla_jepa"]["overrides"]["binarize_gripper_action"] is False


def test_act_accumulation_resolves_to_native_microbatch_units():
    recipe, profile = resolve_recipe(request(gradient_accumulation_steps=3))
    arguments = cli_arguments(recipe, profile, output="out", dataset_root="data")
    assert "--steps=6" in arguments
    assert "--accelerator.gradient_accumulation.steps=3" in arguments
    assert "--save_freq=3" in arguments
    assert "--eval_steps=6" in arguments
    assert recipe["steps"] == 2
