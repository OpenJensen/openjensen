"""CPU contract checks for camera intake, policy configuration, and batch preparation."""

import json
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

from firebird_vla.config import TrainConfig
from firebird_vla.data import load_data, prepare_batch
from firebird_vla.model import configure_policy_features

TOP = "observation.images.top"
WRIST = "observation.images.wrist"
FRONT = "observation.images.front"
FEATURES = {
    "action": {"dtype": "float32", "shape": [2]},
    "observation.state": {"dtype": "float32", "shape": [2]},
    TOP: {"dtype": "video", "shape": [3, 32, 32]},
    WRIST: {"dtype": "image", "shape": [3, 32, 32]},
    FRONT: {"dtype": "video", "shape": [3, 32, 32]},
}


def module(monkeypatch, name, **attributes):
    """Replace dependency boundaries so the worker checks require no ML install."""
    value = ModuleType(name)
    value.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, value)


@pytest.fixture
def dataset_runtime(monkeypatch):
    loaded = []
    downloads = []

    def snapshot_download(repo_id, **kwargs):
        downloads.append((repo_id, kwargs))
        return "/pinned-dataset"

    def metadata(*args, **kwargs):
        return SimpleNamespace(
            features=FEATURES, fps=30, episodes=[{"episode_index": index} for index in range(4)]
        )

    class Dataset:
        def __init__(self, **kwargs):
            loaded.append(kwargs)
            self.features = FEATURES
            self.hf_dataset = self
            self.episodes = kwargs["episodes"]

        def select_columns(self, columns):
            return [{columns[0]: [float(index), 1.0]} for index in self.episodes]

    module(monkeypatch, "huggingface_hub", snapshot_download=snapshot_download)
    module(
        monkeypatch,
        "lerobot.datasets.lerobot_dataset",
        LeRobotDataset=Dataset,
        LeRobotDatasetMetadata=metadata,
    )
    return loaded, downloads


def test_multi_camera_dataset_validates_both_video_and_image(dataset_runtime):
    cfg = TrainConfig.from_dict({"camera_keys": [TOP, WRIST]})
    train, validation, stats, splits = load_data(cfg)
    loaded, downloads = dataset_runtime
    assert len(loaded) == 2
    assert downloads == [
        (cfg.dataset_id, {"repo_type": "dataset", "revision": cfg.dataset_revision})
    ]
    assert train.features[TOP]["dtype"] == "video"
    assert validation.features[WRIST]["dtype"] == "image"
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert stats["action"]["count"] == [len(splits["train"])]


@pytest.mark.parametrize("invalid", ["observation.images.missing", "observation.state"])
def test_unknown_or_nonvisual_second_camera_fails_before_loading_frames(dataset_runtime, invalid):
    cfg = TrainConfig.from_dict({"camera_keys": [TOP, invalid]})
    with pytest.raises(ValueError, match="Missing image/video camera"):
        load_data(cfg)
    assert dataset_runtime[0] == []


@pytest.fixture
def policy_runtime(monkeypatch):
    class FeatureType(Enum):
        VISUAL = "VISUAL"
        STATE = "STATE"
        ACTION = "ACTION"

    def to_policy_features(features):
        return {
            key: SimpleNamespace(
                type=(
                    FeatureType.VISUAL
                    if value["dtype"] in {"video", "image"}
                    else FeatureType.ACTION
                    if key == "action"
                    else FeatureType.STATE
                ),
                shape=value["shape"],
            )
            for key, value in features.items()
        }

    class PolicyConfig:
        empty_cameras = 0
        input_features = {}
        output_features = {}

        @property
        def image_features(self):
            return {
                key: value
                for key, value in self.input_features.items()
                if value.type == FeatureType.VISUAL
            }

    module(monkeypatch, "lerobot.configs.types", FeatureType=FeatureType)
    module(monkeypatch, "lerobot.datasets.utils", dataset_to_policy_features=to_policy_features)
    return PolicyConfig


def test_policy_and_checkpoint_reload_preserve_every_selected_camera_in_order(policy_runtime):
    cfg = TrainConfig.from_dict({"camera_keys": [WRIST, TOP]})
    policy = policy_runtime()
    configure_policy_features(cfg, policy, FEATURES)
    assert list(policy.input_features) == ["observation.state", WRIST, TOP]
    assert list(policy.image_features) == [WRIST, TOP]
    assert list(policy.output_features) == ["action"]
    assert FRONT not in policy.input_features

    # Loading a saved policy without dataset metadata must retain the same views.
    configure_policy_features(cfg, policy, from_checkpoint=True)
    assert list(policy.image_features) == [WRIST, TOP]
    # A checkpoint with fewer cameras or a different order must not silently change inputs.
    for cameras in ([TOP], [TOP, WRIST]):
        with pytest.raises(ValueError, match="Checkpoint camera inputs differ"):
            configure_policy_features(
                TrainConfig.from_dict({"camera_keys": cameras}), policy, from_checkpoint=True
            )


def test_legacy_policy_keeps_single_camera(policy_runtime):
    cfg = TrainConfig.from_dict({"camera_key": TOP})
    policy = policy_runtime()
    configure_policy_features(cfg, policy, FEATURES)
    assert list(policy.image_features) == [TOP]
    configure_policy_features(cfg, policy, from_checkpoint=True)


def test_four_camera_policy_has_no_synthetic_or_dropped_views(policy_runtime):
    fourth = "observation.images.right_wrist"
    cameras = [TOP, WRIST, FRONT, fourth]
    features = {**FEATURES, fourth: {"dtype": "video", "shape": [3, 32, 32]}}
    cfg = TrainConfig.from_dict({"camera_keys": cameras})
    policy = policy_runtime()
    policy.empty_cameras = 2
    configure_policy_features(cfg, policy, features)
    assert list(policy.image_features) == cameras
    assert policy.empty_cameras == 0
    batch = {**camera_batch(), fourth: object()}
    processed = prepare_batch(batch, dict, cameras)
    assert all(processed[key] is batch[key] for key in cameras)
    configure_policy_features(cfg, policy, from_checkpoint=True)


class PaddingMask:
    def to(self, device):
        assert device == "cuda"
        return self


def camera_batch():
    return {
        "task": ["Pick up the object"],
        "observation.state": object(),
        "action": object(),
        "action_is_pad": PaddingMask(),
        TOP: object(),
        WRIST: object(),
        FRONT: object(),
        f"{TOP}_padding_mask": object(),
        f"{FRONT}_padding_mask": object(),
    }


def test_processor_receives_both_selected_views_and_skips_unused_gpu_transfers():
    batch = camera_batch()
    calls = []

    def preprocessor(value):
        calls.append(value)
        return dict(value)

    processed = prepare_batch(batch, preprocessor, [WRIST, TOP])
    assert calls[0][WRIST] is batch[WRIST]
    assert calls[0][TOP] is batch[TOP]
    assert FRONT not in calls[0]
    assert f"{FRONT}_padding_mask" not in calls[0]
    assert processed[f"{TOP}_padding_mask"] is batch[f"{TOP}_padding_mask"]
    assert processed["actions_id_pad"] is batch["action_is_pad"]
    assert FRONT in batch  # The dataset batch itself is not mutated.


def test_missing_selected_camera_cannot_silently_train_on_other_view():
    batch = camera_batch()
    del batch[WRIST]
    with pytest.raises(ValueError, match="Missing selected cameras in batch"):
        prepare_batch(batch, lambda value: value, [TOP, WRIST])

    def broken_processor(value):
        return {key: tensor for key, tensor in value.items() if key != WRIST}

    with pytest.raises(ValueError, match="Preprocessor dropped selected cameras"):
        prepare_batch(camera_batch(), broken_processor, [TOP, WRIST])


def test_application_resume_rejects_changed_camera_selection(monkeypatch, tmp_path):
    from firebird_vla import application, model

    previous = TrainConfig.from_dict({"camera_keys": [TOP, WRIST]})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "recipe.json").write_text(json.dumps(previous.to_dict()))
    output = tmp_path / "output"
    output.mkdir()
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": "camera-resume",
                "operation": "policy.finetune",
                "output_dir": str(output),
                "resume_checkpoint": str(checkpoint),
                "dataset": {
                    "source": "huggingface",
                    "repo_id": previous.dataset_id,
                    "revision": previous.dataset_revision,
                },
                "parameters": {"training_method": "qlora", "training": {"camera_keys": [TOP]}},
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["worker", str(request), str(result)])
    monkeypatch.setattr(model, "require_runtime", lambda: None)
    monkeypatch.setattr(application, "resolve_checkpoint", lambda path: path)
    module(monkeypatch, "torch", cuda=SimpleNamespace(mem_get_info=lambda: (8 * 1024**3,) * 2))
    calls = []
    monkeypatch.setattr(application.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    assert application.main() == 1
    assert "checkpoint's camera selection" in json.loads(result.read_text())["error"]
    assert calls == []
