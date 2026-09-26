"""CPU contract checks; these do not claim a GPU-trained Psi-Zero model."""

import copy
from pathlib import Path

import pytest

from firebird_vla.psi_profile import (
    PSI_EXPERT_DIRECTORY,
    PSI_MODEL_ID,
    PSI_MODEL_REVISION,
    PSI_VLM_DIRECTORY,
    launch_config,
    resolve_recipe,
)


def job():
    return {
        "schema_version": 1,
        "operation": "policy.finetune",
        "parameters": {
            "training_method": "full",
            "training": {
                "model_id": PSI_MODEL_ID,
                "model_revision": PSI_MODEL_REVISION,
                "camera_keys": ["observation.images.front"],
                "steps": 5,
                "warmup_steps": 1,
            },
        },
        "dataset": {
            "source": "huggingface",
            "repo_id": "fixture/robot",
            "revision": "a" * 40,
            "format": "lerobot_v2",
            "features": {
                "action": {"shape": [6]},
                "observation.state": {"shape": [6]},
                "observation.images.front": {"dtype": "video"},
            },
        },
    }


def test_native_recipe_maps_generic_lerobot_to_released_psi_expert():
    recipe = resolve_recipe(job())
    config = launch_config(
        recipe,
        dataset_root=Path("/cloud/dataset"),
        model_root=Path("/cloud/model"),
        stats_path=Path("/cloud/stats.json"),
        output=Path("/cloud/output"),
    )
    assert recipe["save_every"] == 1
    assert recipe["action_dim"] == 6
    assert config["model"]["model_name_or_path"] == "/cloud/model/" + PSI_VLM_DIRECTORY
    assert (
        config["model"]["pretrained_action_header_path"] == "/cloud/model/" + PSI_EXPERT_DIRECTORY
    )
    assert config["model"]["tune_vlm"] is False
    assert config["model"]["action_dim"] == config["model"]["odim"] == 36
    assert config["data"]["transform"]["repack"]["state_key"] == "observation.state"
    assert config["data"]["transform"]["field"]["stat_state_key"] == "observation.state"
    assert config["train"]["lora"] is False
    assert config["log"]["report_to"] is None
    assert config["auto_tag_run"] is False


def test_original_psi_whole_body_state_alias_is_supported():
    request = job()
    features = request["dataset"]["features"]
    features["states"] = features.pop("observation.state")
    features["states"]["shape"] = [36]
    features["action"]["shape"] = [36]
    recipe = resolve_recipe(request)
    assert recipe["state_key"] == "states"
    assert recipe["state_dim"] == recipe["action_dim"] == 36


@pytest.mark.parametrize(
    "update, message",
    [
        ({"model_revision": "main"}, "pinned model"),
        ({"camera_keys": []}, "exactly one"),
        ({"camera_keys": ["observation.images.front", "observation.images.front"]}, "exactly one"),
        ({"camera_keys": ["missing"]}, "exactly one"),
        ({"chunk_size": 5}, "chunk_size=30"),
        ({"num_workers": 1}, "num_workers=0"),
        ({"steps": True}, "positive integer"),
        ({"learning_rate": float("nan")}, "finite"),
        ({"validation_fraction": 1}, "between zero and one"),
        ({"warmup_steps": 5}, "smaller than steps"),
        ({"custom_training_command": "true"}, "Unsupported"),
    ],
)
def test_unsupported_settings_are_rejected_before_loading_gpu_packages(update, message):
    request = job()
    request["parameters"]["training"].update(update)
    with pytest.raises(ValueError, match=message):
        resolve_recipe(request)


def test_v3_loader_mismatch_and_lora_are_not_silently_advertised():
    request = job()
    request["dataset"]["format"] = "lerobot_v3"
    with pytest.raises(ValueError, match="v3 datasets are not supported"):
        resolve_recipe(request)
    request = job()
    request["parameters"]["training_method"] = "lora"
    with pytest.raises(ValueError, match="LoRA/QLoRA are unsupported"):
        resolve_recipe(request)


@pytest.mark.parametrize("key", ["action", "observation.state"])
def test_checkpoint_dimensions_are_not_truncated(key):
    request = job()
    request["dataset"]["features"][key]["shape"] = [37]
    with pytest.raises(ValueError, match="1..36 dimensions"):
        resolve_recipe(request)


def test_resolving_recipe_does_not_change_original_application_request():
    request = job()
    original = copy.deepcopy(request)
    resolve_recipe(request)
    assert request == original
