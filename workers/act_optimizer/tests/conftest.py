"""Generated real ACT fixtures; no external datasets or production weights in unit tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def act_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    torch.set_num_threads(1)
    torch.manual_seed(11)
    config = ACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        chunk_size=100,
        n_action_steps=100,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    stats = {
        "observation.state": {"mean": torch.arange(6).float(), "std": torch.ones(6) * 2},
        "observation.images.front": {
            "mean": torch.ones(3, 1, 1) * 0.5,
            "std": torch.ones(3, 1, 1) * 0.25,
        },
        "action": {"mean": torch.arange(6).float(), "std": torch.ones(6) * 3},
    }
    root = tmp_path_factory.mktemp("act-source")
    ACTPolicy(config).save_pretrained(root)
    pre, post = make_act_pre_post_processors(config, dataset_stats=stats)
    pre.save_pretrained(root, config_filename="policy_preprocessor.json")
    post.save_pretrained(root, config_filename="policy_postprocessor.json")
    (root / "train_config.json").write_text(json.dumps({"provenance_only": True}))
    return root
