"""Opt-in real pinned ACT runtime, generated weights and observations only."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "act_optimizer" / "src"))
from firebird_quant.native_application import run_job
from firebird_quant.native_package import inventory, read_json

pytestmark = [
    pytest.mark.native_act,
    pytest.mark.skipif(
        os.environ.get("FIREBIRD_TEST_NATIVE_ACT") != "1",
        reason="Set FIREBIRD_TEST_NATIVE_ACT=1 in the unchanged pinned ACT0.6.1 CPU environment",
    ),
]


@pytest.fixture(scope="module")
def real_act(tmp_path_factory):
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    torch.set_num_threads(1)
    torch.manual_seed(190)
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
        use_vae=False,
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
    root = tmp_path_factory.mktemp("native-act-source")
    ACTPolicy(config).save_pretrained(root)
    pre, post = make_act_pre_post_processors(config, dataset_stats=stats)
    pre.save_pretrained(root, config_filename="policy_preprocessor.json")
    post.save_pretrained(root, config_filename="policy_postprocessor.json")
    return root


@pytest.mark.parametrize("bits", [8, 4])
def test_generated_real_act_complete_native_package(real_act, tmp_path, bits):
    before = inventory(real_act)
    result = run_job(
        {
            "schema_version": 1,
            "job_id": "generated-real-act",
            "operation": "policy.quantize",
            "source": {
                "path": str(real_act),
                "files": before,
                "manifest_sha256": None,
                "artifact_id": "generated-fixture",
                "artifact_manifest_sha256": "a" * 64,
            },
            "output_dir": str(tmp_path / "operation"),
            "native_quantization": {"format": "firebird_quant", "bits": bits, "group_size": 64},
            "timeout_seconds": 120,
        }
    )
    root = Path(result["artifact"]["path"])
    proof = read_json(root / "verification.json")
    assert proof["fresh_packed_reload_exact"] is True
    assert proof["floating_master_reads_blocked"] is True
    assert proof["full_chunk_queue_reset_verified"] is True
    assert proof["task_success"] is None and proof["quality_verified"] is False
    assert result["report"]["packed_weight_bytes"] < result["report"]["source_weight_bytes"]
    assert len(proof["floating"]) == len(proof["packed"]) == 2
    assert all(len(row["raw"]) == len(row["postprocessed"]) == 100 for row in proof["packed"])
    assert not (root / "policy/model.safetensors").exists()
    assert inventory(real_act) == before
