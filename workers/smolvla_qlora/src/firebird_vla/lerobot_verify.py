"""Fresh-process numerical reload verification of a complete native policy."""

import json
import sys
from pathlib import Path

from .checkpoint import verify_bundle, write_json
from .lerobot_train import probe
from .temporal import resolved_temporal


def main():
    checkpoint, report_path = map(Path, sys.argv[1:])
    verify_bundle(checkpoint)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    recipe = json.loads((checkpoint / "recipe.json").read_text())
    model_dir = checkpoint / "pretrained_model"
    config = PreTrainedConfig.from_pretrained(model_dir)
    temporal_path = checkpoint / "temporal-contract.json"
    temporal = None
    if temporal_path.exists():
        temporal = json.loads(temporal_path.read_text())
        actual_temporal = resolved_temporal(config, temporal["action_fps"], recipe["policy_type"])
        if actual_temporal != temporal:
            raise ValueError("Reloaded policy temporal configuration differs from saved contract")
    config.device = "cuda"
    policy = get_policy_class(config.type).from_pretrained(model_dir, config=config, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=model_dir)
    batch = torch.load(checkpoint / "probe-batch.pt", map_location="cpu", weights_only=True)
    expected = torch.load(checkpoint / "probe-action.pt", map_location="cpu", weights_only=True)
    actual = probe(policy, batch, preprocessor, postprocessor, recipe["seed"])
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    write_json(
        report_path,
        {
            "reload_verified": True,
            "temporal_contract": temporal,
            "max_abs_action_difference": (actual - expected).abs().max().item(),
            "task_success": None,
            "scope": "Fixed held-out observation numerical reload, not robot task success",
        },
    )


if __name__ == "__main__":
    main()
