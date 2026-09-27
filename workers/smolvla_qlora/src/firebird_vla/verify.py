"""Fresh-process checkpoint reload and deterministic action equivalence probe."""

import argparse
import json
from pathlib import Path

from .checkpoint import load_for_inference, resolve_checkpoint, write_json
from .config import TrainConfig
from .temporal import check_dataset_temporal, resolved_temporal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import default_collate

    from .data import load_data, prepare_batch
    from .model import predict

    policy, pre, post = load_for_inference(args.checkpoint)
    cfg = TrainConfig.load(args.checkpoint / "recipe.json")
    splits = json.loads((args.checkpoint / "splits.json").read_text())
    _, validation, _, _ = load_data(cfg, splits)
    temporal = None
    temporal_path = args.checkpoint / "temporal-contract.json"
    if temporal_path.exists():
        temporal = json.loads(temporal_path.read_text())
        # PEFT forwards config to its underlying policy through get_base_model.
        base_policy = policy.get_base_model()
        actual_temporal = resolved_temporal(
            base_policy.config, validation.meta.fps, "smolvla", cfg.to_dict()
        )
        check_dataset_temporal(actual_temporal, validation)
        if actual_temporal != temporal:
            raise ValueError("Reloaded policy temporal contract differs from saved checkpoint")
    batch = prepare_batch(default_collate([validation[0]]), pre, cfg.selected_camera_keys)
    actual = predict(policy, batch, post, cfg.seed)
    expected = load_file(str(args.checkpoint / "probe.safetensors"))["action"]
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    report = {
        "reload_verified": True,
        "temporal_contract": temporal,
        "checkpoint": str(args.checkpoint.resolve()),
        "max_abs_action_difference": (actual - expected).abs().max().item(),
        "task_success": None,
        "scope": "One fixed held-out observation; numerical reload, not task evaluation",
    }
    # Keep immutable checkpoint hashes intact; store verification beside the bundle.
    write_json(args.checkpoint.parent / f"{args.checkpoint.name}-verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
