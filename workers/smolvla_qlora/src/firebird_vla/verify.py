"""Fresh-process checkpoint reload and deterministic action equivalence probe."""

import argparse
import json
from pathlib import Path

from .checkpoint import load_for_inference, write_json
from .config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import default_collate

    from .data import load_data, prepare_batch
    from .model import predict

    policy, pre, post = load_for_inference(args.checkpoint)
    cfg = TrainConfig.load(args.checkpoint / "recipe.json")
    splits = json.loads((args.checkpoint / "splits.json").read_text())
    _, validation, _, _ = load_data(cfg, splits)
    batch = prepare_batch(default_collate([validation[0]]), pre)
    actual = predict(policy, batch, post, cfg.seed)
    expected = load_file(str(args.checkpoint / "probe.safetensors"))["action"]
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    report = {
        "reload_verified": True,
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
