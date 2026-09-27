"""Fresh-process action parity check for a trained Psi-Zero checkpoint."""

import json
import sys
from pathlib import Path

from .checkpoint import verify_bundle, write_json
from .psi_train import build_trainer, model_snapshot, predict, require_runtime


def verify(checkpoint, result_path):
    require_runtime()
    import torch
    from safetensors.torch import load_file

    manifest = verify_bundle(checkpoint)
    recipe = json.loads((checkpoint / "recipe.json").read_text())
    trainer = build_trainer(recipe, model_snapshot(), checkpoint / "stats.json", result_path.parent)
    trainer.model.action_header.load_state_dict(
        load_file(str(checkpoint / "action_header.safetensors")), strict=True
    )
    inputs = {
        key: value.to("cuda:0")
        for key, value in load_file(str(checkpoint / "probe-input.safetensors")).items()
    }
    actual = predict(trainer, inputs)
    expected = load_file(str(checkpoint / "probe.safetensors"))["action"]
    if not torch.isfinite(actual).all():
        raise RuntimeError("Reloaded Psi-Zero actions are not finite")
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    write_json(
        result_path,
        {
            "schema_version": 1,
            "step": manifest["step"],
            "reload_verified": True,
            "action_max_abs_difference": float((actual - expected).abs().max()),
            "task_success": None,
            "scope": "fresh_process_action_parity",
        },
    )


if __name__ == "__main__":
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
