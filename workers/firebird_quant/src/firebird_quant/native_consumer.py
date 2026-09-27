"""Strict CPU packed ACT consumer for the existing local policy HTTP server.

The output is ordinary FP32 eager computation backed by packed weight storage.
This is not a GPU performance claim, calibration result, or task-quality gate.
"""

import os
import tempfile
from pathlib import Path

from .native_package import WEIGHT_LIMIT, canonical, inspect_policy, read, write_new


def load_packed_act(source: Path, *, device: str, expected_model_id: str):
    """Load an exact, private byte snapshot; return model/config/processors/identity."""
    if device != "cpu":
        raise ValueError("Native packed ACT serving is verified for CPU only")
    # Validate before importing Torch or constructing a model.
    admitted = inspect_policy(source)
    if admitted["model_id"] != expected_model_id:
        raise ValueError("Packed model identity differs from the admitted checkpoint")
    from firebird_act.probe import runtime_versions

    runtime_versions()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="firebird-packed-consumer-") as temporary:
        snapshot = Path(temporary).resolve()
        for name in admitted["files"]:
            raw = read(source / name, WEIGHT_LIMIT if name == "model.fbq" else 1024 * 1024)
            write_new(snapshot / name, raw)
        if inspect_policy(snapshot) != admitted or inspect_policy(source) != admitted:
            raise ValueError("Packed policy changed during its private snapshot")

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        from . import load_model
        from .codec import PackedTensor

        config = PreTrainedConfig.from_pretrained(snapshot, local_files_only=True)
        config.device = "cpu"
        config.pretrained_backbone_weights = None
        # No from_pretrained policy load and no model.safetensors are involved.
        architecture = ACTPolicy(config).to(device="cpu", dtype=torch.float32).eval()
        candidate = load_model(architecture, snapshot / "model.fbq")
        recipe = admitted["encoding"]["recipe"]
        if canonical(candidate.state.audit.get("recipe")) != canonical(recipe):
            raise ValueError("Packed tensor recipe differs from the encoding contract")
        packed = [v for v in candidate.state.tensors.values() if isinstance(v, PackedTensor)]
        if not packed or any(
            v.bits != recipe["bits"] or v.group_size != recipe["group_size"] for v in packed
        ):
            raise ValueError("Packed tensors differ from the declared precision")
        pre, post = make_pre_post_processors(
            config,
            pretrained_path=str(snapshot),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        if inspect_policy(snapshot) != admitted or inspect_policy(source) != admitted:
            raise ValueError("Packed policy changed during loading")
        return candidate.model.eval(), config, pre, post, admitted["model_id"]
