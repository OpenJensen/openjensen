"""Fresh-process, offline CPU inference on frozen synthetic observations.

This measures inference equivalence, not dataset transfer, calibration or task success.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .bundle import canonical, inventory, read_json, tensor_header, validate_config, verify_export
from .control_schema import metadata as control_metadata

VERSIONS = {"lerobot": "0.6.1", "torch": "2.11.0", "torchvision": "0.26.0", "safetensors": "0.8.0"}
FIXTURE_SEEDS = (171, 902)


def offline_audit(event: str, args: tuple[Any, ...], forbidden: tuple[Path, ...] = ()) -> None:
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network access is disabled during ACT verification")
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(root) for root in forbidden):
            raise RuntimeError("Source reads are disabled during final ACT package verification")


def runtime_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    for name, expected in VERSIONS.items():
        if versions[name].split("+")[0] != expected:
            raise ValueError(f"ACT verification requires {name}=={expected}")
    return versions


def check_queue(policy: Any, batch: Any, chunk: Any, execution: int, torch: Any) -> None:
    """Check two real queue refills, the execution prefix and a subsequent reset."""
    from unittest.mock import patch

    policy.reset()
    with patch.object(policy, "predict_action_chunk", wraps=policy.predict_action_chunk) as predict:
        queued = torch.stack([policy.select_action(batch) for _ in range(2 * execution + 1)])
        expected = torch.stack([chunk[:, i % execution] for i in range(2 * execution + 1)])
        if predict.call_count != 3 or not torch.equal(queued, expected):
            raise ValueError("ACT queue does not refill at its saved execution horizon")
        policy.reset()
        first = policy.select_action(batch)
        if predict.call_count != 4 or not torch.equal(first, chunk[:, 0]):
            raise ValueError("ACT reset did not discard its queued actions")


def infer(checkpoint: Path) -> dict[str, Any]:
    """Strictly load one checkpoint without another policy process or network."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    forbidden = tuple(
        Path(p).resolve()
        for p in json.loads(os.environ.get("FIREBIRD_ACT_FORBIDDEN_SOURCES", "[]"))
    )
    sys.addaudithook(lambda event, args: offline_audit(event, args, forbidden))
    versions = runtime_versions()
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import (
        PolicyProcessorPipeline,
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    if (checkpoint / "manifest.json").exists():
        verify_export(checkpoint)
    before = inventory(checkpoint)
    raw_config = read_json(checkpoint / "config.json")
    validate_config(raw_config, source=raw_config["use_vae"])
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    policy = (
        ACTPolicy.from_pretrained(checkpoint, config=config, strict=True, local_files_only=True)
        .cpu()
        .eval()
    )
    # Safetensors must not discard a redundant shared key or extra tensor silently.
    header, _ = tensor_header((checkpoint / "model.safetensors").read_bytes())
    state = policy.state_dict()
    if set(state) != set(header) - {"__metadata__"}:
        raise ValueError("Strict ACT tensor inventory differs after loading")
    for name, tensor in state.items():
        if (
            tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or list(tensor.shape) != header[name]["shape"]
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Invalid loaded FP32 tensor: {name}")
    for path in checkpoint.glob("*.safetensors"):
        if path.name != "model.safetensors":
            if any(not torch.isfinite(t).all() for t in load_file(path).values()):
                raise ValueError("Non-finite processor statistics")
    pre = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    camera = next(k for k in raw_config["input_features"] if k.startswith("observation.images."))
    shape = raw_config["input_features"][camera]["shape"]
    prediction, execution = config.chunk_size, config.n_action_steps
    fixtures = []
    for seed in FIXTURE_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rgb = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
        observed_state = torch.randn(6, generator=generator, dtype=torch.float32)
        fixture_hash = hashlib.sha256(
            rgb.numpy().tobytes() + observed_state.numpy().tobytes()
        ).hexdigest()
        policy.reset()
        pre.reset()
        post.reset()
        with torch.inference_mode():
            batch = pre({camera: rgb.float() / 255, "observation.state": observed_state})
            chunk = policy.predict_action_chunk(batch)
            if tuple(chunk.shape) != (1, prediction, 6) or not torch.isfinite(chunk).all():
                raise ValueError("ACT output must be finite [1,prediction,6]")
            processed = torch.stack([post(chunk[:, i, :]) for i in range(prediction)])
            if tuple(processed.shape) != (prediction, 1, 6) or not torch.isfinite(processed).all():
                raise ValueError("ACT postprocessed output must be finite [prediction,1,6]")
            check_queue(policy, batch, chunk, execution, torch)
            fixtures.append(
                {
                    "seed": seed,
                    "input_sha256": fixture_hash,
                    "chunk": chunk[0].tolist(),
                    "postprocessed": processed[:, 0].tolist(),
                    "queue_and_reset_exact": True,
                }
            )
    if inventory(checkpoint) != before:
        raise ValueError("Checkpoint changed during inference")
    return {
        "schema_version": 1,
        **control_metadata(checkpoint, raw_config),
        "versions": versions,
        "python": platform.python_version(),
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": 1,
        "network_disabled": True,
        "checkpoint_files": before,
        "prediction_horizon": prediction,
        "execution_horizon": execution,
        "fixtures": fixtures,
        "fixture_scope": "synthetic inference parity only; not calibration or task success",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.result.resolve().is_relative_to(args.checkpoint.resolve()):
        parser.error("Probe result must be outside the checkpoint/package")
    if os.path.lexists(args.result):
        parser.error("Probe result must be a new file")
    result = infer(args.checkpoint)
    with args.result.open("xb") as stream:
        stream.write(canonical(result))


if __name__ == "__main__":
    main()
