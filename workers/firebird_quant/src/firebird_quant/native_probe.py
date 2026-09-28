"""Offline CPU ACT conversion and independent packed-only execution process."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from .native_package import (
    BASE,
    CONTROL_FIELDS,
    canonical,
    encoding,
    inspect_policy,
    inventory,
    read,
    read_json,
    write_new,
)

SEEDS = (171, 902)


def prepare(*, packed_only):
    from firebird_act.probe import offline_audit, runtime_versions

    def audit(event, args):
        offline_audit(event, args)
        if packed_only and event == "open" and args and isinstance(args[0], (str, bytes)):
            if Path(os.fsdecode(args[0])).name == "model.safetensors":
                raise RuntimeError("Floating master reads are forbidden during packed reload")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    sys.addaudithook(audit)
    versions = runtime_versions()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    return torch, versions


def architecture(root):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    return ACTPolicy, config


def processors(root, config):
    from lerobot.policies.factory import make_pre_post_processors

    return make_pre_post_processors(
        config,
        pretrained_path=str(root),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )


def evaluate(policy, root, config, torch):
    from firebird_act.probe import check_queue

    pre, post = processors(root, config)
    camera = next(key for key in config.input_features if key.startswith("observation.images."))
    shape = config.input_features[camera].shape
    prediction, execution = config.chunk_size, config.n_action_steps
    rows = []
    for seed in SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
        state = torch.randn(6, generator=generator, dtype=torch.float32)
        input_sha = hashlib.sha256(image.numpy().tobytes() + state.numpy().tobytes()).hexdigest()
        policy.reset()
        pre.reset()
        post.reset()
        with torch.inference_mode():
            batch = pre({camera: image.float() / 255, "observation.state": state})
            raw = policy.predict_action_chunk(batch)
            if tuple(raw.shape) != (1, prediction, 6) or not torch.isfinite(raw).all():
                raise ValueError("Expected a finite complete ACT chunk")
            processed = torch.stack([post(raw[:, i, :]) for i in range(prediction)], dim=1)
            if tuple(processed.shape) != (1, prediction, 6) or not torch.isfinite(processed).all():
                raise ValueError("Expected finite postprocessed ACT actions")
            check_queue(policy, batch, raw, execution, torch)
        rows.append(
            {
                "seed": seed,
                "input_sha256": input_sha,
                "image_shape": list(shape),
                "raw": raw[0].tolist(),
                "postprocessed": processed[0].tolist(),
                "queue_and_reset_exact": True,
            }
        )
    return rows


def convert(source, destination, bits):
    from firebird_act.bundle import (
        control_files,
        temporal_files,
        tensor_header,
        validate_processors,
    )
    from safetensors.torch import load_file

    from . import Recipe, quantize

    torch, versions = prepare(packed_only=False)
    before = inventory(source)
    policy_cls, config = architecture(source)
    policy = policy_cls.from_pretrained(
        source, config=config, local_files_only=True, strict=True
    ).eval()
    header, _ = tensor_header(read(source / "model.safetensors", 512 * 1024 * 1024))
    loaded = policy.state_dict()
    if set(loaded) != set(header) - {"__metadata__"}:
        raise ValueError("Loaded source tensor names differ from the exact saved inventory")
    saved = load_file(str(source / "model.safetensors"), device="cpu")
    for key, tensor in loaded.items():
        if (
            tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
            or not torch.equal(tensor, saved[key])
        ):
            raise ValueError("Source must load its exact finite FP32 tensors")
    del saved, loaded
    baseline = evaluate(policy, source, config, torch)
    candidate = quantize(policy, Recipe(bits=bits, group_size=64))
    if candidate.audit["quantized_elements"] <= 0:
        raise ValueError("No weights were packed")
    raw_config = read_json(source / "config.json")
    names = (
        BASE
        | validate_processors(source, raw_config)
        | temporal_files(source, raw_config)
        | control_files(source, raw_config)
    )
    destination.mkdir()
    for name in sorted(names):
        write_new(destination / name, read(source / name))
    write_new(destination / "encoding.json", canonical(encoding(bits)))
    candidate.save(destination / "model.fbq")
    packed = evaluate(candidate.model, destination, config, torch)
    if inventory(source) != before:
        raise ValueError("Source changed during quantization")
    info = inspect_policy(destination)
    return {
        "schema_version": 1,
        "versions": versions,
        "model_id": info["model_id"],
        "policy_files": info["files"],
        **{key: info[key] for key in CONTROL_FIELDS if key in info},
        "baseline": baseline,
        "packed": packed,
        "audit": candidate.audit,
        "network_disabled": True,
        "fixture_scope": "generated observations; not calibration or robotics task quality",
        "source_files": before,
    }


def reload_packed(root):
    from . import load_model

    torch, versions = prepare(packed_only=True)
    before = inspect_policy(root)
    policy_cls, config = architecture(root)
    # Trusted architecture only. There are no floating policy weights in this package.
    empty = policy_cls(config).eval()
    candidate = load_model(empty, root / "model.fbq")
    rows = evaluate(candidate.model, root, config, torch)
    if inspect_policy(root) != before:
        raise ValueError("Packed policy changed during reload")
    return {
        "schema_version": 1,
        "versions": versions,
        "model_id": before["model_id"],
        "policy_files": before["files"],
        **{key: before[key] for key in CONTROL_FIELDS if key in before},
        "packed": rows,
        "network_disabled": True,
        "floating_master_reads_blocked": True,
        "source_read_protection": "Python open audit hook; not an OS filesystem sandbox",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("convert", "reload"))
    parser.add_argument("source", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 8))
    args = parser.parse_args()
    if args.mode == "convert":
        if args.destination is None or args.bits is None:
            parser.error("Conversion needs a new destination and precision")
        result = convert(args.source, args.destination, args.bits)
    else:
        result = reload_packed(args.source)
    write_new(args.result, canonical(result))


if __name__ == "__main__":
    main()
