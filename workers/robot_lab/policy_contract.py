"""Bounded native ACT processor metadata, shared by admission and inference."""

import json
from pathlib import Path

PIPELINES = {
    "policy_preprocessor.json": (
        "rename_observations_processor",
        "to_batch_processor",
        "device_processor",
        "normalizer_processor",
    ),
    "policy_postprocessor.json": ("unnormalizer_processor", "device_processor"),
}
CONFIG_KEYS = {
    "rename_observations_processor": {"rename_map"},
    "to_batch_processor": set(),
    "device_processor": {"device", "float_dtype"},
    "normalizer_processor": {"eps", "features", "norm_map", "device"},
    "unnormalizer_processor": {"eps", "features", "norm_map", "device"},
}


def read_json(path, maximum=1024 * 1024):
    def invalid_constant(value):
        raise ValueError("Policy metadata cannot contain non-finite numbers")

    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Policy metadata must be a bounded regular file")
    return json.loads(path.read_text(), parse_constant=invalid_constant)


def validate_processors(checkpoint):
    """Reject dynamic import fallback, external artifacts and escaping tensor state."""
    checkpoint = Path(checkpoint)
    for filename, expected in PIPELINES.items():
        pipeline = read_json(checkpoint / filename)
        if not isinstance(pipeline, dict) or set(pipeline) - {"name", "steps"}:
            raise ValueError("Only native ACT processor metadata is supported")
        steps = pipeline.get("steps")
        if not isinstance(steps, list) or len(steps) != len(expected):
            raise ValueError("ACT requires its standard saved normalization processors")
        for step, registry in zip(steps, expected, strict=True):
            if (
                not isinstance(step, dict)
                or set(step) - {"registry_name", "config", "state_file"}
                or step.get("registry_name") != registry
                or not isinstance(step.get("config"), dict)
                or set(step["config"]) - CONFIG_KEYS[registry]
            ):
                raise ValueError(
                    "Custom processor classes are unsupported; use standard ACT registry steps"
                )
            if registry == "rename_observations_processor" and step["config"].get("rename_map"):
                raise ValueError("Policy camera names must match directly without custom renaming")
            state = step.get("state_file")
            if state is not None:
                if (
                    not isinstance(state, str)
                    or Path(state).name != state
                    or Path(state).suffix != ".safetensors"
                    or "/" in state
                    or "\\" in state
                ):
                    raise ValueError(
                        "Policy processor state must be a checkpoint safetensors basename"
                    )
                target = checkpoint / state
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or not 0 < target.stat().st_size <= 64 * 1024**2
                ):
                    raise ValueError("Policy processor tensor state is missing or invalid")
            if registry in {"normalizer_processor", "unnormalizer_processor"} and state is None:
                raise ValueError("Policy normalization needs its saved tensor state")


def validate_act_config(checkpoint):
    """Bound native ACT allocation before importing Torch or constructing modules."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Native ACT checkpoint must be a regular directory")
    weights = checkpoint / "model.safetensors"
    if (
        weights.is_symlink()
        or not weights.is_file()
        or not 0 < weights.stat().st_size <= 2 * 1024**3
    ):
        raise ValueError("Native ACT weights exceed local admission limits")
    config = read_json(checkpoint / "config.json")
    if not isinstance(config, dict) or config.get("type") != "act":
        raise ValueError("Only native ACT policies are supported")
    defaults = {
        "dim_model": (512, 512),
        "n_heads": (8, 8),
        "dim_feedforward": (3200, 4096),
        "n_encoder_layers": (4, 6),
        "n_decoder_layers": (1, 2),
        "n_vae_encoder_layers": (4, 6),
        "latent_dim": (32, 64),
        "chunk_size": (100, 100),
        "n_action_steps": (100, 100),
        "n_obs_steps": (1, 1),
    }
    for key, (default, maximum) in defaults.items():
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"ACT {key} exceeds the supported local architecture bounds")
    if config.get("dim_model", 512) % config.get("n_heads", 8):
        raise ValueError("ACT model width must divide evenly into its attention heads")
    if config.get("n_action_steps", 100) > config.get("chunk_size", 100):
        raise ValueError("ACT execution length exceeds its action chunk")
    if config.get("vision_backbone", "resnet18") not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError("Local ACT supports ResNet18, ResNet34 or ResNet50 backbones")
    inputs, outputs = config.get("input_features"), config.get("output_features")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("ACT requires explicit state, camera and action features")
    if any(not isinstance(v, dict) for v in [*inputs.values(), *outputs.values()]):
        raise ValueError("ACT feature metadata must contain bounded shape records")
    images = {k: v for k, v in inputs.items() if k.startswith("observation.images.")}
    if (
        len(images) != 1
        or set(inputs) != {"observation.state", *images}
        or inputs["observation.state"].get("shape") != [6]
        or outputs.get("action", {}).get("shape") != [6]
    ):
        raise ValueError("ACT requires one RGB camera and six state/action values")
    shape = next(iter(images.values())).get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[0] != 3
        or any(type(n) is not int or n < 1 for n in shape)
        or shape[1] > 1024
        or shape[2] > 1280
        or shape[1] * shape[2] > 921600
    ):
        raise ValueError("ACT camera dimensions exceed the supported local image budget")
    return config
