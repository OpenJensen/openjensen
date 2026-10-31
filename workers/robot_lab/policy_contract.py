"""Bounded native ACT processor metadata, shared by admission and inference."""

import json
import math
import re
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


SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLM_ASSET_PATH = "assets/smolvlm"
SMOLVLA_PIPELINES = {
    "policy_preprocessor.json": (
        "rename_observations_processor",
        "to_batch_processor",
        "smolvla_new_line_processor",
        "tokenizer_processor",
        "device_processor",
        "normalizer_processor",
    ),
    "policy_postprocessor.json": PIPELINES["policy_postprocessor.json"],
}
SMOLVLM_ASSET_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.json",
    "generation_config.json",
}
SMOLVLM_REQUIRED_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
}
CONFIG_KEYS.update(
    {
        "smolvla_new_line_processor": set(),
        "tokenizer_processor": {
            "max_length",
            "task_key",
            "padding_side",
            "padding",
            "truncation",
            "tokenizer_name",
        },
    }
)


def read_json(path, maximum=1024 * 1024):
    def invalid_constant(value):
        raise ValueError("Policy metadata cannot contain non-finite numbers")

    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Policy metadata must be a bounded regular file")
    return json.loads(path.read_text(), parse_constant=invalid_constant)


def validate_processors(checkpoint, policy_type="act"):
    """Reject dynamic import fallback, external artifacts and escaping tensor state."""
    checkpoint = Path(checkpoint)
    if policy_type not in {"act", "smolvla"}:
        raise ValueError("Only native ACT and SmolVLA processors are supported")
    pipelines = SMOLVLA_PIPELINES if policy_type == "smolvla" else PIPELINES
    for filename, expected in pipelines.items():
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
            if registry == "tokenizer_processor":
                config = step["config"]
                maximum = read_json(checkpoint / "config.json").get("tokenizer_max_length", 48)
                if (
                    set(config) != CONFIG_KEYS[registry]
                    or type(config["max_length"]) is not int
                    or not 1 <= config["max_length"] <= 128
                    or config["max_length"] != maximum
                    or config["task_key"] != "task"
                    or config["padding_side"] != "right"
                    or config["padding"] != "max_length"
                    or config["truncation"] is not True
                    or config["tokenizer_name"] not in {SMOLVLM_MODEL, SMOLVLM_ASSET_PATH}
                ):
                    raise ValueError("SmolVLA requires its standard bounded local task tokenizer")
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


def _bounded_integer(config, key, default, minimum, maximum):
    value = config.get(key, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"SmolVLA {key} exceeds the supported local architecture bounds")
    return value


def _data_only(value):
    if isinstance(value, dict):
        if any(key in value for key in ("auto_map", "custom_pipelines", "code_revision")):
            raise ValueError("SmolVLM assets cannot load custom code")
        if value.get("trust_remote_code") not in (None, False):
            raise ValueError("SmolVLM assets cannot trust remote code")
        for item in value.values():
            _data_only(item)
    elif isinstance(value, list):
        for item in value:
            _data_only(item)


def _validate_smolvlm_dimensions(config):
    if not isinstance(config, dict) or config.get("model_type") != "smolvlm":
        raise ValueError("SmolVLA requires the native SmolVLM2 500M asset configuration")
    text, vision = config.get("text_config"), config.get("vision_config")
    if (
        not isinstance(text, dict)
        or text.get("model_type") != "llama"
        or not isinstance(vision, dict)
        or vision.get("model_type") != "smolvlm_vision"
        or config.get("scale_factor") != 4
    ):
        raise ValueError("Unsupported SmolVLM text, vision or projection architecture")
    # The native constructor builds the full backbone before truncating retained layers.
    for key, maximum in {
        "hidden_size": 960,
        "intermediate_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 15,
        "num_key_value_heads": 5,
        "vocab_size": 49280,
    }.items():
        _bounded_integer(text, key, None, 1, maximum)
    _bounded_integer(text, "head_dim", 64, 1, 64)
    _bounded_integer(text, "max_position_embeddings", 8192, 1, 8192)
    if (
        text["hidden_size"] % text["num_attention_heads"]
        or text["num_attention_heads"] % text["num_key_value_heads"]
    ):
        raise ValueError("SmolVLM attention dimensions are incompatible")
    for key, default, maximum in (
        ("hidden_size", None, 768),
        ("intermediate_size", 3072, 3072),
        ("num_hidden_layers", 12, 12),
        ("num_attention_heads", None, 12),
        ("num_channels", 3, 3),
        ("image_size", None, 512),
    ):
        _bounded_integer(vision, key, default, 1, maximum)
    patch = _bounded_integer(vision, "patch_size", None, 16, 32)
    if vision["hidden_size"] % vision["num_attention_heads"] or vision["image_size"] % patch:
        raise ValueError("SmolVLM vision dimensions are incompatible")


def validate_smolvla_assets(checkpoint):
    """Return the fixed local data-only backbone/tokenizer closure; never resolve a Hub path."""
    checkpoint = Path(checkpoint)
    assets, root = checkpoint / "assets", checkpoint / SMOLVLM_ASSET_PATH
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or assets.is_symlink()
        or not assets.is_dir()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise ValueError("SmolVLA requires fixed local assets/smolvlm tokenizer assets")
    if {path.name for path in assets.iterdir()} != {"smolvlm"}:
        raise ValueError("SmolVLA assets must contain only the fixed smolvlm directory")
    seen, total, documents = set(), 0, {}
    for path in root.iterdir():
        if path.name not in SMOLVLM_ASSET_FILES or path.is_symlink() or not path.is_file():
            raise ValueError("SmolVLM assets accept only known data files without symbolic links")
        size = path.stat().st_size
        total += size
        if size <= 0 or total >= 32 * 1024**2:
            raise ValueError("SmolVLM data assets must total less than 32 MiB")
        seen.add(path.name)
        if path.suffix == ".json":
            try:
                value = read_json(path, 32 * 1024**2)
                _data_only(value)
            except (RecursionError, UnicodeError) as error:
                raise ValueError("Invalid SmolVLM data-only JSON") from error
            documents[path.name] = value
    if not SMOLVLM_REQUIRED_FILES <= seen:
        raise ValueError(
            "SmolVLA requires complete local configuration, processor and tokenizer assets"
        )
    _validate_smolvlm_dimensions(documents["config.json"])
    if not isinstance(documents["tokenizer_config.json"], dict):
        raise ValueError("SmolVLM tokenizer configuration must be an object")
    if not isinstance(documents["tokenizer.json"], dict):
        raise ValueError("SmolVLM tokenizer data must be an object")
    return root


def validate_smolvla_config(checkpoint):
    """Bound native SmolVLA allocation and semantics before importing its runtime."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Native SmolVLA checkpoint must be a regular directory")
    weights = checkpoint / "model.safetensors"
    if (
        weights.is_symlink()
        or not weights.is_file()
        or not 0 < weights.stat().st_size <= 2 * 1024**3
    ):
        raise ValueError("Native SmolVLA weights exceed local admission limits")
    config = read_json(checkpoint / "config.json")
    if not isinstance(config, dict) or config.get("type") != "smolvla":
        raise ValueError("Only native SmolVLA policies are supported")
    for key, default, minimum, maximum in (
        ("n_obs_steps", 1, 1, 1),
        ("max_state_dim", 32, 6, 32),
        ("max_action_dim", 32, 6, 32),
        ("chunk_size", 50, 1, 50),
        ("n_action_steps", 50, 1, 50),
        ("num_steps", 10, 1, 10),
        ("num_vlm_layers", 16, 1, 16),
        ("tokenizer_max_length", 48, 1, 128),
        ("self_attn_every_n_layers", 2, 1, 16),
        ("empty_cameras", 0, 0, 0),
        ("prefix_length", -1, -1, 128),
    ):
        _bounded_integer(config, key, default, minimum, maximum)
    if config.get("n_action_steps", 50) > config.get("chunk_size", 50):
        raise ValueError("SmolVLA execution length exceeds its action chunk")
    layers = config.get("num_vlm_layers", 16)
    expert = config.get("num_expert_layers", -1)
    if type(expert) is not int or expert < -1 or expert > 16 or expert > 0 and layers % expert:
        raise ValueError("SmolVLA expert layers must divide the retained VLM layers")
    width = config.get("expert_width_multiplier", 0.75)
    if type(width) not in (int, float) or not math.isfinite(width) or not 0.25 <= width <= 0.75:
        raise ValueError("SmolVLA expert width exceeds the supported local architecture bounds")
    resize = config.get("resize_imgs_with_padding", [512, 512])
    if (
        not isinstance(resize, list)
        or len(resize) != 2
        or any(type(n) is not int or not 1 <= n <= 512 for n in resize)
    ):
        raise ValueError("SmolVLA requires bounded image resizing within 512 pixels")
    if (
        config.get("rtc_config") is not None
        or config.get("compile_model", False) is not False
        or config.get("adapt_to_pi_aloha", False) is not False
        or config.get("use_delta_joint_actions_aloha", False) is not False
        or config.get("attention_mode", "cross_attn") != "cross_attn"
        or config.get("vlm_model_name", SMOLVLM_MODEL)
        not in {
            SMOLVLM_MODEL,
            SMOLVLM_ASSET_PATH,
        }
    ):
        raise ValueError(
            "SmolVLA requires native local inference without RTC, compilation or Aloha adaptation"
        )
    inputs, outputs = config.get("input_features"), config.get("output_features")
    if (
        not isinstance(inputs, dict)
        or not isinstance(outputs, dict)
        or any(not isinstance(feature, dict) for feature in [*inputs.values(), *outputs.values()])
    ):
        raise ValueError("SmolVLA requires explicit native feature records")
    cameras = {key: feature for key, feature in inputs.items() if feature.get("type") == "VISUAL"}
    if (
        not 1 <= len(cameras) <= 3
        or set(inputs) != {"observation.state", *cameras}
        or inputs["observation.state"].get("type") != "STATE"
        or inputs["observation.state"].get("shape") != [6]
        or set(outputs) != {"action"}
        or outputs["action"].get("type") != "ACTION"
        or outputs["action"].get("shape") != [6]
    ):
        raise ValueError("SmolVLA requires one to three RGB cameras and six state/action values")
    for key, feature in cameras.items():
        shape = feature.get("shape")
        if (
            re.fullmatch(r"observation\.(?:images\.[a-zA-Z0-9_.-]{1,64}|image[0-9]*)", key) is None
            or not isinstance(shape, list)
            or len(shape) != 3
            or shape[0] != 3
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[1] > 1024
            or shape[2] > 1280
            or shape[1] * shape[2] > 921600
        ):
            raise ValueError("SmolVLA camera features exceed the supported RGB image budget")
    validate_smolvla_assets(checkpoint)
    return config


def validate_policy_config(checkpoint):
    """Closed dispatch to reviewed built-in native policy architectures."""
    config = read_json(Path(checkpoint) / "config.json")
    if not isinstance(config, dict):
        raise ValueError("Policy configuration must be an object")
    if config.get("type") == "act":
        return validate_act_config(checkpoint)
    if config.get("type") == "smolvla":
        return validate_smolvla_config(checkpoint)
    raise ValueError("Only native ACT and SmolVLA policies are supported")
