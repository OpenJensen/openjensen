"""Strict checkpoint loading and explicitly selected LM quantization."""


def load_smolvla_config(path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # Dispatch on the serialized "type" before decoding the concrete dataclass.
    config = PreTrainedConfig.from_pretrained(path)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError("Expected a SmolVLA checkpoint configuration")
    return config


def load_checkpoint_weights(policy, model_file):
    from safetensors.torch import load_file

    # The constructor uses mixed dtypes that can differ from released weights.
    # Preserve checkpoint dtypes while still requiring every key and shape.
    policy.load_state_dict(load_file(str(model_file), device="cpu"), strict=True, assign=True)
    return policy


def quantize_linears(policy, roots):
    """Load FP weights on CPU, then pack selected layers directly onto CUDA one at a time."""
    import bitsandbytes as bnb
    import torch

    names = []
    for name, layer in list(policy.named_modules()):
        if not isinstance(layer, torch.nn.Linear) or not name.startswith(roots):
            continue
        if isinstance(layer, bnb.nn.Linear4bit):
            raise ValueError(f"Already quantized: {name}")
        packed = bnb.nn.Linear4bit(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
            # SmolVLA explicitly casts activations to q/k/v/o_proj.weight.dtype.
            # Floating *storage* preserves those casts; values are still packed 4-bit NF4.
            quant_storage=torch.bfloat16,
        )
        packed.load_state_dict(layer.state_dict())
        packed.requires_grad_(False)
        packed = packed.to("cuda:0")
        parent_name, _, child_name = name.rpartition(".")
        policy.get_submodule(parent_name).set_submodule(child_name, packed)
        names.append(name)
    if not names:
        raise ValueError(
            "No supported SmolVLA linear layers found; refusing an unquantized candidate"
        )
    policy.is_loaded_in_4bit = True  # PEFT dispatches to its bitsandbytes LoRA layer.
    return names
