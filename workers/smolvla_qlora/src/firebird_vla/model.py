"""SmolVLA-specific NF4 conversion and PEFT integration (single BF16 CUDA device)."""

from importlib.metadata import version

QUANT_ROOTS = (
    "model.vlm_with_expert.vlm.model.text_model.layers.",
    "model.vlm_with_expert.lm_expert.layers.",
)
PROJECTIONS = [
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
]


def require_runtime():
    import torch

    expected = {
        "lerobot": "0.4.4",
        "torch": "2.7.1",
        "transformers": "4.57.1",
        "peft": "0.18.0",
        "bitsandbytes": "0.48.2",
    }
    for package, required in expected.items():
        if version(package).split("+")[0] != required:
            raise RuntimeError(f"{package} must be {required}; install the smolvla extra")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires a CUDA GPU with BF16 support (Ampere or newer)")
    torch.cuda.set_device(0)


def quantize_linears(policy, roots=QUANT_ROOTS):
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
        raise ValueError("No supported SmolVLA linear layers found; refusing unquantized training")
    policy.is_loaded_in_4bit = True  # PEFT dispatches to its bitsandbytes LoRA layer.
    return names


def build_policy(cfg, features=None, policy_config_dir=None, adapter_dir=None, trainable=True):
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import LoraConfig, PeftModel, get_peft_model

    base = snapshot_download(
        cfg.model_id,
        revision=cfg.model_revision,
        allow_patterns=["config.json", "model.safetensors"],
    )
    backbone = snapshot_download(
        cfg.backbone_id,
        revision=cfg.backbone_revision,
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
    )
    config = SmolVLAConfig.from_pretrained(policy_config_dir or base)
    config.device = "cpu"  # Never materialize an unquantized model on the target GPU.
    config.vlm_model_name = backbone  # Config/tokenizer are also pinned, no moving main ref.
    config.load_vlm_weights = False  # Full VLA checkpoint below supplies ALL pretrained weights.
    config.push_to_hub = False
    config.pretrained_path = base
    config.chunk_size = config.n_action_steps = cfg.chunk_size
    config.freeze_vision_encoder = config.train_expert_only = True
    if features is not None:
        mapped = dataset_to_policy_features(features)
        config.input_features = {k: mapped[k] for k in ("observation.state", cfg.camera_key)}
        config.output_features = {"action": mapped["action"]}
        if config.output_features["action"].type != FeatureType.ACTION:
            raise ValueError("Dataset has no action feature")
    # Strict loading is essential: an adapter over randomly initialized missing weights is invalid.
    policy = SmolVLAPolicy.from_pretrained(base, config=config, strict=True)
    policy.requires_grad_(False)
    quantized = quantize_linears(policy)
    policy.model.vlm_with_expert.vlm.model.vision_model.to(dtype=torch.bfloat16)
    policy.to("cuda:0")  # Device only; never cast packed weights after quantization.
    config.device = "cuda:0"
    if adapter_dir is None:
        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=r"model\.vlm_with_expert\.lm_expert\.layers\..*\.(q|k|v|o|gate|up|down)_proj",
            modules_to_save=PROJECTIONS,
            bias="none",
        )
        policy = get_peft_model(policy, lora)
    else:
        policy = PeftModel.from_pretrained(policy, str(adapter_dir), is_trainable=trainable)
    if trainable:
        trainable_names = [n for n, p in policy.named_parameters() if p.requires_grad]
        if not trainable_names or not any("lora_" in n for n in trainable_names):
            raise RuntimeError("No trainable LoRA parameters")
        if any("lora_" not in n and ".modules_to_save." not in n for n in trainable_names):
            raise RuntimeError("Unexpected trainable base weights")
    return policy, config, quantized


def make_processors(config, stats):
    import torch
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    tensors = {
        key: {name: torch.tensor(value) for name, value in values.items()}
        for key, values in stats.items()
    }
    return make_smolvla_pre_post_processors(config, dataset_stats=tensors)


def predict(policy, batch, postprocessor, seed):
    """Deterministic single observation probe without disturbing training RNG or action queues."""
    import torch

    was_training = policy.training
    policy.eval()
    policy.reset()
    try:
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action = policy.select_action(batch)
            result = postprocessor(action).float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite action during checkpoint probe")
        return result
    finally:
        policy.reset()
        policy.train(was_training)
