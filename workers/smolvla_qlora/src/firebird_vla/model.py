"""SmolVLA NF4/PEFT integration with native BF16 or scaled FP16 CUDA training."""

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


def runtime_compute_dtype(saved_precision=None):
    """Choose real device support, excluding software-emulated BF16 on T4."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires a CUDA GPU")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 7:
        raise RuntimeError("This recipe requires CUDA compute capability 7.0 or newer")
    if saved_precision is not None:
        if saved_precision not in {"float16", "bfloat16"}:
            raise ValueError("Checkpoint has an unsupported compute precision")
        if saved_precision == "bfloat16" and major < 8:
            raise ValueError("This BF16 checkpoint requires an Ampere or newer GPU (L4/A100)")
        return getattr(torch, saved_precision)
    return torch.bfloat16 if major >= 8 else torch.float16


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
    dtype = runtime_compute_dtype()
    torch.cuda.set_device(0)
    return dtype


def quantize_linears(policy, roots=QUANT_ROOTS, compute_dtype=None):
    """Load FP weights on CPU, then pack selected layers directly onto CUDA one at a time."""
    import bitsandbytes as bnb
    import torch

    compute_dtype = compute_dtype or runtime_compute_dtype()
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
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type="nf4",
            # SmolVLA explicitly casts activations to q/k/v/o_proj.weight.dtype.
            # Floating *storage* preserves those casts; values are still packed 4-bit NF4.
            quant_storage=compute_dtype,
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


def prepare_base_precision(policy, compute_dtype):
    """Convert BF16 base tensors before packing, preserving FP32 normalization."""
    import torch

    if compute_dtype == torch.float16:
        policy._apply(
            lambda tensor: (
                tensor.to(dtype=torch.float16) if tensor.dtype == torch.bfloat16 else tensor
            )
        )


def promote_trainable_parameters(policy):
    """GradScaler unscales FP32 adapter/master gradients, never packed weights."""
    import torch

    for parameter in policy.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()


def preserve_attention_precision(policy, compute_dtype):
    """Honor SmolVLA's explicit FP32 attention scores even under FP16 autocast."""
    import torch

    if compute_dtype == torch.float16:
        expert = policy.model.vlm_with_expert
        original = expert.eager_attention_forward

        def attention(*args, **kwargs):
            with torch.autocast("cuda", enabled=False):
                return original(*args, **kwargs)

        expert.eager_attention_forward = attention


def configure_policy_features(cfg, config, features=None, *, from_checkpoint=False):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features

    cameras = cfg.selected_camera_keys
    if from_checkpoint and tuple(config.image_features) != cameras:
        raise ValueError("Checkpoint camera inputs differ from the training recipe")
    if features is not None:
        mapped = dataset_to_policy_features(features)
        for key in cameras:
            if key not in mapped or mapped[key].type != FeatureType.VISUAL:
                raise ValueError(f"Missing image/video camera {key}")
        config.input_features = {key: mapped[key] for key in ("observation.state", *cameras)}
        config.output_features = {"action": mapped["action"]}
        if config.output_features["action"].type != FeatureType.ACTION:
            raise ValueError("Dataset has no action feature")
        config.empty_cameras = 0


def build_policy(
    cfg,
    features=None,
    policy_config_dir=None,
    adapter_dir=None,
    trainable=True,
    compute_dtype=None,
):
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file

    compute_dtype = compute_dtype or runtime_compute_dtype()
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
    # The registry base dispatches the serialized `type` before subclass decoding.
    config = PreTrainedConfig.from_pretrained(policy_config_dir or base)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("The pinned checkpoint must contain a SmolVLA configuration")
    config.device = "cpu"  # Never materialize an unquantized model on the target GPU.
    config.vlm_model_name = backbone  # Config/tokenizer are also pinned, no moving main ref.
    config.load_vlm_weights = False  # Full VLA checkpoint below supplies ALL pretrained weights.
    config.push_to_hub = False
    config.pretrained_path = base
    config.chunk_size = config.n_action_steps = cfg.chunk_size
    config.freeze_vision_encoder = config.train_expert_only = True
    configure_policy_features(cfg, config, features, from_checkpoint=policy_config_dir is not None)
    # Strict loading is essential: an adapter over randomly initialized missing weights is invalid.
    policy = SmolVLAPolicy(config)
    # Preserve the checkpoint's mixed F32/BF16 dtypes while requiring every key/shape.
    # safetensors.load_model(strict=True) also rejects intentional constructor dtype differences.
    from pathlib import Path

    policy.load_state_dict(
        load_file(str(Path(base) / "model.safetensors")), strict=True, assign=True
    )
    policy.requires_grad_(False)
    prepare_base_precision(policy, compute_dtype)
    preserve_attention_precision(policy, compute_dtype)
    quantized = (
        quantize_linears(policy, compute_dtype=compute_dtype) if cfg.method == "qlora" else []
    )
    policy.model.vlm_with_expert.vlm.model.vision_model.to(dtype=compute_dtype)
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
        if compute_dtype == torch.float16:
            promote_trainable_parameters(policy)
        trainable_names = [n for n, p in policy.named_parameters() if p.requires_grad]
        if not trainable_names or not any("lora_" in n for n in trainable_names):
            raise RuntimeError("No trainable LoRA parameters")
        if any("lora_" not in n and ".modules_to_save." not in n for n in trainable_names):
            raise RuntimeError("Unexpected trainable base weights")
    policy._firebird_compute_dtype = compute_dtype
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
            compute_dtype = (
                getattr(policy, "_firebird_compute_dtype", None) or runtime_compute_dtype()
            )
            with torch.autocast("cuda", dtype=compute_dtype):
                action = policy.select_action(batch)
            result = postprocessor(action).float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite action during checkpoint probe")
        return result
    finally:
        policy.reset()
        policy.train(was_training)
