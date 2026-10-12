"""Materialize the trained native policy, including the actual NF4 base and adapters."""

from pathlib import Path


def floating_state_dict(policy):
    """Export NF4 + LoRA in float32 without a second lossy quantization round."""
    import bitsandbytes as bnb
    import torch
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.other import ModulesToSaveWrapper

    result = {}

    def tensor(value):
        value = value.detach().cpu()
        value = value.float() if value.is_floating_point() else value
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Cannot export nonfinite policy weights")
        return value.contiguous().clone()

    def weight(layer):
        if isinstance(layer, bnb.nn.Linear4bit):
            return (
                bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)
                .float()
                .cpu()
            )
        return layer.weight.detach().float().cpu()

    def visit(module, prefix):
        if isinstance(module, ModulesToSaveWrapper):
            active = module.active_adapter
            visit(module.modules_to_save[active], prefix)
            return
        if isinstance(module, LoraLayer):
            if module.merged or module.disable_adapters:
                raise ValueError("Export requires active, unmerged LoRA adapters")
            base = module.get_base_layer()
            merged = weight(base)
            for adapter in module.active_adapters:
                if getattr(module, "lora_variant", {}).get(adapter) is not None:
                    raise ValueError("This exporter supports standard LoRA only")
                merged += module.get_delta_weight(adapter).detach().float().cpu()
            result[prefix + "weight"] = tensor(merged)
            if base.bias is not None:
                result[prefix + "bias"] = tensor(base.bias)
            return
        if isinstance(module, bnb.nn.Linear4bit):
            result[prefix + "weight"] = tensor(weight(module))
            if module.bias is not None:
                result[prefix + "bias"] = tensor(module.bias)
            return
        for name, value in module._parameters.items():
            if value is not None:
                result[prefix + name] = tensor(value)
        for name, value in module._buffers.items():
            if value is not None and name not in module._non_persistent_buffers_set:
                result[prefix + name] = tensor(value)
        for name, child in module.named_children():
            visit(child, prefix + name + ".")

    visit(policy.get_base_model(), "")
    return result


def export_checkpoint(source, destination):
    import json

    from safetensors.torch import save_file

    from .checkpoint import load_for_inference, verify_bundle
    from .config import TrainConfig

    source, destination = Path(source), Path(destination)
    verify_bundle(source)
    cfg = TrainConfig.load(source / "recipe.json")
    policy, pre, post = load_for_inference(source)
    destination.mkdir(parents=True, exist_ok=False)
    state = floating_state_dict(policy)
    save_file(state, str(destination / "model.safetensors"))
    config = policy.get_base_model().config
    config.device = "cpu"
    config.save_pretrained(destination)
    pre.save_pretrained(destination, config_filename="policy_preprocessor.json")
    post.save_pretrained(destination, config_filename="policy_postprocessor.json")
    (destination / "export-lineage.json").write_text(
        json.dumps(
            {
                "method": cfg.method,
                "source_checkpoint": str(source),
                "representation": "float32 materialization of the trained base plus adapters",
                "task_success": None,
                "native_conversion_requires_evaluation": True,
            },
            indent=2,
        )
    )
    return destination
