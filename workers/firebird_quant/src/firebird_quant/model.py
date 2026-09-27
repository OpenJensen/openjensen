"""Portable eager inference using packed parameter storage and ordinary Torch operators."""

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import parametrize

from .codec import PackedTensor, unpack
from .state import QuantizedState, Recipe, load, quantize_state_dict


class _PackedWeight(nn.Module):
    def __init__(self, packed):
        super().__init__()
        self.register_buffer("codes", packed.codes.clone())
        # Byte storage prevents model.half()/bfloat16() from rounding calibration scales.
        self.register_buffer("scale_bytes", packed.scales.view(torch.uint8).clone())
        self.scale_dtype = packed.scales.dtype
        self.shape, self.bits, self.group_size = packed.shape, packed.bits, packed.group_size

    def forward(self, anchor):
        return unpack(
            self.codes,
            self.scale_bytes.view(self.scale_dtype),
            self.shape,
            anchor.dtype,
            self.bits,
            self.group_size,
        )


@dataclass
class QuantizedModel:
    model: nn.Module
    state: QuantizedState

    @property
    def audit(self):
        return self.state.audit

    def save(self, path):
        """Save the frozen conversion, independent of subsequent model device/dtype moves."""
        return self.state.save(path)


def _check_model(model):
    if not isinstance(model, nn.Module):
        raise TypeError("Expected a torch.nn.Module; use quantize_state_dict for tensor mappings")
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.modules.lazy.LazyModuleMixin)
            and module.has_uninitialized_params()
        ):
            raise ValueError(f"{name or '<root>'}: lazy tensors must be materialized first")
        if parametrize.is_parametrized(module):
            raise ValueError(
                f"{name or '<root>'}: existing parametrizations must be materialized first"
            )
        if isinstance(module, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
            raise ValueError("Use an eager module before quantization, not a TorchScript module")
        if any(
            (
                module._state_dict_hooks,
                module._state_dict_pre_hooks,
                module._load_state_dict_pre_hooks,
                module._load_state_dict_post_hooks,
            )
        ):
            raise ValueError("Custom state_dict hooks require an explicit model adapter")
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)) and module.max_norm is not None:
            raise ValueError("Embedding max_norm mutates weights; use an explicit model adapter")
    storage_owners = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if isinstance(
            tensor, (nn.parameter.UninitializedParameter, nn.parameter.UninitializedBuffer)
        ):
            raise ValueError(f"{name}: lazy tensors must be materialized first")
        if type(tensor) not in (torch.Tensor, nn.Parameter):
            raise ValueError(f"{name}: tensor subclasses require an explicit model adapter")
        if tensor.layout != torch.strided or tensor.is_meta or tensor.is_quantized:
            raise ValueError(f"{name}: expected a dense, materialized, unquantized tensor")
        if tensor.numel():
            key = (tensor.device, tensor.untyped_storage().data_ptr())
            owner = storage_owners.setdefault(key, id(tensor))
            if owner != id(tensor):
                raise ValueError("Shared-storage views require an explicit model adapter")


def _install(model, state):
    _check_model(model)
    original = model.state_dict()
    names = set(state.tensors) | set(state.aliases)
    if set(original) != names:
        raise ValueError("Checkpoint tensor names do not match the supplied model")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if not names <= parameters.keys() | buffers.keys():
        raise ValueError("Custom state_dict hooks require an explicit model adapter")
    for name, tensor in original.items():
        value = state.tensors[state.aliases.get(name, name)]
        dtype = value.dtype if isinstance(value, PackedTensor) else str(value.dtype)
        if (
            tensor.is_meta
            or tuple(tensor.shape) != tuple(value.shape)
            or str(tensor.dtype) != dtype
        ):
            raise ValueError(f"Checkpoint shape/dtype/device mismatch: {name}")
    # A differently tied architecture could change semantics after loading. Require the same ties.
    identities = {}
    for name, tensor in {**parameters, **buffers}.items():
        if name in names:
            identities.setdefault(id(tensor), []).append(name)
    for tied_names in identities.values():
        if len({state.aliases.get(name, name) for name in tied_names}) != 1:
            raise ValueError("Supplied model ties weights that the checkpoint does not tie")
    for name, target in state.aliases.items():
        slots = parameters | buffers
        if slots[name] is not slots[target]:
            raise ValueError("Shared-storage views require an explicit model adapter")
    converted = copy.deepcopy(model).eval()
    converted.requires_grad_(False)
    modules = dict(converted.named_modules(remove_duplicate=False))
    shared = {}
    with torch.no_grad():
        for name in original:
            target = state.aliases.get(name, name)
            value = state.tensors[target]
            path, _, attribute = name.rpartition(".")
            module = modules[path]
            # Shared submodules appear under multiple paths. Install each physical slot once.
            if parametrize.is_parametrized(module, attribute):
                continue
            current = getattr(module, attribute)
            if isinstance(value, PackedTensor):
                if target not in shared:
                    shared[target] = _PackedWeight(value).to(device=current.device)
                device, dtype = current.device, current.dtype
                delattr(module, attribute)
                module.register_buffer(attribute, torch.empty(0, device=device, dtype=dtype))
                # The zero-size anchor records the compute dtype; no floating master is retained.
                parametrize.register_parametrization(module, attribute, shared[target], unsafe=True)
            else:
                current.copy_(value.to(device=current.device))
        # RNNBase caches Parameter references outside its state_dict. Discard the old copies.
        for module in converted.modules():
            if isinstance(module, nn.RNNBase):
                module._init_flat_weights()
    return QuantizedModel(converted, state)


def quantize(model: nn.Module, recipe=Recipe()) -> QuantizedModel:
    """Return an independent inference model; arbitrary registered dense weights are eligible."""
    _check_model(model)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    state = quantize_state_dict(model.state_dict(), recipe, parameter_names=set(parameters))
    return _install(model, state)


def load_model(model: nn.Module, path) -> QuantizedModel:
    """Caller supplies trusted architecture/configuration; the artifact supplies tensors only."""
    return _install(model, load(path))
