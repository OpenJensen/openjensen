"""OPEN JENSEN Quant: one packed-weight API across model families."""

from importlib import import_module

__all__ = [
    "PackedTensor",
    "QuantizedModel",
    "QuantizedState",
    "Recipe",
    "load",
    "load_model",
    "pack",
    "quantize",
    "quantize_state_dict",
]


_EXPORT_MODULES = {
    "PackedTensor": "codec",
    "pack": "codec",
    "QuantizedModel": "model",
    "load_model": "model",
    "quantize": "model",
    "QuantizedState": "state",
    "Recipe": "state",
    "load": "state",
    "quantize_state_dict": "state",
}


def __getattr__(name):
    """Keep structural package inspection free of Torch and model imports."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
