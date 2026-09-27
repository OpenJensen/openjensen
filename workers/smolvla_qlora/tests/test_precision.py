"""CPU contract tests for T4 precision, packing, scaling, and resumable cursors."""

import math
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from firebird_vla.model import (
    prepare_base_precision,
    preserve_attention_precision,
    promote_trainable_parameters,
    quantize_linears,
    runtime_compute_dtype,
)
from firebird_vla.train import optimizer_step, validate_resume_cursor


@pytest.mark.parametrize("major,expected", [(7, "float16"), (8, "bfloat16"), (9, "bfloat16")])
def test_compute_dtype_uses_native_architecture_even_when_bf16_is_emulated(
    monkeypatch, major, expected
):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=lambda _: (major, 5),
                is_bf16_supported=lambda: True,
            ),
            float16="float16",
            bfloat16="bfloat16",
        ),
    )
    assert runtime_compute_dtype() == expected


@pytest.mark.parametrize("major,saved", [(7, "float16"), (8, "float16"), (9, "bfloat16")])
def test_checkpoint_precision_is_preserved_across_compatible_gpus(monkeypatch, major, saved):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, get_device_capability=lambda _: (major, 0)
            ),
            float16="float16",
            bfloat16="bfloat16",
        ),
    )
    assert runtime_compute_dtype(saved) == saved


def test_bf16_checkpoint_requires_native_bf16_instead_of_silent_lossy_conversion(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda _: (7, 5)),
            float16="float16",
            bfloat16="bfloat16",
        ),
    )
    with pytest.raises(ValueError, match="Ampere"):
        runtime_compute_dtype("bfloat16")
    with pytest.raises(ValueError, match="unsupported compute precision"):
        runtime_compute_dtype("float32")


@pytest.mark.parametrize("available,major", [(False, 8), (True, 6)])
def test_unusable_device_rejected_before_model_loading(monkeypatch, available, major):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: available, get_device_capability=lambda _: (major, 0)
            ),
            float16="float16",
            bfloat16="bfloat16",
        ),
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        runtime_compute_dtype()


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def to(self, *, dtype):
        return Tensor(dtype)

    def float(self):
        return Tensor("float32")


def test_fp16_base_conversion_preserves_fp32_and_integer_tensors(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(float16="float16", bfloat16="bfloat16")
    )
    tensors = [Tensor("bfloat16"), Tensor("float32"), Tensor("int64")]
    results = []
    policy = SimpleNamespace(_apply=lambda convert: results.extend(map(convert, tensors)))
    prepare_base_precision(policy, "float16")
    assert [tensor.dtype for tensor in results] == ["float16", "float32", "int64"]
    assert results[1:] == tensors[1:]
    results.clear()
    prepare_base_precision(policy, "bfloat16")
    assert results == []


def test_scaler_master_weights_never_cast_frozen_nf4_storage(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))

    class Parameter:
        def __init__(self, dtype, trainable):
            self.data = Tensor(dtype)
            self.requires_grad = trainable

        @property
        def dtype(self):
            return self.data.dtype

    packed = Parameter("float16", False)
    lora = Parameter("float16", True)
    projection = Parameter("float32", True)
    original_packed = packed.data
    policy = SimpleNamespace(parameters=lambda: [packed, lora, projection])
    promote_trainable_parameters(policy)
    assert packed.data is original_packed
    assert lora.dtype == projection.dtype == "float32"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_nf4_compute_and_floating_storage_always_match(monkeypatch, dtype):
    replacements = {}
    constructors = []

    class Linear:
        in_features, out_features, bias = 64, 32, None

        def state_dict(self):
            return {"weight": "checkpoint weights"}

    class Linear4bit:
        def __init__(self, *args, **kwargs):
            constructors.append(kwargs)

        def load_state_dict(self, value):
            assert value == {"weight": "checkpoint weights"}

        def requires_grad_(self, value):
            assert value is False

        def to(self, device):
            assert device == "cuda:0"
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Linear=Linear)))
    monkeypatch.setitem(
        sys.modules, "bitsandbytes", SimpleNamespace(nn=SimpleNamespace(Linear4bit=Linear4bit))
    )
    policy = SimpleNamespace(
        named_modules=lambda: [("layers.q_proj", Linear())],
        get_submodule=lambda _: SimpleNamespace(
            set_submodule=lambda name, value: replacements.update({name: value})
        ),
    )
    assert quantize_linears(policy, roots=("layers.",), compute_dtype=dtype) == ["layers.q_proj"]
    assert constructors[0]["compute_dtype"] == constructors[0]["quant_storage"] == dtype
    assert constructors[0]["quant_type"] == "nf4"
    assert constructors[0]["compress_statistics"] is True
    assert policy.is_loaded_in_4bit is True
    assert isinstance(replacements["q_proj"], Linear4bit)


def test_fp16_attention_keeps_explicit_fp32_scores_outside_autocast(monkeypatch):
    calls = []

    @contextmanager
    def autocast(device, *, enabled):
        calls.append((device, enabled))
        yield
        calls.append("leave")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16", autocast=autocast))
    expert = SimpleNamespace(
        eager_attention_forward=lambda value: calls.append(value) or "attention"
    )
    policy = SimpleNamespace(model=SimpleNamespace(vlm_with_expert=expert))
    preserve_attention_precision(policy, "float16")
    assert expert.eager_attention_forward("scores") == "attention"
    assert calls == [("cuda", False), "scores", "leave"]


@pytest.mark.parametrize(
    "norm,enabled,updated",
    [
        (2.5, True, True),
        (float("inf"), True, False),
        (float("nan"), True, False),
        (2.5, False, True),
    ],
)
def test_scaled_optimizer_unscales_before_clipping_and_skips_overflow(
    monkeypatch, norm, enabled, updated
):
    calls = []
    optimizer = object()
    parameters = [object()]

    def clip(actual, maximum, *, error_if_nonfinite):
        assert actual is parameters and maximum == 1
        assert error_if_nonfinite is not enabled
        calls.append("clip")
        return norm

    scaler = SimpleNamespace(
        is_enabled=lambda: enabled,
        unscale_=lambda actual: calls.append("unscale") if actual is optimizer else None,
        step=lambda actual: calls.append("step") if actual is optimizer else None,
        get_scale=lambda: 65536,
        update=lambda **kwargs: calls.append(("update", kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=clip)),
            isfinite=math.isfinite,
        ),
    )
    _, actual = optimizer_step(parameters, optimizer, scaler, 1)
    assert actual is updated
    assert calls[:2] == ["unscale", "clip"]
    assert calls[2:] == (
        ["step", ("update", {})] if updated else [("update", {"new_scale": 32768})]
    )


@pytest.mark.parametrize("step,consumed,scaled", [(2, 24, True), (2, 16, False), (2, 16, True)])
def test_resume_cursor_accounts_for_skipped_scaled_update_windows(step, consumed, scaled):
    validate_resume_cursor(step, consumed, 8, scaled)


@pytest.mark.parametrize(
    "step,consumed,scaled",
    [(2, 24, False), (2, 8, True), (2, 17, True), (-1, 0, True), (True, 8, False)],
)
def test_invalid_resume_cursor_is_rejected(step, consumed, scaled):
    with pytest.raises(ValueError, match="cursor"):
        validate_resume_cursor(step, consumed, 8, scaled)
