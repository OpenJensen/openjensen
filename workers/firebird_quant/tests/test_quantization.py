import copy
import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from firebird_quant import Recipe, load, load_model, pack, quantize, quantize_state_dict
from firebird_quant.state import tensor_hash


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(731)
    torch.set_num_threads(1)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(), (0,), (131,), (7, 19), (8, 3, 3, 3), (2, 3, 4, 5, 6)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_arbitrary_rank_dtype_and_tail_have_bounded_rounding_error(bits, shape, dtype):
    source = torch.randn(shape, dtype=dtype)
    result = pack(source, bits=bits, group_size=32)
    restored = result.dequantize()
    assert restored.shape == source.shape and restored.dtype == source.dtype
    assert result.codes.dtype == torch.uint8
    if source.numel():
        bound = result.scales.repeat_interleave(32)[: source.numel()] / 2
        rounding = source.abs().float().flatten() * torch.finfo(dtype).eps
        assert ((source - restored).abs().flatten() <= bound + rounding + 1e-7).all()
    assert (
        result.nbytes == result.codes.numel() + result.scales.numel() * result.scales.element_size()
    )


def test_int4_really_packs_two_codes_per_byte():
    source = torch.tensor(
        [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0]
    )
    packed = pack(source, group_size=16)
    assert packed.codes.tolist() == [0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x8F]
    assert torch.equal(packed.dequantize(), source)


@pytest.mark.parametrize("bits", [4, 8])
def test_zero_and_subnormal_blocks_stay_finite(bits):
    for source in (torch.zeros(128), torch.full((128,), 1e-42)):
        result = pack(source, bits=bits)
        assert torch.isfinite(result.dequantize()).all()
        assert (result.scales > 0).all()


@pytest.mark.parametrize(
    "options",
    [
        {"bits": 3},
        {"bits": True},
        {"group_size": 0},
        {"group_size": 7},
        {"min_elements": -1},
        {"exclude": "head*"},
    ],
)
def test_invalid_recipe_fails(options):
    with pytest.raises(ValueError):
        Recipe(**options)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_fail_before_modifying_model(value):
    model = nn.Linear(32, 32)
    with torch.no_grad():
        model.weight[0, 0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        quantize(model)
    assert isinstance(model.weight, nn.Parameter)


class CustomFunctional(nn.Module):
    """No Linear/Conv child and no conventional weight names."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Parameter(torch.randn(32, 32) / 8)
        self.register_buffer("positions", torch.randn(16, 32))

    def forward(self, x):
        return torch.sin(x @ self.projection) + self.positions[: x.shape[0]]


class TiedLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.head = nn.Linear(16, 32, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Linear(16, 16)
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.norm = nn.GroupNorm(4, 16)

    def forward(self, image, timestep):
        hidden = image + self.time(timestep)[:, :, None, None]
        return image + self.conv(F.silu(self.norm(hidden)))


def cases():
    return [
        (nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8)), (torch.randn(4, 32),)),
        (
            nn.Sequential(
                nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 8, 3, groups=2)
            ),
            (torch.randn(2, 3, 10, 10),),
        ),
        (nn.Conv1d(8, 16, 5, groups=2), (torch.randn(2, 8, 32),)),
        (nn.Conv3d(4, 8, 3), (torch.randn(1, 4, 6, 6, 6),)),
        (nn.ConvTranspose2d(8, 8, 3, groups=2), (torch.randn(1, 8, 5, 5),)),
        (nn.GRU(16, 32, num_layers=2, batch_first=True), (torch.randn(2, 5, 16),)),
        (nn.LSTM(16, 32, batch_first=True, bidirectional=True), (torch.randn(2, 5, 16),)),
        (nn.RNN(16, 32, batch_first=True), (torch.randn(2, 5, 16),)),
        (
            nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, dropout=0),
            (torch.randn(2, 5, 32),),
        ),
        (CustomFunctional(), (torch.randn(4, 32),)),
        (TiedLanguage(), (torch.randint(0, 32, (2, 5)),)),
        (TinyDenoiser(), (torch.randn(2, 16, 8, 8), torch.randn(2, 16))),
    ]


def assert_close_tree(left, right, **kwargs):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, **kwargs)
    else:
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_close_tree(a, b, **kwargs)


@pytest.mark.parametrize("index", range(12))
@pytest.mark.parametrize("bits", [4, 8])
def test_model_families_forward_matches_dequantized_reference_and_reload(index, bits, tmp_path):
    model, inputs = cases()[index]
    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = quantize(model, Recipe(bits=bits))
    reference = copy.deepcopy(model)
    reference.load_state_dict(result.state.dequantize(), strict=True)
    assert result.audit["quantized_elements"] > 0
    assert result.audit["stored_tensor_bytes"] < result.audit["source_tensor_bytes"]
    with torch.inference_mode():
        expected = reference(*inputs)
        actual = result.model(*inputs)
        assert_close_tree(expected, actual, rtol=1e-6, atol=1e-6)
        # Round-trip correctness is separate from model-dependent task-quality loss.
    path = tmp_path / "model.fbq"
    result.save(path)
    restored = load_model(model, path)
    with torch.inference_mode():
        assert_close_tree(restored.model(*inputs), actual, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert all(not parameter.requires_grad for parameter in result.model.parameters())


def test_buffers_biases_and_norms_remain_exact_and_reported():
    model = CustomFunctional()
    result = quantize(model)
    rows = {row["name"]: row for row in result.audit["tensors"]}
    assert rows["projection"]["quantized"]
    assert rows["positions"]["retained_reason"] == "buffer"
    assert torch.equal(result.model.positions, model.positions)
    result = quantize(nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32)))
    rows = {row["name"]: row for row in result.audit["tensors"]}
    assert rows["0.bias"]["retained_reason"] == "rank_below_minimum"
    assert rows["1.weight"]["retained_reason"] == "rank_below_minimum"


def test_tied_exclusion_applies_to_every_alias_and_packed_storage_is_shared():
    model = TiedLanguage()
    retained = quantize(model, Recipe(exclude=("head.*",)))
    assert retained.audit["quantized_elements"] == 0
    result = quantize(model)
    assert result.state.aliases == {"head.weight": "embedding.weight"}
    assert (
        result.model.embedding.parametrizations.weight[0]
        is result.model.head.parametrizations.weight[0]
    )
    assert result.audit["unique_elements"] == model.embedding.weight.numel()


def test_only_packed_weights_and_empty_anchors_remain_in_module():
    result = quantize(nn.Linear(128, 128, bias=False))
    values = result.model.state_dict().values()
    assert sum(value.numel() * value.element_size() for value in values) == 9216
    assert not list(result.model.parameters())
    assert result.model.weight.shape == (128, 128)


def test_dtype_conversion_does_not_round_scales():
    result = quantize(nn.Linear(32, 32, bias=False))
    original = result.model.weight
    scales = result.model.parametrizations.weight[0].scale_bytes.clone()
    result.model.half()
    assert result.model.weight.dtype == torch.float16
    assert torch.equal(result.model.weight, original.half())
    assert torch.equal(scales, result.model.parametrizations.weight[0].scale_bytes)
    result.model.double()
    assert result.model.weight.dtype == torch.float64


def test_include_exclude_small_tensors_and_integer_state():
    state = {
        "encoder.kernel": torch.randn(64, 64),
        "head.kernel": torch.randn(64, 64),
        "tiny": torch.randn(2, 2),
        "step": torch.tensor(3),
    }
    result = quantize_state_dict(state, Recipe(include=("encoder.*", "tiny"), min_elements=0))
    rows = {row["name"]: row for row in result.audit["tensors"]}
    assert rows["encoder.kernel"]["quantized"]
    assert rows["head.kernel"]["retained_reason"] == "outside_include"
    assert rows["tiny"]["retained_reason"] == "packing_would_not_reduce_storage"
    assert torch.equal(result.dequantize()["step"], state["step"])


def test_invalid_model_state_and_existing_parametrization_fail():
    with pytest.raises(ValueError, match="materialized"):
        quantize(nn.Linear(32, 32, device="meta"))
    with pytest.raises(ValueError, match="dense"):
        quantize_state_dict({"sparse": torch.eye(32).to_sparse()})
    from torch.nn.utils.parametrizations import weight_norm

    with pytest.raises(ValueError, match="parametrizations"):
        quantize(weight_norm(nn.Linear(32, 32)))


def test_wrong_model_and_ties_rejected_on_load(tmp_path):
    path = tmp_path / "model.fbq"
    quantize(TiedLanguage()).save(path)
    with pytest.raises(ValueError, match="names"):
        load_model(nn.Linear(32, 32), path)
    wrong = TiedLanguage()
    wrong.head.weight = nn.Parameter(wrong.head.weight.clone())
    with pytest.raises(ValueError, match="Shared-storage"):
        load_model(wrong, path)


def test_publish_does_not_overwrite_and_cleans_partial_files(tmp_path):
    result = quantize(nn.Linear(32, 32))
    path = tmp_path / "model.fbq"
    result.save(path)
    contents = path.read_bytes()
    with pytest.raises(FileExistsError):
        result.save(path)
    assert path.read_bytes() == contents
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("corruption", ["version", "checksum", "scale", "alias", "inventory"])
def test_malformed_artifacts_are_rejected(tmp_path, corruption):
    path = tmp_path / "model.fbq"
    quantize(nn.Linear(32, 32)).save(path)
    with safe_open(path, framework="pt") as reader:
        metadata = reader.metadata()
        storage = {name: reader.get_tensor(name) for name in reader.keys()}
    manifest = json.loads(metadata["firebird_quant"])
    if corruption == "version":
        manifest["version"] = 999
    elif corruption == "checksum":
        storage["t0.codes"][0] ^= 1
    elif corruption == "scale":
        storage["t0.scales"][0] = 0
        manifest["storage_hashes"]["t0.scales"] = tensor_hash(storage["t0.scales"])
    elif corruption == "alias":
        manifest["aliases"] = {"missing": "missing"}
    elif corruption == "inventory":
        storage["unexpected"] = torch.ones(1)
    corrupted = tmp_path / "corrupt.fbq"
    save_file(storage, corrupted, metadata={"firebird_quant": json.dumps(manifest)})
    with pytest.raises(ValueError):
        load(corrupted)


def test_cli_new_process_roundtrip_and_actual_file_compression(tmp_path):
    source, packed, restored = (
        tmp_path / name for name in ("float.safetensors", "q.fbq", "out.safetensors")
    )
    weights = {"image_encoder.kernel": torch.randn(64, 32, 3, 3), "head.bias": torch.randn(64)}
    save_file(weights, source)

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "firebird_quant.cli", *map(str, args)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=30,
        )

    completed = run("quantize", source, "--out", packed)
    assert completed.returncode == 0, completed.stderr
    audit = json.loads(completed.stdout)
    assert audit["quantized_elements"] == weights["image_encoder.kernel"].numel()
    assert packed.stat().st_size < source.stat().st_size * 0.3
    assert not audit["quality_verified"] and not audit["speedup_verified"]
    assert run("inspect", packed).returncode == 0
    assert run("dequantize", packed, "--out", restored).returncode == 0
    assert torch.equal(load_file(restored)["head.bias"], weights["head.bias"])
    assert run("quantize", source, "--out", packed).returncode != 0
    assert (
        run(
            "quantize", source, "--out", tmp_path / "empty.fbq", "--include", "missing.*"
        ).returncode
        != 0
    )


def test_batchnorm_state_dict_metadata_survives_export(tmp_path):
    result = quantize(nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32)))
    path = tmp_path / "model.fbq"
    result.save(path)
    assert load(path).dequantize()._metadata["1"]["version"] == 2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("bits", [4, 8])
def test_largest_finite_values_do_not_overflow_during_dequantization(dtype, bits):
    source = torch.tensor([torch.finfo(dtype).max, -torch.finfo(dtype).max], dtype=dtype)
    restored = pack(source, bits=bits).dequantize()
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored, source, rtol=2 * torch.finfo(dtype).eps, atol=0)


def test_partial_shared_storage_and_mutating_embedding_are_rejected():
    model = nn.Module()
    model.first = nn.Parameter(torch.randn(32, 32))
    model.second = nn.Parameter(model.first[:16])
    with pytest.raises(ValueError, match="Shared-storage"):
        quantize(model)
    with pytest.raises(ValueError, match="max_norm"):
        quantize(nn.Embedding(32, 32, max_norm=1))
    with pytest.raises(ValueError, match="lazy"):
        quantize(nn.LazyLinear(32))


def test_shared_submodule_is_installed_once(tmp_path):
    layer = nn.Linear(32, 32)
    model = nn.Sequential(layer, nn.ReLU(), layer).eval()
    result = quantize(model)
    inputs = torch.randn(2, 32)
    reference = copy.deepcopy(model)
    reference.load_state_dict(result.state.dequantize())
    with torch.inference_mode():
        torch.testing.assert_close(result.model(inputs), reference(inputs))
    result.save(tmp_path / "shared.fbq")
    restored = load_model(model, tmp_path / "shared.fbq")
    with torch.inference_mode():
        torch.testing.assert_close(result.model(inputs), restored.model(inputs), atol=0, rtol=0)


def test_noncontiguous_weights_are_supported():
    model = nn.Linear(32, 32, bias=False)
    model.weight = nn.Parameter(model.weight.T)
    result = quantize(model)
    x = torch.randn(2, 32)
    with torch.inference_mode():
        torch.testing.assert_close(
            result.model(x), F.linear(x, result.state.dequantize()["weight"])
        )
