"""Malformed but checksum-consistent artifacts must fail semantic validation."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from firebird_quant import Recipe, load, quantize
from firebird_quant.state import tensor_hash


def rewrite(path, output, change):
    with safe_open(path, framework="pt") as reader:
        manifest = json.loads(reader.metadata()["firebird_quant"])
        storage = {name: reader.get_tensor(name) for name in reader.keys()}
    change(manifest, storage)
    manifest["storage_hashes"] = {name: tensor_hash(value) for name, value in storage.items()}
    save_file(storage, output, metadata={"firebird_quant": json.dumps(manifest)})


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("position", [0, 1])
def test_reserved_signed_codes_are_rejected_even_with_updated_checksums(bits, position, tmp_path):
    original = tmp_path / "original.fbq"
    quantize(nn.Linear(32, 32), Recipe(bits=bits)).save(original)

    def corrupt(manifest, storage):
        entry = next(value for value in manifest["tensors"].values() if value["kind"] == "packed")
        key = entry["key"]
        qmax = (1 << (bits - 1)) - 1
        scales = storage[key + ".scales"]
        scales.fill_(torch.finfo(scales.dtype).max / qmax)
        while not torch.isfinite(scales * qmax).all():
            scales.copy_(torch.nextafter(scales, torch.zeros_like(scales)))
        codes = storage[key + ".codes"]
        if bits == 8:
            codes[position] = 128
        else:
            # INT4 zero nibble represents forbidden -8; exercise both nibble positions.
            codes[0] = 0x80 if position == 0 else 0x08

    corrupted = tmp_path / "corrupt.fbq"
    rewrite(original, corrupted, corrupt)
    with pytest.raises(ValueError, match="reserved"):
        load(corrupted)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_retained_tensor_rejected_even_with_updated_checksum(value, tmp_path):
    original = tmp_path / "original.fbq"
    quantize(nn.Linear(32, 32)).save(original)

    def corrupt(manifest, storage):
        storage[manifest["tensors"]["bias"]["key"]][0] = value

    corrupted = tmp_path / "corrupt.fbq"
    rewrite(original, corrupted, corrupt)
    with pytest.raises(ValueError, match="nonfinite"):
        load(corrupted)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_format_version_requires_an_exact_integer(version, tmp_path):
    original = tmp_path / "original.fbq"
    quantize(nn.Linear(32, 32)).save(original)
    corrupted = tmp_path / "corrupt.fbq"
    rewrite(original, corrupted, lambda manifest, _: manifest.update(version=version))
    with pytest.raises(ValueError, match="version"):
        load(corrupted)


def test_imported_audit_cannot_promote_quality_or_forge_observed_storage(tmp_path):
    original = tmp_path / "original.fbq"
    result = quantize(nn.Linear(32, 32))
    result.save(original)

    def corrupt(manifest, _):
        manifest["audit"].update(
            quality_verified=True,
            speedup_verified=True,
            source_tensor_bytes=1,
            stored_tensor_bytes=0,
            unique_elements=0,
            quantized_elements=0,
            gpu_memory_saved_bytes=12345,
            tensors=[{"name": "invented", "quantized": True, "source_sha256": "a" * 64}],
        )

    rewritten = tmp_path / "rewritten.fbq"
    rewrite(original, rewritten, corrupt)
    observed = load(rewritten).audit
    assert observed["quality_verified"] is False
    assert observed["speedup_verified"] is False
    assert observed["storage_checksums_verified"] is True
    assert observed["source_hashes_verified"] is False
    assert observed["recipe_verified"] is False
    assert "gpu_memory_saved_bytes" not in observed
    for key in (
        "source_tensor_bytes",
        "stored_tensor_bytes",
        "unique_elements",
        "quantized_elements",
    ):
        assert observed[key] == result.audit[key]
    assert {row["name"] for row in observed["tensors"]} == {"weight", "bias"}
    assert [row["name"] for row in observed["tensors"] if row["quantized"]] == ["weight"]
    assert all(row["source_sha256"] is None for row in observed["tensors"])
