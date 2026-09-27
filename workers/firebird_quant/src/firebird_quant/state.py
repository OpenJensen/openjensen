"""A model-independent tensor inventory, selection policy, and auditable export."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from math import prod
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .codec import DTYPES, PackedTensor, pack


@dataclass(frozen=True)
class Recipe:
    bits: int = 4
    group_size: int = 64
    min_elements: int = 128
    min_ndim: int = 2
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.bits) is not int or self.bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if type(self.group_size) is not int or self.group_size < 2 or self.group_size % 2:
            raise ValueError("group_size must be a positive even integer >= 2")
        for name in ("min_elements", "min_ndim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for patterns in (self.include, self.exclude):
            if isinstance(patterns, str) or not all(isinstance(p, str) and p for p in patterns):
                raise ValueError("include/exclude must be sequences of nonempty glob patterns")

    def retained_reason(self, name, tensor):
        if any(fnmatchcase(name, pattern) for pattern in self.exclude):
            return "excluded"
        if self.include and not any(fnmatchcase(name, pattern) for pattern in self.include):
            return "outside_include"
        if str(tensor.dtype) not in DTYPES:
            return "non_floating"
        if tensor.ndim < self.min_ndim:
            return "rank_below_minimum"
        if tensor.numel() < self.min_elements:
            return "size_below_minimum"
        return None


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps([str(tensor.dtype), list(tensor.shape)]).encode()
    return hashlib.sha256(
        header + tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _identity(tensor):
    # Empty tensors need identity-based keys: their storage pointer is always zero.
    if not tensor.numel():
        return (id(tensor),)
    return (
        tensor.device,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


@dataclass
class QuantizedState:
    tensors: dict[str, Tensor | PackedTensor]
    aliases: dict[str, str]
    audit: dict
    metadata: dict = field(default_factory=dict)

    def dequantize(self):
        result = OrderedDict(
            (name, value.dequantize() if isinstance(value, PackedTensor) else value.clone())
            for name, value in self.tensors.items()
        )
        result.update((name, result[target]) for name, target in self.aliases.items())
        result._metadata = self.metadata
        return result

    def save(self, path):
        """One safetensors file, including a versioned manifest; never pickle model code."""
        path = Path(path)
        storage, entries = {}, {}
        for index, (name, value) in enumerate(self.tensors.items()):
            key = f"t{index}"
            if isinstance(value, PackedTensor):
                value.validate()
                storage[key + ".codes"] = value.codes.detach().cpu().contiguous()
                storage[key + ".scales"] = value.scales.detach().cpu().contiguous()
                entries[name] = {
                    "key": key,
                    "kind": "packed",
                    "shape": list(value.shape),
                    "dtype": value.dtype,
                    "bits": value.bits,
                    "group_size": value.group_size,
                }
            else:
                storage[key] = value.detach().cpu().contiguous().clone()
                entries[name] = {"key": key, "kind": "retained"}
        manifest = {
            "format": "firebird-quant",
            "version": 1,
            "tensors": entries,
            "aliases": self.aliases,
            "audit": self.audit,
            "metadata": self.metadata,
            "storage_hashes": {name: tensor_hash(value) for name, value in storage.items()},
        }
        # Write a sibling temporary file, reload/verify it, then publish without clobbering.
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".firebird-quant-", dir=path.parent)
        os.close(fd)
        try:
            save_file(storage, temporary, metadata={"firebird_quant": json.dumps(manifest)})
            load(temporary)
            os.link(temporary, path)
        finally:
            os.unlink(temporary)
        return path


def quantize_state_dict(state, recipe=Recipe(), *, parameter_names=None):
    """Pack arbitrary tensor names/shapes. A module supplies parameter_names to protect buffers."""
    if not state:
        raise ValueError("The state dictionary is empty")
    groups = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise ValueError(
                "Expected a state dictionary of named tensors; custom extra state unsupported"
            )
        if value.layout != torch.strided or value.is_meta or value.is_quantized:
            raise ValueError(f"{name}: expected a dense, materialized, unquantized tensor")
        if type(value) not in (Tensor, torch.nn.Parameter):
            raise ValueError(f"{name}: tensor subclasses require an explicit adapter")
        if str(value.dtype) not in DTYPES and value.dtype not in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{name}: unsupported checkpoint dtype {value.dtype}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name}: nonfinite weights or buffers")
        groups.setdefault(_identity(value), []).append(name)
    tensors, aliases, rows = {}, {}, []
    for names in groups.values():
        name, *other_names = names
        source = state[name].detach().cpu().contiguous()
        reasons = [recipe.retained_reason(alias, source) for alias in names]
        if parameter_names is not None and any(alias not in parameter_names for alias in names):
            reasons.append("buffer")
        reason = next((reason for reason in reasons if reason), None)
        value = (
            source.clone()
            if reason
            else pack(source, bits=recipe.bits, group_size=recipe.group_size)
        )
        source_bytes = source.numel() * source.element_size()
        if isinstance(value, PackedTensor) and value.nbytes >= source_bytes:
            value, reason = source.clone(), "packing_would_not_reduce_storage"
        tensors[name] = value
        aliases.update((alias, name) for alias in other_names)
        packed = isinstance(value, PackedTensor)
        rows.append(
            {
                "name": name,
                "aliases": other_names,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "elements": source.numel(),
                "source_bytes": source_bytes,
                "stored_bytes": value.nbytes if packed else source_bytes,
                "source_sha256": tensor_hash(source),
                "quantized": packed,
                "retained_reason": reason,
            }
        )
    elements = sum(row["elements"] for row in rows)
    audit = {
        "recipe": asdict(recipe),
        "runtime": "torch-dequantize-on-access",
        "weight_format": f"firebird-symmetric-int{recipe.bits}-v1",
        "source_tensor_bytes": sum(row["source_bytes"] for row in rows),
        "stored_tensor_bytes": sum(row["stored_bytes"] for row in rows),
        "unique_elements": elements,
        "quantized_elements": sum(row["elements"] for row in rows if row["quantized"]),
        "tensors": rows,
        "quality_verified": False,
        "speedup_verified": False,
    }
    return QuantizedState(tensors, aliases, audit, dict(getattr(state, "_metadata", {})))


def _loaded_audit(declared, tensors, aliases):
    """Rebuild observable coverage; imported provenance cannot establish quality."""
    if not isinstance(declared, dict):
        raise ValueError("Invalid source audit metadata")
    declared_rows = declared.get("tensors", [])
    hashes = (
        {
            row["name"]: row.get("source_sha256")
            for row in declared_rows
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        if isinstance(declared_rows, list)
        else {}
    )
    rows, precisions = [], set()
    for name, value in tensors.items():
        packed = isinstance(value, PackedTensor)
        shape = tuple(value.shape)
        elements = prod(shape)
        dtype = DTYPES[value.dtype] if packed else value.dtype
        source_bytes = elements * torch.empty((), dtype=dtype).element_size()
        digest = hashes.get(name)
        if not (
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
        ):
            digest = None
        if packed:
            precisions.add(value.bits)
        rows.append(
            {
                "name": name,
                "aliases": [alias for alias, target in aliases.items() if target == name],
                "shape": list(shape),
                "dtype": str(dtype),
                "elements": elements,
                "source_bytes": source_bytes,
                "stored_bytes": value.nbytes if packed else source_bytes,
                "source_sha256": digest,
                "quantized": packed,
                "retained_reason": None if packed else "retained_in_artifact",
            }
        )
    precision = next(iter(precisions)) if len(precisions) == 1 else "mixed"
    return {
        "recipe": declared.get("recipe"),
        "recipe_verified": False,
        "runtime": "torch-dequantize-on-access",
        "weight_format": f"firebird-symmetric-int{precision}-v1" if precisions else "unquantized",
        "source_tensor_bytes": sum(row["source_bytes"] for row in rows),
        "stored_tensor_bytes": sum(row["stored_bytes"] for row in rows),
        "unique_elements": sum(row["elements"] for row in rows),
        "quantized_elements": sum(row["elements"] for row in rows if row["quantized"]),
        "tensors": rows,
        "quality_verified": False,
        "speedup_verified": False,
        "source_hashes_verified": False,
        "storage_checksums_verified": True,
    }


def load(path) -> QuantizedState:
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if "firebird_quant" not in metadata:
            raise ValueError("Not an OPEN JENSEN Quant artifact")
        manifest = json.loads(metadata["firebird_quant"])
        if (
            manifest.get("format") != "firebird-quant"
            or type(manifest.get("version")) is not int
            or manifest["version"] != 1
        ):
            raise ValueError("Unsupported OPEN JENSEN Quant format version")
        if set(reader.keys()) != set(manifest["storage_hashes"]):
            raise ValueError("Storage inventory mismatch")
        storage = {name: reader.get_tensor(name) for name in reader.keys()}
    for name, value in storage.items():
        if tensor_hash(value) != manifest["storage_hashes"][name]:
            raise ValueError(f"Storage checksum mismatch: {name}")
    tensors, used = {}, []
    for name, entry in manifest["tensors"].items():
        key = entry["key"]
        if entry["kind"] == "packed":
            value = PackedTensor(
                storage[key + ".codes"],
                storage[key + ".scales"],
                tuple(entry["shape"]),
                entry["dtype"],
                entry["bits"],
                entry["group_size"],
            )
            value.validate()
            used.extend((key + ".codes", key + ".scales"))
        elif entry["kind"] == "retained":
            value = storage[key]
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name}: nonfinite retained tensor")
            used.append(key)
        else:
            raise ValueError("Unknown tensor storage kind")
        tensors[name] = value
    if len(used) != len(set(used)) or set(used) != set(storage):
        raise ValueError("Manifest does not describe storage exactly once")
    aliases = manifest["aliases"]
    if set(aliases) & set(tensors) or any(target not in tensors for target in aliases.values()):
        raise ValueError("Invalid tied-tensor aliases")
    audit = _loaded_audit(manifest["audit"], tensors, aliases)
    return QuantizedState(tensors, aliases, audit, manifest["metadata"])
