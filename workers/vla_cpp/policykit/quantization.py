"""Explicit component policy for vla.cpp SmolVLA/pi0 GGUF checkpoints."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path


def tensor_quantization(name: str, shape, language: str, vision: str | None):
    # Allowlist only backbone blocks. Projector, embeddings, and aex stay float.
    if len(shape) != 2 or not name.endswith('.weight') or int(shape[0]) % 32:
        return None
    if name.startswith('vlm.blk.') and 'norm' not in name:
        return language
    if name.startswith('vit.blk.') and not any(s in name for s in ('ln1', 'ln2', 'norm')):
        return vision
    return None


def validate_master(reader):
    import gguf

    # The pinned decoder and protected native tensor paths support F32/BF16.
    # Copying protected F16 tensors unchanged would publish unloadable weights.
    float_types = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16}
    if any(t.tensor_type not in float_types for t in reader.tensors):
        raise ValueError('Input must contain only F32/BF16 master tensors; '
                         'F16 and previously packed candidates are unsupported')


def quantize(source: Path, destination: Path, vendor: Path, language: str, vision: str | None):
    started = time.perf_counter()
    import gguf
    import numpy as np
    spec = importlib.util.spec_from_file_location('upstream_quantizer', vendor)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    reader = gguf.GGUFReader(source)
    validate_master(reader)
    writer = gguf.GGUFWriter(destination, reader.fields['general.architecture'].contents())
    for name, field in reader.fields.items():
        if name in {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count', 'general.architecture'}:
            continue
        if field.types[0] == gguf.GGUFValueType.ARRAY:
            writer.add_array(name, field.contents())
        else:
            writer.add_key_value(name, field.contents(), field.types[0])
    expected = {}
    for tensor in reader.tensors:
        kind = tensor_quantization(tensor.name, tensor.shape, language, vision)
        if kind:
            data = upstream.to_f32(tensor)
            if data is None:
                raise ValueError(f'Expected floating-point master: {tensor.name}')
            dtype = getattr(gguf.GGMLQuantizationType, kind)
            writer.add_tensor(tensor.name, gguf.quants.quantize(data, dtype), raw_dtype=dtype)
        else:
            dtype = tensor.tensor_type
            data = np.ascontiguousarray(tensor.data)
            if dtype == gguf.GGMLQuantizationType.BF16:
                data = data.view(np.uint16)
            writer.add_tensor(tensor.name, data, raw_dtype=dtype)
        expected[tensor.name] = dtype.name
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    actual = {t.name: t.tensor_type.name for t in gguf.GGUFReader(destination).tensors}
    if actual != expected:
        raise RuntimeError('Written tensor precision audit failed')
    audit = {'language': language, 'vision': vision, 'tensor_types': actual,
             'type_counts': dict(Counter(actual.values())),
             'quantize_seconds': time.perf_counter() - started,
             'timing_scope': 'in-process quantization including input read, packing, output write and precision audit; excludes audit JSON write'}
    destination.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vendor-script', type=Path, required=True)
    parser.add_argument('--type', choices=['Q8_0', 'Q4_0'], required=True)
    parser.add_argument('--vision-type', choices=['Q8_0', 'Q4_0'])
    args = parser.parse_args()
    quantize(args.source, args.out, args.vendor_script, args.type, args.vision_type)


if __name__ == '__main__':
    main()
