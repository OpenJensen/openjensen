from policykit.quantization import tensor_quantization
from policykit.benchmark import parse_bench


def test_real_upstream_benchmark_row():
    assert parse_bench('| smolvla | 2 | 512 | 16 | 90.0 | 100.0 | 110.0 | 20.0 |\n') == (100.0, 110.0)


def test_action_expert_and_projector_remain_float():
    for name in ['aex.blk.0.attn_q.weight', 'action_out_proj.weight', 'mm.fc.weight',
                 'token_embd.weight', 'vlm.blk.0.attn_norm.weight']:
        assert tensor_quantization(name, [960, 960], 'Q4_0', 'Q8_0') is None


def test_language_and_vision_have_independent_precision():
    assert tensor_quantization('vlm.blk.0.attn_q.weight', [960, 960], 'Q4_0', 'Q8_0') == 'Q4_0'
    assert tensor_quantization('vit.blk.0.attn_q.weight', [960, 960], 'Q4_0', 'Q8_0') == 'Q8_0'
    assert tensor_quantization('vit.blk.0.attn_q.weight', [960, 960], 'Q4_0', None) is None


def test_written_gguf_preserves_protected_tensors(tmp_path):
    import pytest
    import json
    from pathlib import Path
    gguf = pytest.importorskip('gguf')
    np = pytest.importorskip('numpy')
    from policykit.quantization import quantize
    vendor = Path('artifacts/docker/vendor/vla.cpp/scripts/quantize_gguf.py')
    if not vendor.exists():
        pytest.skip('Integration test requires prepared vendor source')
    source, destination = tmp_path / 'source.gguf', tmp_path / 'candidate.gguf'
    tensors = ['vlm.blk.0.attn_q.weight', 'vit.blk.0.attn_q.weight',
               'aex.blk.0.attn_q.weight', 'mm.fc.weight', 'token_embd.weight']
    writer = gguf.GGUFWriter(source, 'smolvla')
    values = np.arange(1024, dtype=np.float32).reshape(32, 32) / 1024
    for name in tensors:
        writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    audit = quantize(source, destination, vendor, 'Q4_0', 'Q8_0')
    assert audit['quantize_seconds'] > 0
    assert json.loads(destination.with_suffix('.audit.json').read_text())['quantize_seconds'] == audit['quantize_seconds']
    written = {t.name: t for t in gguf.GGUFReader(destination).tensors}
    assert written[tensors[0]].tensor_type.name == 'Q4_0'
    assert written[tensors[1]].tensor_type.name == 'Q8_0'
    for name in tensors[2:]:
        assert written[name].tensor_type.name == 'F32'
        np.testing.assert_array_equal(written[name].data, values)
