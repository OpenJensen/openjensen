from dataclasses import replace
from pathlib import Path
import copy
import platform
import subprocess

import pytest

from policykit.config import load_config
from policykit.runtime import VlaCpp
from policykit.packed_bench import parse_actions


def test_action_metrics_exclude_padding_and_reject_nonfinite():
    values = [str(dim) for _ in range(50) for dim in range(32)]
    result = parse_actions('action_len=1600\n' + '\n'.join(values))
    assert len(result) == 350
    assert result[:7] == list(range(7))
    values[1599] = 'nan'
    with pytest.raises(ValueError, match='finite'):
        parse_actions('action_len=1600\n' + '\n'.join(values))


def test_patch_applies_to_pristine_source_and_is_idempotent(tmp_path):
    config = load_config(Path(__file__).parents[1]/'configs/benchmark.docker.yaml')
    vendor = config.root/'artifacts/docker/vendor/vla.cpp'
    if not vendor.exists():
        pytest.skip('Prepared vendor checkout required')
    checkout = tmp_path/'vendor'
    for relative in ['src/models/smolvla.cpp','src/serving/vla-bench.cpp']:
        path = checkout/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(subprocess.check_output(['git','-C',str(vendor),'show',f'HEAD:{relative}']))
    subprocess.run(['git','init',str(checkout)],capture_output=True,check=True)
    data = copy.deepcopy(config.data)
    data['runtime']['source_dir'] = str(checkout)
    runtime = VlaCpp(replace(config,data=data))
    runtime.apply_patches()
    expected = (checkout/'src/models/smolvla.cpp').read_bytes()
    runtime.apply_patches()
    assert (checkout/'src/models/smolvla.cpp').read_bytes() == expected
    assert b'gst.load_packed' in expected


@pytest.mark.parametrize('name,kind', [('aex.blk.0.attn_q.weight','Q8_0'),
                                    ('vlm.blk.0.attn_q.weight','Q4_1')])
def test_runtime_rejects_protected_or_unsupported_packing(tmp_path,name,kind):
    if platform.system() != 'Linux':
        pytest.skip('Linux runtime integration test')
    gguf = pytest.importorskip('gguf')
    np = pytest.importorskip('numpy')
    binary = Path('artifacts/docker/vendor/vla.cpp/build/vla-bench')
    if not binary.exists():
        pytest.skip('Compiled runtime required')
    model = tmp_path/'invalid.gguf'
    writer = gguf.GGUFWriter(model,'smolvla')
    dtype = getattr(gguf.GGMLQuantizationType,kind)
    writer.add_tensor(name,gguf.quants.quantize(np.ones((32,32),dtype=np.float32),dtype),raw_dtype=dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    result = subprocess.run([str(binary),'--ckpt',str(model)],capture_output=True,text=True,timeout=30)
    assert result.returncode != 0
    assert 'unsupported packed SmolVLA tensor' in result.stderr
