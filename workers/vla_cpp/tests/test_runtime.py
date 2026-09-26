from pathlib import Path
import hashlib

import pytest

from policykit.config import load_config
from policykit.runtime import VlaCpp
from policykit.packed_bench import runtime_patch_sha256


def test_quantization_command_is_reproducible():
    config = load_config(Path(__file__).parents[1] / "configs/benchmark.v1.yaml")
    command = VlaCpp(config).quantize_command(Path("in.gguf"), Path("out.gguf"), {"type": "Q8_0", "vision": False})
    assert command[-6:] == ["--in", "in.gguf", "--out", "out.gguf", "--type", "Q8_0"]


def test_vision_quantization_is_explicit():
    config = load_config(Path(__file__).parents[1] / "configs/benchmark.v1.yaml")
    command = VlaCpp(config).quantize_command(Path("in.gguf"), Path("out.gguf"), config.preset("q4_0_vision"))
    assert command[-2:] == ["--vision-type", "Q8_0"]


@pytest.mark.parametrize('manifest', ['benchmark.v1.yaml', 'benchmark.docker.yaml'])
def test_manifest_and_packed_benchmark_use_the_packaged_patch(manifest):
    config = load_config(Path(__file__).parents[1] / 'configs' / manifest)
    patch_paths = config.data['runtime']['patches']
    assert len(patch_paths) == 1
    patch = config.root / patch_paths[0]
    assert patch.is_file()
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == runtime_patch_sha256()
