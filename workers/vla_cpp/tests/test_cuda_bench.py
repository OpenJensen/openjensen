import pytest

from policykit.cuda_bench import samples_from_log


def test_cuda_samples_preserve_measurement_order():
    text = 'vla: backend = CUDA (device 0)\nvla-bench: samples_ms=3.5,1.5,2.5\n'
    assert samples_from_log(text, 3) == [3.5, 1.5, 2.5]


@pytest.mark.parametrize('text', [
    'vla: backend = CPU\nvla-bench: samples_ms=1,2',
    'backend = CUDA\nfalling back to CPU\nvla-bench: samples_ms=1,2',
    'backend = CUDA\nvla-bench: samples_ms=1,nan',
    'backend = CUDA\nvla-bench: samples_ms=1,-2',
    'backend = CUDA\nvla-bench: samples_ms=1',
])
def test_invalid_cuda_measurements_cannot_pass(text):
    with pytest.raises(ValueError):
        samples_from_log(text, 2)
