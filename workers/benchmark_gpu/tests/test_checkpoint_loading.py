"""Regression checks for real checkpoint loading failures found on the RTX host."""

import pytest


def test_smolvla_serialized_type_dispatch(tmp_path):
    pytest.importorskip("lerobot")
    from benchmark_model import load_smolvla_config
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    expected = SmolVLAConfig(device="cpu", chunk_size=20, n_action_steps=10)
    expected.save_pretrained(tmp_path)
    actual = load_smolvla_config(tmp_path)
    assert isinstance(actual, SmolVLAConfig)
    assert (actual.chunk_size, actual.n_action_steps) == (20, 10)


def test_checkpoint_dtype_preserved_and_incomplete_weights_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from benchmark_model import load_checkpoint_weights
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    source = torch.nn.Linear(4, 2).to(dtype=torch.bfloat16)
    save_file(source.state_dict(), path)
    loaded = load_checkpoint_weights(torch.nn.Linear(4, 2), path)
    assert loaded.weight.dtype == torch.bfloat16
    torch.testing.assert_close(loaded.weight, source.weight, rtol=0, atol=0)
    save_file({"weight": source.weight}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint_weights(torch.nn.Linear(4, 2), path)
    save_file({"weight": torch.zeros(3, 4), "bias": source.bias}, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint_weights(torch.nn.Linear(4, 2), path)
