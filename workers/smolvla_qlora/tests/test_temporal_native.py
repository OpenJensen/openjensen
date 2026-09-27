"""Opt-in genuine pinned ACT CPU behavior; generated observations, no model download."""

import os
import subprocess
import sys
from importlib.metadata import version

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FIREBIRD_TEST_NATIVE_TEMPORAL") != "1",
    reason="Requires existing pinned LeRobot 0.6.2 CPU environment and explicit opt-in",
)


def test_native_act_horizons_dataset_padding_loss_queue_and_reload(tmp_path):
    assert version("lerobot") == "0.6.2"
    import numpy as np
    import torch
    from firebird_vla.temporal import check_dataset_temporal, resolved_temporal
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from torch.utils.data import default_collate

    torch.set_num_threads(1)
    torch.manual_seed(42)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["j" + str(i) for i in range(6)],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {"dtype": "float32", "shape": (6,), "names": ["j" + str(i) for i in range(6)]},
    }
    data_root = tmp_path / "dataset"
    writer = LeRobotDataset.create(
        "fixture/temporal", fps=20, features=features, root=data_root, use_videos=False
    )
    for episode in range(2):
        for frame in range(5):
            writer.add_frame(
                {
                    "observation.state": np.full(6, frame / 10, dtype=np.float32),
                    "observation.images.front": np.full((32, 32, 3), frame * 10, dtype=np.uint8),
                    "action": np.full(6, episode * 100 + frame, dtype=np.float32),
                    "task": "Generated temporal software fixture",
                }
            )
        writer.save_episode()
    writer.finalize()
    cfg = ACTConfig(
        device="cpu",
        chunk_size=8,
        n_action_steps=3,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        dropout=0,
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=0,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    record = resolved_temporal(cfg, 20, "act", {"prediction_horizon": 8, "execution_horizon": 3})
    data = LeRobotDataset(
        "fixture/temporal",
        root=data_root,
        episodes=[0],
        delta_timestamps={"action": record["action_delta_timestamps"]},
    )
    check_dataset_temporal(record, data)
    last = data[4]
    assert last["action"].shape == (8, 6)
    assert last["action_is_pad"].tolist() == [False] + [True] * 7
    assert last["action"][:, 0].tolist() == [4.0] * 8  # Never crosses into episode 1.
    policy = ACTPolicy(cfg)
    assert tuple(policy.model.decoder_pos_embed.weight.shape) == (8, 32)
    batch = default_collate([last])
    original_actions = batch["action"].clone().requires_grad_(True)
    batch["action"] = original_actions
    policy.train()
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(original_actions.grad[:, 1:]) == 0
    assert torch.count_nonzero(original_actions.grad[:, :1]) > 0
    assert policy.model.decoder_pos_embed.weight.grad is not None
    changed = dict(batch, action=batch["action"].detach().clone())
    changed["action"][:, 1:] = 9999
    torch.testing.assert_close(policy(changed)[0], loss.detach(), rtol=0, atol=0)
    policy.eval()
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (1, 8, 6)
    assert torch.isfinite(chunk).all()
    calls = []
    original_predict = policy.predict_action_chunk

    def observed(b):
        calls.append(1)
        return original_predict(b)

    policy.predict_action_chunk = observed
    policy.reset()
    for index in range(7):
        torch.testing.assert_close(policy.select_action(batch), chunk[:, index % 3], rtol=0, atol=0)
    assert len(calls) == 3  # 3 execution steps, independent of 8 predicted steps.
    policy.predict_action_chunk = original_predict
    policy.reset()
    directory = tmp_path / "policy"
    policy.save_pretrained(directory)
    loaded = ACTPolicy.from_pretrained(directory, local_files_only=True)
    assert (loaded.config.chunk_size, loaded.config.n_action_steps) == (8, 3)
    torch.testing.assert_close(loaded.predict_action_chunk(batch), chunk, rtol=0, atol=0)
    # Fresh interpreter reload is independent of constructor objects and queue state.
    tensor_batch = {
        key: value.detach() for key, value in batch.items() if isinstance(value, torch.Tensor)
    }
    torch.save(tensor_batch, tmp_path / "batch.pt")
    code = """
import sys, torch
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
torch.set_num_threads(1)
root = Path(sys.argv[1])
p = ACTPolicy.from_pretrained(root / 'policy', local_files_only=True)
b = torch.load(root / 'batch.pt', weights_only=True)
torch.save(p.predict_action_chunk(b), root / 'fresh.pt')
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        check=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    torch.testing.assert_close(
        torch.load(tmp_path / "fresh.pt", weights_only=True), chunk, rtol=0, atol=0
    )
    wrong = ACTConfig.from_pretrained(directory, local_files_only=True)
    wrong.chunk_size = 7
    incompatible = ACTPolicy(wrong)
    with pytest.raises(RuntimeError, match="size mismatch"):
        incompatible.load_state_dict(policy.state_dict(), strict=True)
