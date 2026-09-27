"""Bounded generated ACT training/resume fixture; invoked in isolated CPU processes."""

import hashlib
import json
import sys
from pathlib import Path


def digest_state(value):
    import torch

    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(k): digest_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [digest_state(v) for v in value]
    return value


def fixture(root):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint{i}" for i in range(6)],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {"dtype": "float32", "shape": (6,), "names": [f"joint{i}" for i in range(6)]},
    }
    writer = LeRobotDataset.create(
        "fixture/accumulation", fps=20, features=features, root=root, use_videos=False
    )
    for episode in range(2):
        for frame in range(5):
            writer.add_frame(
                {
                    "observation.state": np.arange(6, dtype=np.float32) / 10 + frame / 10,
                    "observation.images.front": np.full(
                        (32, 32, 3), 20 + frame * 15 + episode, dtype=np.uint8
                    ),
                    "action": np.arange(6, dtype=np.float32) / 20 + episode + frame / 10,
                    "task": "Generated accumulation software fixture",
                }
            )
        writer.save_episode()
    writer.finalize()


def main():
    import time
    from importlib.metadata import version

    import torch
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_train as trainer

    from firebird_vla.accumulation import checkpoint_optimization
    from firebird_vla.checkpoint import sha256
    from firebird_vla.lerobot_train import install_training_hooks
    from firebird_vla.native_profiles import NATIVE_PROFILES

    torch.set_num_threads(1)
    root, mode = Path(sys.argv[1]), sys.argv[2]
    saved_checkpoint = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    assert mode in {"baseline", "interrupted", "resumed"}
    assert version("lerobot") == "0.6.2"
    root.mkdir(exist_ok=True)
    dataset_root = root / "dataset"
    if not dataset_root.exists():
        fixture(dataset_root)
    # Any attempted Hub fallback must fail; all fixture inputs are local.
    import huggingface_hub

    def no_network(*args, **kwargs):
        raise AssertionError("Generated native proof must not download inputs")

    huggingface_hub.snapshot_download = no_network
    dataset = DatasetConfig(
        repo_id="fixture/accumulation",
        root=dataset_root,
        revision="a" * 40,
        eval_split=0.5,
        video_backend="pyav",
    )
    dataset.image_transforms.enable = False
    config = ACTConfig(
        device="cpu",
        push_to_hub=False,
        chunk_size=8,
        n_action_steps=3,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        use_vae=True,
        dropout=0.1,
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=0,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    cfg = TrainPipelineConfig(
        dataset=dataset,
        policy=config,
        output_dir=root / mode / "native-output",
        job_name="accumulation-proof",
        steps=6,
        batch_size=3,
        save_freq=3,
        log_freq=3,
        eval_steps=3,
        max_eval_samples=3,
        env_eval_freq=0,
        num_workers=0,
        seed=42,
        cudnn_deterministic=True,
        save_checkpoint=mode != "baseline",
    )
    cfg.accelerator.gradient_accumulation.steps = 3
    profile = NATIVE_PROFILES["act"]
    recipe = {
        "steps": 2,
        "batch_size": 3,
        "gradient_accumulation_steps": 3,
        "save_every": 1,
        "eval_every": 1,
        "log_every": 1,
        "warmup_steps": 0,
        "learning_rate": 0.0001,
        "max_grad_norm": 0.5,
        "seed": 42,
        "policy_type": "act",
        "camera_keys": ["observation.images.front"],
        "prediction_horizon": 8,
        "execution_horizon": 3,
        "model_id": profile["model_id"],
        "model_revision": profile["model_revision"],
    }
    checkpoint = saved_checkpoint or root / "interrupted/training/checkpoint-000001"
    resume = checkpoint if mode == "resumed" else None
    before = (
        {
            p.relative_to(checkpoint).as_posix(): sha256(p)
            for p in checkpoint.rglob("*")
            if p.is_file()
        }
        if resume
        else None
    )
    if resume:
        cfg = TrainPipelineConfig.from_pretrained(checkpoint / "pretrained_model")
        cfg.output_dir = root / mode / "native-output"
        cfg.resume = True
        cfg.policy.device = "cpu"
        sys.argv = [
            "native-fixture",
            "--config_path=" + str(checkpoint / "pretrained_model/train_config.json"),
        ]
    else:
        sys.argv = ["native-fixture"]
    training = root / mode / "training"
    training.mkdir(parents=True)
    state = install_training_hooks(
        trainer,
        recipe,
        profile,
        training,
        resume=resume,
        probe_devices=[],
        runtime_environment={
            "fixture": "generated CPU ACT",
            "torch": version("torch"),
            "lerobot": version("lerobot"),
            "accelerate": version("accelerate"),
        },
    )
    original_update = trainer.update_policy
    captured = {}
    model_inputs = root / "probe-input.pt"

    class FixtureInterrupted(Exception):
        pass

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        captured.update(
            policy=kwargs["accelerator"].unwrap_model(args[1]),
            optimizer=args[3],
            scheduler=kwargs["lr_scheduler"],
        )
        if not model_inputs.exists():
            torch.save(
                {k: v[:1].clone() for k, v in args[2].items() if isinstance(v, torch.Tensor)},
                model_inputs,
            )
        if mode == "interrupted" and state["accumulator"].consumed == 4:
            raise FixtureInterrupted("Interrupted after one complete and one partial window")
        return result

    trainer.update_policy = update
    started = time.monotonic()
    interrupted = False
    try:
        trainer.train.__wrapped__(cfg)
    except FixtureInterrupted:
        interrupted = True
    assert interrupted == (mode == "interrupted")
    p = captured["policy"]
    p.eval()
    p.reset()
    with torch.inference_mode():
        chunk = p.predict_action_chunk(torch.load(model_inputs, weights_only=True))
    summary = {
        "schema_version": 1,
        "mode": mode,
        "elapsed_seconds": time.monotonic() - started,
        "source_fixture_kind": "generated",
        "runtime": {k: version(k) for k in ("torch", "lerobot", "accelerate")},
        "updates": state["step"],
        "consumed_microbatches": state["accumulator"].consumed,
        "model": digest_state(p.state_dict()),
        "optimizer": digest_state(captured["optimizer"].state_dict()),
        "scheduler": digest_state(captured["scheduler"].state_dict())
        if captured["scheduler"]
        else None,
        "prediction": digest_state(chunk),
        "shape": list(chunk.shape),
        "interrupted_partial_window": interrupted,
        "task_success": None,
        "cuda_verified": False,
    }
    if mode == "interrupted":
        assert {d.name for d in training.glob("checkpoint-*")} == {"checkpoint-000001"}
        summary["checkpoint"] = checkpoint_optimization(checkpoint, native=True)
    if resume:
        assert before == {
            p.relative_to(checkpoint).as_posix(): sha256(p)
            for p in checkpoint.rglob("*")
            if p.is_file()
        }
        summary["source_unchanged"] = True
        summary["checkpoint"] = checkpoint_optimization(training / "checkpoint-000002", native=True)
    (root / (mode + ".json")).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    # Hardlinks in working/published checkpoints count only once against physical fixture storage.
    inodes = {}
    for pth in root.rglob("*"):
        if pth.is_file():
            st = pth.stat()
            inodes[(st.st_dev, st.st_ino)] = st.st_size
    physical = sum(inodes.values())
    assert physical <= 400 * 1024**2, f"Fixture cap exceeded: {physical}"
    print(
        json.dumps(
            {
                "mode": mode,
                "updates": state["step"],
                "physical_fixture_bytes": physical,
                "logical_bytes": total,
            }
        )
    )


if __name__ == "__main__":
    main()
