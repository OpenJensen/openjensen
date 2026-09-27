"""Optional CPU fixture for the pinned native producer; not the export runtime.

Run in a separate producer overlay with LeRobot0.6.2 at e595b790 and
Accelerate1.14.0. Modes run in separate processes, in order: generate, resume,
bundle. The final mode needs the repository's smolvla_qlora/src on PYTHONPATH.
Never install producer dependencies into the pinned ACT0.6.1 consumer environment.
Only synthetic seeded tensors are used; no datasets, weights or network are fetched.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")


def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network forbidden during synthetic fixture proof")


sys.addaudithook(audit)
import torch
from accelerate import Accelerator
from lerobot.common.train_utils import save_checkpoint, resume_before_prepare, resume_after_prepare
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

PIN = "e595b7902714ba51f91e47523f66f89c5181b649"
EXPECTED = {
    "lerobot": "0.6.2",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "safetensors": "0.8.0",
    "accelerate": "1.14.0",
}
versions = {name: importlib.metadata.version(name) for name in EXPECTED}
assert {name: value.split("+")[0] for name, value in versions.items()} == EXPECTED, versions
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def semantic(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        assert torch.isfinite(data).all()
        return {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {
            str(k): semantic(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [semantic(v) for v in value]
    return value


def state_sha(value):
    return hashlib.sha256(canonical(semantic(value))).hexdigest()


def metadata():
    return {
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "network_disabled": True,
        "upstream_revision": PIN,
        "fixture_scope": "synthetic batches and two optimizer updates; no dataset, trained quality, GPU or task success",
    }


def update(model, optimizer, batch, accelerator):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    accelerator.backward(loss)
    optimizer.step()
    return loss.item()


def probe(model, batch):
    model.eval()
    with torch.inference_mode():
        action = model.predict_action_chunk(batch)
    assert tuple(action.shape) == (1, 100, 6) and torch.isfinite(action).all()
    return action


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["generate", "resume", "bundle"])
parser.add_argument("root", type=Path)
args = parser.parse_args()
root = args.root.resolve()
checkpoint = root / "upstream-checkpoint"
if args.mode == "generate":
    root.mkdir()
    torch.manual_seed(11)
    config = ACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        chunk_size=100,
        n_action_steps=100,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/synthetic-act", revision="a" * 40),
        policy=config,
        batch_size=1,
        steps=2,
        num_workers=0,
        seed=11,
    )
    cfg.accelerator.mixed_precision = "no"
    cfg.optimizer = config.get_optimizer_preset()
    cfg.scheduler = config.get_scheduler_preset()
    policy = ACTPolicy(config).cpu()
    optimizer = cfg.optimizer.build(policy.get_optim_params())
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    policy, optimizer = accelerator.prepare(policy, optimizer)
    stats = {
        "observation.state": {"mean": torch.arange(6).float(), "std": torch.ones(6) * 2},
        "observation.images.front": {
            "mean": torch.ones(3, 1, 1) * 0.5,
            "std": torch.ones(3, 1, 1) * 0.25,
        },
        "action": {"mean": torch.arange(6).float(), "std": torch.ones(6) * 3},
    }
    pre, post = make_act_pre_post_processors(config, dataset_stats=stats)
    generator = torch.Generator().manual_seed(917)
    batch = {
        "observation.state": torch.randn(1, 6, generator=generator),
        "observation.images.front": torch.rand(1, 3, 32, 32, generator=generator),
        "action": torch.randn(1, 100, 6, generator=generator),
        "action_is_pad": torch.zeros(1, 100, dtype=torch.bool),
    }
    first_loss = update(policy, optimizer, batch, accelerator)
    action = probe(policy, batch)
    save_checkpoint(
        checkpoint,
        1,
        cfg,
        policy,
        optimizer,
        preprocessor=pre,
        postprocessor=post,
        accelerator=accelerator,
    )
    torch.save(batch, checkpoint / "probe-batch.pt")
    torch.save(action, checkpoint / "probe-action.pt")
    before = inventory(checkpoint)
    saved_optimizer = state_sha(optimizer.state_dict())
    saved_model = state_sha(policy.state_dict())
    rng_draw = torch.rand(8).tolist()
    second_loss = update(policy, optimizer, batch, accelerator)
    result = metadata() | {
        "schema_version": 1,
        "step": 1,
        "first_loss": first_loss,
        "second_loss": second_loss,
        "saved_optimizer_sha256": saved_optimizer,
        "saved_model_sha256": saved_model,
        "rng_draw": rng_draw,
        "next_optimizer_sha256": state_sha(optimizer.state_dict()),
        "next_model_sha256": state_sha(policy.state_dict()),
        "checkpoint_files": before,
    }
    assert inventory(checkpoint) == before
    (root / "generated.json").write_bytes(canonical(result))
elif args.mode == "resume":
    expected = json.loads((root / "generated.json").read_text())
    assert inventory(checkpoint) == expected["checkpoint_files"]
    cfg = TrainPipelineConfig.from_pretrained(
        checkpoint / "pretrained_model", local_files_only=True
    )
    cfg.checkpoint_path = checkpoint
    cfg.resume = True
    config = PreTrainedConfig.from_pretrained(
        checkpoint / "pretrained_model", local_files_only=True
    )
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(
        checkpoint / "pretrained_model", config=config, strict=True, local_files_only=True
    ).cpu()
    optimizer = cfg.optimizer.build(policy.get_optim_params())
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    step = resume_before_prepare(cfg)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    resume_after_prepare(cfg, accelerator, policy, optimizer, None)
    assert step == 1
    assert state_sha(policy.state_dict()) == expected["saved_model_sha256"]
    assert state_sha(optimizer.state_dict()) == expected["saved_optimizer_sha256"]
    assert torch.rand(8).tolist() == expected["rng_draw"]
    batch = torch.load(checkpoint / "probe-batch.pt", map_location="cpu", weights_only=True)
    action = probe(policy, batch)
    saved_action = torch.load(checkpoint / "probe-action.pt", map_location="cpu", weights_only=True)
    assert torch.equal(action, saved_action)
    second_loss = update(policy, optimizer, batch, accelerator)
    assert second_loss == expected["second_loss"], (second_loss, expected["second_loss"])
    assert state_sha(policy.state_dict()) == expected["next_model_sha256"]
    assert state_sha(optimizer.state_dict()) == expected["next_optimizer_sha256"]
    assert inventory(checkpoint) == expected["checkpoint_files"]
    result = metadata() | {
        "schema_version": 1,
        "step": step,
        "reload_verified": True,
        "max_abs_action_difference": 0.0,
        "rng_resume_exact": True,
        "optimizer_resume_exact": True,
        "next_update_exact": True,
        "next_loss": second_loss,
        "next_model_sha256": expected["next_model_sha256"],
        "next_optimizer_sha256": expected["next_optimizer_sha256"],
        "checkpoint_files": expected["checkpoint_files"],
    }
    (root / "resumed.json").write_bytes(canonical(result))
else:
    from firebird_vla.lerobot_train import commit_checkpoint
    from firebird_vla.application import publish

    proof = json.loads((root / "resumed.json").read_text())
    assert proof["reload_verified"] and proof["next_update_exact"]
    assert inventory(checkpoint) == proof["checkpoint_files"]
    bundle = root / "training-bundle"
    bundle.mkdir()
    recipe = {
        "policy_type": "act",
        "training_backend": "lerobot",
        "method": "full",
        "upstream_revision": PIN,
        "model_revision": PIN,
        "model_id": "code://lerobot/act",
        "steps": 2,
        "chunk_size": 100,
        "camera_keys": ["observation.images.front"],
        "dataset_id": "fixture/synthetic-act",
        "dataset_revision": "a" * 40,
    }
    commit_checkpoint(checkpoint, bundle / "checkpoint", recipe, step=1)
    (bundle / "latest.json").unlink()
    (bundle / "verification.json").write_bytes(canonical(proof))
    publish(
        bundle,
        {
            "architecture": "act",
            "training_backend": "lerobot",
            "method": "full",
            "action_dim": 6,
            "camera_keys": recipe["camera_keys"],
            "reload_verified": True,
            "task": "unverified",
            "task_success": None,
            "base_model": {"repository": recipe["model_id"], "revision": PIN},
            "dataset": {
                "source": "huggingface",
                "repo_id": recipe["dataset_id"],
                "revision": recipe["dataset_revision"],
            },
            "fixture_scope": proof["fixture_scope"],
        },
        "Synthetic CPU ACT native checkpoint",
        "training_checkpoint",
    )
    result = {
        "schema_version": 1,
        "bundle": str(bundle),
        "manifest_sha256": sha(bundle / "manifest.json"),
        "files": inventory(bundle),
        "scope": proof["fixture_scope"],
    }
    (root / "bundle.json").write_bytes(canonical(result))
print(json.dumps({"mode": args.mode, "status": "passed", "root": str(root), "versions": versions}))
