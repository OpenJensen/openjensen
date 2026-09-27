"""Generated native ACT/robotics-shaped fixtures; never claim recorded policy quality."""

from pathlib import Path

import pytest
from firebird_act.bundle import canonical, inventory
from firebird_distill.contracts import digest, teacher_info


@pytest.fixture(scope="session")
def teacher(tmp_path_factory):
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = ACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        dim_model=512,
        n_heads=4,
        dim_feedforward=1024,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        chunk_size=100,
        n_action_steps=100,
        use_vae=False,
        dropout=0.0,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    stats = {
        "observation.state": {"mean": torch.arange(6).float(), "std": torch.ones(6) * 2},
        "observation.images.front": {
            "mean": torch.ones(3, 1, 1) * 0.5,
            "std": torch.ones(3, 1, 1) * 0.25,
        },
        "action": {"mean": torch.arange(6).float(), "std": torch.ones(6) * 3},
    }
    root = tmp_path_factory.mktemp("distill-teacher")
    ACTPolicy(cfg).save_pretrained(root)
    pre, post = make_act_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(root, config_filename="policy_preprocessor.json")
    post.save_pretrained(root, config_filename="policy_postprocessor.json")
    return root.resolve()


@pytest.fixture
def job(teacher, tmp_path):
    import torch
    from safetensors.torch import save

    expected = inventory(teacher)
    cfg, camera, processors = teacher_info(teacher, expected)
    root = (tmp_path / "corpus").resolve()
    root.mkdir()
    generator = torch.Generator().manual_seed(31)
    doc = {
        "schema_version": 1,
        "format": "act-observation-corpus-v1",
        "source": {
            "kind": "generated_fixture",
            "identity": "unit-test-generated-scenes",
            "revision": "seed31",
            "inventory_sha256": "a" * 64,
        },
        "semantics": {
            "state_names": [f"joint{i}" for i in range(6)],
            "action_names": [f"joint{i}" for i in range(6)],
            "units": ["fixture_coordinate"] * 6,
            "compatibility": "generated_fixture",
            "teacher_processors_sha256": processors,
        },
        "camera": camera,
        "image_shape": [3, 32, 32],
        "chunk_size": 100,
        "samples": [],
    }
    for episode, split in enumerate(("train", "validation", "final")):
        for frame in (0, 2):
            tensors = {
                "image": torch.randint(0, 256, (3, 32, 32), generator=generator, dtype=torch.uint8),
                "state": torch.randn(6, generator=generator),
                "actions": torch.randn(100, 6, generator=generator),
                "padding": torch.arange(100) + frame >= 4,
            }
            raw = save(tensors)
            name = f"sample-{len(doc['samples']):06d}.safetensors"
            (root / name).write_bytes(raw)
            doc["samples"].append(
                {
                    "file": name,
                    "sha256": digest(raw),
                    "bytes": len(raw),
                    "episode_id": episode,
                    "lineage_group": f"scene-{episode}",
                    "frame_index": frame,
                    "episode_length": 4,
                    "split": split,
                }
            )
    (root / "manifest.json").write_bytes(canonical(doc))
    return {
        "schema_version": 1,
        "job_id": "generated-proof",
        "operation": "policy.distill",
        "teacher": {
            "path": str(teacher),
            "files": expected,
            "artifact_id": "generated-teacher",
            "artifact_manifest_sha256": "b" * 64,
        },
        "dataset": {"path": str(root), "manifest_sha256": digest(canonical(doc))},
        "recipe": {
            "adapter": "act-act-v1",
            "student": "act-256",
            "steps": 8,
            "learning_rate": 1e-4,
            "seed": 1729,
        },
        "output_dir": str((tmp_path / "operation").resolve()),
        "timeout_seconds": 120,
    }


def rebind_manifest(job, mutate):
    import json

    root = Path(job["dataset"]["path"])
    doc = json.loads((root / "manifest.json").read_bytes())
    mutate(doc)
    raw = canonical(doc)
    (root / "manifest.json").write_bytes(raw)
    job["dataset"]["manifest_sha256"] = digest(raw)
    return doc
