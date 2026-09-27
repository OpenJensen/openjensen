"""Replay generated packing observations through the real isolated policy.run worker.

This uses no recorded/private data and applies no actions. Use an exclusive new
output directory and the existing pinned ACT CPU interpreter, with act_optimizer,
firebird_quant and isaac_sim source roots on PYTHONPATH. The worker owns its child,
its loopback HTTP server and its 120-second deadline.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.package.resolve()):
        parser.error("Output must be new and outside the packed package")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    import torch
    from firebird_quant.native_package import canonical, inspect_policy, read_json, sha
    from sim_worker.rollout.native_replay import run_job

    torch.set_num_threads(1)
    source = args.package / "policy"
    info = inspect_policy(source)
    config = read_json(source / "config.json")
    proof = read_json(args.package / "verification.json")
    args.output.mkdir()
    corpus = args.output / "observations"
    corpus.mkdir()
    camera = next(k for k in config["input_features"] if k.startswith("observation.images."))
    samples = []
    for index, fixture in enumerate(proof["packed"]):
        generator = torch.Generator().manual_seed(fixture["seed"])
        pixels = torch.randint(
            0, 256, fixture["image_shape"], generator=generator, dtype=torch.uint8
        )
        state = torch.randn(6, generator=generator)
        rgb = pixels.permute(1, 2, 0).contiguous().numpy().tobytes()
        name = f"frame-{index:06d}.rgb"
        (corpus / name).write_bytes(rgb)
        samples.append(
            {
                "episode_index": index,
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "task": "Generated temporal software fixture",
                "state": state.tolist(),
                "origin": "synthetic",
                "lineage_group": f"generated-{index}",
                "image": {
                    "file": name,
                    "width": pixels.shape[2],
                    "height": pixels.shape[1],
                    "sha256": sha(rgb),
                    "bytes": len(rgb),
                },
            }
        )
    doc = {
        "schema_version": 1,
        "format": "native-policy-observations-v1",
        "source": {
            "kind": "generated_fixture",
            "identity": "generated-packing-seeds-171-902",
            "manifest_sha256": sha((args.package / "verification.json").read_bytes()),
        },
        "camera_key": camera,
        "semantics": {
            "state_names": [f"coordinate-{i}" for i in range(6)],
            "action_names": [f"coordinate-{i}" for i in range(6)],
            "units": ["generated_fixture"] * 6,
            "compatibility": "generated_fixture",
        },
        "samples": samples,
    }
    (corpus / "manifest.json").write_bytes(canonical(doc))
    job = {
        "schema_version": 1,
        "job_id": "generated-temporal-replay",
        "operation": "policy.run",
        "source": {
            "path": str(source),
            "files": info["files"],
            "model_id": info["model_id"],
            "artifact_id": "generated-temporal-packed",
            "artifact_manifest_sha256": sha((args.package / "manifest.json").read_bytes()),
        },
        "observations": {"path": str(corpus), "manifest_sha256": sha(canonical(doc))},
        "output_dir": str(args.output / "operation"),
        "timeout_seconds": 120,
    }
    (args.output / "request.json").write_bytes(canonical(job))
    result = run_job(job)
    saved = read_json(Path(result["artifact"]["path"]) / "predictions.json")
    assert result["report"]["action_shape"] == [config["chunk_size"], 6]
    for row, expected in zip(saved["records"], proof["packed"], strict=True):
        assert row["actions"] == expected["postprocessed"]
    assert inspect_policy(source) == info
    (args.output / "result.json").write_bytes(canonical(result))
    print(json.dumps({"status": "passed", "action_shape": result["report"]["action_shape"]}))


if __name__ == "__main__":
    main()
