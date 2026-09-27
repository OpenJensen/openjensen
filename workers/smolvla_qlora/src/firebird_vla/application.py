"""Translate application jobs into registered native fine-tuning methods."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .accumulation import checkpoint_optimization
from .checkpoint import (
    resolve_checkpoint,
    sha256,
    verify_bundle,
    verify_training_checkpoint,
    write_json,
)
from .config import TrainConfig
from .telemetry import emit

METHODS = {"lora": "LoRA", "qlora": "QLoRA (NF4)"}


def publish(path, metadata, label, kind):
    write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "metadata": metadata,
            "files": {
                p.relative_to(path).as_posix(): sha256(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and p != path / "manifest.json"
            },
        },
    )
    return {"path": str(path), "label": label, "format": kind}


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    job = json.loads(request_path.read_text())
    result = {"schema_version": 1, "job_id": job["job_id"]}
    output = Path(job["output_dir"])
    try:
        if job.get("schema_version") != 1:
            raise ValueError("Unsupported application protocol")
        if job["operation"] == "policy.finetune":
            parameters = job["parameters"]
            method = parameters["training_method"]
            if method not in METHODS:
                raise ValueError("Training method is not registered")
            data = job["dataset"]
            if data["source"] != "huggingface" or len(data["revision"]) != 40:
                raise ValueError("Training needs an immutable Hub dataset intake")
            recipe = parameters.get("training") or {}
            if any(
                key in recipe for key in ("method", "output_dir", "dataset_id", "dataset_revision")
            ):
                raise ValueError("Method, output and dataset lineage are owned by the application")
            cfg = TrainConfig.from_dict(
                {
                    **recipe,
                    "method": method,
                    "dataset_id": data["repo_id"],
                    "dataset_revision": data["revision"],
                    "output_dir": str(output / "training"),
                }
            )
            resume = Path(job["resume_checkpoint"]) if job.get("resume_checkpoint") else None
            if job.get("artifact"):
                if job["artifact"]["format"] != "training_checkpoint":
                    raise ValueError("Resume requires a native training checkpoint")
                root = Path(job["artifact"]["path"])
                verify_bundle(root)
                resume = root / "checkpoint"
            if resume:
                resume = resolve_checkpoint(resume)
                previous = TrainConfig.load(resume / "recipe.json")
                if (previous.method, previous.dataset_id, previous.dataset_revision) != (
                    method,
                    data["repo_id"],
                    data["revision"],
                ):
                    raise ValueError("Resume must use the checkpoint's method and pinned dataset")
                if any(key in recipe for key in ("camera_key", "camera_keys")):
                    if cfg.selected_camera_keys != previous.selected_camera_keys:
                        raise ValueError("Resume must use the checkpoint's camera selection")
                from .temporal import TEMPORAL_FIELDS

                supplied_temporal = TEMPORAL_FIELDS.union({"chunk_size"}).intersection(recipe)
                if supplied_temporal and any(
                    cfg.temporal[key] != previous.temporal[key] for key in previous.temporal
                ):
                    raise ValueError("Resume must preserve the checkpoint temporal configuration")
                cfg = replace(previous, output_dir=str(output / "training"))
            from .model import require_runtime

            require_runtime()
            emit("preparing", "Checking available GPU memory")
            import torch

            free, total = torch.cuda.mem_get_info()
            minimum = 2 * 1024**3 if method == "qlora" else 4 * 1024**3
            if free < minimum:
                raise RuntimeError(
                    f"Insufficient free GPU memory for {method}: {free / 1024**3:.1f} GiB"
                )
            write_json(
                output / "resource-preflight.json",
                {
                    "free_bytes": free,
                    "total_bytes": total,
                    "method": method,
                    "minimum_free_bytes": minimum,
                    "estimate_only": True,
                },
            )
            recipe_path = output / "recipe.json"
            write_json(recipe_path, cfg.to_dict())
            command = [sys.executable, "-m", "firebird_vla.train", "--config", str(recipe_path)]
            if resume:
                command += ["--resume", str(resume)]
            subprocess.run(command, check=True)
            checkpoint = resolve_checkpoint(output / "training")
            final_step = verify_training_checkpoint(checkpoint)["step"]
            if final_step != cfg.steps:
                raise ValueError("Training returned without a complete final checkpoint")
            emit(
                "verifying",
                "Reloading checkpoint in a fresh process and checking action parity",
                step=final_step,
                total_steps=cfg.steps,
            )
            subprocess.run(
                [sys.executable, "-m", "firebird_vla.verify", str(checkpoint)], check=True
            )
            bundle = output / "bundle"
            bundle.mkdir()
            shutil.copytree(checkpoint, bundle / "checkpoint")
            shutil.copyfile(
                checkpoint.parent / (checkpoint.name + "-verification.json"),
                bundle / "verification.json",
            )
            shutil.copyfile(output / "training/metrics.jsonl", bundle / "training-metrics.jsonl")
            shutil.copyfile(output / "training/environment.json", bundle / "environment.json")
            last_metrics = {}
            with (bundle / "training-metrics.jsonl").open() as stream:
                for line in stream:
                    last_metrics = json.loads(line)
            temporal = json.loads((checkpoint / "temporal-contract.json").read_text())
            if (
                json.loads((bundle / "verification.json").read_text()).get("temporal_contract")
                != temporal
            ):
                raise ValueError("Fresh reload did not verify the checkpoint temporal contract")
            optimization = checkpoint_optimization(checkpoint, native=False)
            if (
                json.loads((bundle / "verification.json").read_text()).get("optimization")
                != optimization
            ):
                raise ValueError("Fresh reload optimization evidence differs from the checkpoint")
            metadata = {
                "optimization": optimization,
                "temporal_contract": temporal,
                "method": method,
                "architecture": "smolvla",
                "task": "unverified",
                "action_dim": data["features"]["action"]["shape"][0],
                "dataset": data,
                "base_model": {"repository": cfg.model_id, "revision": cfg.model_revision},
                "camera_keys": list(cfg.selected_camera_keys),
                "reload_verified": True,
                "task_success": None,
            }
            result["artifact"] = publish(
                bundle, metadata, METHODS[method] + " checkpoint", "training_checkpoint"
            )
            result["report"] = {
                "optimization": optimization,
                "temporal_contract": temporal,
                "scope": "native_training_and_reload",
                "method": method,
                "steps": cfg.steps,
                "final_optimizer_metrics": last_metrics,
                "task_success": None,
            }
            emit(
                "verifying",
                "Checkpoint reload verified; publishing training artifacts",
                step=cfg.steps,
                total_steps=cfg.steps,
                reload_verified=True,
            )
        elif job["operation"] == "policy.export":
            from .export import export_checkpoint

            artifact = job["artifact"]
            if artifact["format"] != "training_checkpoint":
                raise ValueError("Export requires a native training checkpoint")
            source = Path(artifact["path"])
            manifest = verify_bundle(source)
            bundle = output / "bundle"
            bundle.mkdir()
            export_checkpoint(source / "checkpoint", bundle / "policy")
            result["artifact"] = publish(
                bundle, manifest["metadata"], "Native floating checkpoint", "native_checkpoint"
            )
            result["report"] = {"scope": "native_export", "task_success": None}
        else:
            raise ValueError("Unsupported training operation")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    temporary = result_path.with_suffix(".tmp")
    write_json(temporary, result)
    temporary.replace(result_path)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
