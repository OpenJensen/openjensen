"""Firebird worker protocol for the isolated Psi-Zero action-expert trainer."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .application import publish
from .checkpoint import verify_bundle, write_json
from .psi_profile import PSI_FIELDS, resolve_recipe
from .telemetry import emit


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    job = json.loads(request_path.read_text())
    output = Path(job["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "job_id": job["job_id"]}
    try:
        resume = Path(job["resume_checkpoint"]).resolve() if job.get("resume_checkpoint") else None
        if job.get("artifact"):
            if job["artifact"]["format"] != "training_checkpoint":
                raise ValueError("Psi-Zero resume requires a training checkpoint")
            bundle = Path(job["artifact"]["path"]).resolve()
            verify_bundle(bundle)
            resume = bundle / "checkpoint" if (bundle / "checkpoint").is_dir() else bundle
        if resume:
            verify_bundle(resume)
            previous = json.loads((resume / "recipe.json").read_text())
            supplied = job["parameters"].get("training") or {}
            if any(
                key not in PSI_FIELDS or value != previous.get(key)
                for key, value in supplied.items()
            ):
                raise ValueError("Resume cannot change the saved Psi-Zero recipe")
            job = {
                **job,
                "parameters": {
                    **job["parameters"],
                    "training": {
                        key: value for key, value in previous.items() if key in PSI_FIELDS
                    },
                },
            }
            recipe = resolve_recipe(job)
            for key in (
                "method",
                "training_backend",
                "model_id",
                "model_revision",
                "dataset_id",
                "dataset_revision",
                "camera_keys",
            ):
                if recipe.get(key) != previous.get(key):
                    raise ValueError(
                        "Resume must preserve the Psi-Zero model, dataset, method and cameras"
                    )
            recipe = previous
        else:
            recipe = resolve_recipe(job)
        recipe_path = output / "recipe.json"
        write_json(recipe_path, recipe)
        command = [
            sys.executable,
            "-m",
            "firebird_vla.psi_train",
            str(recipe_path),
            str(output / "training"),
        ]
        if resume:
            command.append(str(resume))
        subprocess.run(command, check=True)
        latest = json.loads((output / "training/latest.json").read_text())
        checkpoint = output / "training" / latest["checkpoint"]
        emit(
            "verifying",
            "Reloading the Psi-Zero action expert in a fresh process",
            step=latest["step"],
            total_steps=recipe["steps"],
        )
        verification = output / "verification.json"
        subprocess.run(
            [sys.executable, "-m", "firebird_vla.psi_verify", str(checkpoint), str(verification)],
            check=True,
        )
        bundle = output / "bundle"
        bundle.mkdir()
        shutil.copytree(checkpoint, bundle / "checkpoint")
        shutil.copyfile(verification, bundle / "verification.json")
        for name in ("metrics.jsonl", "environment.json", "splits.json"):
            shutil.copyfile(output / "training" / name, bundle / name)
        metadata = {
            "step": latest["step"],
            "method": "full",
            "architecture": "psi0",
            "training_backend": "psi0",
            "training_scope": "action_expert_with_frozen_vlm",
            "task": "unverified",
            "action_dim": recipe["action_dim"],
            "dataset": job["dataset"],
            "base_model": {"repository": recipe["model_id"], "revision": recipe["model_revision"]},
            "camera_keys": recipe["camera_keys"],
            "reload_verified": True,
            "task_success": None,
        }
        result["artifact"] = publish(
            bundle, metadata, "Psi-Zero action-expert checkpoint", "training_checkpoint"
        )
        result["report"] = {
            "scope": "native_training_and_reload",
            "method": "full",
            "steps": latest["step"],
            "training_scope": "action_expert_with_frozen_vlm",
            "task_success": None,
        }
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(result_path, result)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
