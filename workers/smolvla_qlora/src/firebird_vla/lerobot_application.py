"""Firebird application protocol for isolated, pinned native LeRobot trainers.

SmolVLA continues to use its dedicated PEFT worker. This bridge executes the
upstream trainers for other policy architectures and translates their actual
optimizer metrics and complete checkpoints into Firebird's existing protocol.
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from .application import publish
from .checkpoint import verify_bundle, write_json
from .local_dataset import bind_local_recipe, verify_local_snapshot
from .native_profiles import LEROBOT_REVISION, native_profile_for_recipe, validate_native_dataset
from .telemetry import emit
from .temporal import TEMPORAL_FIELDS, validate_temporal

SHARED_FIELDS = TEMPORAL_FIELDS | {
    "model_id",
    "model_revision",
    "checkpoint_subdirectory",
    "steps",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "warmup_steps",
    "camera_key",
    "camera_keys",
    "chunk_size",
    "validation_fraction",
    "eval_every",
    "eval_batches",
    "save_every",
    "log_every",
    "num_workers",
    "seed",
}


def resolve_recipe(job):
    parameters = job["parameters"]
    recipe = dict(parameters.get("training") or {})
    profile = native_profile_for_recipe(recipe)
    if profile is None:
        raise ValueError("No registered native training adapter for this model")
    if parameters.get("training_method") != "full":
        raise ValueError("Native LeRobot policies use the full training method")
    if set(recipe) - SHARED_FIELDS:
        raise ValueError(
            f"Unsupported native training fields: {sorted(set(recipe) - SHARED_FIELDS)}"
        )
    if recipe.get("model_revision") != profile["model_revision"]:
        raise ValueError("Native training must use the catalog's immutable model revision")
    if recipe.get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("Native training currently requires gradient_accumulation_steps=1")
    validate_temporal(recipe, profile["policy_type"])
    data = job["dataset"]
    if data.get("source") not in {"huggingface", "local"}:
        raise ValueError("Native training requires a pinned Hub dataset or verified local snapshot")
    if data.get("source") == "huggingface" and len(data.get("revision", "")) != 40:
        raise ValueError("Native training requires an immutable Hugging Face dataset")
    steps = recipe.setdefault("steps", 20000)
    for name, default in {
        "batch_size": 4,
        "save_every": max(1, math.ceil(steps / 5)),
        "eval_every": min(100, steps),
        "eval_batches": 20,
        "log_every": 10,
        "num_workers": 0,
        "seed": 42,
        "validation_fraction": 0.2,
    }.items():
        recipe.setdefault(name, default)
    cameras = recipe.get("camera_keys") or (
        [recipe["camera_key"]]
        if recipe.get("camera_key")
        else [
            key
            for key, feature in data["features"].items()
            if feature.get("dtype") in {"video", "image"}
        ]
    )
    if not cameras or len(cameras) != len(set(cameras)):
        raise ValueError("Select at least one unique dataset camera")
    if profile.get("required_cameras") and len(cameras) != profile["required_cameras"]:
        raise ValueError(f"{profile['label']} requires {profile['required_cameras']} camera(s)")
    if any(data["features"].get(key, {}).get("dtype") not in {"video", "image"} for key in cameras):
        raise ValueError("Selected camera is not a dataset image/video feature")
    validate_native_dataset(profile, data["features"], cameras)
    recipe.update(
        {
            "camera_keys": cameras,
            "method": "full",
            "training_backend": "lerobot",
            "upstream_revision": LEROBOT_REVISION,
            "policy_type": profile["policy_type"],
            "dataset_id": data["repo_id"],
            "dataset_revision": data["revision"],
        }
    )
    if data.get("source") == "local":
        bind_local_recipe(job, recipe)
    return recipe, profile


def cli_arguments(
    recipe, profile, *, output, dataset_root, model_root=None, features=None, resume=False
):
    """Build arguments from an operator-owned profile, never user executable strings."""
    values = {
        "dataset.repo_id": recipe["dataset_id"],
        "dataset.revision": recipe["dataset_revision"],
        "dataset.root": str(dataset_root),
        "dataset.video_backend": "pyav",
        "dataset.eval_split": recipe["validation_fraction"],
        "output_dir": str(output),
        "job_name": "firebird-" + profile["policy_type"],
        "policy.device": "cuda",
        "policy.push_to_hub": False,
        "steps": recipe["steps"],
        "batch_size": recipe["batch_size"],
        "save_freq": recipe["save_every"],
        "log_freq": recipe["log_every"],
        "eval_steps": recipe["eval_every"],
        "max_eval_samples": recipe["eval_batches"] * recipe["batch_size"],
        "env_eval_freq": 0,
        "num_workers": recipe["num_workers"],
        "seed": recipe["seed"],
        "wandb.enable": False,
        "cudnn_deterministic": True,
        "accelerator.gradient_accumulation.steps": 1,
    }
    initialization = profile["initialization"]
    if initialization == "pretrained":
        values["policy.path"] = str(model_root)
    else:
        values["policy.type"] = profile["policy_type"]
        if initialization != "scratch":
            values["policy." + initialization] = str(model_root)
    for key, value in profile["overrides"].items():
        values["policy." + key] = value
    # Processor feature names must match the explicitly selected dataset cameras.
    if features is not None:
        values["policy.input_features"] = {
            key: features[key]
            for key in ["observation.state", *recipe["camera_keys"]]
            if key in features
        }
    if "learning_rate" in recipe:
        values["policy.optimizer_lr"] = recipe["learning_rate"]
    if "weight_decay" in recipe:
        values["policy.optimizer_weight_decay"] = recipe["weight_decay"]
    # Resumes load the exact saved train_config; even historical ignored fields
    # must not be reinterpreted as new architecture overrides.
    if not resume:
        temporal = validate_temporal(recipe, profile["policy_type"])
        if temporal is not None:
            values["policy.chunk_size"] = temporal["prediction_horizon"]
            values["policy.n_action_steps"] = temporal["execution_horizon"]
        elif "chunk_size" in recipe:
            values["policy.chunk_size"] = recipe["chunk_size"]
            values["policy.n_action_steps"] = recipe["chunk_size"]
    return [
        "--" + key + "=" + (value if isinstance(value, str) else json.dumps(value))
        for key, value in values.items()
    ]


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    job = json.loads(request_path.read_text())
    output = Path(job["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "job_id": job["job_id"]}
    try:
        if job.get("schema_version") != 1 or job.get("operation") != "policy.finetune":
            raise ValueError("Native LeRobot worker requires the training application protocol")
        resume = Path(job["resume_checkpoint"]).resolve() if job.get("resume_checkpoint") else None
        if job.get("artifact"):
            artifact = Path(job["artifact"]["path"]).resolve()
            verify_bundle(artifact)
            resume = artifact / "checkpoint"
        if resume:
            verify_bundle(resume)
            recipe = json.loads((resume / "recipe.json").read_text())
            profile = native_profile_for_recipe(recipe)
            data = job["dataset"]
            identity = dict(recipe)
            if data.get("source") == "local":
                bind_local_recipe(job, identity)
            else:
                identity.update(dataset_id=data["repo_id"], dataset_revision=data["revision"])
            if (
                profile is None
                or recipe.get("method") != "full"
                or recipe.get("dataset_id") != identity["dataset_id"]
                or recipe.get("dataset_revision") != identity["dataset_revision"]
                or recipe.get("dataset_splits") != identity.get("dataset_splits")
                or recipe.get("upstream_revision") != LEROBOT_REVISION
            ):
                raise ValueError("Resume must preserve the saved worker, model and pinned dataset")
        else:
            recipe, profile = resolve_recipe(job)
        if recipe.get("dataset_source") == "local":
            verify_local_snapshot(job.get("dataset_snapshot"))
            write_json(output / "dataset-snapshot.json", job["dataset_snapshot"])
        recipe_path = output / "recipe.json"
        write_json(recipe_path, recipe)
        command = [
            sys.executable,
            "-m",
            "firebird_vla.lerobot_train",
            str(recipe_path),
            str(output),
        ]
        if resume:
            command += [str(resume)]
        subprocess.run(command, check=True)
        if recipe.get("dataset_source") == "local":
            verify_local_snapshot(job["dataset_snapshot"])
        latest = json.loads((output / "training/latest.json").read_text())
        checkpoint = output / "training" / latest["checkpoint"]
        emit(
            "verifying",
            "Reloading native policy in a fresh process",
            step=latest["step"],
            total_steps=recipe["steps"],
        )
        verification = output / "verification.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "firebird_vla.lerobot_verify",
                str(checkpoint),
                str(verification),
            ],
            check=True,
        )
        bundle = output / "bundle"
        bundle.mkdir()
        shutil.copytree(checkpoint, bundle / "checkpoint")
        shutil.copyfile(verification, bundle / "verification.json")
        for name in ["metrics.jsonl", "environment.json", "splits.json", "resolved-config.json"]:
            source = output / "training" / name
            if source.exists():
                shutil.copyfile(source, bundle / name)
        temporal = json.loads((checkpoint / "temporal-contract.json").read_text())
        if json.loads(verification.read_text()).get("temporal_contract") != temporal:
            raise ValueError("Fresh reload did not verify the checkpoint temporal contract")
        metadata = {
            "temporal_contract": temporal,
            "method": "full",
            "architecture": profile["policy_type"],
            "training_backend": "lerobot",
            "initialization": profile["initialization"],
            "task": "unverified",
            "action_dim": job["dataset"]["features"]["action"]["shape"][0],
            "dataset": job["dataset"],
            "base_model": {"repository": recipe["model_id"], "revision": recipe["model_revision"]},
            "camera_keys": recipe["camera_keys"],
            "reload_verified": True,
            "task_success": None,
        }
        result["artifact"] = publish(
            bundle, metadata, profile["label"] + " checkpoint", "training_checkpoint"
        )
        result["report"] = {
            "temporal_contract": temporal,
            "scope": "native_training_and_reload",
            "method": "full",
            "steps": latest["step"],
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
