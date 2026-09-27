"""Pinned Psi-Zero adapter contract, importable without its optional ML stack."""

import math
import re

PSI_REVISION = "4f3720d45e102b36d7c3e9465ab8062274170518"
PSI_MODEL_ID = "USC-PSI-Lab/psi-model"
PSI_MODEL_REVISION = "4c6f9776fc5b18d87945254175e38bb74b9d7748"
PSI_VLM_DIRECTORY = "psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
PSI_EXPERT_DIRECTORY = "psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"
PSI_FIELDS = {
    "model_id",
    "model_revision",
    "steps",
    "batch_size",
    "learning_rate",
    "gradient_accumulation_steps",
    "warmup_steps",
    "weight_decay",
    "max_grad_norm",
    "camera_key",
    "camera_keys",
    "validation_fraction",
    "eval_every",
    "eval_batches",
    "save_every",
    "log_every",
    "num_workers",
    "seed",
    "chunk_size",
}


def resolve_recipe(job):
    if job.get("schema_version") != 1 or job.get("operation") != "policy.finetune":
        raise ValueError("Psi-Zero requires the version 1 training application protocol")
    parameters = job["parameters"]
    if parameters.get("training_method") != "full":
        raise ValueError(
            "Psi-Zero trains the full action expert with its VLM frozen; LoRA/QLoRA are unsupported"
        )
    recipe = dict(parameters.get("training") or {})
    if set(recipe) - PSI_FIELDS:
        raise ValueError(f"Unsupported Psi-Zero settings: {sorted(set(recipe) - PSI_FIELDS)}")
    if (recipe.get("model_id"), recipe.get("model_revision")) != (PSI_MODEL_ID, PSI_MODEL_REVISION):
        raise ValueError("Psi-Zero requires the catalog's pinned model revision")
    data = job["dataset"]
    if data.get("source") != "huggingface" or not re.fullmatch(
        r"[a-f0-9]{40}", data.get("revision", "")
    ):
        raise ValueError("Psi-Zero requires an immutable Hugging Face dataset inspection")
    if data.get("format") not in {"lerobot_v2", "lerobot_v21"}:
        raise ValueError(
            "Psi-Zero's pinned upstream loader requires LeRobot v2.x; "
            "v3 datasets are not supported by this adapter"
        )
    features = data["features"]
    state_key = "observation.state" if "observation.state" in features else "states"
    for key in ("action", state_key):
        shape = features.get(key, {}).get("shape", [])
        if len(shape) != 1 or type(shape[0]) is not int or not 1 <= shape[0] <= 36:
            raise ValueError(f"Psi-Zero needs {key} vectors with 1..36 dimensions")
    cameras = recipe.get("camera_keys") or (
        [recipe["camera_key"]] if recipe.get("camera_key") else []
    )
    if len(cameras) != 1 or features.get(cameras[0], {}).get("dtype") not in {"image", "video"}:
        raise ValueError("Psi-Zero currently requires exactly one selected image/video camera")
    defaults = {
        "steps": 20000,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 50,
        "weight_decay": 1e-6,
        "max_grad_norm": 1.0,
        "validation_fraction": 0.2,
        "eval_every": 100,
        "eval_batches": 20,
        "log_every": 10,
        "num_workers": 0,
        "seed": 42,
        "chunk_size": 30,
    }
    for key, value in defaults.items():
        recipe.setdefault(key, value)
    recipe.setdefault("save_every", max(1, math.ceil(recipe["steps"] / 5)))
    for key in (
        "steps",
        "batch_size",
        "gradient_accumulation_steps",
        "eval_every",
        "eval_batches",
        "log_every",
        "save_every",
    ):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if recipe["chunk_size"] != 30:
        raise ValueError("The published Psi-Zero action expert requires chunk_size=30")
    if recipe["num_workers"] != 0:
        raise ValueError("The deterministic Psi-Zero adapter requires num_workers=0")
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] <= 2**32 - 1:
        raise ValueError("seed must be a 32-bit non-negative integer")
    if not 0 < recipe["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    for key in ("learning_rate", "weight_decay", "max_grad_norm"):
        if (
            not isinstance(recipe[key], (float, int))
            or not math.isfinite(recipe[key])
            or recipe[key] < 0
        ):
            raise ValueError(f"{key} must be finite and non-negative")
    if not recipe["learning_rate"] or not recipe["max_grad_norm"]:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if type(recipe["warmup_steps"]) is not int or not 0 <= recipe["warmup_steps"] < recipe["steps"]:
        raise ValueError("warmup_steps must be non-negative and smaller than steps")
    recipe.update(
        {
            "camera_keys": cameras,
            "state_key": state_key,
            "method": "full",
            "training_backend": "psi0",
            "policy_type": "psi0",
            "architecture": "psi0",
            "upstream_revision": PSI_REVISION,
            "dataset_id": data["repo_id"],
            "dataset_revision": data["revision"],
            "action_dim": features["action"]["shape"][0],
            "state_dim": features[state_key]["shape"][0],
        }
    )
    return recipe


def launch_config(recipe, *, dataset_root, model_root, stats_path, output):
    """Use the released action-expert architecture and explicit generic LeRobot mappings."""
    return {
        "exp": "firebird",
        "seed": recipe["seed"],
        "auto_tag_run": False,
        "log": {"report_to": None, "log_freq": recipe["log_every"]},
        "wandb": {},
        "train": {
            "name": "finetune",
            "output_dir": str(output),
            "lora": False,
            "mixed_precision": "bf16",
            "data_parallel": "ddp",
            "train_batch_size": recipe["batch_size"],
            "val_batch_size": recipe["batch_size"],
            "learning_rate": recipe["learning_rate"],
            "max_grad_norm": recipe["max_grad_norm"],
            "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
            "max_training_steps": recipe["steps"],
            "warmup_steps": recipe["warmup_steps"],
            "warmup_ratio": None,
            "lr_scheduler_type": "cosine",
            "lr_scheduler_kwargs": {"weight_decay": recipe["weight_decay"], "betas": [0.95, 0.999]},
        },
        "data": {
            "root_dir": str(dataset_root),
            "train_repo_ids": [recipe["dataset_id"]],
            "transform": {
                "repack": {
                    "image_keys": recipe["camera_keys"],
                    "state_key": recipe["state_key"],
                    "action_key": "action",
                    "pad_action_dim": 36,
                    "pad_state_dim": 36,
                    "action_chunk_size": 30,
                },
                "field": {
                    "stat_path": str(stats_path),
                    "stat_action_key": "action",
                    "stat_state_key": recipe["state_key"],
                    "action_norm_type": "bounds",
                    "use_norm_mask": False,
                    "normalize_state": True,
                    "pad_action_dim": 36,
                    "pad_state_dim": 36,
                },
                "model": {
                    "img_aug": False,
                    "resize": {"size": [240, 320]},
                    "center_crop": {"size": [240, 320]},
                },
            },
        },
        "model": {
            "model_name_or_path": str(model_root / PSI_VLM_DIRECTORY),
            "pretrained_action_header_path": str(model_root / PSI_EXPERT_DIRECTORY),
            "noise_scheduler": "flow",
            "train_diffusion_steps": 1000,
            "n_conditions": 0,
            "action_chunk_size": 30,
            "action_dim": 36,
            "action_exec_horizon": 30,
            "observation_horizon": 1,
            "odim": 36,
            "view_feature_dim": 2048,
            "tune_vlm": False,
            "use_film": False,
            "combined_temb": False,
            "rtc": True,
            "max_delay": 8,
        },
    }
