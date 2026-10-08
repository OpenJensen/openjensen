"""Operator-owned native LeRobot profiles; never accepts executable code from requests.

Upstream policy implementation is pinned independently of model weights. Scratch
baselines use an explicit code:// source rather than pretending to fine-tune a
nonexistent model checkpoint. Keep worker/core copies byte-identical.
"""

LEROBOT_REVISION = "e595b7902714ba51f91e47523f66f89c5181b649"
LEROBOT_REQUIREMENT = (
    "lerobot[{extras}] @ git+https://github.com/huggingface/lerobot.git@" + LEROBOT_REVISION
)
NATIVE_PROFILES = {
    "act": {
        "label": "ACT",
        "description": "Action Chunking Transformer baseline",
        "policy_type": "act",
        "extra": "training",
        "initialization": "scratch",
        "model_id": "code://lerobot/act",
        "model_revision": LEROBOT_REVISION,
        "overrides": {},
        "minimum_gpu_memory_gb": 16,
    },
    "diffusion": {
        "label": "Diffusion Policy",
        "description": "Visual diffusion imitation policy",
        "policy_type": "diffusion",
        "extra": "training,diffusion",
        "initialization": "scratch",
        "model_id": "code://lerobot/diffusion",
        "model_revision": LEROBOT_REVISION,
        "overrides": {},
        "minimum_gpu_memory_gb": 16,
    },
    "eo1": {
        "label": "EO-1",
        "description": "Qwen vision-language-action flow policy",
        "policy_type": "eo1",
        "extra": "training,eo1",
        "initialization": "pretrained",
        "model_id": "lerobot/eo1-base",
        "model_revision": "bba306bb0095693f1b8f318450efed3ee7466590",
        "overrides": {
            "attn_implementation": "sdpa",
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        "minimum_gpu_memory_gb": 40,
    },
    "evo1": {
        "label": "EVO-1",
        "description": "InternVL policy with a continuous action head",
        "policy_type": "evo1",
        "extra": "training,evo1",
        "initialization": "pretrained",
        "model_id": "zuoxingdong/evo1_libero",
        "model_revision": "515921f4a2c1d3f3ad523721eafa26fdf2af315b",
        "overrides": {"use_flash_attn": False, "training_stage": "stage1"},
        "minimum_gpu_memory_gb": 24,
    },
    "gr00t_n17": {
        "label": "GR00T N1.7",
        "description": "NVIDIA robot foundation model",
        "policy_type": "groot",
        "extra": "training,groot",
        "initialization": "base_model_path",
        "model_id": "nvidia/GR00T-N1.7-LIBERO",
        "model_revision": "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
        "checkpoint_subdirectory": "libero_spatial",
        "overrides": {"use_flash_attention": False},
        "minimum_gpu_memory_gb": 40,
    },
    "multi_task_dit": {
        "label": "Multi-Task DiT",
        "description": "Text-conditioned diffusion transformer baseline",
        "policy_type": "multi_task_dit",
        "extra": "training,multi_task_dit",
        "initialization": "scratch",
        "model_id": "code://lerobot/multi_task_dit",
        "model_revision": LEROBOT_REVISION,
        "overrides": {},
        "minimum_gpu_memory_gb": 24,
    },
    "pi0": {
        "label": "π₀",
        "description": "Flow-matching robot policy",
        "policy_type": "pi0",
        "extra": "training,pi",
        "initialization": "pretrained",
        "model_id": "lerobot/pi0_libero_finetuned_v044",
        "model_revision": "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
        "overrides": {
            "train_expert_only": True,
            "freeze_vision_encoder": True,
            "gradient_checkpointing": True,
            "dtype": "bfloat16",
        },
        "minimum_gpu_memory_gb": 40,
    },
    "pi0_fast": {
        "label": "π₀-FAST",
        "description": "Autoregressive policy with FAST action tokens",
        "policy_type": "pi0_fast",
        "extra": "training,pi",
        "initialization": "pretrained",
        "model_id": "lerobot/pi0fast-base",
        "model_revision": "acbfee34e383700f64ede42e3ac7e89027dd43c2",
        "overrides": {"gradient_checkpointing": True, "dtype": "bfloat16"},
        "minimum_gpu_memory_gb": 40,
    },
    "pi05": {
        "label": "π₀.₅",
        "description": "Generalist flow-matching robot policy",
        "policy_type": "pi05",
        "extra": "training,pi",
        "initialization": "pretrained",
        "model_id": "lerobot/pi05_base",
        "model_revision": "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
        "overrides": {
            "train_expert_only": True,
            "freeze_vision_encoder": True,
            "gradient_checkpointing": True,
            "dtype": "bfloat16",
        },
        "minimum_gpu_memory_gb": 40,
    },
    "vla_jepa": {
        "label": "VLA-JEPA",
        "description": "Action policy with predictive world-model supervision",
        "policy_type": "vla_jepa",
        "extra": "training,vla_jepa",
        "initialization": "pretrained",
        "model_id": "lerobot/VLA-JEPA-Pretrain",
        "model_revision": "e946c3e5b538d760f4b4ff239d1b1c12090c041d",
        "overrides": {
            "resize_images_to": [224, 224],
            "pre_snap_gripper_action": False,
            "binarize_gripper_action": False,
            "reinit_modules": [
                "model.action_model.action_encoder",
                "model.action_model.action_decoder",
                "model.action_model.state_encoder",
            ],
        },
        "minimum_gpu_memory_gb": 40,
        "required_cameras": 2,
    },
    "vqbet": {
        "label": "VQ-BeT",
        "description": "Vector-quantized behavior transformer baseline",
        "policy_type": "vqbet",
        "extra": "training",
        "initialization": "scratch",
        "model_id": "code://lerobot/vqbet",
        "model_revision": LEROBOT_REVISION,
        "overrides": {},
        "minimum_gpu_memory_gb": 16,
        "required_cameras": 1,
    },
    "wall_x": {
        "label": "WALL-X",
        "description": "WALL-X policy using released WALL-OSS flow weights",
        "policy_type": "wall_x",
        "extra": "training,wallx",
        "initialization": "pretrained_name_or_path",
        "model_id": "x-square-robot/wall-oss-flow",
        "model_revision": "44e827683819957d8c574e8b746a1a97e77f518a",
        "overrides": {"vision_attn_implementation": "sdpa"},
        "minimum_gpu_memory_gb": 40,
    },
    "xvla": {
        "label": "XVLA",
        "description": "Florence-based cross-embodiment policy",
        "policy_type": "xvla",
        "extra": "training,xvla",
        "initialization": "pretrained",
        "model_id": "lerobot/xvla-base",
        "model_revision": "cdb7964e4fe842935d671bfab5a5ebe00a96648c",
        "overrides": {
            "action_mode": "auto",
            "freeze_vision_encoder": True,
            "freeze_language_encoder": True,
            "train_policy_transformer": False,
            "train_soft_prompts": True,
        },
        "minimum_gpu_memory_gb": 24,
    },
}


SMOLVLA_NATIVE_PROFILE = {
    "label": "SmolVLA", "description": "Native supervised action-expert fine-tuning",
    "policy_type": "smolvla", "extra": "training,smolvla", "initialization": "pretrained",
    "model_id": "lerobot/smolvla_base", "model_revision": "d9f33c94a60fb382c90dea2164c96845bd955e28",
    "overrides": {"freeze_vision_encoder": True, "train_expert_only": True},
    "minimum_gpu_memory_gb": 24,
}


def native_profile_for_recipe(recipe, method=None):
    repository = (recipe or {}).get("model_id")
    if repository == SMOLVLA_NATIVE_PROFILE["model_id"] and (method or (recipe or {}).get("method")) == "full":
        return SMOLVLA_NATIVE_PROFILE
    return next(
        (profile for profile in NATIVE_PROFILES.values() if profile["model_id"] == repository), None
    )


def native_requirement(profile):
    return LEROBOT_REQUIREMENT.format(extras=profile["extra"])


# These are checkpoint architecture bounds, not arbitrary UI dimension limits.
NATIVE_DIMENSION_LIMITS = {
    "smolvla": (32, 32),
    "eo1": (32, 32),
    "evo1": (24, 24),
    "groot": (132, 132),
    "pi0": (32, 32),
    "pi05": (32, 32),
    "pi0_fast": (32, 32),
    "wall_x": (20, 20),
    "xvla": (20, 20),
}


def validate_native_dataset(profile, features, cameras):
    policy = profile["policy_type"]
    limits = NATIVE_DIMENSION_LIMITS.get(policy, (None, None))
    for key, maximum in zip(("observation.state", "action"), limits, strict=True):
        shape = features.get(key, {}).get("shape", [])
        if len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1:
            raise ValueError(f"{profile['label']} requires a vector {key} feature")
        if maximum and shape[0] > maximum:
            raise ValueError(f"{profile['label']} supports at most {maximum} {key} dimensions")
    if policy == "evo1" and len(cameras) > 3:
        raise ValueError("EVO-1 supports at most three selected cameras")
