"""Training choices and pinned inputs, independent of optional GPU packages.

The additional checkpoints come from the repository's open_weight_vlas catalog.
A catalog entry is not an executable adapter: only a configured native worker can
make it available. SmolVLA retains its dedicated quantized worker; other native policies use
a separate pinned LeRobot environment.
"""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .native_profiles import NATIVE_PROFILES

if TYPE_CHECKING:
    from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog


@dataclass(frozen=True)
class TrainingModel:
    id: str
    label: str
    description: str
    model_id: str
    model_revision: str | None
    checkpoint_subdirectory: str | None = None
    methods: tuple[str, ...] = ("lora", "qlora")
    backend: str = "smolvla"
    initialization: str = "pretrained"
    minimum_gpu_memory_gb: int | None = None
    required_cameras: int | None = None
    suggested_gpu_memory_gb: int | None = None


TRAINING_MODELS = (
    TrainingModel(
        "smolvla",
        "SmolVLA",
        "Compact vision-language-action policy",
        "lerobot/smolvla_base",
        "d9f33c94a60fb382c90dea2164c96845bd955e28",
        suggested_gpu_memory_gb=16,
    ),
    TrainingModel(
        "openvla_oft",
        "OpenVLA-OFT",
        "Continuous action chunks",
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "6d0231af0e48c5985f1ff86908f4674b84bc049b",
    ),
    TrainingModel(
        "openvla",
        "OpenVLA",
        "Autoregressive action prediction",
        "openvla/openvla-7b-finetuned-libero-spatial",
        "962318cec55ac10993ff0f5f43eda9a270b4c873",
    ),
    TrainingModel(
        "pi0",
        "π₀",
        "Flow-matching robot policy",
        "lerobot/pi0_libero_finetuned_v044",
        "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
    ),
    TrainingModel(
        "pi05",
        "π₀.₅",
        "OpenPI robot policy",
        "gs://openpi-assets/checkpoints/pi05_libero",
        None,
    ),
    TrainingModel(
        "gr00t_n17",
        "GR00T N1.7",
        "NVIDIA robot foundation model",
        "nvidia/GR00T-N1.7-LIBERO",
        "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
        "libero_spatial",
    ),
)
# Generic LeRobot workers use their own pinned dependency environment. Historical
# OpenVLA entries remain operator-adapter choices; all native policies have an
# executable training route rather than receiving readiness from their label.
TRAINING_MODELS = tuple(
    model for model in TRAINING_MODELS if model.id not in NATIVE_PROFILES
) + tuple(
    TrainingModel(
        id=key,
        label=profile["label"],
        description=profile["description"],
        model_id=profile["model_id"],
        model_revision=profile["model_revision"],
        checkpoint_subdirectory=profile.get("checkpoint_subdirectory"),
        methods=("full",),
        backend="lerobot",
        initialization=profile["initialization"],
        minimum_gpu_memory_gb=profile.get("minimum_gpu_memory_gb"),
        required_cameras=profile.get("required_cameras"),
    )
    for key, profile in NATIVE_PROFILES.items()
)
TRAINING_MODELS += (
    TrainingModel(
        id="psi0",
        label="Psi-Zero",
        description="Frozen vision-language backbone and action expert",
        model_id="USC-PSI-Lab/psi-model",
        model_revision="4c6f9776fc5b18d87945254175e38bb74b9d7748",
        methods=("full",),
        backend="psi0",
        initialization="pretrained",
        minimum_gpu_memory_gb=40,
        required_cameras=1,
    ),
)
TRAINING_MODELS = tuple(
    model for model in TRAINING_MODELS if model.id not in {"openvla", "openvla_oft"}
) + tuple(model for model in TRAINING_MODELS if model.id in {"openvla", "openvla_oft"})
TRAINING_MODEL_BY_ID = {model.id: model for model in TRAINING_MODELS}


def supports_gradient_accumulation(model: TrainingModel, runtime: Runtime) -> bool:
    """Only reviewed bundled adapters declare optimizer-window semantics."""
    expected = {
        "smolvla": "firebird_vla.application",
        "act": "firebird_vla.lerobot_application",
    }.get(model.id)
    return bool(
        expected
        and model.id in runtime.training_model_ids
        and (runtime.execution == "skypilot" or runtime.training_module == expected)
    )


def public_training_models(catalog: RuntimeCatalog) -> list[dict]:
    result = []
    for model in TRAINING_MODELS:
        runtime_ids = [
            runtime.id
            for runtime in catalog.runtimes
            if runtime.training_python
            and runtime.training_root
            and runtime.device == "cuda"
            and model.id in runtime.training_model_ids
            and model.model_revision is not None
        ]
        reason = None
        if not runtime_ids:
            reason = (
                "Connect Google Cloud to train"
                if model.backend in {"smolvla", "lerobot", "psi0"}
                and model.id not in {"openvla", "openvla_oft"}
                else "Adapter not installed"
            )
        result.append(
            {
                **asdict(model),
                "methods": list(model.methods),
                "available": bool(runtime_ids),
                "status": "ready"
                if runtime_ids
                else "connect_account"
                if model.id == "smolvla" or model.backend in {"lerobot", "psi0"}
                else "coming_soon",
                "unavailable_reason": reason,
                "runtime_ids": runtime_ids,
                "gradient_accumulation_supported": model.id in {"act", "smolvla"},
                "gradient_accumulation_runtime_ids": [
                    runtime.id
                    for runtime in catalog.runtimes
                    if runtime.id in runtime_ids and supports_gradient_accumulation(model, runtime)
                ],
                "training_step_unit": "optimizer_updates",
                "training_world_size": 1,
            }
        )
    return result


def training_model_for_recipe(recipe: dict | None) -> TrainingModel:
    repository = (recipe or {}).get("model_id", TRAINING_MODELS[0].model_id)
    model = next((model for model in TRAINING_MODELS if model.model_id == repository), None)
    if model is None:
        raise ValueError("Training model is not registered")
    return model
