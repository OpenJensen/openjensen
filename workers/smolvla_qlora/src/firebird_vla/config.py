"""Strict, dependency-free recipe validation and deterministic episode splits."""

import json
import math
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .temporal import TEMPORAL_FIELDS, validate_temporal


@dataclass(frozen=True)
class TrainConfig:
    method: str = "qlora"
    model_id: str = "lerobot/smolvla_base"
    model_revision: str = "d9f33c94a60fb382c90dea2164c96845bd955e28"
    backbone_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    backbone_revision: str = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    dataset_id: str = "codywang/so101_pickup_test"
    dataset_revision: str = "ecef85bc07005f771ad86deeff1427f9d72953ed"
    output_dir: str = "outputs/smolvla-qlora"
    camera_key: str = "observation.images.front"
    camera_keys: list[str] | None = None
    steps: int = 20000
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 50
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    chunk_size: int = 50
    prediction_horizon: int | None = None
    execution_horizon: int | None = None
    observation_history: int | None = None
    frame_stride: int | None = None
    validation_fraction: float = 0.2
    eval_every: int = 100
    eval_batches: int = 20
    save_every: int = 4000
    log_every: int = 10
    num_workers: int = 0
    seed: int = 42

    def validate(self):
        if self.method not in {"lora", "qlora"}:
            raise ValueError("Supported training methods: lora, qlora")
        for name in ("model_revision", "backbone_revision", "dataset_revision"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"[0-9a-f]{40}", getattr(self, name)
            ):
                raise ValueError(f"{name} must be an immutable 40-character Hub commit SHA")
        for name in ("model_id", "backbone_id", "dataset_id", "camera_key", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.camera_keys is not None:
            if not isinstance(self.camera_keys, list) or not self.camera_keys:
                raise ValueError("camera_keys must be a non-empty list")
            if any(not isinstance(key, str) or not key.strip() for key in self.camera_keys):
                raise ValueError("camera_keys must contain non-empty strings")
            if len(set(self.camera_keys)) != len(self.camera_keys):
                raise ValueError("camera_keys must not contain duplicates")
        positive = (
            "steps",
            "batch_size",
            "gradient_accumulation_steps",
            "lora_rank",
            "lora_alpha",
            "chunk_size",
            "eval_every",
            "eval_batches",
            "save_every",
            "log_every",
        )
        for name in positive + ("warmup_steps", "num_workers", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in positive else 0):
                raise ValueError(f"Invalid integer {name}: {value!r}")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be less than steps")
        for name in (
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "lora_dropout",
            "validation_fraction",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate/max_grad_norm must be positive; weight_decay >= 0")
        if not 0 <= self.lora_dropout < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("Require dropout in [0,1) and validation_fraction in (0,1)")
        validate_temporal(self.to_dict(), "smolvla")
        return self

    @property
    def selected_camera_keys(self):
        """Preserve explicit camera order; old recipes keep their single camera."""
        return tuple(self.camera_keys) if self.camera_keys is not None else (self.camera_key,)

    def resume_matches(self, previous):
        """Compare effective recipes across the legacy single-camera representation."""

        def canonical(config):
            data = config.to_dict()
            data.pop("output_dir")
            data.pop("camera_key")
            data["camera_keys"] = list(config.selected_camera_keys)
            return data

        return canonical(self) == canonical(previous)

    @property
    def temporal(self):
        return validate_temporal(self.to_dict(), "smolvla")

    def to_dict(self):
        data = asdict(self)
        for key in TEMPORAL_FIELDS:
            if data[key] is None:
                data.pop(key)
        if self.prediction_horizon is not None or self.execution_horizon is not None:
            # The legacy dataclass default is not a second caller choice. Persist
            # the new representation without a contradictory legacy alias.
            if self.chunk_size != 50:
                raise ValueError("Use either legacy chunk_size or independent horizons, not both")
            data.pop("chunk_size")
        return data

    @classmethod
    def from_dict(cls, data):
        validate_temporal(data, "smolvla")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def split_episodes(episode_ids, fraction, seed):
    ids = list(episode_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Need at least two unique episodes for a disjoint split")
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    ids.sort()
    random.Random(seed).shuffle(ids)
    n_val = max(1, min(len(ids) - 1, round(len(ids) * fraction)))
    return {"train": sorted(ids[n_val:]), "validation": sorted(ids[:n_val])}


def batch_indices(size, batch_size, seed, epoch):
    """Recreate an epoch without serializing a DataLoader or dropping its final batch."""
    if size < 1 or batch_size < 1:
        raise ValueError("Dataset and batch size must be positive")
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, size, batch_size)]
