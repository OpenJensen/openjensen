"""Display names are separate from immutable model manifests and run records."""

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path

from pydantic import Field, field_validator

from .contracts import PolicyArtifact, StrictRecord

logger = logging.getLogger(__name__)


def descriptive_model_name(artifact: PolicyArtifact, artifacts=()) -> str:
    records = {item.id: item for item in artifacts if item.project_id == artifact.project_id}
    lineage, seen, pending = [], set(), [artifact]
    while pending:
        item = pending.pop(0)
        if item.id in seen:
            continue
        seen.add(item.id)
        lineage.append(item)
        pending.extend(records[parent] for parent in item.parent_ids if parent in records)
    architecture, step, variant = None, None, None
    for item in lineage:
        metadata = item.metadata
        architecture = architecture or metadata.get("architecture") or metadata.get("policy_type")
        if not architecture:
            from .training_catalog import TRAINING_MODELS

            base = metadata.get("base_model")
            repository = (
                base.get("repository") if isinstance(base, dict) else metadata.get("model_id")
            )
            architecture = next(
                (model.id for model in TRAINING_MODELS if model.model_id == repository), None
            )
        if step is None:
            step = next(
                (
                    metadata[key]
                    for key in ["step", "checkpoint_step", "completed_steps"]
                    if type(metadata.get(key)) is int and metadata[key] >= 0
                ),
                None,
            )
            if step is None and (
                match := re.search(r"(?:\bstep\s*|checkpoint[-_])(\d+)", item.label, re.I)
            ):
                step = int(match[1])
        precision = metadata.get("precision")
        if isinstance(precision, dict):
            precision = precision.get("language")
        if variant is None and isinstance(precision, str) and precision:
            variant = {"Q4_0": "q4", "Q8_0": "q8"}.get(precision, precision.lower())
    if variant is None:
        method = artifact.metadata.get("method")
        variant = "sft" if method == "full" else method if method in {"lora", "qlora"} else None

    def slug(value):
        return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")

    return "-".join(
        slug(value) for value in [architecture or "policy", step, variant] if value is not None
    )[:80]


class ModelRunName(StrictRecord):
    name: str = Field(min_length=1, max_length=80, strict=True)

    @field_validator("name")
    @classmethod
    def printable_name(cls, value: str) -> str:
        if any(not character.isprintable() for character in value):
            raise ValueError("Model names must contain printable characters only")
        return value


class ModelNames:
    def __init__(self, data_dir: Path):
        self.directory = data_dir / "model-names"

    @staticmethod
    def key(project_id: str, job_id: str) -> str:
        return hashlib.sha256(json.dumps([project_id, job_id]).encode()).hexdigest()

    def name(self, project_id: str, job_id: str, default: str) -> str:
        key = self.key(project_id, job_id)
        path = self.directory / f"{key}.json"
        if path.exists():
            try:
                return ModelRunName.model_validate_json(path.read_text()).name
            except (ValueError, OSError) as exc:
                logger.warning("Unable to read saved model display name %s: %s", key, exc)
        return default

    def rename(self, project_id: str, job_id: str, name: str) -> ModelRunName:
        value = ModelRunName(name=name)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".name-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write(value.model_dump_json() + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.directory / f"{self.key(project_id, job_id)}.json")
        finally:
            Path(temporary).unlink(missing_ok=True)
        return value

    def decorate(self, artifact: PolicyArtifact, artifacts=()) -> PolicyArtifact:
        return artifact.model_copy(
            update={
                "run_name": self.name(
                    artifact.project_id,
                    artifact.job_id,
                    descriptive_model_name(artifact, artifacts),
                )
            }
        )
