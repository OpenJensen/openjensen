"""Display names are separate from immutable model manifests and run records."""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from pydantic import Field, field_validator

from .contracts import PolicyArtifact, StrictRecord

logger = logging.getLogger(__name__)
COLORS = (
    "Amber",
    "Azure",
    "Coral",
    "Crimson",
    "Emerald",
    "Golden",
    "Indigo",
    "Ivory",
    "Jade",
    "Lilac",
    "Mint",
    "Ochre",
    "Pearl",
    "Silver",
    "Teal",
    "Violet",
)
OBJECTS = (
    "Acorn",
    "Comet",
    "Crane",
    "Dune",
    "Falcon",
    "Fern",
    "Finch",
    "Harbor",
    "Kite",
    "Lantern",
    "Maple",
    "Orchid",
    "Pebble",
    "Pine",
    "Sparrow",
    "Willow",
)


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

    def name(self, project_id: str, job_id: str) -> str:
        key = self.key(project_id, job_id)
        path = self.directory / f"{key}.json"
        if path.exists():
            try:
                return ModelRunName.model_validate_json(path.read_text()).name
            except (ValueError, OSError) as exc:
                logger.warning("Unable to read saved model display name %s: %s", key, exc)
        return (
            f"{COLORS[int(key[:2], 16) % len(COLORS)]} {OBJECTS[int(key[2:4], 16) % len(OBJECTS)]}"
        )

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

    def decorate(self, artifact: PolicyArtifact) -> PolicyArtifact:
        return artifact.model_copy(
            update={"run_name": self.name(artifact.project_id, artifact.job_id)}
        )
