"""Immutable operation identities over dataset import and conversion progress."""

import os
import re
import tempfile
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from vla_platform.contracts import Timestamp, now
from vla_platform.job_observability import JobObservability, public_value, redact

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
IDENTITY = r"^[a-f0-9]{32}$"
MAX_BYTES = 1024**2


class DatasetOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    id: str = Field(pattern=IDENTITY)
    dataset_id: str = Field(pattern=IDENTITY)
    project_id: str = Field(min_length=1, max_length=128)
    operation: Literal["dataset.import", "dataset.convert"]
    name: str = Field(min_length=1, max_length=100)
    created_at: Timestamp
    updated_at: Timestamp
    status: Literal["waiting", "running", "succeeded", "failed", "cancelled", "interrupted"]
    stage: str
    error: str | None = None
    inputs: dict = Field(default_factory=dict)
    result: dict | None = None
    redirect_dataset_id: str | None = Field(default=None, pattern=IDENTITY)


class DatasetOperations:
    def __init__(self, data_dir):
        self.directory = Path(data_dir) / "dataset-jobs"
        self.observability = JobObservability(Path(data_dir))

    def path(self, ident):
        if not re.fullmatch(IDENTITY, ident):
            raise ValueError("Dataset operation identity is invalid")
        path = self.directory / f"{ident}.json"
        if self.directory.is_symlink() or path.is_symlink():
            raise ValueError("Dataset operation storage needs administrator attention")
        return path

    def get(self, ident):
        path = self.path(ident)
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError("Dataset operation was not found")
        record = DatasetOperation.model_validate_json(path.read_bytes())
        if record.id != ident:
            raise ValueError("Dataset operation identity differs from its saved record")
        return record.model_dump(mode="json")

    def records(self, project_id=None):
        if not self.directory.exists():
            return []
        if self.directory.is_symlink():
            raise ValueError("Dataset operation storage needs administrator attention")
        rows = []
        for path in self.directory.glob("*.json"):
            try:
                value = self.get(path.stem)
            except OSError, ValueError:
                # One damaged operation does not hide unrelated job history.
                continue
            if project_id is None or value["project_id"] == project_id:
                rows.append(value)
        return sorted(rows, key=lambda row: (row["created_at"], row["id"]), reverse=True)

    def save(self, value):
        record = DatasetOperation.model_validate(value)
        path = self.path(record.id)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = record.model_dump_json().encode() + b"\n"
        if len(encoded) > MAX_BYTES:
            raise ValueError("Dataset operation record exceeds its supported size")
        fd, temporary = tempfile.mkstemp(prefix=".dataset-job-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return record.model_dump(mode="json")

    def create(self, dataset, operation, inputs=None):
        value = DatasetOperation(
            id=uuid4().hex,
            dataset_id=dataset["id"],
            project_id=dataset["project_id"],
            name=dataset["name"],
            operation=operation,
            created_at=now(),
            updated_at=now(),
            status="waiting" if operation == "dataset.import" else "running",
            stage="uploading" if operation == "dataset.import" else "converting",
            inputs={"dataset_id": dataset["id"], "name": dataset["name"], **(inputs or {})},
        ).model_dump(mode="json")
        result = self.save(value)
        self.observability.try_record(
            "dataset:" + value["id"],
            "Dataset operation accepted",
            component="dataset",
            stage=value["stage"],
            data={"operation": operation},
        )
        return result

    def update(self, ident, *, status, stage, error=None, result=None, redirect_dataset_id=None):
        value = self.get(ident)
        if value["status"] in TERMINAL:
            return value
        previous = (value["status"], value["stage"], value["error"])
        value.update(
            status=status,
            stage=stage,
            error=redact(error) if error else None,
            result=public_value(result) if result else None,
            updated_at=now(),
            redirect_dataset_id=redirect_dataset_id,
        )
        saved = self.save(value)
        if previous != (status, stage, error):
            self.observability.try_record(
                "dataset:" + ident,
                error or f"Dataset operation {status}",
                component="dataset",
                stage=stage,
                severity="error" if status == "failed" else "info",
                data={"status": status},
            )
            if status == "failed":
                self.observability.try_failure(
                    "dataset:" + ident,
                    stage=stage,
                    message=error or "Dataset operation failed",
                    code="DATASET_OPERATION_FAILED",
                )
        return saved
