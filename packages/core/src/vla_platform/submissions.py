"""Durable, project/operation-scoped submission identity; no dispatch or retries."""

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import StringConstraints, TypeAdapter
from sqlalchemy import select

from vla_platform.augmentation.contracts import AugmentationRequest
from vla_platform.contracts import IntakeRequest, Job
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.storage import Storage, job_submissions, jobs

IdempotencyKey = Annotated[
    str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
]
SubmissionOperation = Literal[
    "dataset.inspect",
    "dataset.augment",
    "policy.finetune",
    "policy.distill",
    "policy.quantize",
    "policy.evaluate",
    "policy.run",
    "policy.export",
    "policy.workflow",
    "policy.import",
]
SubmissionRequest = IntakeRequest | PolicyRequest | AugmentationRequest
MAX_REQUEST_BYTES = 1024 * 1024


class SubmissionConflict(ValueError):
    """A key is already bound to another normalized client request."""


class SubmissionUnavailable(RuntimeError):
    """Saved acceptance cannot be verified; never create replacement work."""


def validate_key(value: str) -> str:
    return TypeAdapter(IdempotencyKey).validate_python(value)


def canonical(value: dict) -> bytes:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("Normalized submission request exceeds the 1 MiB limit")
    return encoded


def operation(request: SubmissionRequest) -> str:
    return "dataset.inspect" if isinstance(request, IntakeRequest) else request.operation


@dataclass(frozen=True)
class Submission:
    project_id: str
    operation: str
    key: str
    request_record: dict
    request_sha256: str

    @classmethod
    def create(cls, project_id: str, request: SubmissionRequest, key: str) -> Submission:
        record = request.model_dump(mode="json")
        return cls(
            project_id,
            operation(request),
            validate_key(key),
            record,
            hashlib.sha256(canonical(record)).hexdigest(),
        )


async def lookup(
    storage: Storage,
    project_id: str,
    operation_name: str,
    key: str,
    *,
    fingerprint: str | None = None,
) -> Job | None:
    """Return current state only after checking the immutable acceptance and job link."""
    key = validate_key(key)
    async with storage.engine.connect() as connection:
        try:
            row = (
                (
                    await connection.execute(
                        select(job_submissions).where(
                            job_submissions.c.project_id == project_id,
                            job_submissions.c.operation == operation_name,
                            job_submissions.c.idempotency_key == key,
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            if type(row["fingerprint_version"]) is not int or row["fingerprint_version"] != 1:
                raise ValueError("Unknown fingerprint version")
            raw = row["request_record"]
            model = (
                IntakeRequest
                if operation_name == "dataset.inspect"
                else AugmentationRequest
                if operation_name == "dataset.augment"
                else PolicyRequest
            )
            request = model.model_validate(raw)
            if operation(request) != operation_name or canonical(
                request.model_dump(mode="json")
            ) != canonical(raw):
                raise ValueError("Stored request is not normalized")
            if hashlib.sha256(canonical(raw)).hexdigest() != row["request_sha256"]:
                raise ValueError("Stored request digest differs")
            accepted = Job.model_validate(row["accepted_response"])
            value = (
                (await connection.execute(select(jobs).where(jobs.c.id == row["job_id"])))
                .mappings()
                .one_or_none()
            )
            if value is None:
                raise ValueError("Accepted job is missing")
            current = Job.model_validate(value["record"])
            if (
                accepted.id != row["job_id"]
                or current.id != accepted.id
                or accepted.project_id != project_id
                or current.project_id != project_id
                or value["project_id"] != project_id
                or value["status"] != current.status
                or accepted.kind != operation_name
                or current.kind != operation_name
                or current.created_at != accepted.created_at
                or canonical(current.request.model_dump(mode="json"))
                != canonical(accepted.request.model_dump(mode="json"))
            ):
                raise ValueError("Accepted job identity changed")
        except (ValueError, TypeError, KeyError) as exc:
            raise SubmissionUnavailable(
                "Saved submission identity is unavailable; no new work was submitted"
            ) from exc
        if fingerprint is not None and row["request_sha256"] != fingerprint:
            raise SubmissionConflict("Idempotency key already identifies a different request")
        return current
