"""Common work-job reads and admission; maintenance retains its own API."""

from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import Field, JsonValue, ValidationError

from vla_platform.augmentation.contracts import AugmentationRequest
from vla_platform.contracts import IntakeRequest
from vla_platform.decision_contracts import DecisionRequest
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.scenes.agent import SceneError
from vla_platform.scenes.contracts import SceneRequest
from vla_platform.submissions import IdempotencyKey
from vla_platform.teaching_sessions.contracts import TeachingCaptureRequest
from vla_platform.workspace_jobs import (
    OPERATIONS,
    JobOperation,
    WorkEvent,
    WorkJob,
    WorkRecord,
    core_job,
    get_job,
    list_jobs,
)

router = APIRouter(prefix="/api/v1", tags=["Jobs"])


class WorkSubmission(WorkRecord):
    operation: Literal[
        "dataset.inspect",
        "dataset.augment",
        "policy.finetune",
        "policy.distill",
        "policy.quantize",
        "policy.evaluate",
        "policy.run",
        "policy.export",
        "policy.workflow",
        "teaching.capture",
        "decision.score",
        "scene.generate",
    ]
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    target_id: str | None = None


async def project(request, ident):
    if await request.app.state.projects.get(ident) is None:
        raise HTTPException(404, "Project not found")
    return request.app.state


@router.get("/job-operations", response_model=list[JobOperation])
async def operations():
    return OPERATIONS


@router.get("/projects/{project_id}/work-jobs", response_model=list[WorkJob])
async def history(
    project_id: str,
    request: Request,
    response: Response,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=1000000),
    module: str | None = Query(default=None, max_length=100),
    status: str = Query(default="all", max_length=32),
    q: str = Query(default="", max_length=200),
):
    state = await project(request, project_id)
    rows = await list_jobs(state, project_id)
    if status not in {
        "all",
        "active",
        "queued",
        "running",
        "cancelling",
        "waiting",
        "succeeded",
        "failed",
        "cancelled",
        "interrupted",
    }:
        raise HTTPException(422, "Unknown job status")
    rows = [
        job
        for job in rows
        if (not module or job.module == module)
        and (
            status == "all"
            or (status == "active" and job.status in {"queued", "running", "cancelling"})
            or job.status == status
        )
        and q.casefold()
        in f"{job.title} {job.operation} {job.execution.label if job.execution else ''}".casefold()
    ]
    response.headers["X-Total-Count"] = str(len(rows))
    if offset + limit < len(rows):
        response.headers["X-Next-Offset"] = str(offset + limit)
    return [
        job.model_copy(update={"inputs": {}, "result": None, "outputs": [], "events": []})
        for job in rows[offset : offset + limit]
    ]


@router.get("/projects/{project_id}/work-jobs/{key}", response_model=WorkJob)
async def detail(project_id: str, key: str, request: Request):
    state = await project(request, project_id)
    job = await get_job(state, project_id, key)
    if job is None:
        raise HTTPException(404, "Job not found in this project")
    if key.startswith("core:"):
        job.events = [
            WorkEvent(timestamp=e.timestamp, stage=e.stage, message=e.message)
            for e in state.execution.lifecycle.events(job.source_id)[-200:]
        ]
    if job.operation == "scene.generate" and key.startswith("core:"):
        try:
            run = state.scenes.get(job.inputs["request_id"])
            if run.job_id == job.source_id and run.project_id == project_id:
                job.events = [
                    WorkEvent(timestamp=e["at"], stage=e["stage"], message=e["message"])
                    for e in run.events[-200:]
                ]
        except SceneError, ValueError:
            pass
    return job


@router.post("/projects/{project_id}/work-jobs", response_model=WorkJob, status_code=202)
async def submit(
    project_id: str,
    payload: WorkSubmission,
    request: Request,
    response: Response,
    idempotency_key: Annotated[IdempotencyKey, Header()],
):
    state = await project(request, project_id)
    response.headers["Idempotency-Key"] = idempotency_key
    response.headers["Cache-Control"] = "no-store"
    if len(request.headers.getlist("idempotency-key")) != 1:
        raise HTTPException(422, "Supply exactly one Idempotency-Key header")
    raw = dict(payload.inputs)
    if "operation" in raw and raw["operation"] != payload.operation:
        raise HTTPException(422, "Job operation differs from its inputs")
    raw.pop("operation", None)
    try:
        if payload.operation == "scene.generate":
            import hashlib

            if payload.target_id not in {None, "scene"}:
                raise ValueError("This scene worker does not support the selected compute target")
            # Scope the scene's durable request ID to the project and opaque submission key.
            ident = hashlib.sha256(
                f"{project_id}:scene.generate:{idempotency_key}".encode()
            ).hexdigest()[:32]
            if raw.get("request_id") not in {None, ident}:
                raise ValueError("Scene identity is managed by the job submission")
            raw["request_id"] = ident
            body = SceneRequest.model_validate(raw)
            source = await state.execution.get(body.source_job_id)
            if source is None or source.project_id != project_id:
                raise HTTPException(404, "Dataset source not found in this project")
            return core_job(
                await state.execution.submit(project_id, body, idempotency_key=idempotency_key)
            )
        model = {
            "dataset.inspect": IntakeRequest,
            "dataset.augment": AugmentationRequest,
            "teaching.capture": TeachingCaptureRequest,
            "decision.score": DecisionRequest,
        }.get(payload.operation, PolicyRequest)
        if model is PolicyRequest:
            if payload.target_id is not None:
                from vla_platform.compute_targets import ComputeTargets

                target = ComputeTargets(state.execution.settings.data_dir).get(payload.target_id)
                if raw.get("runtime_id") not in {None, target.runtime_id}:
                    raise ValueError("Compute target differs from the requested runtime")
                raw["runtime_id"] = target.runtime_id
                raw["execution_target_id"] = target.id
        elif payload.target_id is not None:
            expected = {
                "dataset.inspect": "datasets",
                "dataset.augment": "augmentation",
                "teaching.capture": "teaching",
                "decision.score": "decision",
            }[payload.operation]
            if payload.target_id != expected:
                raise ValueError("This operation does not support the selected compute target")
        if model is not IntakeRequest:
            raw["operation"] = payload.operation
        body = model.model_validate(raw)
        return core_job(
            await state.execution.submit(project_id, body, idempotency_key=idempotency_key)
        )
    except ValidationError:
        raise HTTPException(422, "Job inputs do not match the selected operation") from None
    except SceneError as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        # Durable submission conflicts keep their registered 409 handler.
        from vla_platform.submissions import SubmissionConflict

        if isinstance(exc, SubmissionConflict):
            raise
        raise HTTPException(422, str(exc)) from None


@router.post("/projects/{project_id}/work-jobs/{key}/cancel", response_model=WorkJob)
async def cancel(project_id: str, key: str, request: Request):
    state = await project(request, project_id)
    job = await get_job(state, project_id, key)
    if job is None:
        raise HTTPException(404, "Job not found in this project")
    if not job.can_cancel:
        raise HTTPException(409, "This job cannot be cancelled in its current state")
    if key.startswith("core:"):
        await state.execution.cancel(job.source_id)
    elif key.startswith("scene:"):
        await state.scenes.cancel(job.source_id)
    elif key.startswith("agent:"):
        await state.agent_control.interrupt(job.source_id, "stop")
    elif key.startswith("dataset:"):
        await state.dataset_library.cancel_conversion(
            job.inputs["dataset_id"], expected_job_id=job.source_id
        )
    return await get_job(state, project_id, key)
