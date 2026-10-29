"""One public work-job vocabulary over the existing durable operation owners.

Adapters project records; they never duplicate execution or rewrite old history.
Maintenance is deliberately outside this service.
"""

from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from vla_platform.compute.contracts import ExecutionBinding
from vla_platform.contracts import TERMINAL, IntakeRequest
from vla_platform.job_observability import redact
from vla_platform.lifecycle.contracts import LifecycleResult
from vla_platform.scenes.contracts import SceneJobResult


class WorkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobLink(WorkRecord):
    label: str
    url: str


class WorkEvent(WorkRecord):
    timestamp: str
    stage: str
    message: str


class WorkJob(WorkRecord):
    id: str
    source_id: str
    project_id: str
    operation: str
    module: str
    title: str
    status: Literal[
        "queued",
        "running",
        "cancelling",
        "waiting",
        "succeeded",
        "failed",
        "cancelled",
        "interrupted",
    ]
    source_status: str
    created_at: str
    updated_at: str
    stage: str | None = None
    message: str | None = None
    execution: ExecutionBinding | None = None
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    result: JsonValue = None
    outputs: list[JobLink] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    can_cancel: bool = False
    events: list[WorkEvent] = Field(default_factory=list)


class JobOperation(WorkRecord):
    id: str
    title: str
    module: str
    path: str
    submission: Literal["job", "interactive", "upload"] = "job"


OPERATIONS = [
    JobOperation(id=ident, title=title, module=module, path=path, submission=submission)
    for ident, title, module, path, submission in (
        ("dataset.inspect", "Inspect dataset", "datasets", "/datasets/", "job"),
        ("dataset.import", "Import dataset", "datasets", "/datasets/", "upload"),
        ("dataset.convert", "Convert dataset", "datasets", "/datasets/", "interactive"),
        ("dataset.augment", "Augment dataset", "augmentation", "/augmentation/", "job"),
        ("scene.generate", "Real to sim", "real-to-sim", "/real-to-sim/", "job"),
        ("policy.finetune", "Fine-tune", "training", "/training/", "job"),
        ("policy.distill", "Distill", "distillation", "/distillation/", "job"),
        ("policy.quantize", "Quantize", "quantization", "/quantization/", "job"),
        ("policy.evaluate", "Evaluate", "evaluation", "/evaluation/", "job"),
        ("policy.run", "Run policy", "simulation", "/simulation/", "job"),
        ("policy.export", "Export model", "models", "/models/", "job"),
        ("policy.import", "Import model", "models", "/models/", "upload"),
        ("policy.workflow", "Policy workflow", "evaluation", "/evaluation/", "job"),
        ("teaching.capture", "Record teaching", "teaching", "/ai-teleoperation/", "job"),
        ("agent.control", "AI teleoperation", "teaching", "/ai-teleoperation/", "interactive"),
        ("decision.score", "Decision scoring", "decision-lab", "/decision-lab/", "job"),
    )
]
CATALOG = {item.id: item for item in OPERATIONS}


def core_job(job):
    operation = CATALOG[job.kind]
    request = job.request.model_dump(mode="json")
    title = operation.title
    if isinstance(job.request, IntakeRequest):
        title = job.request.repo_id or "Local dataset"
        if job.request.recordings is not None:
            title = "Prepare recording"
    elif job.kind == "policy.finetune":
        title = (request.get("training") or {}).get("model_id") or title
    parents = [
        request[key]
        for key in ("dataset_job_id", "source_job_id", "resume_job_id")
        if request.get(key)
    ]
    links = []
    if isinstance(job.result, LifecycleResult):
        links = [
            JobLink(
                label=artifact.label,
                url=(
                    f"/api/v1/projects/{quote(job.project_id, safe='')}/artifacts/"
                    f"{quote(artifact.id, safe='')}/download"
                ),
            )
            for artifact in job.result.artifacts
        ]
    elif isinstance(job.result, SceneJobResult):
        links = [
            JobLink(
                label=name,
                url=(
                    f"/api/v1/jobs/{quote(job.id, safe='')}/scene/artifact?"
                    f"name={quote(name, safe='')}"
                ),
            )
            for name in job.result.scene_artifacts
        ]
    elif job.kind == "dataset.augment" and job.status == "succeeded":
        links = [
            JobLink(
                label="Download augmented dataset",
                url=f"/api/v1/jobs/{quote(job.id, safe='')}/augmentation/download",
            )
        ]
    scene_needs_attention = (
        isinstance(job.result, SceneJobResult)
        and job.status in {"failed", "succeeded"}
        and job.result.validation_status in {"needs_review", "needs_simulation"}
        # The scene owner publishes succeeded only after all required gates and
        # final evidence pass. Its legacy validation label remains needs_review.
        and not (
            job.status == "succeeded"
            and job.stage == "complete"
            and job.result.validation_status == "needs_review"
        )
    )
    return WorkJob(
        id="core:" + job.id,
        source_id=job.id,
        project_id=job.project_id,
        operation=job.kind,
        module=operation.module,
        title=title,
        status=(
            "cancelling"
            if job.cancellation and job.cancellation.cleanup == "pending"
            else "waiting"
            if job.cancellation and job.cancellation.cleanup == "failed"
            else "waiting"
            if scene_needs_attention
            else job.status
        ),
        source_status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        stage=job.stage,
        message=redact(job.error) if job.error else None,
        execution=job.execution,
        inputs=request,
        result=job.result.model_dump(mode="json") if job.result else None,
        outputs=links,
        parent_ids=["core:" + ident for ident in parents]
        + (["scene:" + request["source_scene_id"]] if request.get("source_scene_id") else []),
        can_cancel=job.status not in TERMINAL,
    )


def scene_job(run):
    return WorkJob(
        id="scene:" + run.id,
        source_id=run.id,
        project_id=run.project_id,
        operation="scene.generate",
        module="real-to-sim",
        title="Real to sim",
        status="waiting" if run.status in {"needs_review", "needs_simulation"} else run.status,
        source_status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        stage=run.stage,
        message=run.message,
        execution=ExecutionBinding(
            adapter="scene", infrastructure="local", target_id="scene", label="Scene worker"
        ),
        inputs=run.request.model_dump(mode="json"),
        result={"gates": run.gates, "usage": run.usage, "attempt": run.attempt},
        outputs=[
            JobLink(label=name, url=f"/api/v1/scenes/{run.id}/artifact?name={quote(name, safe='')}")
            for name in run.artifacts
        ],
        parent_ids=["core:" + run.request.source_job_id],
        can_cancel=run.status in {"queued", "running"},
        events=[
            WorkEvent(timestamp=e["at"], stage=e["stage"], message=e["message"])
            for e in run.events[-200:]
        ],
    )


def agent_job(session):
    state = {
        "ready": "waiting",
        "waiting_for_acceptance": "waiting",
        "paused": "waiting",
        "complete": "succeeded",
        "stopped": "cancelled",
        "error": "failed",
        "interrupted": "interrupted",
    }.get(session.status, "running")
    return WorkJob(
        id="agent:" + session.id,
        source_id=session.id,
        project_id=session.project_id,
        operation="agent.control",
        module="teaching",
        title=session.task,
        status=state,
        source_status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        stage=session.status,
        message=session.message,
        execution=ExecutionBinding(
            adapter="agent",
            infrastructure=session.target_kind,
            target_id=session.provider_id,
            label="AI teleoperation",
        ),
        inputs={"task": session.task, "mode": session.mode, "target": session.target_kind},
        result={
            "steps": session.step_count,
            "calls": session.calls_used,
            "task_success_verified": session.task_success_verified,
        },
        parent_ids=["core:" + session.teaching_job_id] if session.teaching_job_id else [],
        can_cancel=state not in {"succeeded", "cancelled", "failed", "interrupted"},
    )


def dataset_job(record):
    if "operation" not in record:
        return legacy_dataset_job(record)
    dataset_id = record.get("redirect_dataset_id") or record["dataset_id"]
    ready = record["status"] == "succeeded"
    return WorkJob(
        id="dataset:" + record["id"],
        source_id=record["id"],
        project_id=record["project_id"],
        operation=record["operation"],
        module="datasets",
        title=record["name"],
        status=record["status"],
        source_status=record["status"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        stage=record["stage"],
        message=record.get("error"),
        execution=ExecutionBinding(
            adapter="local", infrastructure="local", target_id="datasets", label="Dataset worker"
        ),
        inputs={**record["inputs"], "dataset_id": dataset_id},
        result=record.get("result"),
        parent_ids=[record["inputs"]["source_job_id"]]
        if record["inputs"].get("source_job_id")
        else [],
        outputs=[JobLink(label="Download dataset", url=f"/api/v1/datasets/{dataset_id}/download")]
        if ready and record["operation"] == "dataset.convert"
        else [],
        can_cancel=record["status"] == "running" and record["operation"] == "dataset.convert",
    )


def legacy_dataset_job(record):
    converting = bool(record.get("conversion"))
    state = {
        "uploading": "waiting",
        "detected": "succeeded",
        "converting": "running",
        "ready": "succeeded",
        "failed": "failed",
    }[record["status"]]
    return WorkJob(
        id="dataset:" + record["id"],
        source_id=record["id"],
        project_id=record["project_id"],
        operation="dataset.convert" if converting else "dataset.import",
        module="datasets",
        title=record["name"],
        status=state,
        source_status=record["status"],
        created_at=record["created_at"],
        updated_at=record.get("updated_at", record["created_at"]),
        stage=record["status"],
        message=record.get("error"),
        execution=ExecutionBinding(
            adapter="local", infrastructure="local", target_id="datasets", label="Dataset worker"
        ),
        inputs={
            "name": record["name"],
            "bytes": record["total_bytes"],
            "conversion": record.get("conversion"),
        },
        result=record.get("converted") or record.get("detection"),
        outputs=[JobLink(label="Download dataset", url=f"/api/v1/datasets/{record['id']}/download")]
        if record["status"] == "ready"
        else [],
    )


async def list_jobs(state, project_id):
    rows = [core_job(job) for job in await state.execution.list(project_id)]
    core_ids = {job.source_id for job in rows}
    rows.extend(
        scene_job(run)
        for run in state.scenes.list(project_id)
        if not run.job_id or run.job_id not in core_ids
    )
    rows.extend(agent_job(session) for session in await state.agent_control.list(project_id))
    dataset_operations = state.dataset_library.operation_records(project_id)
    rows.extend(dataset_job(record) for record in dataset_operations)
    covered_datasets = {record["dataset_id"] for record in dataset_operations}
    rows.extend(
        legacy_dataset_job(record)
        for record in state.dataset_library.records()
        if record.get("project_id") == project_id
        and record.get("status") != "unavailable"
        and record["id"] not in covered_datasets
    )
    return sorted(rows, key=lambda job: (job.created_at, job.id), reverse=True)


async def get_job(state, project_id, key):
    """Resolve only the requested owner; polling never scans complete histories."""
    from vla_platform.agent_control.service import AgentError
    from vla_platform.scenes.agent import SceneError

    family, separator, ident = key.partition(":")
    if not separator or not ident:
        return None
    try:
        if family == "core":
            source = await state.execution.get(ident)
            result = core_job(source) if source is not None else None
        elif family == "scene":
            result = scene_job(state.scenes.get(ident))
        elif family == "agent":
            result = agent_job(await state.agent_control.get(ident))
        elif family == "dataset":
            try:
                result = dataset_job(state.dataset_library.operations.get(ident))
            except ValueError:
                source = state.dataset_library.get(ident, project_id)
                result = (
                    legacy_dataset_job(source) if source.get("status") != "unavailable" else None
                )
        else:
            return None
    except SceneError, AgentError, ValueError:
        return None
    return result if result is not None and result.project_id == project_id else None
