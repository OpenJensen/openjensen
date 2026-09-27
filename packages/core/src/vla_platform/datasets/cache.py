"""Reuse project inspections only after confirming their source identity."""

import asyncio
import hashlib
import stat
from pathlib import Path

from vla_platform.contracts import DatasetProfile, IntakeRequest, Job
from vla_platform.datasets.inspect import MAX_METADATA_BYTES, resolve_hub_revision


def local_identity(request: IntakeRequest, allowed_root: Path | None) -> IntakeRequest | None:
    """Hash bounded, regular metadata; let the isolated worker report invalid sources."""
    if allowed_root is None:
        return None
    try:
        root = allowed_root.resolve(strict=True)
        candidate = Path(request.path or "")
        dataset = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=True)
        target = (dataset / "meta/info.json").resolve(strict=True)
        if not dataset.is_relative_to(root) or not target.is_relative_to(root):
            return None
        if not stat.S_ISREG(target.stat().st_mode):
            return None
        with target.open("rb") as handle:
            raw = handle.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            return None
    except OSError, ValueError:
        return None
    return request.model_copy(
        update={
            "path": str(dataset),
            "revision": f"metadata-sha256:{hashlib.sha256(raw).hexdigest()}",
        }
    )


class Inspections:
    def __init__(self, execution):
        self.execution = execution
        # The workspace has one owner. Check + insert must still be atomic between
        # concurrent HTTP requests so two clicks cannot launch the same worker.
        self.lock = asyncio.Lock()

    async def prepare(self, request: IntakeRequest) -> IntakeRequest | None:
        if request.source == "huggingface":
            revision = await resolve_hub_revision(request)
            return request.model_copy(update={"revision": revision})
        return await asyncio.to_thread(local_identity, request, self.execution.settings.local_root)

    async def find(self, project_id: str, prepared: IntakeRequest) -> Job | None:
        records = await self.execution.list(project_id)
        for status in ("succeeded", "running", "queued"):
            for job in reversed(records):
                if job.status == status and self.matches(job, prepared):
                    return job
        return None

    async def submit(
        self, project_id: str, request: IntakeRequest, *, idempotency_key: str | None = None
    ) -> Job:
        if idempotency_key is not None:
            # The durable original-request lookup must precede mutable source resolution.
            return await self.execution.submit(project_id, request, idempotency_key=idempotency_key)
        if request.snapshot_for_training:
            # Metadata identity cannot identify changed rows or video bytes.
            # Every explicit snapshot request revalidates the complete source.
            return await self.execution.submit(project_id, request)
        prepared = await self.prepare(request)
        if prepared is None:
            return await self.execution.submit(project_id, request)
        async with self.lock:
            # Prefer a finished inspection when old duplicate jobs are present.
            cached = await self.find(project_id, prepared)
            if cached is not None:
                return cached
            return await self.execution.submit(project_id, prepared)

    def matches(self, job: Job, request: IntakeRequest) -> bool:
        previous = job.request
        if isinstance(previous, IntakeRequest) and previous.snapshot_for_training:
            return False
        if not isinstance(previous, IntakeRequest) or previous.source != request.source:
            return False
        if request.source == "huggingface":
            if previous.repo_id != request.repo_id:
                return False
        else:
            # Resolving an old relative path against a newly configured local root
            # could redirect episode previews. Only persisted absolute paths qualify.
            if previous.path != request.path:
                return False
        if job.status == "succeeded":
            result = job.result
            return (
                isinstance(result, DatasetProfile)
                and result.inspection_scope == "metadata_only"
                and result.source == request.source
                and result.repo_id == request.repo_id
                and result.revision == request.revision
            )
        return previous.revision == request.revision
