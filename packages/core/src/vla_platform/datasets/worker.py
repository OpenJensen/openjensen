"""Fixed metadata-only worker entry point; never writes application metadata."""

import asyncio
import signal
import sys
from pathlib import Path

from vla_platform.contracts import DatasetSnapshot, WorkerRequest, WorkerResult
from vla_platform.datasets.inspect import inspect_hub, inspect_local


def training_blocker(error: Exception) -> str:
    """The validator's reason, without the wrapper that made it read like a crash."""
    text = str(error).strip().removeprefix("Dataset validation failed:").strip()
    return (text or "The dataset did not pass training validation")[:1000]


async def run(request_path: Path, result_path: Path) -> None:
    try:
        request = WorkerRequest.model_validate_json(request_path.read_bytes())
        if request.intake.recordings is not None:
            raise ValueError("Recording selections require the supervised preparation adapter")
        if request.intake.source == "huggingface":
            result = await inspect_hub(request.intake)
        elif request.intake.snapshot_for_training:
            from vla_platform.datasets.snapshots import (
                SnapshotError,
                create_snapshot,
                resolve_snapshot,
            )

            if not request.snapshot_store or not request.local_root:
                raise ValueError("Local training snapshot storage is not configured")
            try:
                # Synchronous inside this isolated process so SIGTERM unwinds the
                # snapshot reader's finally block and reaps its native subprocess.
                value = create_snapshot(
                    request.local_root,
                    request.intake.path,
                    Path(request.snapshot_store),
                    python=request.reader_python,
                )
            except SnapshotError as exc:
                # Training readiness is a property of the dataset, not a reason to hide
                # it. Nothing was published; report the plain inspection and say why
                # no training copy exists. An unreadable dataset still fails below.
                result = inspect_local(request.intake, request.local_root)
                result.training_blocker = training_blocker(exc)
                result = type(result).model_validate(result.model_dump())
            else:
                frozen = resolve_snapshot(Path(request.snapshot_store), value)
                result = inspect_local(
                    request.intake.model_copy(update={"path": str(frozen)}),
                    request.snapshot_store,
                )
                result.snapshot = DatasetSnapshot.model_validate(value)
                result.inspection_scope = "complete_snapshot"
                result.warnings = list(
                    dict.fromkeys(
                        [
                            *[
                                warning
                                for warning in result.warnings
                                if not warning.startswith(
                                    (
                                        "Metadata-only inspection:",
                                        "No videos, frame data, task instructions",
                                        "Local revision hashes meta/info.json only;",
                                    )
                                )
                            ],
                            *value["warnings"],
                        ]
                    )
                )
                result = type(result).model_validate(result.model_dump())
        else:
            result = await asyncio.to_thread(inspect_local, request.intake, request.local_root)
        response = WorkerResult(result=result)
    except Exception as exc:
        response = WorkerResult(error=f"{type(exc).__name__}: {str(exc)[:1000]}")
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(response.model_dump_json(), encoding="utf-8")
    temporary.replace(result_path)


def terminate_worker(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate_worker)
    asyncio.run(run(Path(sys.argv[1]), Path(sys.argv[2])))
