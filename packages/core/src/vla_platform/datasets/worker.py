"""Fixed metadata-only worker entry point; never writes application metadata."""

import asyncio
import sys
from pathlib import Path

from vla_platform.contracts import WorkerRequest, WorkerResult
from vla_platform.datasets.inspect import inspect_hub, inspect_local


async def run(request_path: Path, result_path: Path) -> None:
    try:
        request = WorkerRequest.model_validate_json(request_path.read_bytes())
        if request.intake.source == "huggingface":
            result = await inspect_hub(request.intake)
        else:
            result = await asyncio.to_thread(inspect_local, request.intake, request.local_root)
        response = WorkerResult(result=result)
    except Exception as exc:
        response = WorkerResult(error=f"{type(exc).__name__}: {str(exc)[:1000]}")
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(response.model_dump_json(), encoding="utf-8")
    temporary.replace(result_path)


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1]), Path(sys.argv[2])))
