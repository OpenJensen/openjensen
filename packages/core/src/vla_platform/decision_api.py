"""Optional, manual Muose advisory scoring; never submits jobs or actions."""

import asyncio
import hashlib
import math
import os
import signal
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ValidationError

from vla_platform.decision_contracts import (
    DecisionStatus,
    ScoreRequest,
    ScoreResult,
    canonical,
    strict_json,
)
from vla_platform.environment import getenv
from vla_platform.worker_paths import worker_module, worker_package

router = APIRouter(prefix="/api/v1/decision", tags=["Decision advisory"])
MODEL = "muose/Muose-50M-Decision"
REVISION = "5afb8eeff127621fea2d66fc63f56798ada12eda"
MODEL_HASH = "3d81f0712ea1e9495a5996258dbc9a41e2fc2e1912dd87464683b20950e5c76f"
LICENSE = "CC-BY-NC-SA-4.0"
VERSIONS = {"torch": "2.11.0", "safetensors": "0.8.0", "tokenizers": "0.23.2"}
TEMPLATE = "STATE:\n{state}\n\nINSTRUCTIONS:\n{instructions}\n\nCRITERION:\n{criterion}"
# These strings are part of the existing decision worker's exact wire contract.
CAVEATS = [
    "Relative softmax weights are not calibrated correctness probabilities.",
    "Experimental Firebird prompt; not a reproduction of the upstream benchmark.",
    "Banking-tuned text scorer; out-of-domain and robotics quality are unverified.",
    "Advisory only: this output cannot authorize a job, tool, payment, or robot action.",
]
MAX_BYTES = 65536
DEADLINE = 40
CLEANUP_SECONDS = 6
_busy = threading.Lock()


def configuration():
    names = ("FIREBIRD_DECISION_PYTHON", "FIREBIRD_DECISION_ROOT", "FIREBIRD_DECISION_MODEL_DIR")
    values = [getenv(name) for name in names]
    if not all(values) or getenv("JENSEN_DECISION_ACCEPT_LICENSE") != LICENSE:
        raise HTTPException(
            503, "The operator must configure the local scorer and accept its license."
        )
    python, root, model = (Path(value) for value in values)
    if (
        os.name != "posix"
        or not all(path.is_absolute() for path in (python, root, model))
        or not python.is_file()
        or not os.access(python, os.X_OK)
        or not (worker_package(root, "jensen_decision") / "__main__.py").is_file()
        or not model.is_dir()
    ):
        raise HTTPException(503, "The local scorer configuration is unavailable on this host.")
    return python, root, model


@router.get("/status", response_model=DecisionStatus)
async def status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        configuration()
    except HTTPException as exc:
        return DecisionStatus(
            configured=False, available=False, busy=_busy.locked(), message=exc.detail
        )
    return DecisionStatus(
        configured=True,
        available=not _busy.locked(),
        busy=_busy.locked(),
        message="The local scorer is busy. Refresh after the current request finishes."
        if _busy.locked()
        else (
            "Configured to attempt local scoring. Model integrity and runtime are verified "
            "on each score; quality is experimental."
        ),
    )


def validate_result(raw, payload):
    value = strict_json(raw)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or type(value.get("threads")) is not int
        or value.get("advisory_only") is not True
        or value.get("calibrated") is not False
    ):
        raise ValueError("Invalid identity types")
    result = ScoreResult.model_validate(value)
    expected_hash = hashlib.sha256(canonical(payload)).hexdigest()
    if (
        result.model_sha256 != MODEL_HASH
        or result.request_sha256 != expected_hash
        or result.prompt_template_sha256 != hashlib.sha256(TEMPLATE.encode()).hexdigest()
        or result.runtime_versions != VERSIONS
        or result.caveats != CAVEATS
        or [row.id for row in result.scores] != [row["id"] for row in payload["criteria"]]
    ):
        raise ValueError("Unbound result")
    logits = [row.logit for row in result.scores]
    exponentials = [math.exp(value - max(logits)) for value in logits]
    total = sum(exponentials)
    if any(
        abs(row.relative_weight - value / total) > 1e-10
        for row, value in zip(result.scores, exponentials, strict=True)
    ):
        raise ValueError("Invalid weights")
    if result.selected_id != result.scores[max(range(len(logits)), key=logits.__getitem__)].id:
        raise ValueError("Invalid selection")
    return result


async def read_bounded(stream):
    output = bytearray()
    while chunk := await stream.read(4096):
        output.extend(chunk)
        if len(output) > MAX_BYTES:
            raise ValueError("Oversized worker output")
    return bytes(output)


async def cleanup(process):
    # Drain without retaining output while the public CLI reaps its own scoring child.
    async def discard(stream):
        while await stream.read(4096):
            pass

    drains = [asyncio.create_task(discard(stream)) for stream in (process.stdout, process.stderr)]
    try:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), CLEANUP_SECONDS)
            except TimeoutError:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await asyncio.wait_for(asyncio.gather(process.wait(), *drains), 5)
    finally:
        for task in drains:
            if not task.done():
                task.cancel()
        await asyncio.gather(*drains, return_exceptions=True)


async def supervised(payload, paths, request):
    python, root, model = paths
    process = None
    reads = []
    with tempfile.TemporaryDirectory(prefix="jensen-decision-api-") as temporary:
        temporary = str(Path(temporary).resolve())
        path = Path(temporary) / "request.json"
        path.write_bytes(canonical(payload))
        env = {
            "PATH": os.defpath,
            "HOME": temporary,
            "TMPDIR": temporary,
            "PYTHONPATH": str(root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        command = [
            str(python),
            "-m",
            worker_module(root, "jensen_decision"),
            "--model-dir",
            str(model),
            "--request",
            str(path),
            "--accept-license",
            LICENSE,
            "--timeout-seconds",
            "30",
        ]
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        )
        try:
            async with asyncio.timeout(DEADLINE):
                # Repeated cancellation must not strand a spawn before its handle is registered.
                interrupted = False
                while not spawn.done():
                    try:
                        await asyncio.shield(spawn)
                    except asyncio.CancelledError:
                        interrupted = True
                process = spawn.result()
                if interrupted:
                    raise asyncio.CancelledError
                reads = [
                    asyncio.create_task(read_bounded(process.stdout)),
                    asyncio.create_task(read_bounded(process.stderr)),
                ]
                while process.returncode is None or not all(task.done() for task in reads):
                    for task in reads:
                        if task.done():
                            task.result()
                    if await request.is_disconnected():
                        raise HTTPException(
                            499, "Scoring request disconnected; no result accepted."
                        )
                    await asyncio.sleep(0.02)
                stdout, _ = await asyncio.gather(*reads)
                if await process.wait() != 0:
                    raise ValueError("Worker failed")
                return validate_result(stdout, payload)
        finally:

            async def finish_owned():
                for task in reads:
                    if not task.done():
                        task.cancel()
                if reads:
                    await asyncio.gather(*reads, return_exceptions=True)
                if process is not None:
                    await cleanup(process)

            finish = asyncio.create_task(finish_owned())
            while not finish.done():
                try:
                    await asyncio.shield(finish)
                except asyncio.CancelledError:
                    continue
            finish.result()


def request_schema():
    schema = ScoreRequest.model_json_schema()
    definitions = schema.pop("$defs", {})

    def inline(value):
        if isinstance(value, dict):
            if "$ref" in value:
                return inline(definitions[value["$ref"].removeprefix("#/$defs/")])
            return {key: inline(item) for key, item in value.items()}
        if isinstance(value, list):
            return [inline(item) for item in value]
        return value

    return inline(schema)


@router.post(
    "/score",
    response_model=ScoreResult,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": request_schema()}},
        }
    },
)
async def score(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        body = bytearray()
        async with asyncio.timeout(4):
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise ValueError("Oversized input")
        decoded = strict_json(body)
        payload = ScoreRequest.model_validate(decoded).model_dump()
        if type(decoded.get("schema_version")) is not int:
            raise ValueError("Invalid schema type")
    except ValueError, UnicodeError, RecursionError, ValidationError, TimeoutError:
        raise HTTPException(422, "Supply bounded UTF-8 text and 2–8 unique criteria.") from None
    paths = configuration()
    if not _busy.acquire(blocking=False):
        raise HTTPException(
            409, "The local scorer is busy. Wait for its current request to finish."
        )
    try:
        return await supervised(payload, paths, request)
    except OSError, ValueError, RecursionError, ValidationError, TimeoutError:
        raise HTTPException(
            503,
            "No score was accepted. Check the local runtime, model files "
            "and 512-token input limit.",
        ) from None
    finally:
        _busy.release()
