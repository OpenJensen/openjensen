"""Supervise fixed native Parquet readers without importing native code into core."""

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Literal

from vla_platform.datasets.local_preview import reader_python

WORKER_SCRIPT = Path(__file__).with_name("hub_parquet_worker.py")
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_STDERR_BYTES = 16 * 1024
READER_TIMEOUT_SECONDS = 10.0


class ReaderError(ValueError):
    """Stable bounded errors, with no native stderr or dataset code in API output."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


async def _bounded_read(stream: asyncio.StreamReader, limit: int, label: str) -> bytes:
    result = bytearray()
    while block := await stream.read(min(65536, limit + 1 - len(result))):
        result.extend(block)
        if len(result) > limit:
            raise ReaderError("output_limit", f"Reader {label} exceeds the output byte budget")
    return bytes(result)


async def read_parquet(
    raw: bytes,
    operation: Literal["index", "frames"],
    *,
    episode_index: int = 0,
    python: str | Path | None = None,
    timeout_seconds: float | None = None,
) -> list[dict]:
    """Decode already bounded Hub bytes in a disposable process, on POSIX or Windows.

    The executable is trusted operator configuration, never dataset metadata. It
    defaults to the same source-checkout CPU reader used by local previews. Unlike
    local previews there are no untrusted filesystem paths to open. Cancellation
    and deadlines kill and reap the reader before releasing the explorer slot.
    Declared decode limits are not an OS-enforced native-memory sandbox.
    """
    timeout = READER_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
        raise ReaderError("invalid_request", "Reader timeout must be finite and in (0, 30]")
    if operation not in {"index", "frames"} or type(episode_index) is not int or episode_index < 0:
        raise ReaderError("invalid_request", "Unsupported Parquet preview request")
    input_limit = 8 * 1024 * 1024 if operation == "index" else MAX_INPUT_BYTES
    if not isinstance(raw, bytes) or len(raw) > input_limit:
        raise ReaderError("input_limit", "Dataset file exceeds the preview input size limit")
    executable = (
        Path(python)
        if python is not None
        else Path(os.environ.get("FIREBIRD_CPU_READER_PYTHON") or reader_python())
    )
    process = None
    tasks: list[asyncio.Task] = []
    try:
        async with asyncio.timeout(timeout):
            process = await asyncio.create_subprocess_exec(
                str(executable),
                "-I",
                "-B",
                str(WORKER_SCRIPT.resolve()),
                operation,
                str(episode_index),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def send() -> None:
                try:
                    process.stdin.write(raw)
                    await process.stdin.drain()
                except BrokenPipeError, ConnectionResetError:
                    # A failed child is handled by its bounded result and exit code.
                    pass
                finally:
                    process.stdin.close()
                    try:
                        await process.stdin.wait_closed()
                    except BrokenPipeError, ConnectionResetError:
                        pass

            tasks = [
                asyncio.create_task(send()),
                asyncio.create_task(_bounded_read(process.stdout, MAX_OUTPUT_BYTES, "response")),
                asyncio.create_task(_bounded_read(process.stderr, MAX_STDERR_BYTES, "diagnostic")),
                asyncio.create_task(process.wait()),
            ]
            _, output, _, returncode = await asyncio.gather(*tasks)
    except TimeoutError as exc:
        raise ReaderError("timeout", "Parquet preview exceeded its wall-clock budget") from exc
    except OSError as exc:
        raise ReaderError(
            "reader_unavailable", "Install or configure the isolated CPU reader"
        ) from exc
    finally:

        async def cleanup() -> None:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            # Stop competing readers before draining the killed process. Waiting
            # on PIPEs without draining can deadlock on paused transports.
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if process is not None:

                async def discard(stream):
                    while await stream.read(65536):
                        pass

                await asyncio.gather(
                    discard(process.stdout), discard(process.stderr), process.wait()
                )

        # A second cancellation during shutdown must not abandon the child.
        cleanup_task = asyncio.create_task(cleanup())
        cancelled = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancelled = True
        cleanup_task.result()
        if cancelled:
            raise asyncio.CancelledError
    if returncode:
        raise ReaderError("reader_failed", "Isolated Parquet reader exited unsuccessfully")
    try:
        response = json.loads(output)
        if not isinstance(response, dict):
            raise ValueError("Invalid response")
        if "error" in response:
            if not isinstance(response["error"], str):
                raise ValueError("Invalid error")
            raise ReaderError("invalid_parquet", response["error"][:300])
        rows = response["rows"]
        if response.get("schema_version") != 1 or not isinstance(rows, list):
            raise ValueError("Invalid response")
        if len(rows) > (100_000 if operation == "index" else 5) or not all(
            isinstance(row, dict) for row in rows
        ):
            raise ValueError("Invalid rows")
        return rows
    except ReaderError:
        raise
    except (ValueError, KeyError, UnicodeError) as exc:
        raise ReaderError(
            "reader_failed", "Isolated Parquet reader returned an invalid result"
        ) from exc
