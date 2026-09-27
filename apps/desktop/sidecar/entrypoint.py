"""Fixed desktop child modes. This is not a general Python execution bridge."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import socket
import sys
import threading
from pathlib import Path

from sidecar_resources import absolute_directory, json_object, load_resources, read_regular

MAX_CONTROL = 4096
START_TIMEOUT = 10
READY_TIMEOUT = 60


def control_record(raw: bytes, command: str, nonce: str | None = None) -> str:
    if len(raw) > MAX_CONTROL:
        raise ValueError("Control message is too large")
    value = json_object(raw)
    if (
        set(value) != {"schema_version", "command", "nonce"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["command"] != command
        or not isinstance(value["nonce"], str)
        or not re.fullmatch(r"[a-f0-9]{32}", value["nonce"])
        or (nonce is not None and value["nonce"] != nonce)
    ):
        raise ValueError("Invalid private control message")
    return value["nonce"]


def emit(value: dict) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_CONTROL:
        raise ValueError("Control response is too large")
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


def control_reader(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    """At most two records. Raw reads avoid a buffered-stdin lock at interpreter exit."""

    def deliver(value):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, value)
        except RuntimeError:  # The owned server already closed its event loop.
            pass

    pending = bytearray()
    records = 0
    try:
        while records < 2:
            chunk = os.read(sys.stdin.fileno(), min(1024, MAX_CONTROL + 1 - len(pending)))
            if not chunk:
                deliver(ValueError("Incomplete control message") if pending else None)
                return
            pending.extend(chunk)
            while b"\n" in pending:
                raw, _, rest = pending.partition(b"\n")
                if len(raw) > MAX_CONTROL:
                    raise ValueError("Control message is too large")
                deliver(bytes(raw))
                records += 1
                pending = bytearray(rest)
                if records == 2:
                    return
            if len(pending) > MAX_CONTROL:
                raise ValueError("Control message is too large")
    except (OSError, ValueError) as exc:
        deliver(exc)


def shutdown_reason(message, nonce: str) -> str:
    if message is None:
        return "parent_eof"
    if isinstance(message, Exception):
        raise message
    control_record(message, "shutdown", nonce)
    return "shutdown"


async def early_shutdown(queue: asyncio.Queue, nonce: str) -> bool:
    # Deliver a pipe-close callback queued while synchronous imports/validation ran.
    await asyncio.sleep(0)
    if queue.empty():
        return False
    reason = shutdown_reason(queue.get_nowait(), nonce)
    emit({"schema_version": 1, "event": "stopped", "nonce": nonce, "reason": reason})
    return True


def clear_optional_environment() -> None:
    """Child-local only: optional routes must not activate from inherited app settings."""
    for name in tuple(os.environ):
        if name.startswith("FIREBIRD_") or name == "GEMINI_API_KEY":
            os.environ.pop(name, None)


async def serve(args: argparse.Namespace) -> int:
    clear_optional_environment()
    queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    thread = threading.Thread(
        target=control_reader, args=(asyncio.get_running_loop(), queue), daemon=True
    )
    thread.start()
    first = await asyncio.wait_for(queue.get(), START_TIMEOUT)
    if first is None:
        return 0
    if isinstance(first, Exception):
        raise first
    nonce = control_record(first, "start")
    if await early_shutdown(queue, nonce):
        return 0
    resources = load_resources(args.resources)
    data_dir = absolute_directory(args.data_dir, existing=False)
    if (
        resources.root == data_dir
        or resources.root in data_dir.parents
        or data_dir in resources.root.parents
    ):
        raise ValueError("Application data and bundled resources must be disjoint")
    local_root = absolute_directory(args.local_root) if args.local_root else None
    # Explicit paths only. Do not read settings from the current working directory or environment.
    import uvicorn
    from vla_platform import __version__
    from vla_platform.api import create_app
    from vla_platform.settings import Settings

    if resources.app_version != __version__:
        raise ValueError("Bundled resources and backend version disagree")
    if await early_shutdown(queue, nonce):
        return 0
    app = create_app(Settings(data_dir=data_dir, static_dir=resources.web, local_root=local_root))
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
        access_log=False,
        log_level="warning",
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)

    async def run_server() -> None:
        try:
            await server.serve(sockets=[sock])
        except SystemExit as exc:
            # Uvicorn uses SystemExit on failed lifespan startup. Convert it inside
            # this task so asyncio cannot bypass the owned socket/control cleanup.
            raise RuntimeError("Backend lifespan startup failed") from exc

    control = asyncio.create_task(queue.get())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    running = None
    reason = "shutdown"
    error = None
    ready = False
    try:
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
        running = asyncio.create_task(run_server())
        deadline = asyncio.get_running_loop().time() + READY_TIMEOUT
        while not server.started and not running.done() and not control.done():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Backend did not become ready within its startup deadline")
            await asyncio.sleep(0.02)
        # A parent that exited while imports/startup ran must never receive a late ready record.
        if not control.done() and not running.done() and server.started:
            emit(
                {
                    "schema_version": 1,
                    "event": "ready",
                    "nonce": nonce,
                    "host": "127.0.0.1",
                    "port": sock.getsockname()[1],
                    "build_id": resources.build_id,
                    "app_version": __version__,
                    "resources_sha256": resources.manifest_sha256,
                    "workspace_id": hashlib.sha256(os.fsencode(data_dir)).hexdigest(),
                }
            )
            ready = True
        if not running.done() and not control.done():
            await asyncio.wait({running, control}, return_when=asyncio.FIRST_COMPLETED)
        if control.done():
            reason = shutdown_reason(control.result(), nonce)
        elif not ready:
            raise RuntimeError("Backend exited before readiness")
    except (ValueError, OSError, TimeoutError, RuntimeError) as exc:
        error = exc
    finally:
        server.should_exit = True
        # Retain the existing application's job cancellation/owner-lock lifecycle. The future
        # native parent owns any final hard-stop deadline; do not falsely report cleanup here.
        try:
            if running is not None:
                await running
        finally:
            control.cancel()
            await asyncio.gather(control, return_exceptions=True)
            sock.close()
    if error:
        raise error
    emit({"schema_version": 1, "event": "stopped", "nonce": nonce, "reason": reason})
    return 0


def intake_worker(request_path: Path, result_path: Path) -> int:
    if (
        not request_path.is_absolute()
        or not result_path.is_absolute()
        or request_path.parent != result_path.parent
        or request_path.name != "request.json"
        or result_path.name != "result.json"
    ):
        raise ValueError("Intake requires the fixed absolute job request/result paths")
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("Intake result already exists")
    # Bound and type-check the app-owned envelope before invoking the identical source worker.
    from vla_platform.contracts import WorkerRequest
    from vla_platform.datasets.worker import run, terminate_worker

    WorkerRequest.model_validate_json(read_regular(request_path, 1024 * 1024))
    signal.signal(signal.SIGTERM, terminate_worker)
    asyncio.run(run(request_path, result_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixed OPEN JENSEN desktop child")
    modes = parser.add_subparsers(dest="mode", required=True)
    child = modes.add_parser("serve")
    child.add_argument("--resources", type=Path, required=True)
    child.add_argument("--data-dir", type=Path, required=True)
    child.add_argument("--local-root", type=Path)
    worker = modes.add_parser("intake-worker")
    worker.add_argument("request", type=Path)
    worker.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "serve" and any(
        path is not None and not path.is_absolute()
        for path in (args.resources, args.data_dir, args.local_root)
    ):
        parser.error("Desktop paths must be absolute")
    try:
        return (
            asyncio.run(serve(args))
            if args.mode == "serve"
            else intake_worker(args.request, args.result)
        )
    except (ValueError, OSError, TimeoutError, RuntimeError) as exc:
        # Private stderr is diagnostic only; stdout never contains arbitrary exceptions/paths.
        print(f"Desktop child failed: {type(exc).__name__}: {str(exc)[:1000]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
