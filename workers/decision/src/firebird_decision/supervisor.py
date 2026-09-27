"""One owned child, bounded pipes/deadline, no implicit retry or arbitrary command API."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import threading
import time

from .contracts import ERROR_MESSAGES, MAX_RESPONSE_BYTES, DecisionError, parse_json


def run_owned(command, *, timeout: float, env=None):
    if os.name != "posix":
        raise DecisionError(
            "The decision subprocess supervisor currently supports Linux and macOS."
        )
    selector = selectors.DefaultSelector()
    process = None
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    old_handlers = {}
    interrupted = []
    cleaning = False

    def defer(signum, frame):
        if not cleaning:
            interrupted.append(signum)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, defer)
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        for name in outputs:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            if interrupted:
                raise DecisionError(
                    "Decision scoring interrupted; the owned scoring process was stopped."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DecisionError(
                    "Decision scoring exceeded its deadline; no result was accepted."
                )
            for key, _ in selector.select(min(remaining, 0.05)):
                chunk = os.read(key.fd, 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                else:
                    outputs[key.data].extend(chunk)
                    if len(outputs[key.data]) > MAX_RESPONSE_BYTES:
                        raise DecisionError(
                            "Decision worker output exceeded its bounded response limit."
                        )
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if process.returncode:
            # Only a known code crosses this boundary; never reflect arbitrary child error text.
            if process.returncode == 2 and not outputs["stdout"]:
                try:
                    error = parse_json(bytes(outputs["stderr"]))
                except DecisionError:
                    error = None
                if (
                    isinstance(error, dict)
                    and set(error) == {"schema_version", "code", "error"}
                    and type(error["schema_version"]) is int
                    and error["schema_version"] == 1
                    and isinstance(error["code"], str)
                    and error["code"] in ERROR_MESSAGES
                ):
                    raise DecisionError(ERROR_MESSAGES[error["code"]], error["code"])
            # Do not reflect unexpected tracebacks, user input, or local paths.
            raise DecisionError(
                "Decision worker failed; verify the pinned installation, inputs, and model files."
            )
        return bytes(outputs["stdout"])
    finally:
        cleaning = True  # repeated signals cannot interrupt kill/reap or handler restoration
        try:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                for name in outputs:
                    getattr(process, name).close()
        finally:
            selector.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
