"""Process-isolated planner transport. No executor credentials enter its environment."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class RuntimeError(ValueError):
    pass


def _invalid_constant(_value):
    raise RuntimeError("Provider returned invalid numeric data; no motion admitted")


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise RuntimeError("Provider returned duplicate properties; no motion admitted")
        result[key] = value
    return result


async def run(profile, payload):
    root = Path(__file__).resolve().parents[5] / "workers/agent_control/src"
    if not (root / "jensen_agent_control/__main__.py").is_file():
        raise RuntimeError("Agent planning runtime is unavailable in this application release")
    if profile.get("provider") not in {"mock", "openai_responses"}:
        raise RuntimeError("Unsupported provider transport; no motion admitted")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT"}
    }
    environment.update(PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    if profile.get("api_key") and profile["provider"] == "openai_responses":
        environment["OPENJENSEN_AGENT_API_KEY"] = profile["api_key"]
    value = dict(
        payload,
        provider=profile["provider"],
        model=profile["model"],
        reasoning_effort=profile["reasoning_effort"],
    )
    try:
        request = json.dumps(value, allow_nan=False).encode()
    except ValueError, TypeError:
        raise RuntimeError("Provider input is invalid; no motion admitted") from None
    if len(request) > MAX_REQUEST_BYTES:
        raise RuntimeError("Provider input exceeds its size limit; no motion admitted")
    timeout = payload.get("timeout_seconds", 30)
    if type(timeout) not in {int, float} or not 0 < timeout <= 120:
        raise RuntimeError("Provider timeout is invalid; no motion admitted")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "jensen_agent_control",
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )

    async def send():
        try:
            process.stdin.write(request)
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
        except BrokenPipeError, ConnectionResetError:
            pass

    async def receive():
        chunks, size = [], 0
        while chunk := await process.stdout.read(65536):
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise RuntimeError("Provider response exceeded limits; no motion admitted")
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        async with asyncio.timeout(timeout + 2):
            _sent, raw = await asyncio.gather(send(), receive())
            await process.wait()
        try:
            result = json.loads(
                raw, parse_constant=_invalid_constant, object_pairs_hook=_unique_object
            )
        except ValueError, TypeError, UnicodeDecodeError:
            raise RuntimeError("Provider returned invalid data; no motion admitted") from None
        if isinstance(result, dict) and result.get("status") == "error":
            # Preserve independently bounded usage, never provider error text.
            usage = result.get("usage")
            safe = {"status": "error"}
            if isinstance(usage, dict) and all(
                type(usage.get(k)) is int and 0 <= usage[k] <= 1000000
                for k in ("input_tokens", "output_tokens")
            ):
                safe["usage"] = {k: usage[k] for k in ("input_tokens", "output_tokens")}
                return safe
            raise RuntimeError(
                "Provider request failed; no motion admitted. Billing may be unverified."
            )
        if (
            process.returncode != 0
            or not isinstance(result, dict)
            or result.get("status") not in {"proposed", "complete", "checked", "assessed"}
        ):
            raise RuntimeError(
                "Provider request failed; no motion admitted. Billing may be unverified."
            )
        return result
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                async with asyncio.timeout(2):
                    await process.wait()
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
