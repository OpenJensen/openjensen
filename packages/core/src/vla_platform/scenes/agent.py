"""Ephemeral read-only Codex roles. No generated Python or shell is executed."""

import asyncio
import json
import os
import shutil
import signal
from pathlib import Path

from vla_platform.api_connections import ApiConnections


class SceneError(ValueError):
    pass


def codex_binary():
    configured = os.environ.get("JENSEN_CODEX_BIN")
    return configured or shutil.which("codex") or str(Path.home() / ".local/bin/codex")


async def command(args, *, cwd, prompt=None, env=None, timeout=180, limit=2 * 1024**2):
    process = await asyncio.create_subprocess_exec(
        *map(str, args),
        cwd=cwd,
        env=env,
        start_new_session=True,
        stdin=asyncio.subprocess.PIPE if prompt is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def output():
        chunks = []
        size = 0
        while part := await process.stdout.read(65536):
            size += len(part)
            if size > limit:
                raise SceneError("Worker output exceeded the configured limit.")
            chunks.append(part)
        await process.wait()
        return b"".join(chunks)

    try:
        if prompt is not None:
            process.stdin.write(prompt.encode())
            await process.stdin.drain()
            process.stdin.close()
        async with asyncio.timeout(timeout):
            raw = await output()
        if process.returncode:
            raise SceneError(
                "Worker did not complete. Check its runtime or provider configuration."
            )
        return raw
    except TimeoutError:
        raise SceneError("Worker exceeded its time budget; no replacement was launched.") from None
    finally:
        # Kill the entire owned group, including any descendants after normal exit.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def run_role(directory, role, schema, prompt, frames, data_dir):
    schema_path = directory / f"{role}.schema.json"
    answer = directory / f"{role}.json"
    schema_path.write_text(json.dumps(schema.model_json_schema()))
    # Isolated config prevents unrelated MCP servers, hooks and user instructions loading.
    home = directory / ".codex-runtime"
    home.mkdir(mode=0o700, exist_ok=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            "PATH",
            "HOME",
            "LANG",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "NO_PROXY",
        }
    }
    env["CODEX_HOME"] = str(home)
    key = ApiConnections(data_dir).key("openai")
    if key:
        env["CODEX_API_KEY"] = key
        # API key is only passed in the child environment, never command arguments.
        (home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": key}))
        (home / "auth.json").chmod(0o600)
    else:
        auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        if not auth.is_file():
            raise SceneError(
                "Save an OpenAI key in Cloud & APIs or sign in to Codex on the service host."
            )
        shutil.copyfile(auth, home / "auth.json")
        (home / "auth.json").chmod(0o600)
    args = [
        codex_binary(),
        "--ask-for-approval",
        "never",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--json",
        "-c",
        'model_reasoning_effort="medium"',
        "--output-schema",
        schema_path,
        "--output-last-message",
        answer,
    ]
    model = os.environ.get("JENSEN_SCENE_MODEL")
    if model:
        args += ["--model", model]
    for frame in frames:
        args += ["--image", frame]
    args += ["-"]
    try:
        raw = await command(args, cwd=directory, prompt=prompt, env=env, timeout=240)
        if answer.is_symlink() or not answer.is_file() or answer.stat().st_size > 65536:
            raise SceneError("Agent did not produce a bounded structured response.")
        result = schema.model_validate_json(answer.read_bytes())
        usage = {}
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "turn.completed":
                for k, v in event.get("usage", {}).items():
                    if isinstance(v, int) and v >= 0:
                        usage[k] = usage.get(k, 0) + v
        return result, usage
    finally:
        # Never include the short-lived auth copy in the scene bundle or saved evidence.
        (home / "auth.json").unlink(missing_ok=True)
