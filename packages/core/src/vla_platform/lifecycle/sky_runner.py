"""Bounded per-job GCP training through SkyPilot CLI, including artifact return.

No global SkyPilot configuration or provider credential files are modified.
Launch uses a private task bundle and verifies the server billing project.
Results travel through supported ``sky logs --sync-down`` before ``sky down``.
See https://docs.skypilot.co/en/latest/reference/cli.html and sky_bootstrap.py.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import tarfile
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from uuid import uuid4

from vla_platform import cloud_compute_catalog

MAX_ARTIFACT_BYTES = 20 * 1024**3
MAX_ARTIFACT_FILES = 10000
MAX_PART_BYTES = 32 * 1024**2
MAX_LOG_BYTES = 5 * 1024**2
CHECKPOINT_POLL_SECONDS = 15
STATE_NAME = "sky-state.json"
Event = Callable[[str], Awaitable[None]]
CLUSTER_PATTERN = r"fb-[a-f0-9]{12}-[a-f0-9]{8}"
_RUN_SECRETS: ContextVar[tuple[str, ...]] = ContextVar("firebird_run_secrets", default=())


def validate_endpoint(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 2048:
        raise ValueError("Invalid SkyPilot server endpoint")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and not any(character.isspace() for character in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Invalid SkyPilot server endpoint")
    return value.rstrip("/")


def executable() -> str:
    path = shutil.which("sky")
    fallback = Path.home() / ".local" / "bin" / "sky"
    if path:
        return path
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    raise ValueError("Install SkyPilot with GCP support on the application server.")


def worker_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "workers" / "smolvla_qlora"
        if (candidate / "src" / "firebird_vla" / "application.py").is_file():
            return candidate
    raise ValueError("The bundled SmolVLA worker sources are missing on the application server.")


def _write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _copy_tree(source: Path, target: Path, *, maximum: int):
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Training input must be a regular directory")
    total, count = 0, 0
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Training input symlinks are not permitted")
        if path.is_file():
            total += path.stat().st_size
            count += 1
        elif not path.is_dir():
            raise ValueError("Training input contains an unsupported file")
        if total > maximum or count > MAX_ARTIFACT_FILES:
            raise ValueError("Training input exceeds the supported transfer limit")
    shutil.copytree(source, target)


def effective_training_recipe(payload: dict) -> dict:
    """Resolve only worker-selection metadata; the remote checkpoint owns resume state."""
    recipe = payload.get("parameters", {}).get("training")
    artifact = payload.get("artifact") or {}
    # Admission normalizes new resume requests, but persisted queued requests may
    # still contain caller overrides. A checkpoint always owns worker selection.
    if not artifact and not payload.get("resume_checkpoint"):
        return recipe or {}
    candidates = []
    if artifact.get("path"):
        candidates.append(Path(artifact["path"]) / "checkpoint" / "recipe.json")
    if payload.get("resume_checkpoint"):
        candidates.append(Path(payload["resume_checkpoint"]) / "recipe.json")
    for path in candidates:
        if path.is_file():
            if path.stat().st_size > 1024**2:
                raise ValueError("Checkpoint recipe exceeds the supported size")
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("Invalid checkpoint recipe")
            return value
    base = artifact.get("metadata", {}).get("base_model", {})
    if isinstance(base, dict) and base.get("repository"):
        return {"model_id": base["repository"], "model_revision": base.get("revision")}
    return {}


def prepare(payload: dict, stage_dir: Path, target: dict) -> tuple[Path, dict]:
    """Create a reviewable fixed task; payload paths never become shell commands."""
    project = target.get("project_id", "")
    region = target.get("region", "")
    accelerator = target.get("accelerator", "")
    count = target.get("gpu_count", 1)
    disk = target.get("disk_size_gb", 100)
    idle = target.get("idle_minutes", 10)
    endpoint = validate_endpoint(target.get("sky_api_endpoint"))
    workspace = target.get("workspace", "default")
    if not isinstance(workspace, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", workspace):
        raise ValueError("Invalid SkyPilot workspace")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project):
        raise ValueError("Invalid Google Cloud project")
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,62}[0-9]", region):
        raise ValueError("Invalid Google Cloud region")
    if (
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", accelerator)
        or type(count) is not int
        or count != 1
    ):
        raise ValueError("Cloud training requires exactly one registered GPU")
    if type(disk) is not int or not 100 <= disk <= 2048:
        raise ValueError("Cloud disk must be between 100 and 2048 GiB")
    if type(idle) is not int or not 1 <= idle <= 60:
        raise ValueError("Cloud idle teardown must be between 1 and 60 minutes")
    if payload.get("operation") not in {"policy.finetune", "policy.quantize"}:
        raise ValueError("SkyPilot supports fine-tuning and quantization")
    recipe = effective_training_recipe(payload)
    steps = recipe.get("steps", 20000)
    every = recipe.get("save_every", max(1, (steps + 4) // 5) if type(steps) is int else 4000)
    if type(steps) is not int or type(every) is not int or steps < 1 or every < 1:
        raise ValueError("Invalid checkpoint save cadence")
    if (steps + every - 1) // every > 10000:
        raise ValueError("Choose a save interval that produces at most 10000 local checkpoints")
    timeout = payload.get("parameters", {}).get("timeout_seconds", 7200)
    if type(timeout) is not int or not 30 <= timeout <= 86400:
        raise ValueError("Invalid remote training deadline")
    if not isinstance(payload.get("job_id"), str):
        raise ValueError("Cloud training requires an application job identity")
    stage_dir.mkdir(parents=True, exist_ok=True)
    state_path = stage_dir / STATE_NAME
    if state_path.exists():
        raise ValueError("This job already has a SkyPilot dispatch; create a new run")
    cluster = (
        "fb-" + hashlib.sha256(payload["job_id"].encode()).hexdigest()[:12] + "-" + uuid4().hex[:8]
    )
    task_name = "firebird-training"
    bundle = stage_dir / "sky-bundle"
    bundle.mkdir(exist_ok=False)
    root = worker_root()
    (bundle / "worker").mkdir()
    _copy_tree(root / "src", bundle / "worker" / "src", maximum=20 * 1024**2)
    for name in ("pyproject.toml", "requirements-smolvla-linux.txt"):
        shutil.copyfile(root / name, bundle / "worker" / name)
    shutil.copyfile(Path(__file__).with_name("sky_bootstrap.py"), bundle / "bootstrap.py")
    shutil.copyfile(Path(__file__).with_name("cloud_storage.py"), bundle / "cloud_storage.py")
    remote_payload = json.loads(json.dumps(payload))
    # The bundled worker does not consume runtime commands, mounts or env.
    remote_payload["runtime"] = {"provider": "gcp", "device": "cuda"}
    remote_payload["output_dir"] = "output"
    remote_payload["source"] = None
    (bundle / "inputs").mkdir()
    if payload.get("artifact"):
        _copy_tree(
            Path(payload["artifact"]["path"]),
            bundle / "inputs" / "artifact",
            maximum=MAX_ARTIFACT_BYTES,
        )
        remote_payload["artifact"]["path"] = "inputs/artifact"
    if payload.get("resume_checkpoint"):
        _copy_tree(
            Path(payload["resume_checkpoint"]),
            bundle / "inputs" / "resume",
            maximum=MAX_ARTIFACT_BYTES,
        )
        remote_payload["resume_checkpoint"] = "inputs/resume"
    from vla_platform.lifecycle.native_profiles import native_profile_for_recipe, native_requirement
    from vla_platform.lifecycle.training_catalog import training_model_for_recipe

    profile = (
        native_profile_for_recipe(recipe) if payload["operation"] == "policy.finetune" else None
    )
    psi_training = (
        payload["operation"] == "policy.finetune"
        and training_model_for_recipe(recipe).backend == "psi0"
    )
    worker_module = "firebird_vla.lerobot_application" if profile else "firebird_vla.application"
    extra_setup = []
    if psi_training:
        from vla_platform.lifecycle.sky_psi import WORKER_MODULE, stage_psi

        worker_module = WORKER_MODULE
        extra_setup = stage_psi(bundle, root)
    elif payload["operation"] == "policy.quantize":
        from vla_platform.lifecycle.sky_quantization import WORKER_MODULE, stage_quantization

        worker_module = WORKER_MODULE
        extra_setup = stage_quantization(bundle, root)
    dependency_setup = []
    if profile:
        # The native upstream also uses CUDA 12.8. PyPI's default torch wheel
        # may require newer drivers than SkyPilot's image, so preserve the
        # CUDA build as well as the Python API versions throughout resolution.
        (bundle / "native-cu128-constraints.txt").write_text(
            "torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n"
        )
        dependency_setup = [
            "python3 -m uv pip install --python .venv/bin/python "
            "--index-url https://download.pytorch.org/whl/cu128 "
            "torch==2.11.0+cu128 torchvision==0.26.0+cu128",
            "python3 -m uv pip install --python .venv/bin/python "
            "--constraint native-cu128-constraints.txt "
            "--index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match "
            + shlex.quote(native_requirement(profile)),
            '.venv/bin/python -c "import torch, torchvision; '
            "assert torch.__version__ == '2.11.0+cu128'; "
            "assert torchvision.__version__ == '0.26.0+cu128'; "
            "assert torch.version.cuda == '12.8'\"",
        ]
    elif not psi_training:
        dependency_setup = [
            "python3 -m uv pip install --python .venv/bin/python "
            "-r worker/requirements-smolvla-linux.txt"
        ]
    storage_prefix = target.get("storage_uri")
    _write_json(bundle / "request.json", remote_payload)
    _write_json(
        bundle / "dispatch.json",
        {
            "task_name": task_name,
            "timeout_seconds": timeout,
            "worker_module": worker_module,
            "storage_prefix": storage_prefix,
        },
    )
    # JSON is valid YAML, so no YAML dependency is needed in the application.
    task_path = stage_dir / "sky-task.yaml"
    resources = {
        "infra": f"gcp/{region}",
        "accelerators": f"{accelerator}:1",
        "cpus": "2+",
        "memory": "32+"
        if psi_training
        else "16+"
        if payload["operation"] == "policy.quantize"
        else "4+",
        "disk_size": disk,
        "use_spot": False,
    }
    instance = target.get("instance_type")
    if instance:
        if not isinstance(instance, str) or not re.fullmatch(r"[a-z][a-z0-9-]{2,79}", instance):
            raise ValueError("Invalid Google Cloud machine type")
        resources["instance_type"] = instance
    _write_json(
        task_path,
        {
            "name": task_name,
            "resources": resources,
            "workdir": str(bundle.resolve()),
            "envs": {
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "WANDB_MODE": "disabled",
                **({"FIREBIRD_PSI_ROOT": "psi"} if psi_training else {}),
            },
            "setup": "\n".join(
                [
                    "set -euo pipefail",
                    "sudo apt-get update -qq",
                    "sudo apt-get install -y ffmpeg libgl1 libglib2.0-0",
                    "python3 -m pip install --user uv==0.12.19",
                    "python3 -m uv venv --python " + ("3.12" if profile else "3.11") + " .venv",
                    *dependency_setup,
                    "python3 -m uv pip install --python .venv/bin/python "
                    "google-cloud-storage==3.4.1 setuptools==80.10.2",
                    "python3 -m uv pip install --python .venv/bin/python "
                    "--no-deps --no-build-isolation -e worker",
                ]
                + extra_setup
            ),
            "run": ".venv/bin/python bootstrap.py",
        },
    )
    config_path = stage_dir / "sky-config.yaml"
    _write_json(
        config_path,
        {
            "active_workspace": workspace,
            "api_server": {"endpoint": endpoint},
        },
    )
    state = {
        "version": 1,
        "job_id": payload["job_id"],
        "cluster": cluster,
        "target": target,
        "phase": "prepared",
        "request_id": None,
        "cleanup_error": None,
        "config_path": str(config_path.resolve()),
    }
    _write_json(state_path, state)
    return task_path, state


async def _stop_process(process):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        pass


async def _safe_output(stream, secrets: tuple[str, ...] = ()):
    values = [value.encode() for value in secrets if value]
    pattern = re.compile(b"|".join(re.escape(value) for value in values)) if values else None
    carry = b""
    keep = max((len(value) for value in values), default=1) - 1
    while raw := await stream.read(8192):
        if pattern is None:
            yield raw
            continue
        buffer = carry + raw
        boundary = max(0, len(buffer) - keep)
        for match in pattern.finditer(buffer):
            if match.start() < boundary < match.end():
                boundary = match.end()
        visible, carry = buffer[:boundary], buffer[boundary:]
        if visible:
            yield pattern.sub(b"[REDACTED]", visible)
    if carry:
        yield pattern.sub(b"[REDACTED]", carry)


async def _command(
    argv: list[str],
    stage_dir: Path,
    *,
    timeout: int,
    on_event: Event | None = None,
    on_line: Event | None = None,
    env_overrides: dict[str, str] | None = None,
    secrets: tuple[str, ...] = (),
) -> tuple[int, str]:
    """Stream bounded diagnostics, never use a shell on the application host."""
    state = json.loads((stage_dir / STATE_NAME).read_text())
    endpoint = validate_endpoint(state.get("target", {}).get("sky_api_endpoint"))
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=stage_dir,
        start_new_session=os.name == "posix",
        env={
            **os.environ,
            **(env_overrides or {}),
            "NO_COLOR": "1",
            "CLICOLOR": "0",
            "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
            "SKYPILOT_API_SERVER_ENDPOINT": endpoint,
        },
    )
    captured = bytearray()
    pending = ""
    log_path = stage_dir / "worker.log"
    written = log_path.stat().st_size if log_path.exists() else 0
    try:
        async with asyncio.timeout(timeout):
            with log_path.open("ab") as log:
                async for chunk in _safe_output(process.stdout, (*_RUN_SECRETS.get(), *secrets)):
                    if len(captured) < MAX_LOG_BYTES:
                        captured.extend(chunk[: MAX_LOG_BYTES - len(captured)])
                    if written < MAX_LOG_BYTES:
                        part = chunk[: MAX_LOG_BYTES - written]
                        log.write(part)
                        log.flush()
                        written += len(part)
                    if on_event or on_line:
                        pending += chunk.decode("utf-8", errors="replace")
                        lines = pending.split("\n")
                        pending = lines.pop()[-16384:]
                        for line in lines:
                            if on_line:
                                await on_line(line)
                            if not on_event:
                                continue
                            # SkyPilot may prefix job-log lines. Progress content is
                            # parsed strictly and never treated as executable text.
                            start = line.find("{")
                            if start < 0:
                                continue
                            try:
                                item = json.loads(line[start:])
                            except ValueError:
                                continue
                            if isinstance(item, dict) and (
                                type(item.get("step")) is int or isinstance(item.get("phase"), str)
                            ):
                                prefix = (
                                    "_quantization:"
                                    if item.get("operation") == "policy.quantize"
                                    else "_telemetry:"
                                )
                                await on_event(prefix + json.dumps(item))
                                if item.get("checkpoint_saved") and type(item.get("step")) is int:
                                    await on_event(f"_checkpoint_ready:{item['step']}")
                await process.wait()
        return process.returncode, captured.decode("utf-8", errors="replace")
    finally:
        await _stop_process(process)


def _args(state: dict) -> list[str]:
    return ["--config", state["config_path"]]


def verify_checkpoint(directory: Path, expected_sha: str | None = None) -> tuple[int, str]:
    manifest_path = directory / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > 4 * 1024**2
    ):
        raise ValueError("Checkpoint manifest is missing or too large")
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha and manifest_sha != expected_sha:
        raise ValueError("Checkpoint manifest checksum mismatch")
    manifest = json.loads(raw)
    files = manifest.get("files")
    step = manifest.get("step")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(files, dict)
        or not files
        or len(files) > MAX_ARTIFACT_FILES
        or type(step) is not int
        or step < 1
    ):
        raise ValueError("Invalid checkpoint manifest")
    actual, size = set(), 0
    for path in directory.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("Unsafe checkpoint file")
        if not path.is_file() or path == manifest_path:
            continue
        name = path.relative_to(directory).as_posix()
        actual.add(name)
        size += path.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            raise ValueError("Checkpoint exceeds the transfer limit")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != files.get(name):
                raise ValueError("Checkpoint file checksum mismatch")
    if actual != set(files):
        raise ValueError("Checkpoint file inventory differs from manifest")
    return step, manifest_sha


def _advance_latest(training: Path, name: str, step: int):
    latest = training / "latest.json"
    prior = json.loads(latest.read_text()) if latest.exists() else {}
    if prior.get("step", -1) < step:
        _write_json(latest, {"checkpoint": name, "step": step})


def install_checkpoint(stage_dir: Path, cluster: str, item: dict) -> int | None:
    name = item.get("name", "")
    if not re.fullmatch(r"checkpoint-\d{6,}", name):
        raise ValueError("Invalid checkpoint transfer identity")
    descriptor = Path(item["descriptor"])
    expected_parent = (
        Path.home() / "sky_logs" / cluster / "1-firebird-training" / "checkpoint-snapshots" / name
    )
    if (
        descriptor.name != "firebird-output.json"
        or descriptor.parent != expected_parent
        or descriptor.is_symlink()
        or descriptor.stat().st_size > 256 * 1024
    ):
        raise ValueError("Invalid checkpoint transfer location")
    expected_sha = item.get("manifest_sha256", "")
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha):
        raise ValueError("Invalid checkpoint transfer manifest identity")
    training = stage_dir / "training"
    training.mkdir(exist_ok=True)
    destination = training / name
    created = not destination.exists()
    if created:
        with extracted_archive(descriptor, training) as extracted:
            step, _ = verify_checkpoint(extracted, expected_sha)
            if step != item.get("step") or step != int(name[11:]):
                raise ValueError("Checkpoint optimizer step mismatch")
            os.rename(extracted, destination)
    else:
        step, _ = verify_checkpoint(destination, expected_sha)
    _advance_latest(training, name, step)
    receipt_path = stage_dir / "sky-checkpoints.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipts[name] = expected_sha
    _write_json(receipt_path, receipts)
    # Only this run's downloaded transfer copy is discarded. Canonical local
    # checkpoints and SkyPilot's server cache are never pruned here.
    shutil.rmtree(descriptor.parent)
    return step if created else None


async def sync_checkpoints(sky: str, stage_dir: Path, state: dict, on_event: Event, *, final=False):
    python = cloud_compute_catalog.sky_python(sky)
    if state.get("target", {}).get("storage_uri"):

        async def received_cloud(line):
            if line.startswith("FIREBIRD_CLOUD_CHECKPOINT="):
                step = int(line.partition("=")[2])
                await on_event(f"Checkpoint {step} saved on Google Cloud")

        code, output = await _command(
            [
                python,
                str(Path(__file__).with_name("cloud_storage.py")),
                "sync",
                state["target"]["storage_uri"],
                str(stage_dir.resolve()),
            ],
            stage_dir,
            timeout=180,
            on_line=received_cloud,
        )
        if code:
            raise RuntimeError(
                "Cloud checkpoint metadata synchronization pending: " + output[-1000:]
            )
        if final:
            state["final_metadata_synced"] = True
            state["metadata_sync_error"] = None
            _write_json(stage_dir / STATE_NAME, state)
        return
    if not python:
        raise ValueError(
            "Could not locate the SkyPilot Python environment for checkpoint downloads"
        )
    receipt_path = stage_dir / "sky-checkpoints.json"
    if not receipt_path.exists():
        _write_json(receipt_path, {})

    async def received(line):
        if line.startswith("FIREBIRD_CHECKPOINT="):
            item = json.loads(line.partition("=")[2])
            install = asyncio.create_task(
                asyncio.to_thread(
                    install_checkpoint,
                    stage_dir,
                    state["cluster"],
                    item,
                )
            )
            try:
                step = await asyncio.shield(install)
            except asyncio.CancelledError:
                await install  # Finish the local atomic commit before cleanup starts.
                raise
            if step is not None:
                await on_event(f"Checkpoint {step} saved locally")

    code, _ = await _command(
        [
            python,
            str(Path(__file__).with_name("sky_checkpoint_sync.py")),
            state["cluster"],
            str(receipt_path.resolve()),
            "final" if final else "live",
        ],
        stage_dir,
        timeout=1800,
        on_line=received,
    )
    if code:
        raise RuntimeError("Checkpoint download pending")
    if final:
        return  # The cluster is about to be torn down; no retention task is needed.
    receipts = json.loads(receipt_path.read_text())
    ack_path = stage_dir / "sky-checkpoints-acknowledged.json"
    acknowledged = json.loads(ack_path.read_text()) if ack_path.exists() else {}
    pending = {key: value for key, value in receipts.items() if acknowledged.get(key) != value}
    for start in range(0, len(pending), 256):
        batch = dict(list(pending.items())[start : start + 256])
        encoded = base64.urlsafe_b64encode(json.dumps(batch).encode()).decode()
        try:
            ack_code, _ = await _command(
                [
                    sky,
                    "exec",
                    state["cluster"],
                    "--name",
                    "firebird-checkpoint-ack",
                    "--gpus",
                    "none",
                    "--cpus",
                    "0.1",
                    "--memory",
                    "0.1",
                    *_args(state),
                    "--",
                    ".venv/bin/python",
                    "-m",
                    "firebird_vla.snapshots",
                    encoded,
                ],
                stage_dir,
                timeout=120,
            )
        except OSError, TimeoutError:
            ack_code = 1
        if ack_code:
            await on_event("Checkpoints saved locally; cloud copy cleanup pending")
            break
        acknowledged.update(batch)
        _write_json(ack_path, acknowledged)


async def follow_training(sky: str, stage_dir: Path, state: dict, timeout: int, on_event: Event):
    wake = asyncio.Event()

    async def progress(message):
        if message.startswith("_checkpoint_ready:"):
            wake.set()
        else:
            await on_event(message)

    tail = asyncio.create_task(
        _command(
            [sky, "logs", state["cluster"], "1", "--follow", "--tail", "2000", *_args(state)],
            stage_dir,
            timeout=timeout,
            on_event=progress,
        )
    )
    pending_notice = False
    try:
        while not tail.done():
            waiter = asyncio.create_task(wake.wait())
            done, _ = await asyncio.wait(
                {tail, waiter},
                timeout=CHECKPOINT_POLL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            wake.clear()
            if tail in done:
                break
            try:
                await sync_checkpoints(sky, stage_dir, state, on_event)
                pending_notice = False
            except OSError, ValueError, RuntimeError, TimeoutError:
                if not pending_notice:
                    await on_event("Checkpoint download pending")
                    pending_notice = True
        await tail
    finally:
        if not tail.done():
            tail.cancel()
        await asyncio.gather(tail, return_exceptions=True)


@contextmanager
def extracted_archive(descriptor: Path, stage_dir: Path):
    expected = json.loads(descriptor.read_text())
    parts = expected.get("parts")
    limit = MAX_ARTIFACT_BYTES + MAX_ARTIFACT_FILES * 2048
    if (
        type(expected.get("size")) is not int
        or not 1 <= expected["size"] <= limit
        or not isinstance(parts, list)
        or not 1 <= len(parts) <= 1024
    ):
        raise ValueError("SkyPilot artifact archive exceeds its declared size")
    if shutil.disk_usage(stage_dir).free < expected["size"] * 2 + 1024**3:
        raise ValueError("Not enough local disk space to save the next checkpoint safely")
    with tempfile.TemporaryDirectory(prefix=".sky-extract-", dir=stage_dir) as temporary:
        destination = Path(temporary) / "output"
        destination.mkdir()
        archive_path = Path(temporary) / "download.tar"
        digest, total = hashlib.sha256(), 0
        with archive_path.open("xb") as output:
            for index, part in enumerate(parts):
                name = f"firebird-output.part-{index:05d}"
                if (
                    not isinstance(part, dict)
                    or part.get("name") != name
                    or type(part.get("size")) is not int
                    or not 1 <= part["size"] <= MAX_PART_BYTES
                ):
                    raise ValueError("Invalid SkyPilot artifact archive part")
                path = descriptor.parent / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size != part["size"]:
                    raise ValueError("Missing or changed SkyPilot artifact archive part")
                total += part["size"]
                if total > expected["size"]:
                    raise ValueError("SkyPilot artifact archive exceeds its declared size")
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
        if total != expected["size"] or digest.hexdigest() != expected.get("sha256"):
            raise ValueError("SkyPilot artifact archive checksum mismatch")
        seen, size = set(), 0
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or member.name in seen
                    or not path.parts
                ):
                    raise ValueError("Unsafe entry in SkyPilot artifact bundle")
                seen.add(member.name)
                size += member.size
                if size > MAX_ARTIFACT_BYTES or len(seen) > MAX_ARTIFACT_FILES:
                    raise ValueError("SkyPilot artifacts exceed the transfer limit")
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        yield destination


def collect(stage_dir: Path, cluster: str):
    logs_root = Path.home() / "sky_logs" / cluster
    candidates = list(logs_root.glob("*/final-output/firebird-output.json"))
    if not candidates:
        candidates = list(logs_root.glob("*/firebird-output.json"))
    if len(candidates) != 1:
        raise ValueError("SkyPilot did not return a unique training artifact bundle")
    descriptor = candidates[0]
    if (
        descriptor.is_symlink()
        or descriptor.stat().st_size > 256 * 1024
        or not descriptor.resolve().is_relative_to(logs_root.resolve())
    ):
        raise ValueError("Invalid SkyPilot artifact integrity record")
    with extracted_archive(descriptor, stage_dir) as destination:
        result_path = destination / "result.json"
        if not result_path.is_file() or result_path.stat().st_size > 4 * 1024**2:
            raise ValueError("Cloud worker did not produce a bounded result")
        result = json.loads(result_path.read_text())
        if result.get("artifact"):
            relative = PurePosixPath(result["artifact"]["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("Cloud artifact escaped its job directory")
            result["artifact"]["path"] = str(stage_dir.joinpath(*relative.parts).resolve())
            result_path.write_text(json.dumps(result))
        # Preserve app-owned request/config/state. Only known worker outputs may
        # be published, so a remote file can never replace dispatcher state.
        allowed = {"result.json", "bundle", "training", "recipe.json", "resource-preflight.json"}
        for path in destination.iterdir():
            if path.name not in allowed:
                continue
            target = stage_dir / path.name
            if path.name == "training":
                target.mkdir(exist_ok=True)
                for child in path.iterdir():
                    child_target = target / child.name
                    if re.fullmatch(r"checkpoint-\d{6,}", child.name):
                        step, sha = verify_checkpoint(child)
                        if step != int(child.name[11:]):
                            raise ValueError("Checkpoint optimizer step mismatch")
                        if child_target.exists():
                            verify_checkpoint(child_target, sha)
                        else:
                            shutil.move(str(child), child_target)
                        _advance_latest(target, child.name, step)
                    elif child.name != "latest.json" and child.is_file():
                        os.replace(child, child_target)
                continue
            if target.exists():
                raise ValueError("Cloud output would overwrite an existing job file")
            shutil.move(str(path), target)


async def sync_final_metadata(sky, stage_dir, state, on_event=None):
    """Preserve already-committed checkpoints without delaying teardown indefinitely."""
    if not state.get("target", {}).get("storage_uri") or state.get("final_metadata_synced"):
        return True

    async def quiet_event(message):
        if on_event:
            await on_event(message)

    try:
        await asyncio.wait_for(sync_checkpoints(sky, stage_dir, state, quiet_event, final=True), 60)
        state["final_metadata_synced"] = True
        state["metadata_sync_error"] = None
    except OSError, TimeoutError, RuntimeError, ValueError:
        state["metadata_sync_error"] = (
            "Cloud checkpoints are stored in GCS; their local metadata needs synchronization."
        )
        if on_event:
            await on_event(state["metadata_sync_error"])
    _write_json(stage_dir / STATE_NAME, state)
    return state.get("final_metadata_synced", False)


async def cleanup(stage_dir: Path, state: dict, *, on_event: Event | None = None) -> bool:
    cluster = state.get("cluster", "")
    job_id = state.get("job_id", "")
    prefix = "fb-" + hashlib.sha256(str(job_id).encode()).hexdigest()[:12] + "-"
    if (
        not isinstance(job_id, str)
        or not re.fullmatch(CLUSTER_PATTERN, cluster)
        or not cluster.startswith(prefix)
    ):
        raise ValueError("Invalid saved SkyPilot cluster identity")
    state["config_path"] = str((stage_dir / "sky-config.yaml").resolve())
    was_prepared = state.get("phase") == "prepared"
    state["phase"] = "cleaning_up"
    _write_json(stage_dir / STATE_NAME, state)
    if on_event:
        await on_event(f"Stopping Google Cloud resources for {cluster}")
    try:
        sky = executable()
        discovery_failed = False
        unknown_submission = not was_prepared and not state.get("request_id")
        request_ids = set()

        async def launch_requests():
            code, output = await _command(
                [
                    sky,
                    "api",
                    "status",
                    "--cluster",
                    cluster,
                    "--limit",
                    "all",
                    "--all-status",
                    "--verbose",
                    "--output",
                    "json",
                    *_args(state),
                ],
                stage_dir,
                timeout=60,
            )
            # SkyPilot 0.13's older API server prints this known warning on stdout
            # when --cluster is unsupported. Keep parsing strict for every other
            # diagnostic, and independently scope records below before cancelling.
            output = output.removeprefix(
                "The flag is ignored because the server does not support it yet.\n"
            )
            records = json.loads(output)
            if code or not isinstance(records, list):
                raise ValueError("Cannot verify pending SkyPilot launch requests")
            # Older servers can ignore the cluster filter. Independently check
            # every record before cancelling; never touch another user's launch.
            return [
                item
                for item in records
                if isinstance(item, dict)
                and item.get("cluster_name") == cluster
                and item.get("name") in {"launch", "sky.launch"}
                and isinstance(item.get("request_id"), str)
                and re.fullmatch(r"[a-f0-9-]{8,64}", item["request_id"])
            ]

        try:
            records = await launch_requests()
            if records:
                unknown_submission = False
                request_ids.update(item["request_id"] for item in records)
                state["request_id"] = records[0]["request_id"]
                _write_json(stage_dir / STATE_NAME, state)
        except OSError, TimeoutError, ValueError:
            discovery_failed = True
        cancellations = [[sky, "cancel", cluster, "--all", "-y", *_args(state)]]
        request_id = state.get("request_id")
        if isinstance(request_id, str) and re.fullmatch(r"[a-f0-9-]{8,64}", request_id):
            request_ids.add(request_id)
        for request_id in sorted(request_ids):
            cancellations.insert(0, [sky, "api", "cancel", request_id, "-y", *_args(state)])
        for argv in cancellations:
            try:
                await _command(argv, stage_dir, timeout=60)
            except OSError, TimeoutError:
                pass
        try:
            if any(
                item.get("status") not in {"SUCCEEDED", "FAILED", "CANCELLED"}
                for item in await launch_requests()
            ):
                discovery_failed = True
        except OSError, TimeoutError, ValueError:
            discovery_failed = True
        await sync_final_metadata(sky, stage_dir, state, on_event)
        code, _ = await _command(
            [sky, "down", cluster, "-y", *_args(state)], stage_dir, timeout=300
        )
        # SkyPilot reports absent clusters as successful teardown in current CLI.
        if code or discovery_failed or unknown_submission:
            raise RuntimeError("SkyPilot could not confirm cloud teardown")
        state["phase"], state["cleanup_error"] = "cleaned_up", None
        _write_json(stage_dir / STATE_NAME, state)
        return True
    except OSError, TimeoutError, RuntimeError, ValueError:
        state["cleanup_error"] = (
            f"Cloud cleanup needs attention. Run sky down -y {cluster} on this host."
        )
        state["phase"] = "cleanup_failed"
        _write_json(stage_dir / STATE_NAME, state)
        if on_event:
            await on_event(state["cleanup_error"])
        return False


async def run(
    payload: dict, stage_dir: Path, target: dict, on_event: Event, *, hf_token: str | None = None
) -> int:
    context = _RUN_SECRETS.set((hf_token,) if hf_token else ())
    try:
        return await _run(payload, stage_dir, target, on_event, hf_token=hf_token)
    finally:
        _RUN_SECRETS.reset(context)


async def ensure_cloud_storage(sky, target):
    python = cloud_compute_catalog.sky_python(sky)
    if not python:
        raise ValueError("SkyPilot Python environment is missing")
    process = await asyncio.create_subprocess_exec(
        python,
        str(Path(__file__).with_name("cloud_storage.py")),
        "ensure",
        target["project_id"],
        target["region"],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 120)
    finally:
        await _stop_process(process)
    bucket = stdout.decode().strip()
    if process.returncode or not re.fullmatch(r"gs://[a-z0-9._-]+", bucket):
        raise ValueError(
            "Could not prepare private Google Cloud artifact storage: " + stderr.decode()[-1000:]
        )
    return bucket


def failed_worker_diagnostic(stage_dir: Path, target: dict) -> str:
    """Keep detached setup failures actionable even when bootstrap never wrote a result."""
    from vla_platform.lifecycle.cloud_errors import cloud_launch_error
    from vla_platform.lifecycle.telemetry import sanitize_text

    path = stage_dir / "worker.log"
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 128 * 1024))
            output = stream.read().decode("utf-8", errors="replace")
    except OSError:
        output = ""
    for secret in _RUN_SECRETS.get():
        output = output.replace(secret, "[redacted]")
    output = sanitize_text(output)
    # Authentication can also appear in signed query strings; diagnostics do
    # not need complete URLs to explain a compiler or dependency failure.
    output = re.sub(r"https?://[^\s<>\"']+", "[redacted URL]", output)
    classified = cloud_launch_error(output, target)
    if not classified.startswith("Cloud training could not start."):
        return classified
    lines = [
        line.strip()
        for line in output.splitlines()
        if re.search(
            r"error|exception|failed|could not find|unable to locate|missing|cmake", line, re.I
        )
    ]
    detail = "\n".join(lines[-12:])[-1800:]
    return "Cloud worker setup or execution failed before publishing its result. " + (
        detail if detail else "Open the worker log for the underlying error."
    )


async def _run(
    payload: dict, stage_dir: Path, target: dict, on_event: Event, *, hf_token: str | None = None
) -> int:
    sky = executable()
    endpoint = validate_endpoint(target.get("sky_api_endpoint"))
    verified = await cloud_compute_catalog.verify_sky_target(
        sky,
        target["project_id"],
        endpoint=endpoint,
    )
    if (
        verified["workspace"] != target.get("workspace")
        or validate_endpoint(verified["sky_api_endpoint"]) != endpoint
    ):
        raise ValueError("The queued run's SkyPilot workspace no longer matches its cloud project")
    await on_event("Preparing private Google Cloud checkpoint storage")
    bucket = await ensure_cloud_storage(sky, target)
    target = {
        **target,
        "storage_uri": bucket.strip() + "/jobs/" + payload["job_id"] + "/" + stage_dir.name,
    }
    task_path, state = await asyncio.to_thread(prepare, payload, stage_dir, target)
    state_path = stage_dir / STATE_NAME
    timeout = payload["parameters"].get("timeout_seconds", 7200)
    try:
        state["phase"] = "launching"
        _write_json(state_path, state)
        await on_event(
            f"Provisioning {target['accelerator']}:1 on Google Cloud · "
            f"{target['project_id']} · {target['region']}"
        )
        code, output = await _command(
            [
                sky,
                "launch",
                "-y",
                "--async",
                "--detach-run",
                "-c",
                state["cluster"],
                "--down",
                "--idle-minutes-to-autostop",
                str(target["idle_minutes"]),
                "--wait-for",
                "jobs_and_ssh",
                *(["--secret", "HF_TOKEN"] if hf_token else []),
                *_args(state),
                str(task_path.resolve()),
            ],
            stage_dir,
            timeout=120,
            **(
                {"env_overrides": {"HF_TOKEN": hf_token}, "secrets": (hf_token,)}
                if hf_token
                else {}
            ),
        )
        request = re.search(r"Submitted sky\.launch request: ([a-f0-9-]{8,64})", output)
        if code or request is None:
            raise RuntimeError(
                "SkyPilot could not submit the cloud launch. Check the worker log and GCP setup."
            )
        state["request_id"] = request[1]
        _write_json(state_path, state)
        code, setup_output = await _command(
            [sky, "api", "logs", request[1], *_args(state)],
            stage_dir,
            timeout=timeout,
            on_event=on_event,
        )
        if code:
            from vla_platform.lifecycle.cloud_errors import cloud_launch_error

            raise RuntimeError(cloud_launch_error(setup_output, target))
        state["phase"] = "training"
        _write_json(state_path, state)
        await on_event(
            "Quantizing on Google Cloud"
            if payload["operation"] == "policy.quantize"
            else "Training on Google Cloud"
        )
        await follow_training(sky, stage_dir, state, timeout, on_event)
        status, _ = await _command(
            [sky, "logs", state["cluster"], "1", "--status", *_args(state)], stage_dir, timeout=60
        )
        state["phase"] = "collecting"
        _write_json(state_path, state)
        await on_event("Saving cloud artifacts and retrieving result metadata")
        await sync_checkpoints(sky, stage_dir, state, on_event, final=True)
        if state["target"].get("storage_uri"):
            state["final_metadata_synced"] = True
            _write_json(state_path, state)
        if not state["target"].get("storage_uri"):
            await asyncio.to_thread(collect, stage_dir, state["cluster"])
        if status and not (stage_dir / "result.json").is_file():
            raise RuntimeError(failed_worker_diagnostic(stage_dir, target))
        return status
    finally:
        # Shield cleanup from the job timeout/cancel while awaiting it. State is
        # durable so process death is reconciled by recover() on next startup.
        task = asyncio.create_task(cleanup(stage_dir, state, on_event=on_event))
        try:
            cleaned = await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        if not cleaned:
            raise RuntimeError(state["cleanup_error"])


async def recover(data_dir: Path) -> list[dict]:
    """Retry teardown for interrupted dispatches; return visible cleanup failures."""
    failures = []
    for state_path in sorted((data_dir / "jobs").glob(f"*/*/{STATE_NAME}")):
        if state_path.is_symlink() or state_path.stat().st_size > 65536:
            continue
        try:
            state = json.loads(state_path.read_text())
            if not isinstance(state, dict) or state.get("job_id") != state_path.parent.parent.name:
                raise ValueError("Saved dispatch does not belong to this job")
            if state.get("version") != 1:
                continue
            if state.get("phase") == "cleaned_up":
                if (
                    state.get("target", {}).get("storage_uri")
                    and not state.get("final_metadata_synced")
                    and not await sync_final_metadata(executable(), state_path.parent, state)
                ):
                    failures.append(
                        {
                            "job_id": state.get("job_id"),
                            "cluster": state.get("cluster"),
                            "error": state["metadata_sync_error"],
                        }
                    )
                continue
            if not await cleanup(state_path.parent, state):
                failures.append(
                    {
                        "job_id": state.get("job_id"),
                        "cluster": state.get("cluster"),
                        "error": state["cleanup_error"],
                    }
                )
        except ValueError, OSError:
            failures.append(
                {
                    "job_id": state_path.parent.parent.name,
                    "cluster": None,
                    "error": "Saved SkyPilot cleanup requires attention; "
                    "check this job's sky-state.json.",
                }
            )
    return failures
