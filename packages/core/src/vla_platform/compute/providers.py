"""Execution adapters. Operation inputs and result validation stay in Lifecycle.

A registered adapter owns process/provider submission, progress and cleanup.
Its return is the worker exit status; a successful process alone never publishes
an artifact. Cancellation must drain the owned process/resource before returning.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vla_platform.contracts import TERMINAL
from vla_platform.lifecycle import telemetry
from vla_platform.lifecycle.runtime import command


@dataclass(frozen=True)
class WorkerInvocation:
    lifecycle: Any
    job: Any
    runtime: Any
    operation: str
    payload: dict
    stage_dir: Path
    output: str
    training: bool
    act_export: bool


class ExecutionAdapter(Protocol):
    async def run(self, invocation: WorkerInvocation) -> int: ...


class SkyPilotAdapter:
    async def run(self, invocation: WorkerInvocation) -> int:
        owner = invocation.lifecycle
        job, runtime = invocation.job, invocation.runtime
        payload, stage_dir = invocation.payload, invocation.stage_dir
        from vla_platform.lifecycle import sky_runner

        if job.compute_target is None:
            raise ValueError("Cloud run has no saved compute selection; submit a new job")
        target = job.compute_target.model_dump()
        owner.require_compute(job.request, runtime)
        await owner.event(job, "preparing", "Preparing your GPU…")
        await owner.compute.ensure_cloud_ready(runtime.id, target)
        current = await owner.execution.get(job.id)
        if current is None or current.status in TERMINAL:
            raise asyncio.CancelledError
        await owner.event(
            job,
            "preparing",
            f"Launching {target['accelerator']} on Google Cloud via SkyPilot "
            f"({target['project_id']}, {target['region']})",
        )

        cloud_stage = "preparing"

        async def progress(message):
            nonlocal cloud_stage
            if message.startswith("_cloud_setup:"):
                detail = json.loads(message.split(":", 1)[1])
                cloud_stage = detail["phase"]
                await owner.event(job, cloud_stage, detail["message"], detail)
                return
            if message.startswith(("_quantization:", "_inference:")):
                detail = json.loads(message.split(":", 1)[1])
                cloud_stage = detail.get("phase", "quantizing")
                await owner.event(job, cloud_stage, detail.get("message", "Quantizing checkpoint"))
                return
            if message.startswith("_telemetry:"):
                line = message[len("_telemetry:") :]
                observed = telemetry.parse_observation(line)
                if observed:
                    cloud_stage = observed.get("phase") or (
                        "checkpoint" if observed.get("checkpoint_saved") else "training"
                    )
                await owner.training_observation(job, line)
                return
            if message.startswith("Optimizer step"):
                cloud_stage = "training"
            elif message.startswith(("Downloading training", "Saving cloud artifacts")):
                cloud_stage = "saving"
            elif message == "Quantizing on Google Cloud":
                cloud_stage = "quantizing"
            elif message == "Evaluating on Google Cloud":
                cloud_stage = "evaluating"
            elif message == "Running policy on Google Cloud":
                cloud_stage = "running"
            elif message == "Training on Google Cloud":
                cloud_stage = "training"
            elif message.startswith("Stopping Google Cloud"):
                cloud_stage = "finishing"
            await owner.event(job, cloud_stage, message)

        from vla_platform.huggingface_connection import HuggingFaceConnection

        hf_token = HuggingFaceConnection(owner.settings.data_dir).token()
        secret_options = {"hf_token": hf_token} if hf_token else {}
        returncode = await sky_runner.run(
            payload,
            stage_dir,
            target,
            progress,
            on_log=lambda line: _record_remote(owner, job, invocation.output, line),
            **secret_options,
        )
        return returncode


class LocalAdapter:
    async def run(self, invocation: WorkerInvocation) -> int:
        owner = invocation.lifecycle
        job, runtime, operation = invocation.job, invocation.runtime, invocation.operation
        stage_dir, output = invocation.stage_dir, invocation.output
        training, act_export = invocation.training, invocation.act_export
        request_path, result_path = stage_dir / "request.json", stage_dir / "result.json"
        container_name = "firebird-" + job.id + "-" + output
        argv, cwd, env = command(
            runtime,
            request_path.resolve(),
            result_path.resolve(),
            owner.settings.data_dir.resolve(),
            container_name,
            training,
            operation in {"policy.import", "policy.quantize"},
            operation in {"policy.evaluate", "policy.run"},
            act_export=act_export,
            operation=operation,
        )
        image = None if act_export else runtime.training_image if training else runtime.image
        if operation in {"policy.import", "policy.quantize"} and runtime.conversion_python:
            image = runtime.conversion_image
        if operation in {"policy.evaluate", "policy.run"} and (
            runtime.evaluation_python or runtime.evaluation_image
        ):
            image = runtime.evaluation_image
        await owner.event(job, output, f"Starting {operation} on {runtime.label}")
        # Settings may change while a queued run waits, or during an earlier
        # workflow stage. Check again immediately before starting each worker.
        owner.require_compute(job.request, runtime)
        if job.execution and job.execution.target_snapshot:
            import shutil

            resources = job.execution.target_snapshot["definition"]["resources"]
            if resources["cpus"] > (os.cpu_count() or 1):
                raise ValueError("The local target requests more CPUs than this machine provides")
            if shutil.disk_usage(stage_dir).free < resources["disk_gib"] * 1024**3:
                raise ValueError("The local target's scratch-space request exceeds available disk")
            memory_path = Path("/proc/meminfo")
            if memory_path.is_file():
                available = next(
                    (
                        int(line.split()[1]) * 1024
                        for line in memory_path.read_text().splitlines()
                        if line.startswith("MemAvailable:")
                    ),
                    None,
                )
                if available is not None and available < resources["memory_gib"] * 1024**3:
                    raise ValueError("The local target's memory request exceeds available RAM")
            env = {
                **(env or os.environ),
                **{
                    key: str(resources["cpus"])
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            }
        spawn = owner.act_spawn_owned
        process = await spawn(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        from vla_platform.compute.streams import output_lines

        async def consume(stream, name):
            written = 0
            with (stage_dir / ("worker.log" if name == "stdout" else "worker-stderr.log")).open(
                "w"
            ) as log:

                async def line(value):
                    nonlocal written
                    saved = owner.execution.observability.try_record(
                        job.id, value, kind="log", component="worker", stage=output, stream=name
                    )
                    safe = (
                        saved.message if saved is not None else "[diagnostic storage unavailable]"
                    )
                    if written < 5 * 1024**2:
                        part = safe[: 5 * 1024**2 - written]
                        log.write(part + "\n")
                        log.flush()
                        written += len(part) + 1
                    if name == "stdout" and operation == "policy.finetune":
                        await owner.training_observation(job, safe)
                    elif name == "stdout":
                        try:
                            progress = json.loads(safe)
                        except ValueError, TypeError:
                            progress = None
                        if isinstance(progress, dict) and type(progress.get("step")) is int:
                            await owner.event(job, output, f"Optimizer step {progress['step']}")

                async for value in output_lines(stream):
                    await line(value)

        drains = [
            asyncio.create_task(consume(process.stdout, "stdout")),
            asyncio.create_task(consume(process.stderr, "stderr")),
        ]
        try:
            await asyncio.gather(*drains, process.wait())
        finally:
            if act_export:
                await owner.act_stop_owned(process)
            else:
                await owner.stop(process, container_name if image else None)
            await asyncio.gather(*drains, return_exceptions=True)
        owner.execution.observability.try_record(
            job.id,
            f"Worker exited with code {process.returncode}",
            component="worker",
            stage=output,
            severity="error" if process.returncode else "info",
            data={"exit_code": process.returncode},
        )
        if process.returncode:
            owner.execution.observability.try_failure(
                job.id,
                stage=output,
                message=f"Worker exited with code {process.returncode}",
                exit_code=process.returncode,
                cleanup="succeeded",
            )
        return process.returncode


async def _record_remote(owner, job, stage, line):
    saved = owner.execution.observability.try_record(
        job.id, line, kind="log", component="provider", stage=stage, stream="stdout"
    )
    return saved.message if saved is not None else "[diagnostic storage unavailable]"


class SkyPilotTargetAdapter:
    async def run(self, invocation: WorkerInvocation) -> int:
        from vla_platform.compute_targets import ComputeTargets
        from vla_platform.huggingface_connection import HuggingFaceConnection
        from vla_platform.lifecycle import sky_runner

        owner, job = invocation.lifecycle, invocation.job
        registry = ComputeTargets(owner.settings.data_dir)
        snapshot = registry.snapshot(job.request.execution_target_id, job.kind, owner.catalog)
        if snapshot.model_dump(mode="json") != job.execution.target_snapshot:
            raise ValueError("The accepted compute target changed; submit a new job")
        definition = snapshot.definition
        import hashlib

        path = owner.settings.data_dir / "jobs" / job.id / "execution-connection.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
            raise ValueError("The accepted compute connection is unavailable")
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != job.execution.connection_sha256:
            raise ValueError("The accepted compute connection changed")
        connection = json.loads(encoded)
        endpoint = sky_runner.validate_endpoint(connection["sky_api_endpoint"])
        target = {
            **definition.model_dump(mode="json"),
            "sky_api_endpoint": endpoint,
            "workspace": connection.get("workspace", "default"),
            "accelerator": definition.resources.accelerator,
            "gpu_count": definition.resources.gpu_count,
            "disk_size_gb": definition.resources.disk_gib,
            "idle_minutes": 10,
            "region": definition.location,
        }
        if definition.infrastructure == "gcp":
            target["project_id"] = connection["project_id"]

        async def progress(message):
            if message.startswith("_telemetry:"):
                await owner.training_observation(job, message[len("_telemetry:") :])
            else:
                await owner.event(job, invocation.output, message)

        token = HuggingFaceConnection(owner.settings.data_dir).token()
        return await sky_runner.run(
            invocation.payload,
            invocation.stage_dir,
            target,
            progress,
            hf_token=token,
            on_log=lambda line: _record_remote(owner, job, invocation.output, line),
        )


class AdapterRegistry:
    def __init__(self):
        self._adapters: dict[str, ExecutionAdapter] = {}

    def register(self, name: str, adapter: ExecutionAdapter):
        if not name or name in self._adapters:
            raise ValueError("Execution adapter already registered or unnamed")
        self._adapters[name] = adapter

    async def run(self, name: str, invocation: WorkerInvocation) -> int:
        adapter = self._adapters.get(name)
        if adapter is None:
            raise ValueError(f"Execution adapter is not installed: {name}")
        return await adapter.run(invocation)


adapters = AdapterRegistry()
adapters.register("local", LocalAdapter())
adapters.register("skypilot", SkyPilotAdapter())
adapters.register("skypilot-target", SkyPilotTargetAdapter())
