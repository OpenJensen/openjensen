"""Lifecycle stages use the application's existing job owner and persistence."""

import asyncio
import hashlib
import json
import os
import re
import signal
import tarfile
import tempfile
import threading
from pathlib import Path

from vla_platform.contracts import TERMINAL, DatasetProfile, Job, now
from vla_platform.lifecycle.contracts import (
    JobEvent,
    LifecycleResult,
    PolicyArtifact,
    PolicyRequest,
    Precision,
)
from vla_platform.lifecycle.runtime import RuntimeCatalog, command


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_bundle(directory: Path, job_dir: Path):
    directory = directory.resolve()
    if not directory.is_relative_to(job_dir.resolve()) or not directory.is_dir():
        raise ValueError("Worker artifact must stay in its application job directory")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    if not files:
        raise ValueError("Empty artifact manifest")
    actual = set()
    total = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Artifact symlinks are not permitted")
        if path.is_file() and path != manifest_path:
            name = path.relative_to(directory).as_posix()
            actual.add(name)
            total += path.stat().st_size
            if digest(path) != files.get(name):
                raise ValueError(f"Artifact hash mismatch: {name}")
    if actual != set(files):
        raise ValueError("Artifact inventory differs from manifest")
    return manifest, digest(manifest_path), total


class Lifecycle:
    def __init__(self, execution):
        self.execution = execution
        self.settings = execution.settings
        self.catalog = RuntimeCatalog.load(self.settings.runtime_config)
        self.sequences: dict[str, int] = {}
        self.export_lock = threading.Lock()

    async def artifacts(self, project_id: str):
        return [
            artifact
            for job in await self.execution.list(project_id)
            if isinstance(job.result, LifecycleResult)
            for artifact in job.result.artifacts
        ]

    async def artifact(self, project_id: str, artifact_id: str):
        value = next((x for x in await self.artifacts(project_id) if x.id == artifact_id), None)
        if value is None:
            raise ValueError("Artifact does not exist in this project")
        return value

    async def resume_checkpoint(self, project_id, job_id):
        job = await self.execution.get(job_id)
        if (
            not job
            or job.project_id != project_id
            or job.status not in {"failed", "cancelled", "interrupted"}
            or job.kind not in {"policy.finetune", "policy.workflow"}
        ):
            raise ValueError("Resume requires an interrupted training job in this project")
        stage = "operation" if job.kind == "policy.finetune" else "training"
        directory = self.settings.data_dir / "jobs" / job.id / stage / "training"
        latest = directory / "latest.json"
        if not latest.is_file():
            raise ValueError("This run has no completed checkpoint to resume")
        name = json.loads(latest.read_text()).get("checkpoint")
        if not isinstance(name, str) or not re.fullmatch(r"checkpoint-\d{6,}", name):
            raise ValueError("Invalid checkpoint name")
        checkpoint = directory / name
        if not checkpoint.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint escapes its run directory")
        if not (checkpoint / "manifest.json").is_file():
            raise ValueError("Last checkpoint is incomplete or missing")
        return checkpoint

    async def download(self, project_id, artifact_id):
        artifact = await self.artifact(project_id, artifact_id)
        directory = self.settings.data_dir / artifact.path
        if digest(directory / "manifest.json") != artifact.manifest_sha256:
            raise ValueError("Registered artifact manifest changed")
        await asyncio.to_thread(validate_bundle, directory, directory.parent)
        destination = self.settings.data_dir / "exports" / (artifact.id.replace(":", "-") + ".tar")
        destination.parent.mkdir(exist_ok=True)

        def archive():
            # Windows cannot replace an archive while another FileResponse reads it.
            # The thread lock also survives cancellation of the awaiting HTTP request.
            with self.export_lock:
                if destination.exists():
                    return
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    with tarfile.open(temporary, "w") as tar:
                        tar.add(directory, arcname="policy")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)

        if not destination.exists():
            await asyncio.to_thread(archive)
        return destination

    async def validate(self, project_id: str, request: PolicyRequest):
        runtime = self.catalog.runtime(request.runtime_id)
        if runtime is None:
            raise ValueError("Runtime is not configured on this application host")
        if request.source_id and self.catalog.source(request.source_id) is None:
            raise ValueError("Policy source is not configured")
        if request.resume_job_id:
            if request.artifact_id:
                raise ValueError("Choose either a checkpoint artifact or an interrupted job")
            await self.resume_checkpoint(project_id, request.resume_job_id)
        if request.artifact_id:
            await self.artifact(project_id, request.artifact_id)
        if request.operation == "policy.finetune" or request.training is not None:
            if request.training_method not in {"lora", "qlora"}:
                raise ValueError("Training method is not registered")
            if not runtime.training_python or not runtime.training_root:
                raise ValueError("This runtime has no training environment")
            dataset = await self.execution.get(request.dataset_job_id)
            if (
                not dataset
                or dataset.project_id != project_id
                or dataset.status != "succeeded"
                or not isinstance(dataset.result, DatasetProfile)
            ):
                raise ValueError("Training requires a successful dataset intake in this project")
            if dataset.result.source != "huggingface":
                raise ValueError("The current native recipe requires a pinned Hugging Face dataset")
        if (
            request.operation in {"policy.evaluate", "policy.run", "policy.workflow"}
            and request.evaluation.mode == "libero"
            and not runtime.simulator_lane
        ):
            raise ValueError("LIBERO is not configured for this runtime")

    def events(self, job_id: str, after: int = 0):
        path = self.settings.data_dir / "jobs" / job_id / "events.jsonl"
        if not path.is_file():
            return []
        result = []
        with path.open() as stream:
            for line in stream:
                event = JobEvent.model_validate_json(line)
                if event.sequence > after:
                    result.append(event)
        return result[-200:]

    async def event(self, job: Job, stage: str, message: str):
        directory = self.settings.data_dir / "jobs" / job.id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "events.jsonl"
        if job.id not in self.sequences:
            prior = self.events(job.id)
            self.sequences[job.id] = prior[-1].sequence if prior else 0
        self.sequences[job.id] += 1
        record = JobEvent(
            sequence=self.sequences[job.id],
            stage=stage,
            message=message[:2000],
            timestamp=now(),
        )
        with path.open("a") as stream:
            stream.write(record.model_dump_json() + "\n")
        async with self.execution.lock:
            current = await self.execution.get(job.id)
            if current and current.status not in TERMINAL:
                current.stage = stage
                await self.execution.save(current)

    async def publish(self, job: Job, result: LifecycleResult):
        async with self.execution.lock:
            current = await self.execution.get(job.id)
            if current and current.status not in TERMINAL:
                current.result = result
                await self.execution.save(current)
            else:
                raise asyncio.CancelledError

    async def stop(self, process, container_name: str | None):
        if container_name:
            cleanup = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(cleanup.wait(), 15)
            except TimeoutError:
                cleanup.kill()
                await cleanup.wait()
        if process.returncode is None or os.name == "posix":
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
            else:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                await process.wait()

    async def native(
        self, job, operation, artifact, output, result, *, final=False, precision=None
    ):
        request = job.request
        runtime = self.catalog.runtime(request.runtime_id)
        stage_dir = self.settings.data_dir / "jobs" / job.id / output
        stage_dir.mkdir(parents=True, exist_ok=False)
        training = operation in {"policy.finetune", "policy.export"}
        payload = {
            "schema_version": 1,
            "operation": operation,
            "job_id": job.id,
            "runtime": runtime.model_dump(),
            "output_dir": str(stage_dir.resolve()),
            "parameters": request.model_dump(),
            "final": final,
            "artifact": artifact.model_dump() if artifact else None,
            "source": None,
        }
        if artifact:
            artifact_dir = self.settings.data_dir / artifact.path
            manifest = artifact_dir / "manifest.json"
            if digest(manifest) != artifact.manifest_sha256:
                raise ValueError("Registered artifact manifest changed")
            payload["artifact"]["path"] = str(artifact_dir.resolve())
        if request.resume_job_id:
            payload["resume_checkpoint"] = str(
                (await self.resume_checkpoint(job.project_id, request.resume_job_id)).resolve()
            )
        payload["prior_reports"] = result.reports
        if request.source_id:
            payload["source"] = self.catalog.source(request.source_id).model_dump()
        chosen_precision = (
            precision
            or request.precision
            or Precision(language="Q4_0" if runtime.device == "cuda" else "Q8_0")
        )
        payload["parameters"]["precision"] = chosen_precision.model_dump()
        if training and operation == "policy.finetune":
            dataset = await self.execution.get(request.dataset_job_id)
            payload["dataset"] = dataset.result.model_dump()
        request_path, result_path = stage_dir / "request.json", stage_dir / "result.json"
        request_path.write_text(json.dumps(payload, allow_nan=False))
        container_name = "firebird-" + job.id + "-" + output
        argv, cwd, env = command(
            runtime,
            request_path.resolve(),
            result_path.resolve(),
            self.settings.data_dir.resolve(),
            container_name,
            training,
            operation in {"policy.import", "policy.quantize"},
        )
        image = runtime.training_image if training else runtime.image
        if operation in {"policy.import", "policy.quantize"} and runtime.conversion_python:
            image = runtime.conversion_image
        await self.event(job, output, f"Starting {operation} on {runtime.label}")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        written = 0
        pending = ""
        try:
            with (stage_dir / "worker.log").open("wb") as log:
                while chunk := await process.stdout.read(8192):
                    if written < 5 * 1024 * 1024:
                        log.write(chunk)
                        log.flush()
                        written += len(chunk)
                    pending += chunk.decode("utf-8", errors="replace")
                    lines = pending.split("\n")
                    pending = lines.pop()[-16384:]
                    for line in lines:
                        try:
                            progress = json.loads(line)
                        except ValueError, TypeError:
                            continue
                        if isinstance(progress, dict) and type(progress.get("step")) is int:
                            await self.event(job, output, f"Optimizer step {progress['step']}")
            await process.wait()
        finally:
            await self.stop(process, container_name if image else None)
        if not result_path.exists() or result_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Worker did not produce a bounded result")
        response = json.loads(result_path.read_text())
        if response.get("schema_version") != 1 or response.get("job_id") != job.id:
            raise ValueError("Worker response identity mismatch")
        if process.returncode or response.get("error"):
            raise RuntimeError(str(response.get("error", "Native worker failed"))[:2000])
        report = {
            **response.get("report", {}),
            "stage": output,
            "operation": operation,
            "artifact_id": artifact.id if artifact else None,
            "final": final,
        }
        result.reports.append(report)
        created = None
        if response.get("artifact"):
            info = response["artifact"]
            path = Path(info["path"])
            manifest, sha, total = await asyncio.to_thread(validate_bundle, path, stage_dir)
            created = PolicyArtifact(
                id=f"{job.id}:{output}",
                project_id=job.project_id,
                job_id=job.id,
                label=info["label"],
                format=info["format"],
                path=path.resolve().relative_to(self.settings.data_dir.resolve()).as_posix(),
                manifest_sha256=sha,
                file_bytes=total,
                parent_ids=[artifact.id] if artifact else [],
                metadata=manifest.get("metadata", {}),
            )
            result.artifacts.append(created)
        await self.publish(job, result)
        await self.event(job, output, f"Completed {operation}")
        return created, report

    async def run(self, job: Job):
        request = job.request
        result = LifecycleResult()
        artifact = (
            await self.artifact(job.project_id, request.artifact_id)
            if request.artifact_id
            else None
        )
        if request.operation != "policy.workflow":
            _, report = await self.native(job, request.operation, artifact, "operation", result)
            if report.get("scope") == "engine_diagnostics":
                result.decision = "diagnostics_only"
            await self.publish(job, result)
            return result
        if request.training is not None:
            artifact, _ = await self.native(job, "policy.finetune", artifact, "training", result)
        elif request.source_id:
            artifact, _ = await self.native(job, "policy.import", None, "import", result)
        if artifact.format == "training_checkpoint":
            artifact, _ = await self.native(
                job, "policy.export", artifact, "training-export", result
            )
        if artifact.format == "native_checkpoint":
            artifact, _ = await self.native(
                job, "policy.import", artifact, "floating-conversion", result
            )
        if artifact.format != "gguf" or artifact.metadata.get("precision") != "float":
            raise ValueError("Workflow baseline must be a floating GGUF")
        baseline = artifact
        _, reference = await self.native(job, "policy.evaluate", baseline, "baseline", result)
        measured = [(baseline, reference)]
        for index, precision in enumerate(request.candidates):
            try:
                candidate, _ = await self.native(
                    job,
                    "policy.quantize",
                    baseline,
                    f"quantize-{index}",
                    result,
                    precision=precision,
                )
                _, measurement = await self.native(
                    job, "policy.evaluate", candidate, f"evaluate-{index}", result
                )
                if measurement.get("runtime") != reference.get("runtime"):
                    raise ValueError("Candidate runtime differs from the measured reference")
                measured.append((candidate, measurement))
            except Exception as exc:
                result.reports.append(
                    {"stage": f"candidate-{index}", "status": "failed", "error": str(exc)[:2000]}
                )
                await self.event(job, f"candidate-{index}", f"Candidate failed: {exc}")
                await self.publish(job, result)
        if request.evaluation.mode == "engine" or request.limits is None:
            result.decision = "diagnostics_only"
            await self.event(
                job,
                "selection",
                "Measurements complete; no automatic promotion without task-quality constraints",
            )
            await self.publish(job, result)
            return result
        limits = request.limits
        eligible = []
        if reference.get("success_rate") is None:
            result.decision = "no_feasible_candidate"
            await self.publish(job, result)
            return result
        for candidate, measurement in measured:
            if (
                measurement.get("complete_episodes") == len(request.evaluation.initial_states)
                and measurement.get("success_rate") is not None
                and measurement["success_rate"] >= limits.min_success_rate
                and measurement["success_rate"]
                >= reference["success_rate"] - limits.max_success_drop
                and measurement["p95_ms"] <= limits.max_p95_ms
                and measurement.get("peak_device_mib") is not None
                and measurement["peak_device_mib"] <= limits.max_peak_device_mib
            ):
                eligible.append((candidate, measurement))
        if not eligible:
            result.decision = "no_feasible_candidate"
            await self.event(
                job, "selection", "No tested candidate meets all requested constraints"
            )
        else:
            chosen, _ = min(eligible, key=lambda item: item[0].metadata["weight_bytes"])
            await self.event(
                job, "selection", f"Frozen candidate: {chosen.label}; checking unused states"
            )
            _, final_reference = await self.native(
                job, "policy.evaluate", baseline, "final-reference", result, final=True
            )
            _, final = await self.native(
                job, "policy.evaluate", chosen, "final-evaluation", result, final=True
            )
            if (
                final_reference.get("complete_episodes") != len(request.evaluation.final_states)
                or final_reference.get("success_rate") is None
                or final.get("success_rate") is None
                or final.get("runtime") != reference.get("runtime")
                or final_reference.get("runtime") != reference.get("runtime")
                or final.get("complete_episodes") != len(request.evaluation.final_states)
                or final["success_rate"] < final_reference["success_rate"] - limits.max_success_drop
                or final.get("success_rate", -1) < limits.min_success_rate
                or final["p95_ms"] > limits.max_p95_ms
                or final.get("peak_device_mib") is None
                or final["peak_device_mib"] > limits.max_peak_device_mib
            ):
                result.decision = "no_feasible_candidate"
                await self.event(
                    job,
                    "final-evaluation",
                    "Frozen candidate failed final acceptance; no replacement selected",
                )
            else:
                package, reloaded = await self.native(
                    job, "policy.run", chosen, "package-and-reload", result, final=True
                )
                if (
                    reloaded.get("runtime") != reference.get("runtime")
                    or reloaded.get("complete_episodes") != len(request.evaluation.final_states)
                    or reloaded.get("success_rate", -1)
                    < final_reference["success_rate"] - limits.max_success_drop
                    or reloaded.get("success_rate", -1) < limits.min_success_rate
                    or reloaded.get("p95_ms", float("inf")) > limits.max_p95_ms
                    or reloaded.get("peak_device_mib") is None
                    or reloaded["peak_device_mib"] > limits.max_peak_device_mib
                ):
                    result.decision = "no_feasible_candidate"
                    await self.event(job, "package-and-reload", "Package failed final acceptance")
                else:
                    result.selected_artifact_id = package.id
                    result.decision = "validated"
        await self.publish(job, result)
        return result
