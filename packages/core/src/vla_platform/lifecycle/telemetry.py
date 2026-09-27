"""Durable observations and reproducibility evidence, without importing a GPU runtime."""

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from vla_platform.contracts import TERMINAL, now
from vla_platform.lifecycle.contracts import (
    PolicyRequest,
    TrainingCheckpoint,
    TrainingMetric,
    TrainingTelemetry,
)

PHASES = {
    "preparing",
    "training",
    "validation",
    "checkpoint",
    "verifying",
    "completed",
    "failed",
    "interrupted",
}
METRICS = {
    "train_loss",
    "validation_loss",
    "validation_action_mse",
    "learning_rate",
    "grad_norm",
    "elapsed_seconds",
}
MAX_JSON_BYTES = 4 * 1024 * 1024


def training_job(job):
    return isinstance(job.request, PolicyRequest) and (
        job.kind == "policy.finetune"
        or job.kind == "policy.workflow"
        and job.request.training is not None
    )


def read_json(path, default=None):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        return default
    try:
        return json.loads(path.read_text())
    except ValueError, OSError:
        return default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def public_value(value):
    """Worker/runtime evidence must not publish credentials or host-only file paths."""
    hidden = {
        "env",
        "output_dir",
        "python",
        "training_python",
        "act_export_python",
        "act_export_root",
        "conversion_python",
        "evaluation_python",
        "conversion_vendor",
        "worker_root",
        "training_root",
        "mounts",
        "vendor",
        "build",
        "simulator_lane",
        "tokenizer",
        "sky_api_endpoint",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
    }
    if isinstance(value, dict):
        return {
            str(key): public_value(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in hidden
            and not re.search(
                r"(^|_)(token|secret|password|credential|api_key)(_|$)", str(key), re.I
            )
        }
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def sanitize_text(value):
    """Redact credential values wherever diagnostics or nested evidence carry them."""
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = re.sub(r"\b(?:hf_[A-Za-z0-9]+|ya29\.[A-Za-z0-9._-]+)\b", "[redacted]", value)
    value = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", value)
    value = re.sub(
        r"(?i)((?:token|secret|password|api_key|authorization|cookie)[\"']?\s*[:=]\s*)"
        r"(?:[\"'][^\"']*[\"']|[^\s,]+)",
        r"\1[redacted]",
        value,
    )
    value = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[redacted]@", value)
    return value


def parse_observation(line):
    """Accept strict JSON after an optional SkyPilot log prefix; ignore diagnostic text."""
    start = line.find("{")
    if start < 0:
        return None
    try:
        value = json.loads(line[start:])
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    result = {}
    for key in ("step", "total_steps"):
        number = value.get(key)
        if type(number) is int and number >= 0:
            result[key] = number
    for key in METRICS:
        number = value.get(key)
        if type(number) in (int, float) and math.isfinite(number):
            result[key] = number
    if isinstance(value.get("phase"), str) and value["phase"] in PHASES:
        result["phase"] = value["phase"]
    if isinstance(value.get("message"), str):
        result["message"] = sanitize_text(value["message"])[:2000]
    if isinstance(value.get("timestamp"), str) and len(value["timestamp"]) < 100:
        try:
            datetime.fromisoformat(value["timestamp"])
            result["timestamp"] = value["timestamp"]
        except ValueError:
            pass
    checkpoint = value.get("checkpoint_saved")
    if isinstance(checkpoint, str) and re.fullmatch(r"checkpoint-\d{6,}", checkpoint):
        result["checkpoint_saved"] = checkpoint
    for key in ("recipe", "environment", "splits"):
        if isinstance(value.get(key), dict):
            result[key] = public_value(value[key])
    if "step" not in result and "phase" not in result:
        return None
    return result


def metric_rows(path, limit=1000, *, prefixed=False):
    """Bound reads even for long-running jobs; a partial trailing write is harmless."""
    if path.is_symlink() or not path.is_file():
        return [], False
    with path.open("rb") as stream:
        size = path.stat().st_size
        offset = max(0, size - MAX_JSON_BYTES)
        stream.seek(offset)
        if offset:
            stream.readline()
        lines = stream.read(MAX_JSON_BYTES).splitlines()
    rows = []
    for line in lines:
        try:
            item = (
                parse_observation(line.decode("utf-8", errors="replace"))
                if prefixed
                else json.loads(line)
            )
        except ValueError:
            continue
        if isinstance(item, dict) and type(item.get("step")) is int and item["step"] >= 0:
            rows.append(item)
    return rows[-limit:], bool(offset or len(rows) > limit)


def log_tail(path):
    if path.is_symlink() or not path.is_file():
        return []
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        lines = stream.read(32768).decode("utf-8", errors="replace").splitlines()[-100:]
    result = []
    for line in lines:
        result.append(sanitize_text(line)[:2000])
    return result


async def capture(lifecycle, job):
    """Freeze accepted lineage before queuing; later settings cannot rewrite history."""
    if not training_job(job):
        return
    request = job.request
    runtime = lifecycle.runtime(request.runtime_id)
    dataset = await lifecycle.execution.get(request.dataset_job_id)
    recipe = request.training
    if request.resume_job_id or request.artifact_id:
        recipe, _, _ = await lifecycle.resumed_training_contract(job.project_id, request)
    evidence = {}
    if runtime and runtime.training_root:
        root = Path(runtime.training_root)
        if runtime.execution == "skypilot":
            from vla_platform.lifecycle import sky_runner

            if hasattr(sky_runner, "worker_root"):
                root = sky_runner.worker_root()
        paths = [root / "pyproject.toml", root / "uv.lock", root / "requirements-smolvla-linux.txt"]
        paths += sorted((root / "src").rglob("*.py"))[:1000]
        evidence["worker_file_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON_BYTES
        }
    value = {
        "schema_version": 1,
        "job_id": job.id,
        "created_at": job.created_at,
        "request": public_value(request.model_dump()),
        "recipe": public_value(recipe or {}),
        "recipe_source": "checkpoint"
        if request.resume_job_id or request.artifact_id
        else "accepted_request",
        "dataset": dataset.result.model_dump() if dataset and dataset.result else None,
        "runtime": public_value(runtime.model_dump()) if runtime else None,
        "compute_target": public_value(job.compute_target.model_dump())
        if job.compute_target
        else None,
        "lineage": {"resume_job_id": request.resume_job_id, "artifact_id": request.artifact_id},
        "evidence": evidence,
    }
    write_json(lifecycle.settings.data_dir / "jobs" / job.id / "reproducibility.json", value)


async def reproducibility(lifecycle, job):
    directory = lifecycle.settings.data_dir / "jobs" / job.id
    value = read_json(directory / "reproducibility.json")
    if not isinstance(value, dict):
        # Legacy jobs must not claim today's mutable runtime config was their runtime.
        dataset = await lifecycle.execution.get(job.request.dataset_job_id)
        value = {
            "schema_version": 1,
            "job_id": job.id,
            "created_at": job.created_at,
            "request": public_value(job.request.model_dump()),
            "recipe": public_value(job.request.training or {}),
            "recipe_source": "accepted_request",
            "runtime": None,
            "dataset": dataset.result.model_dump() if dataset and dataset.result else None,
            "compute_target": public_value(job.compute_target.model_dump())
            if job.compute_target
            else None,
            "lineage": {
                "resume_job_id": job.request.resume_job_id,
                "artifact_id": job.request.artifact_id,
            },
            "evidence": {},
        }
    stage = directory / ("operation" if job.kind == "policy.finetune" else "training")
    evidence = value.setdefault("evidence", {})
    # These are returned by the worker or captured before launch, never current defaults.
    request = read_json(stage / "request.json")
    value.setdefault("accepted_runtime", value.get("runtime"))
    value["runtime_source"] = "accepted_configuration" if value.get("runtime") else "unavailable"
    if isinstance(request, dict) and isinstance(request.get("runtime"), dict):
        value["runtime"] = public_value(request.get("runtime"))
        value["runtime_source"] = "worker_request"
    latest = read_json(stage / "training/latest.json", {})
    checkpoint = latest.get("checkpoint") if isinstance(latest, dict) else None
    checkpoint_dir = stage / "bundle/checkpoint"
    if isinstance(checkpoint, str) and re.fullmatch(r"checkpoint-\d{6,}", checkpoint):
        checkpoint_dir = stage / "training" / checkpoint
    sources = {
        "recipe": [
            stage / "recipe.json",
            stage / "training/recipe.json",
            checkpoint_dir / "recipe.json",
        ],
        "splits": [stage / "training/splits.json", checkpoint_dir / "splits.json"],
        "normalization_stats": [stage / "training/stats.json", checkpoint_dir / "stats.json"],
        "model_report": [stage / "training/model_report.json"],
        "environment": [stage / "training/environment.json", stage / "environment.json"],
        "checkpoint_manifest": [checkpoint_dir / "manifest.json"],
    }
    hashes = evidence.setdefault("file_sha256", {})
    for key, paths in sources.items():
        for path in paths:
            item = read_json(path)
            if isinstance(item, dict):
                if key == "recipe":
                    value["recipe"] = public_value(item)
                    value["recipe_source"] = "worker_resolved"
                else:
                    evidence[key] = public_value(item)
                hashes[str(path.relative_to(directory))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                break
    recipe = value.get("recipe") or {}
    value["model"] = {
        "repository": recipe.get("model_id"),
        "revision": recipe.get("model_revision"),
    }
    value["limitations"] = [
        "Pinned inputs and saved state support reproduction; identical results across hardware "
        "and library versions are not guaranteed.",
        "Validation loss is held-out imitation loss, not robot task success.",
    ]
    if value["recipe_source"] == "accepted_request":
        value["limitations"].append("Worker-resolved defaults have not been observed yet.")
    return public_value(value)


async def snapshot(lifecycle, job):
    if not training_job(job):
        raise ValueError("This job is not a training run")
    directory = lifecycle.settings.data_dir / "jobs" / job.id
    stage = directory / ("operation" if job.kind == "policy.finetune" else "training")
    config = await reproducibility(lifecycle, job)
    events = [
        event.model_copy(
            update={"message": sanitize_text(event.message), "data": public_value(event.data)}
        )
        for event in lifecycle.events(job.id)
    ]
    total = config.get("recipe", {}).get("steps")
    if type(total) is not int or total < 1:
        total = None
    rows, truncated = metric_rows(directory / "training-metrics.jsonl")
    observed, local_truncated = metric_rows(stage / "training/metrics.jsonl")
    legacy_rows = []
    if not rows and not observed:
        # Existing runs predate structured persistence. Their SkyPilot-prefixed
        # worker log is still direct measurement evidence; do not mutate it.
        legacy_rows, truncated = metric_rows(stage / "worker.log", prefixed=True)
        rows = legacy_rows
    combined = {}
    for row in rows + observed:
        clean = parse_observation(json.dumps(row))
        if clean and "step" in clean and any(key in clean for key in METRICS):
            previous = combined.get(clean["step"])
            values = {key: getattr(previous, key) for key in METRICS} if previous else {}
            values.update({key: clean[key] for key in METRICS if key in clean})
            combined[clean["step"]] = TrainingMetric(
                step=clean["step"],
                timestamp=row.get("timestamp") or (previous.timestamp if previous else None),
                **values,
            )
    metrics = sorted(combined.values(), key=lambda item: item.step)[-1000:]
    latest = metrics[-1] if metrics else None
    completed = latest.step if latest else None
    phase = "queued" if job.status == "queued" else "preparing"
    action = "Waiting for a worker" if phase == "queued" else "Preparing training"
    if latest:
        phase, action = "training", f"Optimizer step {latest.step}"
    checkpoints = {}
    for row in legacy_rows:
        completed = max(completed or 0, row["step"])
        if row.get("checkpoint_saved"):
            checkpoints[row["checkpoint_saved"]] = TrainingCheckpoint(
                step=row["step"], name=row["checkpoint_saved"], timestamp=row.get("timestamp")
            )
    for event in events:
        data = event.data
        if type(data.get("total_steps")) is int and data["total_steps"] > 0:
            total = data["total_steps"]
        if type(data.get("step")) is int:
            completed = max(completed or 0, data["step"])
        if data.get("phase") in PHASES:
            phase, action = data["phase"], event.message
        elif event.stage in {"preparing", "training", "saving", "finishing"} and not (
            event.stage == "preparing" and completed
        ):
            phase, action = (
                {"saving": "checkpoint", "finishing": "verifying"}.get(event.stage, event.stage),
                event.message,
            )
        if data.get("checkpoint_saved"):
            checkpoints[data["checkpoint_saved"]] = TrainingCheckpoint(
                step=data.get("step", 0), name=data["checkpoint_saved"], timestamp=event.timestamp
            )
    for path in (stage / "training").glob("checkpoint-*/manifest.json"):
        if re.fullmatch(r"checkpoint-\d{6,}", path.parent.name):
            step = int(path.parent.name[11:])
            checkpoints.setdefault(
                path.parent.name, TrainingCheckpoint(step=step, name=path.parent.name)
            )
            completed = max(completed or 0, step)
    cloud_index = read_json(stage / "cloud-checkpoints.json", {})
    for item in cloud_index.get("checkpoints", []):
        name = item.get("name", "")
        if re.fullmatch(r"checkpoint-\d{6,}", name):
            prior = checkpoints.get(name)
            checkpoints[name] = TrainingCheckpoint(
                step=item["step"],
                name=name,
                timestamp=prior.timestamp if prior else None,
                artifact_id=f"{job.id}:{name}",
                storage="gcs",
                remote_uri=item["uri"],
            )
    if job.status in TERMINAL:
        phase = "completed" if job.status == "succeeded" else job.status
        action = (
            sanitize_text(job.error)
            if job.error
            else ("Training completed" if phase == "completed" else f"Run {phase}")
        )
    elapsed = latest.elapsed_seconds if latest else None
    eta = None
    timed = [item for item in metrics if item.elapsed_seconds is not None]
    if job.status not in TERMINAL and total and len(timed) >= 2 and completed is not None:
        first, last = timed[max(0, len(timed) - 20)], timed[-1]
        if last.step > first.step and last.elapsed_seconds > first.elapsed_seconds:
            eta = (
                max(0, total - completed)
                * (last.elapsed_seconds - first.elapsed_seconds)
                / (last.step - first.step)
            )
    end = job.updated_at if job.status in TERMINAL else now()
    wall = max(
        0, (datetime.fromisoformat(end) - datetime.fromisoformat(job.created_at)).total_seconds()
    )
    return TrainingTelemetry(
        job_id=job.id,
        status=job.status,
        phase=phase,
        current_action=action,
        updated_at=job.updated_at,
        completed_steps=completed,
        total_steps=total,
        percent=min(100, completed / total * 100) if total and completed is not None else None,
        elapsed_seconds=elapsed,
        wall_seconds=wall,
        eta_seconds=eta,
        latest=latest,
        metrics=metrics,
        metrics_truncated=truncated or local_truncated or len(combined) > 1000,
        checkpoints=sorted(checkpoints.values(), key=lambda item: item.step),
        events=events,
        logs=log_tail(stage / "worker.log"),
        reproducibility=config,
    )
