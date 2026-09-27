"""Terminal review of saved training lineage and the existing export/engine adapters.

The application still verifies source files and checkpoint-owned recipes at admission.
This module neither opens server paths nor executes a worker.
"""

import re

from vla_platform.lifecycle.contracts import Evaluation, Precision
from vla_platform.tui_client import ApiError

EXTENDED_MODES = {
    "resume": ("Resume saved training", "policy.finetune", "training"),
    "export": ("Export inference policy", "policy.export", "act_export"),
    "gguf_quantize": ("SmolVLA GGUF quantization", "policy.quantize", "engine_evaluation"),
    "evaluate": ("Evaluate engine / LIBERO", "policy.evaluate", "engine_evaluation"),
    "engine_run": ("Run SmolVLA engine / LIBERO", "policy.run", "run"),
}
EXTENDED_GUIDANCE = {
    "resume": "Choose one saved training checkpoint OR an interrupted training job. "
    "The server restores the saved model, dataset, optimizer and recipe; this is not a "
    "new budget or a retry. A failed job may have no usable checkpoint. The server "
    "verifies its latest complete checkpoint before admission.",
    "export": "Export a training checkpoint with its registered adapter. ACT export "
    "uses a separate local CPU worker and currently requires Hugging Face dataset lineage. "
    "A completed cloud ACT checkpoint is downloaded to the application host (up to 4 GiB). "
    "Export proves packaging/reload, not task success or simulator compatibility.",
    "gguf_quantize": "Choose a SmolVLA checkpoint or floating GGUF and an explicit precision. "
    "ACT uses the separate native packed workflow. Compression alone does not prove quality.",
    "evaluate": "Engine checks measure loading, finite actions and timing, not task success. "
    "LIBERO requires a configured simulator and explicit compatible suite, tasks, disjoint "
    "state sets, seed and horizon. These adapters do not score native ACT/Isaac policies.",
    "engine_run": "Run a compatible SmolVLA GGUF/package using engine checks or a configured "
    "LIBERO simulator. CPU ACT observation replay is a different workflow. Engine output "
    "does not establish robot task success.",
}


def interrupted_training(job):
    return job["kind"] in {"policy.finetune", "policy.workflow"} and job["status"] in {
        "failed",
        "cancelled",
        "interrupted",
    }


def metadata_object(metadata, key):
    value = metadata.get(key)
    return value if isinstance(value, dict) else {}


def visible_resume_recipe(saved, metadata):
    """Apply the service's visible metadata fallback, never guess a default model."""
    base = metadata_object(metadata, "base_model")
    fallback = {
        key: value
        for key, value in {
            "model_id": base.get("repository"),
            "model_revision": base.get("revision"),
            "method": metadata.get("method", saved["training_method"]),
            "camera_keys": metadata.get("camera_keys"),
        }.items()
        if value is not None
    }
    return {**fallback, **(saved.get("training") or {})}


def extended_artifact_allowed(item, mode):
    metadata, fmt = item["metadata"], item["format"]
    architecture = metadata.get("architecture")
    if mode == "resume":
        return (
            fmt == "training_checkpoint" and metadata.get("training_resume_supported") is not False
        )
    if mode == "export":
        return fmt == "training_checkpoint" and architecture in {"act", "smolvla"}
    if mode == "gguf_quantize":
        return (architecture == "smolvla" or (architecture is None and fmt == "gguf")) and (
            fmt in {"training_checkpoint", "native_checkpoint"}
            or (fmt == "gguf" and metadata.get("precision") == "float")
        )
    return fmt in {"gguf", "deployment_package"} and architecture in {None, "smolvla"}


def resume_context(context, artifact_id, resume_job_id):
    """Bind visible saved lineage; do not claim to read the server's checkpoint files."""
    if bool(artifact_id) == bool(resume_job_id):
        raise ApiError("Choose exactly one checkpoint artifact or interrupted training job.")
    jobs = {job["id"]: job for job in context.jobs}
    artifacts = {item["id"]: item for item in context.artifacts}
    lineage, visited, metadata = [], set(), {}
    for _ in range(32):
        source = None
        if artifact_id:
            source = artifacts.get(artifact_id)
            if source is None or not extended_artifact_allowed(source, "resume"):
                raise ApiError("Resume requires a project-owned training checkpoint.")
            if not re.fullmatch(r"[a-f0-9]{64}", source["manifest_sha256"]):
                raise ApiError("Checkpoint manifest identity is invalid.")
            metadata = {**source["metadata"], **metadata}
            job = jobs.get(source["job_id"])
        else:
            job = jobs.get(resume_job_id)
            if job is None or not interrupted_training(job):
                raise ApiError("Choose an interrupted training job in this project.")
        if job is None or job["kind"] not in {"policy.finetune", "policy.workflow"}:
            raise ApiError("The checkpoint's original training job is unavailable.")
        if job["id"] in visited:
            raise ApiError("Checkpoint lineage contains a cycle.")
        visited.add(job["id"])
        # Bind every referenced checkpoint, not just the first artifact. This is
        # private review context; it is never sent as an application Job record.
        lineage.append({**job, "selected_checkpoint": source} if source else job)
        saved = job["request"]
        if saved.get("training") is not None:
            return saved, metadata, lineage
        artifact_id, resume_job_id = saved.get("artifact_id"), saved.get("resume_job_id")
        if not artifact_id and not resume_job_id:
            # Metadata may identify a historical catalog-default training recipe.
            return saved, metadata, lineage
    raise ApiError("Checkpoint lineage is too deep.")


def review_extended(context, mode, raw, request, runtime, artifact, local_artifact):
    if (
        request["source_id"]
        or request["simulation"]
        or any(
            request[field] is not None
            for field in ("native_quantization", "native_distillation", "native_replay", "limits")
        )
    ):
        raise ApiError("This workflow cannot include another adapter or optimizer limits.")
    if request["candidates"] != [Precision().model_dump(mode="json")]:
        raise ApiError("Candidate search belongs to the optimizer workflow.")
    if mode != "gguf_quantize" and request["precision"] is not None:
        raise ApiError("Choose precision in the quantization workflow.")
    if mode not in {"evaluate", "engine_run"} and request["evaluation"] != Evaluation().model_dump(
        mode="json"
    ):
        raise ApiError("This operation does not accept an evaluation protocol.")
    if request["training"] is not None:
        raise ApiError(
            "Resume preserves saved settings; these workflows do not accept training edits."
        )
    if mode != "resume" and (request["resume_job_id"] or request["dataset_job_id"]):
        raise ApiError("This workflow consumes one policy, not a dataset or interrupted job.")
    if mode == "resume":
        saved, metadata, lineage = resume_context(
            context, request["artifact_id"], request["resume_job_id"]
        )
        if artifact and not local_artifact(artifact) and runtime["execution"] != "skypilot":
            raise ApiError("Cloud checkpoint resume requires the Google Cloud training runtime.")
        if not artifact and lineage[0].get("compute_target") and runtime["execution"] != "skypilot":
            raise ApiError("Cloud checkpoint resume requires the Google Cloud training runtime.")
        recipe = visible_resume_recipe(saved, metadata)
        method = recipe.get("method", saved["training_method"])
        if (
            request["dataset_job_id"] != saved.get("dataset_job_id")
            or request["training_method"] != method
        ):
            raise ApiError("Resume must preserve the original dataset and training method.")
        repository = recipe.get("model_id")
        model = next((item for item in context.models if item["model_id"] == repository), None)
        if (
            model is None
            or model["id"] not in runtime["training_model_ids"]
            or method not in model["methods"]
        ):
            raise ApiError("The saved model/method is unavailable on this training runtime.")
        dataset = next(
            (item for item in context.jobs if item["id"] == saved.get("dataset_job_id")), None
        )
        if (
            dataset is None
            or dataset["kind"] != "dataset.inspect"
            or dataset["status"] != "succeeded"
        ):
            raise ApiError("The original completed dataset intake is unavailable.")
        return dataset, model, lineage
    if artifact is None or not extended_artifact_allowed(artifact, mode):
        raise ApiError("Choose a compatible project-owned policy for this workflow.")
    if mode == "export":
        metadata = artifact["metadata"]
        if metadata.get("architecture") == "act":
            if not runtime["act_export"]:
                raise ApiError("This runtime has no ACT export adapter.")
            if metadata.get("training_backend") != "lerobot" or metadata.get("method") != "full":
                raise ApiError("ACT export requires a native full-training checkpoint.")
            if metadata_object(metadata, "dataset").get("source") != "huggingface":
                raise ApiError("ACT export requires pinned Hugging Face dataset lineage.")
            if not local_artifact(artifact):
                owner = next((job for job in context.jobs if job["id"] == artifact["job_id"]), None)
                if (
                    owner is None
                    or owner["status"] != "succeeded"
                    or metadata.get("reload_verified") is not True
                ):
                    raise ApiError(
                        "Cloud ACT export requires a completed, "
                        "reload-verified training checkpoint."
                    )
                return None, None, [owner]
        elif (
            not runtime["training"]
            or "smolvla" not in runtime["training_model_ids"]
            or not local_artifact(artifact)
        ):
            raise ApiError(
                "SmolVLA export requires its local training worker and local checkpoint."
            )
        return None, None, []
    if mode == "gguf_quantize":
        if not isinstance(raw.get("precision"), dict) or "language" not in raw["precision"]:
            raise ApiError("Choose an explicit GGUF language precision.")
        if not local_artifact(artifact) and runtime["execution"] != "skypilot":
            raise ApiError("This policy is stored on Google Cloud; choose a cloud runtime.")
        return None, None, []
    if not local_artifact(artifact) and runtime["execution"] != "skypilot":
        raise ApiError("This policy is stored on Google Cloud; choose a cloud runtime.")
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict) or any(
        key not in evaluation for key in ("mode", "suite", "warmups", "repetitions")
    ):
        raise ApiError("Choose the explicit evaluation mode, suite and inference sample budgets.")
    if any(type(evaluation[key]) is not int for key in ("warmups", "repetitions")):
        raise ApiError("Evaluation sample budgets must be explicit integers, not booleans.")
    if evaluation["mode"] == "libero":
        if runtime["execution"] != "native" or not runtime["simulation"]:
            raise ApiError(
                "LIBERO requires a configured local simulator; "
                "cloud engine checks do not score tasks."
            )
        fields = ("initial_states", "final_states", "seed", "steps")
        task_field = "task_ids" if evaluation["suite"] == "libero_spatial" else "task_id"
        if any(key not in evaluation for key in (*fields, task_field)):
            raise ApiError("Choose explicit LIBERO tasks, disjoint states, seed and step horizon.")
        integer_values = [evaluation["seed"], evaluation["steps"]]
        integer_values += evaluation["initial_states"] + evaluation["final_states"]
        integer_values += (
            evaluation[task_field] if task_field == "task_ids" else [evaluation[task_field]]
        )
        if any(type(value) is not int for value in integer_values):
            raise ApiError("LIBERO task/state IDs, seed and horizon must be explicit integers.")
        if artifact["metadata"].get("task") != evaluation["suite"]:
            raise ApiError("The policy must declare compatibility with the selected LIBERO suite.")
        if evaluation["suite"] == "libero_spatial" and (
            runtime["device"] != "cuda" or not evaluation.get("parity_limits")
        ):
            raise ApiError(
                "Spatial evaluation requires CUDA and explicit approved parity tolerances."
            )
    return None, None, []
