"""API-only lifecycle context, recipe review and durable uncertain-write evidence."""

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
from pydantic import ValidationError

from vla_platform.contracts import now
from vla_platform.lifecycle.contracts import PolicyArtifact, PolicyRequest
from vla_platform.lifecycle.runtime import PublicRuntime
from vla_platform.tui_client import ApiClient, ApiError, endpoint, segment

MAX_RECIPE = 65536
OPERATIONS = {
    "train": ("Fine-tune / train", "policy.finetune", "training"),
    "distill": ("ACT teacher → smaller student", "policy.distill", "native_distillation"),
    "quantize": ("ACT packed INT8 / INT4", "policy.quantize", "native_quantization"),
    "replay": ("CPU observation replay", "policy.run", "native_replay"),
}
GUIDANCE = {
    "train": "Choose a registered model and method, camera keys and explicit training budget. "
    "Model workers still enforce their own dataset/hardware requirements.",
    "distill": "Choose distinct train/validation/final episodes with independent lineage, "
    "six recorded coordinate units. coordinate_attestation must be "
    "teacher_recorded_coordinates, or generated_fixture only for generated data. "
    "No task-quality guarantee.",
    "quantize": "Choose bits 8 or 4. This measures generated-observation drift and packed CPU "
    "reload; it does not establish robot task quality, GPU memory or simulator support.",
    "replay": "Choose explicit episode/frame pairs and six recorded coordinate units. "
    "coordinate_attestation must be policy_recorded_coordinates, or generated_fixture "
    "only for generated data. This is not a simulator or scored task.",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_recipe(raw: str) -> dict:
    if len(raw.encode()) > MAX_RECIPE:
        raise ApiError("Recipe exceeds the 64 KiB limit.")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def forbidden(_):
        raise ValueError("Nonfinite JSON number")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=forbidden)
        canonical(value)  # Also rejects finite-looking exponents that overflow to infinity.
        if not isinstance(value, dict):
            raise ValueError("Recipe must be an object")
        return value
    except ValueError, RecursionError:
        raise ApiError("Enter a JSON object with unique fields and finite numbers.") from None


def local_artifact(item):
    metadata = item["metadata"]
    return metadata.get("storage") != "gcs" and not any(
        metadata.get(key) for key in ("remote", "remote_uri")
    )


def artifact_allowed(item, mode):
    metadata = item["metadata"]
    if not local_artifact(item) or metadata.get("architecture") != "act":
        return False
    if mode == "replay":
        return item["format"] == "native_quantized"
    if mode in {"distill", "quantize"}:
        allowed = item["format"] in {"native_checkpoint", "inference_export"}
        if mode == "quantize":
            allowed = (
                allowed
                and metadata.get("inference_only") is not False
                and (
                    metadata.get("method") != "full"
                    and metadata.get("training_backend") != "lerobot"
                    and metadata.get("use_vae") is not True
                )
            )
        return allowed
    return False


def dataset_allowed(job, mode):
    result = job.get("result") or {}
    if job["kind"] != "dataset.inspect" or job["status"] != "succeeded":
        return False
    if mode == "train":
        return result.get("source") == "huggingface" or bool(result.get("snapshot"))
    snapshot = result.get("snapshot") or {}
    complete = result.get("inspection_scope") == "complete_snapshot" and (
        result.get("format") == "lerobot_v3" and bool(snapshot)
    )
    if mode == "distill":
        complete = (
            complete
            and snapshot.get("lineage_validated") is True
            and (snapshot.get("total_episodes", 0) >= 3)
        )
    return complete


def runtime_allowed(runtime, mode):
    return (
        runtime[OPERATIONS[mode][2]] is True
        and runtime["enabled"] is True
        and runtime["launchable"] is True
        and not runtime.get("unavailable_reason")
        and (
            mode == "train" or (runtime["provider"] == "local" and runtime["execution"] == "native")
        )
    )


@dataclass
class Context:
    project: dict
    jobs: list[dict]
    artifacts: list[dict]
    runtimes: list[dict]
    models: list[dict]

    @classmethod
    async def fetch(cls, client: ApiClient, project_id: str):
        projects = await client.projects()
        project = next((p for p in projects if p["id"] == project_id), None)
        if project is None:
            raise ApiError("Selected project is no longer available.")
        jobs = await client.jobs(project_id)
        artifacts = client.validate(
            list[PolicyArtifact],
            await client.request("GET", f"/projects/{segment(project_id)}/artifacts"),
        )
        client.unique(artifacts)
        if any(a["project_id"] != project_id for a in artifacts):
            raise ApiError("Artifact context belongs to another project.")
        options = await client.request("GET", "/policy-options")
        if not isinstance(options, dict):
            raise ApiError("Configured capabilities are unavailable.")
        runtimes = client.validate(list[PublicRuntime], options.get("runtimes"))
        client.unique(runtimes)
        models = options.get("training_models", [])
        if not isinstance(models, list) or any(
            not isinstance(m, dict)
            or not isinstance(m.get("model_id"), str)
            or not isinstance(m.get("id"), str)
            or not isinstance(m.get("methods"), list)
            or not all(isinstance(method, str) for method in m["methods"])
            for m in models
        ):
            raise ApiError("Training model catalog is invalid.")
        if len({m["id"] for m in models}) != len(models) or len(
            {m["model_id"] for m in models}
        ) != len(models):
            raise ApiError("Training model catalog has duplicate identities.")
        return cls(project, jobs, artifacts, runtimes, models)

    def review(self, mode: str, raw: str):
        value = parse_recipe(raw)
        # All budgets must be chosen, not silently inherited from a template.
        if type(value.get("timeout_seconds")) is not int:
            raise ApiError("Choose an explicit timeout_seconds budget.")
        try:
            request = PolicyRequest.model_validate(value).model_dump(mode="json")
        except ValidationError as exc:
            details = [
                ".".join(str(part) for part in error["loc"])[:120] + ": " + error["msg"][:200]
                for error in exc.errors(
                    include_input=False, include_context=False, include_url=False
                )[:6]
            ]
            raise ApiError("Recipe needs correction: " + "; ".join(details)) from None
        if request["operation"] != OPERATIONS[mode][1]:
            raise ApiError("Recipe operation must match the selected workflow.")
        if request["simulation"] or request["source_id"] or request["resume_job_id"]:
            raise ApiError(
                "This composer creates new lifecycle jobs; simulation/resume use other flows."
            )
        runtime = next((r for r in self.runtimes if r["id"] == request["runtime_id"]), None)
        if runtime is None or not runtime_allowed(runtime, mode):
            raise ApiError("Selected runtime is not currently configured for this operation.")
        artifact = next((a for a in self.artifacts if a["id"] == request["artifact_id"]), None)
        dataset = next((j for j in self.jobs if j["id"] == request["dataset_job_id"]), None)
        model = None
        if mode != "train":
            if artifact is None or not artifact_allowed(artifact, mode):
                raise ApiError("Choose a compatible project-owned local ACT artifact.")
            if not re.fullmatch(r"[a-f0-9]{64}", artifact["manifest_sha256"]):
                raise ApiError("Artifact manifest identity is invalid.")
            field = OPERATIONS[mode][2]
            if not isinstance(request[field], dict):
                raise ApiError("Choose the explicit native operation recipe.")
        elif request["artifact_id"]:
            raise ApiError("Checkpoint resume is not part of this new-training composer.")
        if mode != "quantize" and (dataset is None or not dataset_allowed(dataset, mode)):
            raise ApiError(
                "Choose a completed compatible dataset; native flows need an immutable snapshot."
            )
        if mode == "train":
            if not isinstance(value.get("training_method"), str):
                raise ApiError("Choose an explicit training_method.")
            recipe = request["training"]
            if not isinstance(recipe, dict):
                raise ApiError("Enter the complete training recipe.")
            for key in ("steps", "batch_size"):
                if type(recipe.get(key)) is not int or recipe[key] <= 0:
                    raise ApiError(f"Choose an explicit positive training {key}.")
            rate = recipe.get("learning_rate")
            if type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0:
                raise ApiError("Choose an explicit positive training learning_rate.")
            model = next((m for m in self.models if m["model_id"] == recipe.get("model_id")), None)
            if (
                model is None
                or model["id"] not in runtime["training_model_ids"]
                or request["training_method"] not in model["methods"]
            ):
                raise ApiError("Choose a model/method registered for this runtime.")
            cameras = recipe.get("camera_keys") or [recipe.get("camera_key")]
            features = dataset["result"]["features"]
            if (
                not isinstance(cameras, list)
                or not cameras
                or any(not isinstance(key, str) or key not in features for key in cameras)
                or len(cameras) != len(set(cameras))
            ):
                raise ApiError("Choose explicit camera key(s) present in the selected dataset.")
            count = model.get("required_cameras")
            if type(count) is int and count > 0 and len(cameras) != count:
                raise ApiError(f"This registered model requires {count} camera(s).")
        bound = {
            "project": self.project,
            "runtime": runtime,
            "artifact": artifact,
            "dataset": dataset,
            "model": model,
            "request": request,
        }
        return Review(mode, request, runtime, artifact, dataset, digest(bound))


@dataclass
class Review:
    mode: str
    request: dict
    runtime: dict
    artifact: dict | None
    dataset: dict | None
    context_sha256: str


def template(mode, runtime_id=None, artifact_id=None, dataset_job_id=None):
    value = {"operation": OPERATIONS[mode][1], "runtime_id": runtime_id, "timeout_seconds": None}
    if mode != "train":
        value["artifact_id"] = artifact_id
    if mode != "quantize":
        value["dataset_job_id"] = dataset_job_id
    if mode == "train":
        value.update(
            training_method=None,
            training={
                "model_id": None,
                "steps": None,
                "batch_size": None,
                "learning_rate": None,
                "camera_keys": [],
            },
        )
    elif mode == "distill":
        value["native_distillation"] = {
            "adapter": "act-act-v1",
            "student": "act-256",
            "steps": None,
            "learning_rate": None,
            "seed": None,
            "frame_stride": None,
            "splits": {"train": [], "validation": [], "final": []},
            "coordinate_attestation": None,
            "units": [],
        }
    elif mode == "quantize":
        value["native_quantization"] = {"format": "firebird_quant", "bits": None, "group_size": 64}
    else:
        value["native_replay"] = {
            "adapter": "act-packed-observation-v1",
            "selection": [],
            "coordinate_attestation": None,
            "units": [],
        }
    return value


class AttemptJournal:
    """One private record per endpoint/project. It stores identities, never complete recipes."""

    def __init__(self, root: Path, base: str, project_id: str):
        self.root = root.expanduser().absolute()
        self.identity = digest({"endpoint": endpoint(base), "project_id": project_id})
        self.project_id = project_id
        self.path = self.root / (self.identity + ".json")
        self.lock = FileLock(str(self.root / (self.identity + ".lock")), timeout=1, mode=0o600)

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or self.root.is_symlink():
            raise ApiError("Attempt journal must use a private real directory.")
        if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ApiError("Attempt journal directory must be owned by you with mode0700.")
        lock_path = Path(self.lock.lock_file)
        try:
            lock_info = lock_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(lock_info.st_mode) or (
            os.name == "posix" and (lock_info.st_uid != os.getuid() or lock_info.st_mode & 0o077)
        ):
            raise ApiError("Attempt journal lock is not a regular private file.")

    def _read(self):
        try:
            fd = os.open(
                self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > 4096
                or (os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077))
            ):
                raise ApiError("Attempt journal is invalid; preserve it and inspect saved jobs.")
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ApiError("Attempt journal exceeds its bound.")
        try:
            value = parse_recipe(raw.decode("utf-8"))
            if (
                set(value)
                != {
                    "schema_version",
                    "identity",
                    "project_id",
                    "attempt_id",
                    "state",
                    "created_at",
                    "recipe_sha256",
                    "context_sha256",
                    "operation",
                    "runtime_id",
                }
                or type(value.get("schema_version")) is not int
                or value.get("schema_version") != 1
                or value.get("identity") != self.identity
                or value.get("project_id") != self.project_id
                or value.get("state") not in {"pending", "uncertain"}
                or not re.fullmatch(r"[a-f0-9]{32}", str(value.get("attempt_id")))
                or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("recipe_sha256")))
                or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("context_sha256")))
                or value.get("operation") not in {x[1] for x in OPERATIONS.values()}
                or not isinstance(value.get("runtime_id"), str)
                or not isinstance(value.get("created_at"), str)
            ):
                raise ValueError("Invalid record")
            return value
        except UnicodeError, ValueError, ApiError:
            raise ApiError(
                "Attempt journal is unreadable; preserve it and inspect saved jobs."
            ) from None

    def read(self):
        self.prepare()
        with self.lock:
            return self._read()

    def _write(self, value):
        raw = canonical(value).encode()
        if len(raw) > 4096:
            raise ApiError("Attempt journal record exceeds its bound.")
        fd, name = tempfile.mkstemp(prefix=".attempt-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            self._sync()
        finally:
            Path(name).unlink(missing_ok=True)

    def _sync(self):
        if os.name == "posix":
            fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def begin(self, review):
        self.prepare()
        with self.lock:
            if self._read() is not None:
                raise ApiError(
                    "An earlier attempt needs explicit history review before another submission."
                )
            record = {
                "schema_version": 1,
                "identity": self.identity,
                "project_id": self.project_id,
                "attempt_id": uuid4().hex,
                "state": "pending",
                "created_at": now(),
                "recipe_sha256": digest(review.request),
                "context_sha256": review.context_sha256,
                "operation": review.request["operation"],
                "runtime_id": review.request["runtime_id"],
            }
            self._write(record)
            return record

    @contextmanager
    def claim(self, review):
        # Hold the OS lock across the single HTTP write. Another TUI cannot clear
        # an in-flight attempt; process exit releases the lock, not its evidence.
        self.prepare()
        with self.lock:
            yield self.begin(review)

    def finish(self, record, *, uncertain=False):
        self.prepare()
        with self.lock:
            current = self._read()
            if current is None or digest(current) != digest(record):
                raise ApiError("Attempt record changed; inspect history before acting.")
            if uncertain:
                self._write({**current, "state": "uncertain"})
            else:
                self.path.unlink()
                self._sync()


def journal_directory():
    if os.name == "nt":
        return (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
            / "OpenJensen/state/tui"
        )
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "openjensen/tui"
