"""CLI follow-through using the same bounded, credential-free API transport as the TUI."""

import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote

import httpx

from vla_platform.contracts import TERMINAL, IntakeRequest, Job, Project, ProjectCreate
from vla_platform.lifecycle.contracts import PolicyArtifact
from vla_platform.tui_client import ApiClient, ApiError, segment

DEFAULT_DOWNLOAD_LIMIT = 4 * 1024**3
MAX_DOWNLOAD_LIMIT = 100 * 1024**3


class WaitDeadline(ApiError):
    """Only observation timed out; no job cancellation was requested."""


def client() -> ApiClient:
    return ApiClient(os.getenv("FIREBIRD_API_URL", "http://127.0.0.1:8000"))


async def request_json(method: str, path: str, payload=None):
    api = client()
    try:
        value = await api.request(method, path, payload)
        if method == "POST":
            validate_acknowledgment(api, path, payload, value)
        return value
    finally:
        await api.close()


def validate_acknowledgment(api: ApiClient, path: str, payload, value) -> None:
    """Validate write receipts without retrying or changing their public JSON."""
    parts = [unquote(part) for part in path.strip("/").split("/")]
    try:
        if parts == ["projects"]:
            project = api.validate(Project, value)
            expected = api.validate(ProjectCreate, payload)
            if project["name"] != expected["name"]:
                raise ApiError("Project acknowledgment does not match the request")
            return
        # Explicit fields must not be silently supplied by model defaults in a receipt.
        if not isinstance(value, dict) or not {"kind", "status", "request"} <= value.keys():
            raise ApiError("Incomplete job acknowledgment")
        job = api.validate(Job, value)
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
            if job["id"] != parts[1]:
                raise ApiError("Cancellation acknowledgment identifies another job")
            return
        operations = {
            "intakes": "dataset.inspect",
            "augmentations": "dataset.augment",
            "policy-jobs": (payload or {}).get("operation"),
        }
        if (
            len(parts) != 3
            or parts[0] != "projects"
            or parts[2] not in operations
            or job["project_id"] != parts[1]
            or job["kind"] != operations[parts[2]]
        ):
            raise ApiError("Job acknowledgment does not match the submission")
        recorded = job["request"]
        if parts[2] == "intakes":
            expected = api.validate(IntakeRequest, payload)
            keys = ["source", "repo_id", "snapshot_for_training"]
            # The server resolves Hub branches and local paths against its own data root.
            # They cannot be resolved independently on the CLI computer.
            if expected["source"] == "huggingface" and re.fullmatch(
                r"[a-f0-9]{40}", expected["revision"]
            ):
                keys.append("revision")
        elif parts[2] == "augmentations":
            from vla_platform.augmentation.contracts import AugmentationRequest

            expected = api.validate(AugmentationRequest, payload)
            keys = list(expected)
        else:
            from vla_platform.lifecycle.contracts import PolicyRequest

            expected = api.validate(PolicyRequest, payload)
            keys = [
                "operation",
                "runtime_id",
                "artifact_id",
                "source_id",
                "resume_job_id",
                "precision",
                "candidates",
                "evaluation",
                "limits",
                "timeout_seconds",
                "simulation",
                "native_quantization",
                "native_distillation",
                "native_replay",
            ]
            # Resume recipes/method/dataset are checkpoint-owned; new training recipes
            # may be enriched with the catalog's immutable model identity by the server.
            if not (
                expected["resume_job_id"]
                or (expected["operation"] == "policy.finetune" and expected["artifact_id"])
            ):
                keys.extend(["dataset_job_id", "training_method"])
                if expected["training"] is not None:
                    recipe = recorded.get("training")
                    if not isinstance(recipe, dict) or any(
                        recipe.get(key) != value
                        for key, value in expected["training"].items()
                        if key != "checkpoint_subdirectory"  # Catalog-owned path, not a budget.
                    ):
                        raise ApiError("Acknowledged training recipe changed submitted values")
        if any(recorded.get(key) != expected[key] for key in keys):
            raise ApiError("Acknowledged input does not match the submitted identity")
    except ApiError:
        raise ApiError(
            "Application returned an invalid or nonmatching acknowledgment. "
            "Outcome unknown; the request was sent once. Inspect recorded jobs/projects "
            "before submitting again."
        ) from None


async def wait_job(job_id: str, timeout: int, interval: float):
    if not 1 <= timeout <= 86400 or not 0.1 <= interval <= 30:
        raise ApiError("Wait requires a 1–86400 second timeout and 0.1–30 second interval.")
    api = client()
    try:
        try:
            async with asyncio.timeout(timeout):
                while True:
                    job = await api.job(job_id)
                    if job["status"] in TERMINAL:
                        return job
                    await asyncio.sleep(interval)
        except TimeoutError:
            raise WaitDeadline(
                "Wait deadline reached. The job was not cancelled. "
                "Inspect its status before acting."
            ) from None
    finally:
        await api.close()


async def _artifact(api: ApiClient, project: str, artifact_id: str) -> dict:
    records = api.validate(
        list[PolicyArtifact], await api.request("GET", f"/projects/{segment(project)}/artifacts")
    )
    api.unique(records)
    if any(item["project_id"] != project for item in records):
        raise ApiError("Application returned artifacts for another project.")
    artifact = next((item for item in records if item["id"] == artifact_id), None)
    if artifact is None:
        raise ApiError("Artifact is not registered in the selected project.")
    if not re.fullmatch(r"[a-f0-9]{64}", artifact["manifest_sha256"]):
        raise ApiError("Artifact registry returned an invalid manifest identity.")
    return artifact


async def download_artifact(
    project: str,
    artifact_id: str,
    output: Path,
    max_bytes: int = DEFAULT_DOWNLOAD_LIMIT,
    timeout: int = 600,
    expected_sha256: str | None = None,
):
    if not 1 <= max_bytes <= MAX_DOWNLOAD_LIMIT or not 1 <= timeout <= 3600:
        raise ApiError("Download requires a 1-byte–100 GiB byte bound and 1–3600 second deadline.")
    if expected_sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ApiError("Expected SHA256 must contain exactly 64 hexadecimal characters.")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ApiError("Output already exists; choose a new file. Nothing was overwritten.")
    if not output.parent.is_dir():
        raise ApiError("Output parent directory must already exist.")
    api = client()
    temporary: Path | None = None
    written_identity = None
    try:
        async with asyncio.timeout(timeout):
            before = await _artifact(api, project, artifact_id)
            descriptor, name = tempfile.mkstemp(
                prefix=".firebird-download-", suffix=".partial", dir=output.parent
            )
            temporary = Path(name)
            digest, received = hashlib.sha256(), 0
            with os.fdopen(descriptor, "wb") as target:
                address = (
                    f"{api.base}/api/v1/projects/{segment(project)}/artifacts/"
                    f"{segment(artifact_id)}/download"
                )
                async with api.http.stream(
                    "GET", address, headers={"Accept-Encoding": "identity"}
                ) as response:
                    if response.status_code != 200:
                        raise ApiError(
                            f"Download returned HTTP {response.status_code}; "
                            "no output was published."
                        )
                    if (
                        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        != "application/x-tar"
                    ):
                        raise ApiError("Download did not return the expected TAR media type.")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ApiError(
                            "Compressed HTTP responses are not accepted for bounded downloads."
                        )
                    length = response.headers.get("content-length")
                    if length is not None and (
                        not re.fullmatch(r"[0-9]+", length) or int(length) > max_bytes
                    ):
                        raise ApiError(
                            "Download Content-Length is invalid or exceeds the byte bound."
                        )
                    async for chunk in response.aiter_raw(chunk_size=64 * 1024):
                        received += len(chunk)
                        if received > max_bytes:
                            raise ApiError(
                                "Download exceeded the byte bound; no output was published."
                            )
                        target.write(chunk)
                        digest.update(chunk)
                    if not received or (length is not None and received != int(length)):
                        raise ApiError(
                            "Download was empty or its size did not match Content-Length."
                        )
                target.flush()
                os.fsync(target.fileno())
                written = os.fstat(target.fileno())
                written_identity = (written.st_dev, written.st_ino)
            received_hash = digest.hexdigest()
            if expected_sha256 is not None and received_hash != expected_sha256.lower():
                raise ApiError("Downloaded bytes do not match the expected SHA256.")
            if await _artifact(api, project, artifact_id) != before:
                raise ApiError(
                    "Artifact registry changed during download; no output was published."
                )
            # A hard link publishes atomically and fails if any destination appeared meanwhile.
            # Never replace an existing file, directory or symlink, even under a race.
            os.link(temporary, output)
            if os.name == "posix":
                directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return {
                "project_id": project,
                "artifact_id": artifact_id,
                "output": str(output),
                "bytes": received,
                "sha256": received_hash,
                "expected_sha256_matched": True if expected_sha256 is not None else None,
                "registered_manifest_sha256": before["manifest_sha256"],
                "verification_scope": (
                    "Received bytes and stable artifact registration; model quality "
                    "and archive contents were not independently verified by the CLI."
                ),
            }
    except BaseException as exc:
        if written_identity is not None:
            # Also covers a signal immediately after link(), before the next Python line.
            # Undo only our own publication, never an observed concurrent replacement.
            try:
                current = output.lstat()
                if (current.st_dev, current.st_ino) == written_identity:
                    output.unlink()
            except FileNotFoundError:
                pass
        if isinstance(exc, (httpx.HTTPError, TimeoutError)):
            raise ApiError(
                "Download interrupted or timed out; no completed output was published."
            ) from None
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        await api.close()
