"""Install a verified project ACT artifact on the paired USB companion without running it."""

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from vla_platform.robot_lab_bridge import MAX_JSON, RobotLabError, strict_json

MAX_ARCHIVE = 2 * 1024**3
HEADROOM = 10 * 1024**3
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
UNITS = ["degrees"] * 5 + ["percent"]


class PolicyInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    fps: StrictInt | None = Field(default=None, ge=1, le=40)
    confirm_so101_units: StrictBool = False

    @field_validator("name")
    @classmethod
    def printable_name(cls, value):
        value = value.strip()
        if not value or any(not character.isprintable() for character in value):
            raise ValueError("Choose a printable policy name")
        return value


def _json(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate policy artifact metadata field")
            value[key] = item
        return value

    return json.loads(data, object_pairs_hook=unique)


def _archive_metadata(path, expected_sha):
    """Verify the selected registered bundle again before sending any bytes."""
    observed, documents, seen = {}, {}, set()
    metadata_names = {"manifest.json", "checkpoint/recipe.json", "checkpoint/manifest.json"}
    with tarfile.open(path, "r:") as archive:
        for entry in archive:
            name = entry.name.rstrip("/") if entry.isdir() else entry.name
            pure = PurePosixPath(name)
            if (
                len(seen) >= 256
                or name in seen
                or len(name) > 1024
                or pure.is_absolute()
                or ".." in pure.parts
                or "\\" in name
                or "\x00" in name
                or pure.as_posix() != name
                or not pure.parts
                or pure.parts[0] != "policy"
            ):
                raise ValueError("Invalid policy artifact archive inventory")
            seen.add(name)
            if entry.isdir():
                continue
            if not entry.isfile() or len(pure.parts) < 2 or not 0 <= entry.size <= MAX_ARCHIVE:
                raise ValueError("Policy artifact archive must contain regular files only")
            relative = "/".join(pure.parts[1:])
            digest = hashlib.sha256()
            document = bytearray() if relative in metadata_names else None
            if document is not None and entry.size > 8 * 1024**2:
                raise ValueError("Policy artifact metadata is too large")
            with archive.extractfile(entry) as stream:
                while chunk := stream.read(1024**2):
                    digest.update(chunk)
                    if document is not None:
                        document.extend(chunk)
            observed[relative] = digest.hexdigest()
            if document is not None:
                documents[relative] = bytes(document)
    if observed.pop("manifest.json", None) != expected_sha:
        raise ValueError("Selected policy artifact manifest changed")
    manifest = _json(documents["manifest.json"])
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest.get("files"), dict)
        or manifest["files"] != observed
        or not isinstance(manifest.get("metadata"), dict)
    ):
        raise ValueError("Policy artifact contents differ from its registered manifest")
    recipe = (
        _json(documents["checkpoint/recipe.json"]) if "checkpoint/recipe.json" in documents else {}
    )
    if not isinstance(recipe, dict):
        raise ValueError("Invalid policy training recipe")
    return manifest["metadata"], recipe


def _descriptor(artifact, metadata, recipe, payload):
    if (
        artifact.format not in {"training_checkpoint", "native_checkpoint"}
        or metadata.get("architecture") != "act"
        or metadata.get("training_backend") != "lerobot"
        or metadata.get("reload_verified") is not True
    ):
        raise ValueError("Choose a reload-verified native LeRobot ACT checkpoint")
    if metadata != artifact.metadata:
        raise ValueError("Registered policy artifact metadata changed")
    records = [metadata, recipe]
    for source in list(records):
        for key in ("control_contract", "temporal_contract", "dataset"):
            if isinstance(source.get(key), dict):
                records.append(source[key])
    rates, units_known = set(), False
    for record in records:
        for key in ("fps", "action_fps", "dataset_fps"):
            if key in record:
                value = record[key]
                if type(value) is not int or not 1 <= value <= 40:
                    raise ValueError("Saved policy FPS is unsupported by this arm")
                rates.add(value)
        state, action = record.get("state_units"), record.get("action_units")
        if state is not None or action is not None:
            if state != UNITS or action != UNITS:
                raise ValueError(
                    "Saved policy units are incompatible; expected degrees and gripper percent"
                )
            units_known = True
        if "units" in record:
            if record["units"] != "degrees; gripper percent":
                raise ValueError("Saved policy units are incompatible with this SO-101 controller")
            units_known = True
        for key in ("joint_names", "joint_order", "state_names", "action_names"):
            if key in record:
                names = record[key]
                if (
                    not isinstance(names, list)
                    or not all(isinstance(name, str) for name in names)
                    or [name.removesuffix(".pos") for name in names] != JOINTS
                ):
                    raise ValueError("Saved policy joint order differs from this SO-101 controller")
    if len(rates) > 1:
        raise ValueError("Saved policy frame rates disagree")
    known_fps = next(iter(rates), None)
    if known_fps is not None and payload.fps is not None and payload.fps != known_fps:
        raise ValueError("Keep the saved training FPS unchanged")
    fps = known_fps if known_fps is not None else payload.fps
    if fps is None:
        raise ValueError("Supply the frame rate used when this policy was trained")
    if not units_known and not payload.confirm_so101_units:
        raise ValueError("Confirm six ordered SO-101 joints in degrees/gripper percent")
    camera = metadata.get("camera_keys")
    camera = camera[0] if isinstance(camera, list) and len(camera) == 1 else None
    if camera is not None and (
        not isinstance(camera, str)
        or not re.fullmatch(r"observation\.images\.[a-zA-Z0-9_-]{1,64}", camera)
    ):
        raise ValueError("Saved policy camera key is unsupported")
    identity = "\n".join([artifact.project_id, artifact.id, artifact.manifest_sha256])
    return {
        "schema_version": 1,
        "id": "artifact-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        "name": payload.name,
        "policy_type": "act",
        "fps": fps,
        "camera_key": camera,
        "joint_names": JOINTS,
        "state_units": UNITS,
        "action_units": UNITS,
        "checkpoint": "pretrained_model",
        "source_manifest_sha256": artifact.manifest_sha256,
    }


async def _spool(source, destination):
    size = 0
    try:
        with destination.open("xb") as output:
            if isinstance(source, Path):
                descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_ARCHIVE:
                        raise ValueError("Policy artifact archive exceeds its 2 GiB limit")
                    while chunk := await asyncio.to_thread(stream.read, 1024**2):
                        size += len(chunk)
                        if size > MAX_ARCHIVE:
                            raise ValueError("Policy artifact archive exceeds its 2 GiB limit")
                        await asyncio.to_thread(output.write, chunk)
            else:
                async for chunk in source:
                    if not isinstance(chunk, bytes):
                        raise ValueError("Invalid policy artifact stream")
                    size += len(chunk)
                    if size > MAX_ARCHIVE:
                        raise ValueError("Policy artifact archive exceeds its 2 GiB limit")
                    await asyncio.to_thread(output.write, chunk)
    finally:
        if not isinstance(source, Path) and hasattr(source, "aclose"):
            await source.aclose()
    if not size:
        raise ValueError("Policy artifact archive is empty")
    return size


async def _chunks(path):
    with path.open("rb") as stream:
        while chunk := await asyncio.to_thread(stream.read, 1024**2):
            yield chunk


async def install_artifact(app, project_id, artifact_id, payload: PolicyInstallRequest):
    """Stage a bounded verified archive, then install; this never loads or runs a model."""
    if await app.state.projects.get(project_id) is None:
        raise RobotLabError("Project not found", 404)
    lifecycle, bridge = app.state.execution.lifecycle, app.state.robot_lab
    try:
        artifact = await lifecycle.artifact(project_id, artifact_id)
        if (
            artifact.project_id != project_id
            or artifact.format not in {"training_checkpoint", "native_checkpoint"}
            or artifact.metadata.get("architecture") != "act"
            or artifact.metadata.get("training_backend") != "lerobot"
            or artifact.metadata.get("reload_verified") is not True
            or not re.fullmatch(r"[a-f0-9]{64}", artifact.manifest_sha256)
            or type(artifact.file_bytes) is not int
            or not 0 < artifact.file_bytes <= MAX_ARCHIVE
        ):
            raise ValueError("Choose a reload-verified native LeRobot ACT checkpoint")
        config = bridge.config()
        async with bridge.import_lock, asyncio.timeout(300):
            spool_root = bridge.data_dir / "policy-install"
            spool_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if (
                spool_root.is_symlink()
                or shutil.disk_usage(spool_root).free < HEADROOM + MAX_ARCHIVE
            ):
                raise ValueError("Policy installation needs its archive space plus 10 GiB free")
            with tempfile.TemporaryDirectory(prefix="artifact-", dir=spool_root) as temporary:
                path = Path(temporary) / "policy.tar"
                source = await lifecycle.download(project_id, artifact_id)
                size = await _spool(source, path)
                metadata, recipe = await asyncio.to_thread(
                    _archive_metadata, path, artifact.manifest_sha256
                )
                try:
                    descriptor = _descriptor(artifact, metadata, recipe, payload)
                except ValueError as error:
                    raise RobotLabError(str(error), 422) from None
                encoded = base64.b64encode(
                    json.dumps(descriptor, separators=(",", ":"), allow_nan=False).encode()
                ).decode()
                async with bridge.client.stream(
                    "POST",
                    bridge.url(config, "policy/install"),
                    headers=bridge.headers(config)
                    | {
                        "Content-Type": "application/x-tar",
                        "Content-Length": str(size),
                        "X-OpenJensen-Policy": encoded,
                    },
                    content=_chunks(path),
                    timeout=httpx.Timeout(120, connect=3),
                    follow_redirects=False,
                ) as response:
                    raw = await bridge.read(response, MAX_JSON)
                    try:
                        result = bridge.sanitize(strict_json(raw), config)
                    except ValueError, UnicodeError, RecursionError:
                        raise RobotLabError(
                            "Policy companion returned an invalid installation result"
                        ) from None
                    if not response.is_success:
                        message = (
                            result.get("error", result.get("detail"))
                            if isinstance(result, dict)
                            else None
                        )
                        status = (
                            response.status_code
                            if response.status_code in {400, 409, 413, 422, 503}
                            else 502
                        )
                        raise RobotLabError(
                            str(message)[:1000]
                            if isinstance(message, str)
                            else "Policy installation failed",
                            status,
                        )
                    if not isinstance(result, dict):
                        raise RobotLabError(
                            "Policy companion returned an invalid installation result"
                        )
                    return result
    except RobotLabError:
        raise
    except TimeoutError, httpx.HTTPError, OSError, RuntimeError:
        raise RobotLabError(
            "Policy installation could not complete; the previous policies are retained"
        ) from None
    except ValueError, tarfile.TarError, KeyError:
        raise RobotLabError(
            "Selected policy archive or compatibility settings are invalid", 422
        ) from None
