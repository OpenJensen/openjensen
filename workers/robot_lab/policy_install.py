"""Private, bounded installation of a verified application policy artifact."""

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from hardware import JOINTS
from policy import UNITS, admit
from policy_contract import SMOLVLM_ASSET_FILES, validate_policy_config

MAX_ARCHIVE = 2 * 1024**3


def safe_name(name):
    path = PurePosixPath(name)
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 256
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != name.rstrip("/")
    ):
        raise ValueError("Policy archive has an unsafe member name")
    return str(path)


def install_archive(lab, source, length, descriptor):
    allowed = {
        "schema_version",
        "id",
        "name",
        "policy_type",
        "offline_only",
        "fps",
        "camera_key",
        "joint_names",
        "state_units",
        "action_units",
        "checkpoint",
        "source_manifest_sha256",
    }
    offline = descriptor.get("offline_only", False) if isinstance(descriptor, dict) else False
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) - allowed
        or type(descriptor.get("schema_version")) is not int
        or descriptor["schema_version"] != 1
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(descriptor.get("id", "")))
        or descriptor.get("policy_type") not in {"act", "smolvla"}
        or type(offline) is not bool
        or (
            not offline
            and (
                descriptor.get("joint_names") != list(JOINTS)
                or descriptor.get("state_units") != UNITS
                or descriptor.get("action_units") != UNITS
            )
        )
        or descriptor.get("checkpoint") != "pretrained_model"
        or (
            not offline
            and (type(descriptor.get("fps")) is not int or not 1 <= descriptor["fps"] <= 40)
        )
        or (
            offline
            and descriptor.get("fps") is not None
            and (type(descriptor["fps"]) is not int or not 1 <= descriptor["fps"] <= 240)
        )
        or not re.fullmatch(r"[a-f0-9]{64}", str(descriptor.get("source_manifest_sha256", "")))
    ):
        raise ValueError(
            "Install a verified native ACT artifact with its SO-101 units and frame rate"
        )
    if type(length) is not int or not 1 <= length <= MAX_ARCHIVE:
        raise ValueError("Policy archive must be a complete TAR of at most 2 GiB")
    if shutil.disk_usage(lab.data_dir).free < 10 * 1024**3 + 2 * length:
        raise ValueError(
            "Policy installation needs space for staging while retaining 10 GiB headroom"
        )
    runner = lab.policy
    with runner.lock, lab.lifecycle_lock:
        if lab.mode != "idle" or runner.phase in {"running", "loading", "checking", "installing"}:
            raise ValueError("Stop hardware and policy operations before installing")
        previous = "ready" if runner.runtime and runner.model else "idle"
        runner.operation += 1
        operation = runner.operation
        runner.phase, runner.error = "installing", None
    try:
        with tempfile.TemporaryDirectory(prefix=".policy-install-", dir=lab.data_dir) as temporary:
            temporary = Path(temporary)
            archive_path = temporary / "upload.tar"
            with archive_path.open("xb") as output:
                os.chmod(archive_path, 0o600)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024**2, remaining))
                    if not chunk:
                        raise ValueError("Policy upload ended before its declared length")
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            staging = temporary / "policies" / descriptor["id"]
            checkpoint = staging / "pretrained_model"
            checkpoint.mkdir(parents=True, mode=0o700)
            os.chmod(staging, 0o700)
            seen, hashes, outer, total = set(), {}, None, 0
            with tarfile.open(archive_path, "r:") as archive:
                for member in archive:
                    name = safe_name(member.name)
                    if name in seen or len(seen) >= 512:
                        raise ValueError("Policy archive has duplicate or excessive members")
                    seen.add(name)
                    if member.isdir():
                        if name != "policy" and not name.startswith("policy/"):
                            raise ValueError("Unexpected policy archive directory")
                        continue
                    if not member.isfile() or not name.startswith("policy/"):
                        raise ValueError("Policy archive accepts regular artifact files only")
                    total += member.size
                    if member.size < 0 or total > MAX_ARCHIVE or len(hashes) >= 256:
                        raise ValueError("Policy archive exceeds its payload bounds")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("Policy archive member is unavailable")
                    relative = name.removeprefix("policy/")
                    if relative == "manifest.json":
                        if member.size > 1024**2:
                            raise ValueError("Policy artifact manifest is oversized")
                        outer = stream.read()
                        hashes[relative] = hashlib.sha256(outer).hexdigest()
                        continue
                    target = None
                    for prefix in ("checkpoint/pretrained_model/", "pretrained_model/"):
                        if relative.startswith(prefix):
                            basename = relative.removeprefix(prefix)
                            parts = PurePosixPath(basename).parts
                            if not basename or (
                                len(parts) != 1
                                and (
                                    len(parts) != 3
                                    or parts[:2] != ("assets", "smolvlm")
                                    or parts[2] not in SMOLVLM_ASSET_FILES
                                )
                            ):
                                raise ValueError(
                                    "Policy checkpoint has an unsupported nested asset"
                                )
                            target = checkpoint / basename
                            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                            break
                    checksum = hashlib.sha256()
                    output = target.open("xb") if target else None
                    try:
                        if target:
                            os.chmod(target, 0o600)
                        while chunk := stream.read(1024**2):
                            checksum.update(chunk)
                            if output:
                                output.write(chunk)
                        if output:
                            output.flush()
                            os.fsync(output.fileno())
                    finally:
                        if output:
                            output.close()
                        stream.close()
                    hashes[relative] = checksum.hexdigest()
                archive.fileobj.seek(archive.offset)
                if archive.fileobj.read(1024) != bytes(1024):
                    raise ValueError("Policy archive is missing its closing blocks")
                while trailing := archive.fileobj.read(1024**2):
                    if trailing.strip(b"\0"):
                        raise ValueError("Policy archive has trailing payload")
            expected_manifest = descriptor["source_manifest_sha256"]
            if outer is None or hashlib.sha256(outer).hexdigest() != expected_manifest:
                raise ValueError("Selected policy artifact manifest changed")
            manifest = json.loads(outer)
            expected = manifest.get("files") if isinstance(manifest, dict) else None
            if (
                not isinstance(expected, dict)
                or not expected
                or len(expected) > 256
                or any(
                    not isinstance(k, str) or not re.fullmatch(r"[a-f0-9]{64}", str(v))
                    for k, v in expected.items()
                )
                or hashes != expected | {"manifest.json": expected_manifest}
            ):
                raise ValueError("Policy artifact inventory or hashes differ from its manifest")
            directories = {"policy"}
            for name in hashes:
                directories.update(
                    str(p) for p in PurePosixPath("policy/" + name).parents if str(p) != "."
                )
            if seen - {"policy/" + name for name in hashes} - directories:
                raise ValueError("Unexpected policy archive directories")
            config = validate_policy_config(checkpoint)
            images = [
                k
                for k, value in config.get("input_features", {}).items()
                if value.get("type") == "VISUAL"
                or k.startswith(("observation.images.", "observation.image"))
            ]
            if not images or descriptor.get("camera_key") not in {None, *images}:
                raise ValueError("Policy requires one matching camera feature")
            saved = dict(descriptor, camera_key=descriptor.get("camera_key") or images[0])
            saved.pop("id")
            (staging / "run_policy.json").write_text(json.dumps(saved, allow_nan=False) + "\n")
            os.chmod(staging / "run_policy.json", 0o600)
            admit(staging.parent, descriptor["id"])
            destination = runner.root / descriptor["id"]
            with runner.lock, lab.lifecycle_lock:
                if operation != runner.operation or lab.mode != "idle":
                    raise ValueError("Policy installation was cancelled")
                if destination.exists() or destination.is_symlink():
                    raise ValueError("This policy is already installed")
                staging.rename(destination)
            return runner.catalog() | {"installed_policy_id": descriptor["id"]}
    except Exception as error:
        with runner.lock:
            if operation == runner.operation:
                runner.error = str(error)
        raise
    finally:
        with runner.lock:
            if operation == runner.operation:
                runner.phase = previous
