import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import httpx

from vla_platform.contracts import DatasetProfile, IntakeRequest, now

MAX_METADATA_BYTES = 2 * 1024 * 1024


def profile(
    raw: bytes, request: IntakeRequest, revision: str, license_name: str | None = None
) -> DatasetProfile:
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Metadata exceeds the 2 MiB inspection limit")
    info = json.loads(raw)
    if not isinstance(info, dict):
        raise ValueError("Expected a LeRobot metadata object")
    version = str(info.get("codebase_version", ""))
    if not re.fullmatch(r"v[23]\.\d+(\.\d+)?", version):
        raise ValueError("Unsupported dataset format: expected LeRobot v2/v3 meta/info.json")
    features = info.get("features")
    if not isinstance(features, dict) or "action" not in features:
        raise ValueError("Robotics metadata must declare an action feature")
    if not any(key.startswith("observation.") for key in features):
        raise ValueError("Robotics metadata must declare observation features")
    for key, feature in features.items():
        if not isinstance(feature, dict) or not isinstance(feature.get("dtype"), str):
            raise ValueError(f"Invalid feature definition: {key}")
        shape = feature.get("shape")
        if not isinstance(shape, list) or not all(type(n) is int and n > 0 for n in shape):
            raise ValueError(f"Invalid feature shape: {key}")
    warnings = [
        "Metadata-only inspection: episode counts and schemas are source-declared, "
        "not validated against frames.",
        "Action units, reference frames, controller semantics and calibration are not verified.",
        "Compatibility with a policy, robot or simulator is not established by this inspection.",
        "No videos, frame data, task instructions or evaluation results "
        "were downloaded or inferred.",
    ]
    if not license_name:
        warnings.append("License metadata is missing; inspect the source terms before reuse.")
    if request.source == "local":
        warnings.append(
            "Local revision hashes meta/info.json only; it is not a full dataset snapshot."
        )
    return DatasetProfile(
        source=request.source,
        repo_id=request.repo_id,
        revision=revision,
        format=f"lerobot_v{version[1]}",
        robot_type=info.get("robot_type"),
        total_episodes=info["total_episodes"],
        total_frames=info["total_frames"],
        fps=info["fps"],
        features=features,
        license=license_name,
        metadata_sha256=hashlib.sha256(raw).hexdigest(),
        inspected_at=now(),
        warnings=warnings,
    )


async def bounded_get(client: httpx.AsyncClient, url: str) -> bytes:
    async with client.stream("GET", url) as response:
        if response.status_code in {401, 403}:
            raise ValueError(
                "This source is gated/private; initial intake supports public datasets only"
            )
        if response.status_code == 404:
            raise ValueError("Dataset, revision or LeRobot metadata was not found")
        response.raise_for_status()
        content = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_METADATA_BYTES:
                raise ValueError("Metadata exceeds the 2 MiB inspection limit")
        return bytes(content)


async def inspect_hub(request: IntakeRequest) -> DatasetProfile:
    repo = quote(request.repo_id or "", safe="/")
    revision = quote(request.revision, safe="")
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, max_redirects=5) as client:
        response = json.loads(
            await bounded_get(
                client, f"https://huggingface.co/api/datasets/{repo}/revision/{revision}"
            )
        )
        sha = response.get("sha", "")
        if not re.fullmatch(r"[a-f0-9]{40}", sha):
            raise ValueError("Hub did not provide an immutable dataset revision")
        raw = await bounded_get(
            client, f"https://huggingface.co/datasets/{repo}/resolve/{sha}/meta/info.json"
        )
        card = response.get("cardData") or {}
        license_name = card.get("license") if isinstance(card, dict) else None
        return profile(raw, request, sha, license_name if isinstance(license_name, str) else None)


def inspect_local(request: IntakeRequest, allowed_root: str | None) -> DatasetProfile:
    if not allowed_root:
        raise ValueError("Local intake is disabled; configure FIREBIRD_LOCAL_DATA_ROOT first")
    root = Path(allowed_root).resolve(strict=True)
    candidate = Path(request.path or "")
    dataset = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=True)
    target = (dataset / "meta/info.json").resolve(strict=True)
    if not dataset.is_relative_to(root) or not target.is_relative_to(root):
        raise ValueError("Dataset metadata must remain within FIREBIRD_LOCAL_DATA_ROOT")
    with target.open("rb") as handle:
        raw = handle.read(MAX_METADATA_BYTES + 1)
    digest = hashlib.sha256(raw).hexdigest()
    return profile(raw, request, f"metadata-sha256:{digest}")
