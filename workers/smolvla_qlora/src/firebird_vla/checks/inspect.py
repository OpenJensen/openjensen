"""Small public metadata reads only; never download checkpoint tensors during inspection."""

import json
import urllib.request

from .catalog import PROFILES


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "firebird-vla-qlora-check/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Metadata exceeds the 4 MiB inspection limit")
    return json.loads(raw)


def inspect_entry(entry):
    cp = entry["checkpoint"]
    if cp["source"].startswith("gs://"):
        return {
            "status": "blocked",
            "reason": "Requires hashed OpenPI source and a verified PyTorch conversion",
            "qlora_verified": False,
        }
    prefix = cp.get("subdirectory")
    prefix = f"{prefix}/" if prefix else ""
    info = get_json(
        f"https://huggingface.co/api/models/{cp['source']}/revision/{cp['revision']}?blobs=true"
    )
    if info["sha"] != cp["revision"]:
        raise ValueError("Hub resolved another checkpoint revision")
    files = {
        item["rfilename"][len(prefix) :]: item
        for item in info["siblings"]
        if item["rfilename"].startswith(prefix)
    }
    for name in ["config.json", *PROFILES[entry["id"]].get("auxiliary_files", [])]:
        if name not in files:
            raise ValueError(f"Required native checkpoint asset is missing: {name}")
    config = get_json(
        f"https://huggingface.co/{cp['source']}/resolve/{cp['revision']}/{prefix}config.json"
    )
    return {
        "status": "metadata_inspected",
        "qlora_verified": False,
        "license": (info.get("cardData") or {}).get("license"),
        "architecture": config.get("architectures", [config.get("type")]),
        "checkpoint_files": {
            name: {
                "bytes": item.get("size"),
                "lfs_sha256": (item.get("lfs") or {}).get("sha256"),
                "git_blob_id": item.get("blobId"),
            }
            for name, item in files.items()
        },
        "input_features": config.get("input_features"),
        "output_features": config.get("output_features"),
        "action_horizon": config.get("chunk_size", config.get("action_horizon")),
        "reason": "Metadata/asset presence checked; QLoRA still requires CUDA and a fixture",
    }
