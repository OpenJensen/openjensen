"""Validate explicit, project-owned dataset combinations before dispatch."""

from vla_platform.contracts import DatasetProfile


def validate_mixture(profiles: list[DatasetProfile], camera_keys: list[str], mappings: list[dict[str, str]]):
    if not 2 <= len(profiles) <= 8 or len(mappings) != len(profiles):
        raise ValueError("Choose two to eight datasets with explicit camera mappings")
    first = profiles[0]
    seen = set()
    for data, mapping in zip(profiles, mappings, strict=True):
        identity = (data.source, data.repo_id, data.revision, data.snapshot.manifest_sha256 if data.snapshot else None)
        if identity in seen:
            raise ValueError("The same immutable dataset cannot be included twice")
        seen.add(identity)
        if data.source != "huggingface":
            raise ValueError("Combined training currently supports pinned Hub datasets; train local snapshots separately")
        if data.format != "lerobot_v3" or data.fps != first.fps:
            raise ValueError("Combined training requires LeRobot v3 datasets with the same frame rate")
        for key in ("action", "observation.state"):
            feature, reference = data.features.get(key, {}), first.features.get(key, {})
            if not feature.get("shape") or feature.get("shape") != reference.get("shape") or feature.get("dtype") != reference.get("dtype"):
                raise ValueError(f"Combined datasets must have matching {key} shapes and types")
            if feature.get("names") != reference.get("names"):
                raise ValueError(f"Combined datasets must use the same ordered {key} names")
        if set(mapping) != set(camera_keys) or len(set(mapping.values())) != len(mapping):
            raise ValueError("Map each selected camera to a distinct camera in every dataset")
        for key, source_key in mapping.items():
            feature, reference = data.features.get(source_key, {}), first.features.get(key, {})
            if feature.get("dtype") not in {"video", "image"} or feature.get("shape") != reference.get("shape"):
                raise ValueError(f"Mapped camera {source_key} must match {key}'s image dimensions")
        if data.source == "local" and data.snapshot is None:
            raise ValueError("Local combinations require verified immutable training copies")
