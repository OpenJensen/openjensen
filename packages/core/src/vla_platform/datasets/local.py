"""Small, local-only reads. Dataset metadata never supplies executable code."""

import json
import math
import os
import re
import stat
from pathlib import Path, PureWindowsPath
from string import Formatter

MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_METADATA_NODES = 16_384
MAX_METADATA_DEPTH = 24
MAX_FEATURES = 128
MAX_PATH = 1024
_TEMPLATE_FIELDS = {"episode_chunk", "episode_index", "chunk_index", "file_index", "video_key"}


def _local_path(value: str) -> Path:
    """Reject remote, device, ADS and ambiguous Windows names on every host."""
    if not value or len(value) > MAX_PATH or any(ord(c) < 32 for c in value):
        raise ValueError("Invalid local path")
    normalized = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    if normalized.startswith("//") or (windows.drive and not windows.root):
        raise ValueError("UNC, device and drive-relative paths are unsupported")
    if os.name != "nt" and (windows.drive or "\\" in value):
        raise ValueError("Windows paths are unsupported on this host")
    for index, part in enumerate(normalized.split("/")):
        if index == 0 and windows.is_absolute() and re.fullmatch(r"[a-zA-Z]:", part):
            continue
        if part == ".." or ":" in part:
            raise ValueError("Paths must remain within the configured local root")
        if part not in {"", "."} and (
            part.endswith((" ", "."))
            or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])(?:\..*)?", part)
        ):
            raise ValueError("Reserved or ambiguous local path component")
    return Path(value)


def _reject_links(path: Path) -> None:
    # Walk lexically, without resolve(): resolving a junction could access a network share.
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlinks, junctions and reparse points are unsupported")


def local_root(value: str | None) -> Path:
    if not value:
        raise ValueError("Local intake is disabled; configure FIREBIRD_LOCAL_DATA_ROOT first")
    root = Path(os.path.abspath(_local_path(value)))
    _reject_links(root)
    if not root.is_dir():
        raise ValueError("Configured local root must be an existing directory")
    return root


def confined_path(root: Path, value: str) -> Path:
    candidate = _local_path(value)
    candidate = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
    # Check containment BEFORE filesystem access, including for nonexistent targets.
    if not candidate.is_relative_to(root):
        raise ValueError("Paths must remain within the configured local root")
    _reject_links(candidate)
    return candidate


def safe_metadata(raw: bytes) -> dict:
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Metadata exceeds the 2 MiB inspection limit")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate metadata keys are unsupported")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("Non-finite metadata numbers are unsupported")

    try:
        info = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid or excessively nested JSON metadata") from exc
    stack = [(info, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > MAX_METADATA_DEPTH or nodes > MAX_METADATA_NODES:
            raise ValueError("Metadata structure exceeds the inspection limit")
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((value, depth + 1) for value in item.values())
        elif isinstance(item, list):
            stack.extend((value, depth + 1) for value in item)
        elif isinstance(item, str) and len(item) > 16_384:
            raise ValueError("Metadata string exceeds the inspection limit")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Non-finite metadata numbers are unsupported")
    if not isinstance(info, dict):
        raise ValueError("Expected a LeRobot metadata object")
    required = {"codebase_version", "features", "total_episodes", "total_frames", "fps"}
    if missing := required - info.keys():
        raise ValueError(f"Missing required local metadata fields: {', '.join(sorted(missing))}")
    features = info.get("features")
    if not isinstance(features, dict) or len(features) > MAX_FEATURES:
        raise ValueError("Metadata feature count exceeds the inspection limit")
    for key in features:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", key):
            raise ValueError("Unsupported metadata feature name")
    for key in ("data_path", "video_path"):
        if info.get(key) is not None:
            initial_path(info[key], "observation.images.camera")
    return info


def initial_path(template: str, video_key: str = "") -> str:
    """Only simple known LeRobot substitutions; no attribute/index traversal or conversion."""
    if not isinstance(template, str) or len(template) > MAX_PATH:
        raise ValueError("Invalid metadata path template")
    try:
        for _, field, spec, conversion in Formatter().parse(template):
            if field is not None and (
                field not in _TEMPLATE_FIELDS
                or conversion is not None
                or (spec and not re.fullmatch(r"0?[1-6]?d", spec))
                or (field == "video_key" and spec)
            ):
                raise ValueError("Unsafe metadata path template")
        values = dict.fromkeys(_TEMPLATE_FIELDS, 0)
        values["video_key"] = video_key
        value = template.format_map(values)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unsafe metadata path template") from exc
    candidate = _local_path(value)
    if candidate.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError("Metadata paths must be relative to the dataset")
    return value


def read_local_metadata(path: str, allowed_root: str | None) -> tuple[Path, bytes, dict]:
    dataset = confined_path(local_root(allowed_root), path)
    target = confined_path(dataset, "meta/info.json")
    if not target.is_file():
        raise ValueError("Missing local metadata file: meta/info.json")
    with target.open("rb") as handle:
        raw = handle.read(MAX_METADATA_BYTES + 1)
    return dataset, raw, safe_metadata(raw)


def declared_files(dataset: Path, info: dict) -> list[dict]:
    """Bounded probes only: initial declared shard and up to eight video paths, no scan."""
    checks = []
    if info.get("data_path"):
        checks.append(("data", initial_path(info["data_path"])))
    video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    if info.get("video_path"):
        checks.extend(("video", initial_path(info["video_path"], key)) for key in video_keys[:8])
    result = [
        {
            "kind": kind,
            "path": relative.replace("\\", "/"),
            "status": "present" if confined_path(dataset, relative).is_file() else "missing",
        }
        for kind, relative in checks
    ]
    for kind, missing in (
        ("data", not info.get("data_path")),
        ("video", video_keys and not info.get("video_path")),
    ):
        if missing:
            result.append(
                {"kind": kind, "path": None, "status": "unverified", "reason": "No path template"}
            )
    return result
