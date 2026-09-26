"""Shared preview boundary checks, without a native reader dependency."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError
from vla_platform.contracts import DatasetProfile, LocalDatasetPreview, LocalPreviewLimits
from vla_platform.datasets.local_preview import PreviewLimits

FIXTURE = Path(__file__).parent / "fixtures/lerobot_v3_preview"


@pytest.fixture
def payload():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())["files"]
    path = "data/chunk-000/file-000.parquet"
    return {
        "kind": "frames",
        "metadata_sha256": manifest["meta/info.json"]["sha256"],
        "file": {
            "path": path,
            "size_bytes": manifest[path]["bytes"],
            "sha256": manifest[path]["sha256"],
        },
        "reader": "pyarrow==25.0.1",
        "total_file_rows": 6,
        "rows": [{"action": [0.0, 0.0]}],
        "returned_rows": 1,
        "truncated": True,
        "columns": ["action"],
        "omitted_columns": [],
        "read_bytes": 4915,
        "declared_decoded_bytes": 1206,
        "limits": asdict(PreviewLimits()),
        "warnings": ["Structural contract fixture; not runtime or platform evidence."],
    }


def profile_data():
    raw = (FIXTURE / "meta/info.json").read_bytes()
    info = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "source": "local",
        "revision": f"metadata-sha256:{digest}",
        "format": "lerobot_v3",
        "metadata_sha256": digest,
        "inspected_at": "2026-09-26T14:00:00Z",
        "warnings": ["Structural contract fixture only."],
        **{key: info[key] for key in ("total_episodes", "total_frames", "fps", "features")},
    }


def test_preview_limits_match_reader_defaults():
    assert LocalPreviewLimits().model_dump() == asdict(PreviewLimits())


def test_metadata_profile_serialization_stays_metadata_only():
    result = DatasetProfile.model_validate(profile_data()).model_dump(mode="json")
    assert result["inspection_scope"] == "metadata_only"
    assert "preview" not in result


def test_bounded_profile_round_trip_requires_matching_identity(payload):
    data = {**profile_data(), "inspection_scope": "bounded_parquet_rows", "preview": payload}
    profile = DatasetProfile.model_validate(data)
    assert profile.preview.metadata_sha256 == profile.metadata_sha256
    assert DatasetProfile.model_validate_json(profile.model_dump_json()) == profile


@pytest.mark.parametrize("change", ["missing", "wrong_hash", "wrong_format", "hf", "wrong_scope"])
def test_profile_rejects_unproven_preview_scope(payload, change):
    data = {**profile_data(), "inspection_scope": "bounded_parquet_rows", "preview": payload}
    if change == "missing":
        del data["preview"]
    elif change == "wrong_hash":
        data["preview"]["metadata_sha256"] = "a" * 64
    elif change == "wrong_format":
        data["format"] = "lerobot_v2"
    elif change == "hf":
        data.update(source="huggingface", repo_id="fixture/data", revision="a" * 40)
    else:
        data["inspection_scope"] = "metadata_only"
    with pytest.raises(ValidationError):
        DatasetProfile.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("returned_rows", 2),
        ("total_file_rows", 0),
        ("truncated", False),
        ("columns", ["other"]),
        ("columns", ["action", "action"]),
        ("omitted_columns", ["action"]),
        ("read_bytes", 134217729),
        ("declared_decoded_bytes", 67108865),
        ("warnings", []),
        ("reader", "unknown"),
        ("metadata_sha256", ""),
    ],
)
def test_preview_rejects_inconsistent_claims(payload, field, value):
    payload[field] = value
    with pytest.raises(ValidationError):
        LocalDatasetPreview.model_validate(payload)


@pytest.mark.parametrize(
    "path", ["/data/file.parquet", "data/../file.parquet", "meta/file.parquet"]
)
def test_preview_file_identity_rejects_unsafe_paths(payload, path):
    payload["file"]["path"] = path
    with pytest.raises(ValidationError):
        LocalDatasetPreview.model_validate(payload)


def test_preview_preserves_row_text_and_rejects_excess_output(payload):
    payload.update(columns=["tasks"], rows=[{"tasks": ["  synthetic task  "]}])
    result = LocalDatasetPreview.model_validate(payload)
    assert result.rows[0]["tasks"] == ["  synthetic task  "]
    oversized = deepcopy(payload)
    oversized["rows"][0]["tasks"] = ["x" * 2000]
    oversized["limits"]["max_output_bytes"] = 1024
    with pytest.raises(ValidationError, match="serialized JSON byte budget"):
        LocalDatasetPreview.model_validate(oversized)
