"""Strict metadata-only local admission; no Arrow, model, worker or network calls."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from vla_platform.contracts import IntakeRequest
from vla_platform.datasets.inspect import (
    MAX_LOCAL_METADATA_DEPTH,
    MAX_LOCAL_METADATA_FEATURES,
    MAX_LOCAL_METADATA_NODES,
    MAX_LOCAL_METADATA_STRING,
    MAX_METADATA_BYTES,
    inspect_local,
    local_metadata,
    profile,
)

INFO = {
    "codebase_version": "v3.0",
    "total_episodes": 2,
    "total_frames": 20,
    "fps": 10,
    "features": {
        "action": {"dtype": "float32", "shape": [6]},
        "observation.state": {"dtype": "float32", "shape": [6]},
        "observation.images.front": {"dtype": "video", "shape": [32, 32, 3]},
    },
}
RAW = json.dumps(INFO).encode()
REQUEST = IntakeRequest(source="local", path="robot")


def with_extra(value):
    return RAW[:-1] + b', "extra":' + value + b"}"


@pytest.mark.parametrize(
    "raw,reason",
    [
        (RAW[:-1] + b', "total_frames":20}', "Duplicate"),
        (with_extra(b'{"note":1,"note":2}'), "Duplicate"),
        (with_extra(b"NaN"), "Non-finite"),
        (with_extra(b"Infinity"), "Non-finite"),
        (with_extra(b"-Infinity"), "Non-finite"),
        (with_extra(b"1e999"), "Non-finite"),
        (with_extra(b'"' + b"x" * (MAX_LOCAL_METADATA_STRING + 1) + b'"'), "string"),
        (
            with_extra(b"[" * MAX_LOCAL_METADATA_DEPTH + b"0" + b"]" * MAX_LOCAL_METADATA_DEPTH),
            "structure",
        ),
        (with_extra(b"[" + b"0," * MAX_LOCAL_METADATA_NODES + b"0]"), "structure"),
    ],
    ids=[
        "duplicate-top",
        "duplicate-nested",
        "nan",
        "inf",
        "negative-inf",
        "overflow",
        "string",
        "depth",
        "nodes",
    ],
)
def test_single_unsafe_value_in_otherwise_valid_local_metadata_is_refused(raw, reason):
    # Every input otherwise satisfies the current LeRobot profile contract.
    with pytest.raises(ValueError, match=reason):
        profile(raw, REQUEST, "metadata-sha256:" + hashlib.sha256(raw).hexdigest())


def test_deep_parser_failure_is_a_bounded_validation_error():
    raw = with_extra(b"[" * 2000 + b"0" + b"]" * 2000)
    with pytest.raises(ValueError, match="nested|structure"):
        local_metadata(raw)


def test_structural_limits_accept_exact_boundaries():
    depth = MAX_LOCAL_METADATA_DEPTH - 1
    raw = b'{"padding":' + b"[" * depth + b"0" + b"]" * depth + b"}"
    assert local_metadata(raw)["padding"]
    nodes = b'{"padding":[' + b"0," * (MAX_LOCAL_METADATA_NODES - 4) + b"0]}"
    assert len(local_metadata(nodes)["padding"]) == MAX_LOCAL_METADATA_NODES - 3
    assert (
        local_metadata(with_extra(b'"' + b"x" * MAX_LOCAL_METADATA_STRING + b'"'))["extra"]
        == "x" * MAX_LOCAL_METADATA_STRING
    )
    exact_bytes = RAW + b" " * (MAX_METADATA_BYTES - len(RAW))
    assert local_metadata(exact_bytes) == INFO
    with pytest.raises(ValueError, match="2 MiB"):
        local_metadata(exact_bytes + b" ")


def test_feature_limit_is_explicit_without_restricting_valid_feature_names():
    features = {
        **INFO["features"],
        **{
            f"observation.extra:{index}": {"dtype": "float32", "shape": [1]}
            for index in range(MAX_LOCAL_METADATA_FEATURES - len(INFO["features"]))
        },
    }
    raw = json.dumps({**INFO, "features": features}).encode()
    result = profile(raw, REQUEST, "metadata-sha256:" + hashlib.sha256(raw).hexdigest())
    assert len(result.features) == MAX_LOCAL_METADATA_FEATURES
    features["observation.too_many"] = {"dtype": "float32", "shape": [1]}
    with pytest.raises(ValueError, match="feature count"):
        local_metadata(json.dumps({**INFO, "features": features}).encode())


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_valid_local_metadata_preserves_raw_identity_and_declared_only_scope(tmp_path, version):
    dataset = tmp_path / "robot"
    (dataset / "meta").mkdir(parents=True)
    raw = (
        json.dumps(
            {
                **INFO,
                "codebase_version": version,
                "extra": {"task": "Pick up the cup", "optional": None, "enabled": True},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            },
            indent=2,
        ).encode()
        + b"\n"
    )
    target = dataset / "meta/info.json"
    target.write_bytes(raw)
    before = target.stat()
    result = inspect_local(REQUEST, str(tmp_path))
    assert result.metadata_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.revision == "metadata-sha256:" + result.metadata_sha256
    assert result.inspection_scope == "metadata_only"
    assert result.format == f"lerobot_v{version[1]}"
    assert result.total_frames == 20 and result.features == INFO["features"]
    assert any("not validated against frames" in item for item in result.warnings)
    assert any("not established" in item for item in result.warnings)
    assert target.read_bytes() == raw and target.stat().st_mtime_ns == before.st_mtime_ns
    assert not (dataset / "data").exists()


def test_existing_in_root_symlinks_remain_allowed_and_outside_targets_refused(tmp_path):
    allowed = tmp_path / "allowed"
    dataset = allowed / "robot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_bytes(RAW)
    (allowed / "alias").symlink_to(dataset, target_is_directory=True)
    inside = inspect_local(IntakeRequest(source="local", path="alias"), str(allowed))
    assert inside.metadata_sha256 == hashlib.sha256(RAW).hexdigest()
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(allowed, target_is_directory=True)
    assert inspect_local(REQUEST, str(root_alias)).metadata_sha256 == inside.metadata_sha256
    shared_metadata = allowed / "shared-info.json"
    shared_metadata.write_bytes(RAW)
    (dataset / "meta/info.json").unlink()
    (dataset / "meta/info.json").symlink_to(shared_metadata)
    assert inspect_local(REQUEST, str(allowed)).metadata_sha256 == inside.metadata_sha256
    outside = tmp_path / "outside"
    (outside / "meta").mkdir(parents=True)
    (outside / "meta/info.json").write_bytes(RAW)
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="within"):
        inspect_local(IntakeRequest(source="local", path="escape"), str(allowed))
    (dataset / "meta/info.json").unlink()
    (dataset / "meta/info.json").symlink_to(outside / "meta/info.json")
    with pytest.raises(ValueError, match="within"):
        inspect_local(REQUEST, str(allowed))


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_local_metadata_is_rejected_before_open(tmp_path, monkeypatch, kind):
    dataset = tmp_path / "robot"
    (dataset / "meta").mkdir(parents=True)
    target = dataset / "meta/info.json"
    if kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO fixture requires POSIX; directory admission still runs")
        os.mkfifo(target)
    else:
        target.mkdir()

    def forbidden_open(*args, **kwargs):
        raise AssertionError("Nonregular metadata must be refused before a possibly blocking open")

    # A missing guard fails promptly; this regression cannot block in a FIFO open.
    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(ValueError, match="regular file"):
        inspect_local(REQUEST, str(tmp_path))
