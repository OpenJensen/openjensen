"""Snapshot-derived semantics and resume/export identity, without ML or cloud."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from firebird_vla.control_contract import check_checkpoint, check_resolved, derive, save
from firebird_vla.control_schema import FILE, canonical, optional, validate
from firebird_vla.local_dataset import bind_local_recipe, load_operation_snapshot


def snapshot(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    joints = ["joint_" + str(i) for i in range(6)]
    features = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": joints},
        "action": {"dtype": "float32", "shape": [6], "names": joints},
        "observation.images.front": {"dtype": "video", "shape": [32, 32, 3]},
    }
    provenance = {
        "schema_version": 1,
        "controller": "joint_position_targets",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "action_column": "action",
        "requested_action_column": "teaching.requested_action",
        "joint_order": joints,
        "camera_prim": "/World/Camera",
        "task_success_verified": False,
        "scene_hash_scope": "root USD bytes; referenced assets not inventoried",
        "sources": [
            {
                "source_session_id": f"s{i}",
                "scene_sha256": "a" * 64,
                "origin": "synthetic",
                "lineage_group": f"g{i}",
            }
            for i in range(2)
        ],
        "episodes": [{"episode_index": i, "source_session_id": f"s{i}"} for i in range(2)],
    }
    (root / "meta/info.json").write_bytes(canonical({"fps": 20, "features": features}))
    (root / "meta/firebird-demonstrations.json").write_bytes(canonical(provenance))
    manifest = {
        "schema_version": 1,
        "format": "lerobot_v3",
        "features": features,
        "total_episodes": 2,
        "total_frames": 4,
        "lineage_validated": True,
        "lineage": [
            {"episode_index": i, "origin": "synthetic", "lineage_group": f"g{i}"} for i in range(2)
        ],
        "files": [
            {
                "path": p.relative_to(root).as_posix(),
                "size": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ],
    }
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "firebird-snapshot.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    job = {
        "dataset_snapshot": {"path": str(root), "manifest_sha256": sha, "id": "sha256:" + sha},
        "dataset": {"format": "lerobot_v3", "features": features},
    }
    recipe = {"camera_keys": ["observation.images.front"], "validation_fraction": 0.5, "seed": 42}
    return root, manifest, job, recipe


def test_verified_snapshot_contract_and_resume_keep_exact_coordinate_identity(tmp_path):
    root, manifest, job, recipe = snapshot(tmp_path)
    bound = bind_local_recipe(job, recipe)
    record = bound["control_contract"]
    assert record["action_fps"] == 20
    assert record["source"]["origins"] == ["synthetic"]
    assert record["state_key"] == "observation.state" and record["action_key"] == "action"
    assert str(root) not in json.dumps(record)
    (tmp_path / "dataset-snapshot.json").write_text(json.dumps(job["dataset_snapshot"]))
    assert load_operation_snapshot(tmp_path, bound) == (root, manifest)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "pretrained_model").mkdir(parents=True)
    save(checkpoint, record)
    save(checkpoint / "pretrained_model", record)
    assert check_checkpoint(checkpoint, bound) == optional(checkpoint)
    (checkpoint / "pretrained_model" / FILE).unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        check_checkpoint(checkpoint, bound)
    del bound["control_contract"]
    with pytest.raises(ValueError, match="provenance differs"):
        load_operation_snapshot(tmp_path, bound)


@pytest.mark.parametrize(
    "change", ["units", "names", "fps", "camera", "lineage", "origin", "scene"]
)
def test_declared_recording_mismatches_refused(tmp_path, change):
    root, manifest, job, recipe = snapshot(tmp_path)
    # Rehash a changed dataset as a distinct legitimate snapshot. Its semantics must still fail.
    name = "meta/firebird-demonstrations.json"
    value = json.loads((root / name).read_text())
    if change == "units":
        value["action_units"] = "degrees"
    elif change == "names":
        value["joint_order"].reverse()
    elif change == "camera":
        value["camera_prim"] = "not-a-prim"
    elif change == "lineage":
        value["sources"][0]["lineage_group"] = "wrong"
    elif change == "origin":
        value["sources"][0]["origin"] = "recorded"
    elif change == "scene":
        value["sources"][0]["scene_sha256"] = "unknown"
    else:
        name = "meta/info.json"
        value = json.loads((root / name).read_text())
        value["fps"] = True
    raw = canonical(value)
    (root / name).write_bytes(raw)
    for entry in manifest["files"]:
        if entry["path"] == name:
            entry.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        derive(root, manifest, job["dataset_snapshot"], recipe)


def test_mutated_metadata_after_snapshot_verification_refused(tmp_path):
    root, manifest, job, recipe = snapshot(tmp_path)
    (root / "meta/info.json").write_text("{}")
    with pytest.raises(ValueError, match="differs from verified"):
        derive(root, manifest, job["dataset_snapshot"], recipe)


def test_resolved_policy_and_clock_bound(tmp_path):
    root, manifest, job, recipe = snapshot(tmp_path)
    record = derive(root, manifest, job["dataset_snapshot"], recipe)
    config = SimpleNamespace(
        type="act",
        input_features={
            "observation.state": SimpleNamespace(type="STATE", shape=(6,)),
            "observation.images.front": SimpleNamespace(type="VISUAL", shape=(3, 32, 32)),
        },
        output_features={"action": SimpleNamespace(type="ACTION", shape=(6,))},
    )
    check_resolved(record, config, 20)
    with pytest.raises(ValueError, match="FPS"):
        check_resolved(record, config, 30)
    config.input_features["observation.state"].shape = (5,)
    with pytest.raises(ValueError, match="features"):
        check_resolved(record, config, 20)
    bad = copy.deepcopy(record)
    bad["schema_version"] = True
    with pytest.raises(ValueError):
        validate(bad)


def test_schema_copies_identical_without_runtime_package_dependency():
    root = Path(__file__).resolve().parents[3]
    files = [
        root / p
        for p in (
            "workers/smolvla_qlora/src/firebird_vla/control_schema.py",
            "workers/act_optimizer/src/firebird_act/control_schema.py",
            "workers/isaac_sim/sim_worker/rollout/control_schema.py",
        )
    ]
    assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in files}) == 1
