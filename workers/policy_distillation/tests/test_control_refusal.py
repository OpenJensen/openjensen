"""Dependency-light admission/identity checks, not a native distillation proof."""

import copy
from pathlib import Path

import pytest
from firebird_act.bundle import canonical, inventory
from firebird_distill.application import native_model_id
from firebird_distill.contracts import corpus, digest, read, teacher_info
from firebird_distill.provenance import (
    action_fps, check_snapshot_control, inherited_files, policy_metadata,
)
from semantics_fixture import control_record, policy, temporal_record


def observation_corpus(root, config, metadata, processors, *, legacy=False):
    root.mkdir()
    doc = {
        "schema_version": 1, "format": "act-observation-corpus-v1",
        "source": {"kind": "generated_fixture", "identity": "sha256:" + "a" * 64,
                   "revision": "a" * 64, "inventory_sha256": "d" * 64},
        "semantics": {"state_names": [f"joint_{i}" for i in range(6)],
                      "action_names": [f"joint_{i}" for i in range(6)], "units": ["radians"] * 6,
                      "compatibility": "generated_fixture", "teacher_processors_sha256": processors},
        "camera": "observation.images.front", "image_shape": [3, 32, 32],
        "chunk_size": config["chunk_size"], "samples": [],
    }
    if not legacy:
        doc.update({k: metadata[k] for k in (
            "execution_horizon", "temporal_contract_sha256", "control_contract",
            "control_contract_sha256",
        )}, action_fps=20)
    for i, split in enumerate(("train", "validation", "final")):
        name, raw = f"sample-{i:06d}.safetensors", b"metadata-only sample"
        (root / name).write_bytes(raw)
        doc["samples"].append({"file": name, "sha256": digest(raw), "bytes": len(raw),
                               "episode_id": i, "lineage_group": f"group-{i}",
                               "frame_index": 0, "episode_length": 4, "split": split})
    (root / "manifest.json").write_bytes(canonical(doc))
    return doc


def check(root, config, metadata, processors, doc):
    (root / "manifest.json").write_bytes(canonical(doc))
    return corpus(root, digest(canonical(doc)), config, "observation.images.front", processors,
                  metadata=metadata, expected_fps=20 if metadata["control_contract"] else None)


@pytest.mark.parametrize("prediction,execution", [(1, 1), (8, 3), (100, 100), (1024, 8)])
def test_teacher_and_corpus_preserve_inherited_horizons(tmp_path, prediction, execution):
    root = tmp_path / "teacher"
    config = policy(root, prediction, execution)
    _, _, processors = teacher_info(root, inventory(root))
    metadata = policy_metadata(root, config)
    assert metadata["prediction_horizon"] == prediction
    assert metadata["execution_horizon"] == execution
    assert action_fps(root, metadata) == 20
    assert {"control-contract.json", "temporal-contract.json", "pre-stats.safetensors"} <= inherited_files(root, config)
    data = tmp_path / "corpus"
    doc = observation_corpus(data, config, metadata, processors)
    assert check(data, config, metadata, processors, doc) == doc


@pytest.mark.parametrize("change", [
    lambda d: d.update(chunk_size=100),
    lambda d: d.update(execution_horizon=True),
    lambda d: d.update(execution_horizon=8),
    lambda d: d.update(action_fps=30),
    lambda d: d.update(action_fps=True),
    lambda d: d.update(temporal_contract_sha256=None),
    lambda d: d.pop("execution_horizon"),
    lambda d: d.update(control_contract=None, control_contract_sha256=None),
    lambda d: d["source"].update(identity="sha256:" + "e" * 64),
    lambda d: d["source"].update(revision="e" * 64),
    lambda d: d["semantics"].update(units=["degrees"] * 6),
    lambda d: d["semantics"]["state_names"].reverse(),
    lambda d: d["samples"][2].update(lineage_group="group-0"),
])
def test_contract_or_partition_changes_fail_closed(tmp_path, change):
    teacher = tmp_path / "teacher"
    config = policy(teacher)
    _, _, processors = teacher_info(teacher, inventory(teacher))
    metadata = policy_metadata(teacher, config)
    root = tmp_path / "corpus"
    doc = observation_corpus(root, config, metadata, processors)
    change(doc)
    with pytest.raises(ValueError):
        check(root, config, metadata, processors, doc)


def test_legacy_omission_only_for_100_100_without_sidecars(tmp_path):
    teacher = tmp_path / "teacher"
    config = policy(teacher, 100, 100, sidecars=False)
    _, _, processors = teacher_info(teacher, inventory(teacher))
    metadata = policy_metadata(teacher, config)
    root = tmp_path / "corpus"
    doc = observation_corpus(root, config, metadata, processors, legacy=True)
    assert check(root, config, metadata, processors, doc) == doc
    config["n_action_steps"] = 3
    metadata = policy_metadata(teacher, config)
    with pytest.raises(ValueError, match="Legacy corpus"):
        check(root, config, metadata, processors, doc)


def test_missing_or_changed_sidecars_cannot_match_source_inventory(tmp_path):
    root = tmp_path / "teacher"
    config = policy(root)
    before = inventory(root)
    (root / "control-contract.json").unlink()
    with pytest.raises(ValueError, match="inventory changed"):
        teacher_info(root, before)
    (root / "control-contract.json").write_bytes(canonical(control_record()))
    temporal = temporal_record(config, fps=30)
    (root / "temporal-contract.json").write_bytes(canonical(temporal))
    with pytest.raises(ValueError, match="FPS"):
        teacher_info(root, inventory(root))


def test_native_identity_matches_isaac_control_but_excludes_temporal(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "workers/isaac_sim"))
    from sim_worker.rollout.checkpoint import inspect_checkpoint

    root = tmp_path / "policy"
    config = policy(root)
    before = native_model_id(root)
    assert before == inspect_checkpoint(root).model_id
    temporal = temporal_record(config)
    temporal.update(observation_delta_indices=[0], observation_delta_timestamps=[0.0])
    (root / "temporal-contract.json").write_bytes(canonical(temporal))
    assert native_model_id(root) == before == inspect_checkpoint(root).model_id
    control = control_record()
    control["source"]["scene_sha256"] = ["d" * 64]
    (root / "control-contract.json").write_bytes(canonical(control))
    assert native_model_id(root) == inspect_checkpoint(root).model_id != before


def snapshot_metadata(root):
    (root / "meta").mkdir(parents=True)
    record = control_record()
    features = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": record["joint_order"]},
        "action": {"dtype": "float32", "shape": [6], "names": record["joint_order"]},
        "observation.images.front": {"dtype": "video", "shape": [32, 32, 3]},
    }
    source = {"source_session_id": "s", "scene_sha256": "c" * 64,
              "origin": "synthetic", "lineage_group": "group"}
    provenance = {"schema_version": 1, "controller": "joint_position_targets",
                  "state_units": "radians", "action_units": "radians", "timebase": "simulation_seconds",
                  "action_column": "action", "requested_action_column": "teaching.requested_action",
                  "scene_hash_scope": record["source"]["scene_hash_scope"], "task_success_verified": False,
                  "joint_order": record["joint_order"], "camera_prim": record["camera"]["prim"],
                  "sources": [source], "episodes": [{"episode_index": 0, "source_session_id": "s"}]}
    for name, value in (("meta/info.json", {"fps": 20, "features": features}),
                        ("meta/firebird-demonstrations.json", provenance)):
        (root / name).write_bytes(canonical(value))
    files = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size,
              "sha256": digest(p.read_bytes())} for p in sorted((root / "meta").iterdir())]
    record["source"]["demonstrations_sha256"] = digest((root / "meta/firebird-demonstrations.json").read_bytes())
    manifest = {"fps": 20, "features": features, "files": files, "lineage_validated": True,
                "lineage": [{"episode_index": 0, "origin": "synthetic", "lineage_group": "group"}]}
    pointer = {"path": str(root), "id": "sha256:" + "a" * 64, "manifest_sha256": "a" * 64}
    return manifest, pointer, record


def test_snapshot_contract_derivation_requires_exact_source_and_cadence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "workers/smolvla_qlora/src"))
    root = tmp_path / "snapshot"
    manifest, pointer, record = snapshot_metadata(root)
    metadata = {"control_contract": record}
    check_snapshot_control(root, manifest, pointer, "observation.images.front", metadata, 20)
    changed = copy.deepcopy(pointer)
    changed["id"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="exact teacher source"):
        check_snapshot_control(root, manifest, changed, "observation.images.front", metadata, 20)
    with pytest.raises(ValueError, match="FPS"):
        check_snapshot_control(root, manifest | {"fps": 30}, pointer, "observation.images.front", metadata, 20)
    info = read(root / "meta/info.json")
    info["features"]["action"]["names"] = list(reversed(record["joint_order"]))
    (root / "meta/info.json").write_bytes(canonical(info))
    with pytest.raises(ValueError, match="metadata differs"):
        check_snapshot_control(root, manifest, pointer, "observation.images.front", metadata, 20)


def test_implementation_identity_includes_exact_reader_derivation_sources():
    from firebird_distill.application import implementation_identity

    workers = Path(__file__).resolve().parents[2]
    paths = {
        **{"distillation/" + name: workers / "policy_distillation/src/firebird_distill" / name
           for name in ("__init__.py", "application.py", "contracts.py", "prepare.py",
                        "runtime.py", "provenance.py")},
        **{"act/" + name: workers / "act_optimizer/src/firebird_act" / name
           for name in ("bundle.py", "probe.py", "control_schema.py")},
        **{"data/" + name: workers / "smolvla_qlora/src/firebird_vla" / name
           for name in ("control_contract.py", "control_schema.py")},
    }
    assert implementation_identity() == {key: digest(path.read_bytes()) for key, path in paths.items()}


def test_temporal_only_fps_and_nondefault_no_sidecar_corpus(tmp_path):
    teacher = tmp_path / "teacher"
    config = policy(teacher)
    (teacher / "control-contract.json").unlink()
    _, _, processors = teacher_info(teacher, inventory(teacher))
    metadata = policy_metadata(teacher, config)
    assert action_fps(teacher, metadata) == 20
    with pytest.raises(ValueError, match="FPS"):
        check_snapshot_control(tmp_path, {"fps": 30}, {}, "observation.images.front", metadata, 20)
    (teacher / "temporal-contract.json").unlink()
    metadata = policy_metadata(teacher, config)
    assert action_fps(teacher, metadata) is None
    data = tmp_path / "corpus"
    doc = observation_corpus(data, config, metadata, processors)
    assert check(data, config, metadata, processors, doc) == doc
    for key in ("execution_horizon", "action_fps", "temporal_contract_sha256",
                "control_contract", "control_contract_sha256"):
        doc.pop(key)
    with pytest.raises(ValueError, match="Legacy corpus"):
        check(data, config, metadata, processors, doc)


def test_100_100_corpus_cannot_omit_present_sidecars(tmp_path):
    teacher = tmp_path / "teacher"
    config = policy(teacher, 100, 100)
    _, _, processors = teacher_info(teacher, inventory(teacher))
    metadata = policy_metadata(teacher, config)
    data = tmp_path / "corpus"
    doc = observation_corpus(data, config, metadata, processors, legacy=True)
    with pytest.raises(ValueError, match="Legacy corpus"):
        check(data, config, metadata, processors, doc)


@pytest.mark.parametrize("kind,compatibility", [
    ("generated_fixture", "generated_fixture"),
    ("lerobot", "operator_attested_teacher_recorded_coordinates"),
])
def test_homogeneous_selected_origin_can_use_mixed_source_snapshot(tmp_path, kind, compatibility):
    teacher = tmp_path / "teacher"
    config = policy(teacher)
    record = control_record()
    record["source"]["origins"] = ["recorded", "synthetic"]
    (teacher / "control-contract.json").write_bytes(canonical(record))
    _, _, processors = teacher_info(teacher, inventory(teacher))
    metadata = policy_metadata(teacher, config)
    data = tmp_path / "corpus"
    doc = observation_corpus(data, config, metadata, processors)
    doc["source"]["kind"] = kind
    doc["semantics"]["compatibility"] = compatibility
    assert check(data, config, metadata, processors, doc) == doc
    record["source"]["origins"] = ["recorded"] if kind == "generated_fixture" else ["synthetic"]
    metadata["control_contract"] = record
    metadata["control_contract_sha256"] = digest(canonical(record))
    doc["control_contract"] = record
    doc["control_contract_sha256"] = metadata["control_contract_sha256"]
    with pytest.raises(ValueError, match="origin differs"):
        check(data, config, metadata, processors, doc)


@pytest.mark.parametrize("name", [
    "manifest.json", "recipe.json", "parity.json", "train_config.json", "export-lineage.json",
])
def test_existing_export_teacher_metadata_is_retained_but_not_student_payload(tmp_path, name):
    from firebird_distill.contracts import policy_info

    teacher = tmp_path / "teacher"
    config = policy(teacher)
    (teacher / name).write_bytes(canonical({"scope": "metadata-only regression"}))
    cfg, _, _ = teacher_info(teacher, inventory(teacher))
    assert cfg == config
    assert name not in inherited_files(teacher, config)
    with pytest.raises(ValueError, match="Unexpected files"):
        policy_info(teacher, inventory(teacher))
    (teacher / "arbitrary.py").write_bytes(b"not a supported metadata file")
    with pytest.raises(ValueError, match="Unexpected files"):
        teacher_info(teacher, inventory(teacher))
