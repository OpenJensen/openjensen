"""Generated independent scene fixtures exercise the genuine multi-source writer."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import SimulationFixture, act, settings
from sim_worker.rollout.contracts import Frame, Observation

from firebird_teaching import dataset
from firebird_teaching.dataset import CaptureSelection, additional_captures, convert_captures
from firebird_teaching.journal import Journal
from firebird_teaching.session import Session


def record_scene(path: Path, seed: int, *, frames=12):
    """Declare distinct generated scene inputs before recording, never relabel captures."""
    scene = f"#usda 1.0\n# Generated CPU fixture seed={seed}; not an Isaac task\n".encode()
    scene_path = path.parent / f"generated-scene-{seed}.usda"
    scene_path.write_bytes(scene)
    cfg = replace(
        settings(),
        sim=replace(settings().sim, scene=str(scene_path)),
        scene_sha256=hashlib.sha256(scene).hexdigest(),
        lineage_group=f"synthetic-scene-{seed}",
    )

    class GeneratedScene(SimulationFixture):
        def reset(self, episode_id):
            super().reset(episode_id)
            self.state = (0.01 * (seed % 5),) * len(self.spec.joints)

        def observe(self):
            observed = super().observe()
            return Observation(
                observed.episode_id,
                observed.step,
                observed.sim_time,
                observed.state,
                Frame(32, 32, bytes([seed % 256, self.counter % 256, 128]) * 32 * 32),
            )

    session = Session(cfg, GeneratedScene(cfg.sim), Journal(path, cfg))
    act(session, "task", {"instruction": f"Generated scene {seed}; task success unknown"})
    ids = []
    for _ in range(2):
        act(session, "start")
        ids.append(session.episode_id)
        for frame in range(frames):
            if frame == 1:
                act(session, "correct", {"joint": "joint_0", "delta_rad": 0.15})
            session.tick()
        act(session, "finish")
    session.close()
    return CaptureSelection(path, tuple(ids))


def captures(tmp_path, second_seed=4919):
    return [record_scene(tmp_path / "first", 1729), record_scene(tmp_path / "second", second_seed)]


@pytest.mark.parametrize("second_seed,groups", [(4919, 2), (1729, 1)])
def test_genuine_multi_capture_writer_preserves_declared_groups(tmp_path, second_seed, groups):
    selections = captures(tmp_path, second_seed)
    before = [dataset._inventory(s.root) for s in selections]
    output = tmp_path / "dataset"
    result = convert_captures(selections, output)
    assert result["readback_verified"] and result["frames"] == 48
    assert result["sources"] == 2 and result["lineage_groups"] == groups
    assert result["task_success_verified"] is False
    lineage = json.loads((output / "meta/firebird-lineage.json").read_text())["episodes"]
    assert [e["episode_index"] for e in lineage] == list(range(4))
    assert [e["lineage_group"] for e in lineage] == [
        "synthetic-scene-1729",
        "synthetic-scene-1729",
        f"synthetic-scene-{second_seed}",
        f"synthetic-scene-{second_seed}",
    ]
    assert {e["origin"] for e in lineage} == {"synthetic"}
    provenance = json.loads((output / "meta/firebird-demonstrations.json").read_text())
    assert len(provenance["sources"]) == 2
    assert [s["source_files"] for s in provenance["sources"]] == before
    assert [e["source_session_id"] for e in provenance["episodes"]] == [
        s["source_session_id"] for s in provenance["sources"] for _ in range(2)
    ]
    assert "source_session_id" not in provenance and "scene_sha256" not in provenance
    assert before == [dataset._inventory(s.root) for s in selections]


@pytest.mark.parametrize(
    "key,value",
    [
        ("joint_names", ["other_joint"] + [f"joint_{n}" for n in range(1, 6)]),
        ("camera_prim", "/World/OtherCamera"),
        ("physics_hz", 240),
        ("fps", 25),
        ("state_units", "degrees"),
    ],
)
def test_incompatible_source_schema_rejected_before_writer(tmp_path, key, value):
    selections = captures(tmp_path)
    path = selections[1].root / "session.json"
    meta = json.loads(path.read_text())
    meta[key] = value
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        dataset._admit_captures(selections, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("duplicate", ["root", "nested", "session", "episode"])
def test_duplicate_capture_identities_rejected(tmp_path, duplicate):
    selections = captures(tmp_path)
    if duplicate == "root":
        selections[1] = selections[0]
    elif duplicate == "nested":
        selections[1] = CaptureSelection(selections[0].root / "child", selections[1].episodes)
    elif duplicate == "session":
        first = json.loads((selections[0].root / "session.json").read_text())
        path = selections[1].root / "session.json"
        meta = json.loads(path.read_text())
        meta["session_id"] = first["session_id"]
        path.write_text(json.dumps(meta))
    else:
        # Copy a real capture to a different location, retaining its episode IDs.
        import shutil

        shutil.rmtree(selections[1].root)
        shutil.copytree(selections[0].root, selections[1].root)
        path = selections[1].root / "session.json"
        meta = json.loads(path.read_text())
        meta["session_id"] = "separate-session"
        path.write_text(json.dumps(meta))
        selections[1] = CaptureSelection(selections[1].root, selections[0].episodes)
    with pytest.raises(ValueError, match="distinct|globally unique|non-overlapping"):
        dataset._admit_captures(selections, tmp_path / "dataset")


@pytest.mark.parametrize("bound", ["episodes", "entries", "bytes"])
def test_limits_apply_across_all_sources(tmp_path, monkeypatch, bound):
    selections = captures(tmp_path)
    if bound == "episodes":
        monkeypatch.setattr(dataset, "MAX_EPISODES", 3)
    elif bound == "entries":
        monkeypatch.setattr(
            dataset, "MAX_CAPTURE_ENTRIES", len(list(selections[0].root.rglob("*"))) + 1
        )
    else:
        monkeypatch.setattr(
            dataset,
            "MAX_CAPTURE_BYTES",
            sum(p.stat().st_size for p in selections[0].root.rglob("*") if p.is_file()) + 1,
        )
    with pytest.raises(ValueError, match="1..100|entry limit|byte bound"):
        dataset._admit_captures(selections, tmp_path / "dataset")


def test_explicit_additional_capture_manifest(tmp_path):
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "captures": [{"path": "../second", "episodes": ["episode-1"]}]}
        )
    )
    assert additional_captures(manifest) == [
        CaptureSelection(tmp_path.parent / "second", ("episode-1",))
    ]


@pytest.mark.parametrize(
    "value",
    [
        {"schema_version": True, "captures": []},
        {"schema_version": 1, "captures": [], "extra": True},
        {"schema_version": 1, "captures": [{"path": "x", "episodes": "all"}]},
        {"schema_version": 1, "captures": [{"path": "x", "episodes": ["../escape"]}]},
        {"schema_version": 1, "captures": [{"path": "x", "episodes": ["a"], "group": "invented"}]},
    ],
)
def test_manifest_is_strict(tmp_path, value):
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        additional_captures(manifest)


def test_manifest_file_bounds_and_symlinks(tmp_path):
    manifest = tmp_path / "selection.json"
    manifest.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="bound"):
        additional_captures(manifest)
    link = tmp_path / "linked.json"
    link.symlink_to(manifest)
    with pytest.raises(ValueError, match="linked"):
        additional_captures(link)


def test_second_source_changed_during_readback_never_publishes(tmp_path, monkeypatch):
    selections = captures(tmp_path)
    original = dataset.read_local_dataset

    def read_and_change(repo_id, root):
        loaded = original(repo_id, root)
        (selections[1].root / "session.json").write_bytes(b"changed")
        return loaded

    monkeypatch.setattr(dataset, "read_local_dataset", read_and_change)
    with pytest.raises(ValueError, match="changed during conversion"):
        convert_captures(selections, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("ancestor", [False, True])
def test_source_symlink_is_not_resolved_into_admissibility(tmp_path, ancestor):
    selections = captures(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(
        selections[1].root.parent if ancestor else selections[1].root, target_is_directory=True
    )
    linked_root = link / selections[1].root.name if ancestor else link
    selections[1] = CaptureSelection(linked_root, selections[1].episodes)
    with pytest.raises(ValueError, match="linked|real directory"):
        dataset._admit_captures(selections, tmp_path / "dataset")
