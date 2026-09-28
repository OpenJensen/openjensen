"""Real LeRobot 0.6.2 preparation, without a native teacher allocation.

This module is deliberately separate from the model worker's LeRobot 0.6.1 suite.
When not requested, all 15 cases skip before importing the reader or Torch. When
requested, missing dependencies or a version other than 0.6.2 FAIL fixture setup;
15 passed, 0 skipped (including three positive cases) is the acceptance threshold.

After resource admission, from workers/policy_distillation, use the existing
dataset-reader Python 3.12 and an existing compatible pytest installation/overlay:

  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    FIREBIRD_DISTILL_PREPARE_NATIVE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    <dataset-reader-python> -B -m pytest tests/test_prepare.py -q -p no:cacheprovider

Do not substitute the 0.6.1 model environment or install packages automatically.
One owned temporary dataset has 36 generated frames and three synthetic lineage
groups. The genuine writer produces Parquet/video; prepare() uses the genuine
reader. Tiny metadata-only teacher weights are intentionally not loadable: this
proves preparation admission/decoding, not teacher inference, Isaac, or quality.
All dataset, cache, teacher, and corpus fixtures are deleted on fixture teardown.
"""

import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
from firebird_act.bundle import canonical, inventory
from firebird_act.control_schema import SCOPE
from semantics_fixture import policy, temporal_record

pytestmark = pytest.mark.skipif(
    os.environ.get("FIREBIRD_DISTILL_PREPARE_NATIVE") != "1",
    reason="Opt in explicitly with the existing LeRobot 0.6.2 dataset-reader environment",
)

CAMERA = "observation.images.front"
JOINTS = [f"joint_{i}" for i in range(6)]
FPS = 20
LENGTH = 9
GROUPS = ["synthetic-a", "synthetic-a", "synthetic-b", "synthetic-c"]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _files(root):
    """Exact bytes/inventory, also detecting a reader that writes into its source."""
    result = []
    for entry in sorted(root.rglob("*")):
        assert not entry.is_symlink(), "Generated fixture unexpectedly contains a symlink"
        if entry.is_file():
            raw = entry.read_bytes()
            result.append(
                {"path": entry.relative_to(root).as_posix(), "size": len(raw), "sha256": _sha(raw)}
            )
    return result


def _vector(episode, frame, *, action):
    # Binary fractions survive float32 serialization exactly. Distinct episode
    # and frame terms expose both incorrect subset offsets and wrong delta rows.
    base = (episode * 64 + frame * 4) / 128 + (0.125 if action else 0)
    return [base + coordinate / 128 for coordinate in range(6)]


def _intensity(episode, frame):
    return 16 + episode * 40 + frame * 12


@pytest.fixture(scope="module")
def snapshot():
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("Requested native preparation requires installed LeRobot 0.6.2", pytrace=False)
    if version != "0.6.2":
        pytest.fail(f"Requested preparation requires LeRobot 0.6.2, found {version}", pytrace=False)

    from firebird_act.probe import offline_audit

    # This opt-in module runs in its own process; the audit hook deliberately
    # remains installed for the entire proof, including genuine writer setup.
    sys.addaudithook(offline_audit)
    with tempfile.TemporaryDirectory(prefix="firebird-distill-prepare-") as directory:
        root = Path(directory).resolve()
        with pytest.MonkeyPatch.context() as env:
            for name, value in {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": str(root / "cache/hf"),
                "HF_DATASETS_CACHE": str(root / "cache/datasets"),
                "HF_LEROBOT_HOME": str(root / "cache/lerobot"),
                "XDG_CACHE_HOME": str(root / "cache/xdg"),
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
            }.items():
                env.setenv(name, value)

            import numpy as np
            import torch
            from firebird_vla.control_contract import derive
            from firebird_vla.local_dataset import verify_local_snapshot
            from lerobot.configs.video import RGBEncoderConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                source = root / "snapshot"
                features = {
                    name: {"dtype": "float32", "shape": (6,), "names": JOINTS}
                    for name in ("observation.state", "action", "teaching.requested_action")
                }
                features[CAMERA] = {
                    "dtype": "video",
                    "shape": (32, 32, 3),
                    "names": ["height", "width", "channels"],
                }
                writer = LeRobotDataset.create(
                    repo_id="firebird/generated-preparation-proof",
                    root=source,
                    fps=FPS,
                    robot_type="isaac_joint_position",
                    features=features,
                    use_videos=True,
                    video_backend="pyav",
                    rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
                    image_writer_processes=0,
                    image_writer_threads=0,
                    encoder_threads=1,
                )
                try:
                    for episode in range(4):
                        for frame in range(LENGTH):
                            actions = np.array(
                                _vector(episode, frame, action=True), dtype=np.float32
                            )
                            writer.add_frame(
                                {
                                    CAMERA: np.full(
                                        (32, 32, 3), _intensity(episode, frame), dtype=np.uint8
                                    ),
                                    "observation.state": np.array(
                                        _vector(episode, frame, action=False), dtype=np.float32
                                    ),
                                    "action": actions,
                                    "teaching.requested_action": actions.copy(),
                                    "task": "Generated preparation boundary; no task-quality claim",
                                }
                            )
                        writer.save_episode(parallel_encoding=False)
                finally:
                    writer.finalize()
                del writer

                lineage = [
                    {"episode_index": i, "origin": "synthetic", "lineage_group": group}
                    for i, group in enumerate(GROUPS)
                ]
                demonstrations = {
                    "schema_version": 1,
                    "controller": "joint_position_targets",
                    "state_units": "radians",
                    "action_units": "radians",
                    "timebase": "simulation_seconds",
                    "action_column": "action",
                    "requested_action_column": "teaching.requested_action",
                    "scene_hash_scope": SCOPE,
                    "task_success_verified": False,
                    "joint_order": JOINTS,
                    "camera_prim": "/Generated/Camera",
                    "sources": [
                        {
                            "source_session_id": group,
                            "scene_sha256": _sha(b"generated scene identity, not an Isaac asset"),
                            "origin": "synthetic",
                            "lineage_group": group,
                        }
                        for group in sorted(set(GROUPS))
                    ],
                    "episodes": [
                        {"episode_index": i, "source_session_id": group}
                        for i, group in enumerate(GROUPS)
                    ],
                }
                (source / "meta/firebird-demonstrations.json").write_bytes(
                    canonical(demonstrations)
                )
                (source / "meta/firebird-lineage.json").write_bytes(
                    canonical({"schema_version": 1, "episodes": lineage})
                )
                info = json.loads((source / "meta/info.json").read_bytes())
                assert info["total_episodes"] == 4 and info["total_frames"] == 36
                manifest = {
                    "schema_version": 1,
                    "format": "lerobot_v3",
                    "fps": info["fps"],
                    "features": info["features"],
                    "total_episodes": info["total_episodes"],
                    "total_frames": info["total_frames"],
                    "lineage_validated": True,
                    "lineage": lineage,
                    "files": _files(source),
                }
                raw = (
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    + "\n"
                ).encode("utf-8")
                (source / "firebird-snapshot.json").write_bytes(raw)
                pointer = {
                    "path": str(source),
                    "id": "sha256:" + _sha(raw),
                    "manifest_sha256": _sha(raw),
                }
                checked, admitted = verify_local_snapshot(pointer)
                record = derive(checked, admitted, pointer, {"camera_keys": [CAMERA]})
                assert record["source"]["origins"] == ["synthetic"]
                assert record["joint_order"] == JOINTS and record["action_fps"] == FPS
                assert record["camera"] == {
                    "key": CAMERA,
                    "width": 32,
                    "height": 32,
                    "prim": "/Generated/Camera",
                }
                assert record["source"]["demonstrations_sha256"] == _sha(canonical(demonstrations))
                assert sum(item["size"] for item in _files(root)) < 16 * 1024**2
                before = _files(source)
                try:
                    yield {"root": root, "source": source, "pointer": pointer, "control": record}
                finally:
                    assert _files(source) == before, (
                        "Preparation mutated the shared source snapshot"
                    )
            finally:
                torch.set_num_threads(previous_threads)


@pytest.fixture
def case(snapshot):
    before = _files(snapshot["source"])
    with tempfile.TemporaryDirectory(prefix="case-", dir=snapshot["root"]) as directory:
        case_root = Path(directory)
        teacher = case_root / "teacher"
        config = policy(teacher, prediction=8, execution=3)
        (teacher / "control-contract.json").write_bytes(canonical(snapshot["control"]))
        request = {
            "schema_version": 1,
            "teacher": str(teacher),
            "dataset_snapshot": deepcopy(snapshot["pointer"]),
            "splits": {"train": [3], "validation": [0], "final": [2]},
            "frame_stride": 4,
            "semantics": {
                "state_names": list(JOINTS),
                "action_names": list(JOINTS),
                "units": ["radians"] * 6,
                "compatibility": "generated_fixture",
            },
            "output_dir": str(case_root / "corpus"),
        }
        try:
            yield request, config
        finally:
            assert _files(snapshot["source"]) == before


def _check_corpus(request, prediction):
    import torch
    from firebird_distill.prepare import prepare
    from safetensors.torch import load

    teacher = Path(request["teacher"])
    teacher_before = inventory(teacher)
    result = prepare(request)
    output = Path(result["path"])
    raw = (output / "manifest.json").read_bytes()
    assert result["manifest_sha256"] == _sha(raw)
    doc = json.loads(raw)
    assert result["samples"] == len(doc["samples"]) == 9
    assert doc["source"]["identity"] == request["dataset_snapshot"]["id"]
    assert doc["source"]["revision"] == request["dataset_snapshot"]["manifest_sha256"]
    assert doc["source"]["kind"] == "generated_fixture"
    assert doc["chunk_size"] == prediction and doc["action_fps"] == FPS
    expected_rows = [(episode, frame) for episode in (0, 2, 3) for frame in (0, 4, 8)]
    assert [(s["episode_id"], s["frame_index"]) for s in doc["samples"]] == expected_rows
    for sample, (episode, frame) in zip(doc["samples"], expected_rows, strict=True):
        assert sample["split"] == {0: "validation", 2: "final", 3: "train"}[episode]
        assert sample["lineage_group"] == GROUPS[episode]
        assert sample["episode_length"] == LENGTH
        sample_raw = (output / sample["file"]).read_bytes()
        assert sample["bytes"] == len(sample_raw) and sample["sha256"] == _sha(sample_raw)
        data = load(sample_raw)
        assert set(data) == {"image", "state", "actions", "padding"}
        assert torch.equal(
            data["state"], torch.tensor(_vector(episode, frame, action=False), dtype=torch.float32)
        )
        expected_actions = [
            _vector(episode, min(frame + offset, LENGTH - 1), action=True)
            for offset in range(prediction)
        ]
        assert torch.equal(data["actions"], torch.tensor(expected_actions, dtype=torch.float32))
        assert data["padding"].dtype == torch.bool
        assert data["padding"].tolist() == [
            frame + offset >= LENGTH for offset in range(prediction)
        ]
        assert data["image"].dtype == torch.uint8 and list(data["image"].shape) == [3, 32, 32]
        # Real H.264 is lossy. Constant grayscale has an independent, bounded
        # pixel oracle; even adjacent source frames differ by 12 intensity levels.
        error = (data["image"].to(torch.int16) - _intensity(episode, frame)).abs()
        assert int(error.max()) <= 4
    assert inventory(teacher) == teacher_before
    assert not list(output.parent.glob(".corpus-*"))
    return doc


def test_real_reader_preserves_nonadjacent_rows_control_and_8_3_padding(case, snapshot):
    request, _ = case
    doc = _check_corpus(request, 8)
    assert doc["execution_horizon"] == 3
    assert doc["control_contract"] == snapshot["control"]
    assert doc["control_contract_sha256"] == _sha(canonical(snapshot["control"]))
    teacher = Path(request["teacher"])
    assert doc["temporal_contract_sha256"] == _sha(
        (teacher / "temporal-contract.json").read_bytes()
    )


def test_real_reader_keeps_legacy_100_100_without_optional_sidecars(case):
    request, _ = case
    teacher = Path(request["teacher"]).parent / "legacy-teacher"
    policy(teacher, prediction=100, execution=100, sidecars=False)
    request["teacher"] = str(teacher)
    doc = _check_corpus(request, 100)
    assert doc["execution_horizon"] == 100
    assert doc["temporal_contract_sha256"] is None
    assert doc["control_contract"] is None and doc["control_contract_sha256"] is None


def test_real_native_episode_metadata_reports_selected_lengths(snapshot):
    from firebird_distill.prepare import verify_lengths

    result = verify_lengths(
        {"schema_version": 1, "dataset_snapshot": snapshot["pointer"], "episodes": [3, 0, 2]}
    )
    assert result == {
        "schema_version": 1,
        "snapshot_id": snapshot["pointer"]["id"],
        "snapshot_manifest_sha256": snapshot["pointer"]["manifest_sha256"],
        "episode_lengths": [{"episode_id": episode, "length": LENGTH} for episode in (0, 2, 3)],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate-episode", "Episode appears in multiple splits"),
        ("shared-lineage", "Lineage group leaks between splits"),
        ("state-order", "Coordinate names/order differ"),
        ("action-order", "Coordinate names/order differ"),
        ("units", "Corpus source or coordinates differ"),
        ("fps", "Dataset FPS differs"),
        ("joint-order", "Dataset simulator joints, camera, cadence or provenance differ"),
        ("camera-prim", "Dataset simulator joints, camera, cadence or provenance differ"),
        ("source-identity", "exact teacher source snapshot"),
        ("snapshot-digest", "snapshot manifest identity mismatch"),
        ("camera-resolution", "Dataset camera resolution differs"),
    ],
)
def test_real_preparation_refuses_mismatch_without_publication(case, mutation, message):
    from firebird_distill.prepare import prepare

    request, config = case
    teacher = Path(request["teacher"])
    control = json.loads((teacher / "control-contract.json").read_bytes())
    if mutation == "duplicate-episode":
        request["splits"]["final"] = [0]
    elif mutation == "shared-lineage":
        request["splits"] = {"train": [0], "validation": [1], "final": [2]}
    elif mutation in {"state-order", "action-order"}:
        request["semantics"][mutation.split("-")[0] + "_names"].reverse()
    elif mutation == "units":
        request["semantics"]["units"] = ["degrees"] * 6
    elif mutation == "fps":
        control["action_fps"] = 10
        (teacher / "temporal-contract.json").write_bytes(canonical(temporal_record(config, fps=10)))
    elif mutation == "joint-order":
        control["joint_order"].reverse()
    elif mutation == "camera-prim":
        control["camera"]["prim"] = "/Generated/DifferentCamera"
    elif mutation == "source-identity":
        control["source"]["dataset_snapshot_id"] = "sha256:" + "f" * 64
        control["source"]["dataset_manifest_sha256"] = "f" * 64
    elif mutation == "snapshot-digest":
        request["dataset_snapshot"].update(id="sha256:" + "f" * 64, manifest_sha256="f" * 64)
    elif mutation == "camera-resolution":
        other = teacher.parent / "different-resolution"
        other_config = policy(other, prediction=100, execution=100, sidecars=False)
        other_config["input_features"][CAMERA]["shape"] = [3, 64, 64]
        (other / "config.json").write_bytes(canonical(other_config))
        pre = json.loads((other / "policy_preprocessor.json").read_bytes())
        pre["steps"][-1]["config"]["features"][CAMERA]["shape"] = [3, 64, 64]
        (other / "policy_preprocessor.json").write_bytes(canonical(pre))
        request["teacher"] = str(other)
    (teacher / "control-contract.json").write_bytes(canonical(control))
    selected_teacher = Path(request["teacher"])
    before = inventory(selected_teacher)
    output = Path(request["output_dir"])
    with pytest.raises(ValueError, match=message):
        prepare(request)
    assert not output.exists()
    assert not list(output.parent.glob(".corpus-*"))
    assert inventory(selected_teacher) == before


def test_preparation_never_overwrites_existing_output(case):
    from firebird_distill.prepare import prepare

    request, _ = case
    output = Path(request["output_dir"])
    output.mkdir()
    (output / "owner.txt").write_bytes(b"existing owner output")
    before = _files(output)
    with pytest.raises(FileExistsError, match="Corpus output already exists"):
        prepare(request)
    assert _files(output) == before
    assert not list(output.parent.glob(".corpus-*"))
