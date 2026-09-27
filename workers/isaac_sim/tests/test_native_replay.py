"""Bounded replay ownership and exact recorded-input admission; fixtures are generated."""

# ruff: noqa: E402 -- isolated optional sibling worker source paths.
import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

WORKERS = Path(__file__).resolve().parents[2]
for p in (
    WORKERS / "firebird_quant/src",
    WORKERS / "act_optimizer/src",
    WORKERS / "firebird_quant/tests",
):
    sys.path.insert(0, str(p))
from firebird_quant.native_package import canonical, inspect_policy, sha
from test_native_runtime import real_act as _real_act
from test_packed_checkpoint import packed_fixture

native_act_source = _real_act

from sim_worker.rollout.native_replay import Owner, environment, run_job
from sim_worker.rollout.native_replay_contracts import observations, policy, request


def corpus(root, config, *, count=1):
    root.mkdir()
    camera = next(k for k in config["input_features"] if k.startswith("observation.images."))
    _, height, width = config["input_features"][camera]["shape"]
    rows = []
    for i in range(count):
        pixels = bytes([i % 256]) * (width * height * 3)
        name = f"frame-{i:06d}.rgb"
        (root / name).write_bytes(pixels)
        rows.append(
            {
                "episode_index": 0,
                "frame_index": i,
                "timestamp_seconds": i / 30,
                "task": "Generated runtime test, no task quality",
                "state": [0.0] * 6,
                "origin": "synthetic",
                "lineage_group": "generated-scene",
                "image": {
                    "file": name,
                    "width": width,
                    "height": height,
                    "bytes": len(pixels),
                    "sha256": sha(pixels),
                },
            }
        )
    value = {
        "schema_version": 1,
        "format": "native-policy-observations-v1",
        "camera_key": camera,
        "source": {
            "kind": "generated_fixture",
            "identity": "generated-test",
            "manifest_sha256": "c" * 64,
        },
        "semantics": {
            "state_names": [f"joint{i}" for i in range(6)],
            "action_names": [f"joint{i}" for i in range(6)],
            "units": ["fixture_coordinate"] * 6,
            "compatibility": "generated_fixture",
        },
        "samples": rows,
    }
    (root / "manifest.json").write_bytes(canonical(value))
    return value


@pytest.fixture
def job(tmp_path):
    source = packed_fixture(tmp_path / "policy")
    info = inspect_policy(source)
    doc = corpus(tmp_path / "observations", json.loads((source / "config.json").read_bytes()))
    return {
        "schema_version": 1,
        "job_id": "generated-test",
        "operation": "policy.run",
        "source": {
            "path": str(source.resolve()),
            "files": info["files"],
            "model_id": info["model_id"],
            "artifact_id": "generated",
            "artifact_manifest_sha256": "b" * 64,
        },
        "observations": {
            "path": str((tmp_path / "observations").resolve()),
            "manifest_sha256": sha(canonical(doc)),
        },
        "output_dir": str((tmp_path / "output").resolve()),
        "timeout_seconds": 120,
    }


def validate(job):
    source, data, _ = request(job)
    info = policy(source, job["source"])
    return observations(data, job["observations"]["manifest_sha256"], info["config"])


def mutate(job, fn):
    path = Path(job["observations"]["path"]) / "manifest.json"
    value = json.loads(path.read_bytes())
    fn(value)
    path.write_bytes(canonical(value))
    job["observations"]["manifest_sha256"] = sha(path.read_bytes())


@pytest.mark.parametrize(
    "fn",
    [
        lambda j: j.update(schema_version=True),
        lambda j: j.update(timeout_seconds=True),
        lambda j: j.update(timeout_seconds=601),
        lambda j: j.update(operation="policy.evaluate"),
        lambda j: j.update(cuda=True),
        lambda j: j["source"].update(model_id="wrong"),
        lambda j: j["source"].update(artifact_manifest_sha256="x"),
    ],
)
def test_strict_request(job, fn):
    fn(job)
    with pytest.raises(ValueError):
        validate(job)


@pytest.mark.parametrize(
    "fn",
    [
        lambda d: d["samples"][0].update(state=[True] * 6),
        lambda d: d["samples"][0].update(origin="recorded"),
        lambda d: d["samples"][0].update(timestamp_seconds=-1),
        lambda d: d["samples"][0]["image"].update(file="../image.rgb"),
        lambda d: d["samples"][0]["image"].update(width=34),
        lambda d: d["semantics"].update(compatibility="assume_radians"),
        lambda d: d["semantics"].update(action_names=["x"] * 6),
        lambda d: d.update(samples=[]),
        lambda d: d["samples"].append(copy.deepcopy(d["samples"][0])),
    ],
)
def test_rehashed_invalid_observations_fail(job, fn):
    mutate(job, fn)
    with pytest.raises(ValueError):
        validate(job)


def test_hashes_and_extra_files_fail(job):
    path = Path(job["observations"]["path"])
    (path / "extra").write_text("not admitted")
    with pytest.raises(ValueError, match="unexpected"):
        validate(job)
    (path / "extra").unlink()
    (path / "frame-000000.rgb").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        validate(job)


def test_limits_reject_before_rgb_io(job):
    from sim_worker.rollout import native_replay_contracts as contract

    info = policy(Path(job["source"]["path"]), job["source"])
    config = copy.deepcopy(info["config"])
    camera = next(k for k in config["input_features"] if k.startswith("observation.images."))
    config["input_features"][camera]["shape"] = [3, 1920, 1920]

    def oversized(d):
        original = d["samples"][0]
        d["samples"] = []
        for index in range(32):
            row = copy.deepcopy(original)
            row.update(frame_index=index)
            row["image"].update(
                file=f"frame-{index:06d}.rgb", width=1920, height=1920, bytes=1920 * 1920 * 3
            )
            d["samples"].append(row)

    mutate(job, oversized)
    original = contract.read

    def guarded(path, *args):
        assert path.name == "manifest.json", "RGB IO occurred before aggregate rejection"
        return original(path, *args)

    with patch.object(contract, "read", guarded), pytest.raises(ValueError, match="128MiB"):
        observations(
            Path(job["observations"]["path"]), job["observations"]["manifest_sha256"], config
        )


def test_link_overlap_and_existing_result_rejected(job, tmp_path):
    linked = tmp_path / "link"
    linked.symlink_to(job["observations"]["path"], target_is_directory=True)
    bad = copy.deepcopy(job)
    bad["observations"]["path"] = str(linked)
    with pytest.raises(ValueError):
        request(bad)
    bad = copy.deepcopy(job)
    bad["output_dir"] = job["source"]["path"] + "/child"
    with pytest.raises(ValueError):
        request(bad)
    (Path(job["output_dir"]) / "native-run").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run_job(job)


def test_model_identity_mismatch(job):
    job["source"]["model_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="identity"):
        validate(job)


def test_child_failure_no_publication_or_left_snapshots(job):
    with (
        patch.object(Owner, "run", side_effect=TimeoutError("bounded")),
        pytest.raises(TimeoutError),
    ):
        run_job(job)
    assert not (Path(job["output_dir"]) / "native-run").exists()
    assert not list(Path(job["output_dir"]).glob(".native-replay-*"))


def test_source_mutation_during_failure_visible(job):
    def changed(*args):
        (Path(job["source"]["path"]) / "model.fbq").write_bytes(b"changed")
        raise InterruptedError("cancelled")

    with patch.object(Owner, "run", changed), pytest.raises(ValueError, match="admitted"):
        run_job(job)
    assert not (Path(job["output_dir"]) / "native-run").exists()


def test_environment_no_credentials(monkeypatch):
    for key in ("OPENROUTER_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HF_TOKEN", "HOME"):
        monkeypatch.setenv(key, "private")
        assert key not in environment([])


def test_owned_timeout_and_spawn_signal(tmp_path):
    children, original = [], subprocess.Popen

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGTERM)
        return child

    previous = signal.getsignal(signal.SIGTERM)
    with (tmp_path / "log").open("wb") as log:
        with patch("subprocess.Popen", spawn), pytest.raises(InterruptedError):
            with Owner(10) as owner:
                owner.run(
                    [sys.executable, "-c", "import time;time.sleep(30)"], environment([]), log
                )
        with pytest.raises(TimeoutError):
            with Owner(0.1) as owner:
                owner.run(
                    [sys.executable, "-c", "import time;time.sleep(30)"], environment([]), log
                )
    assert children[0].poll() is not None
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.skipif(
    os.environ.get("FIREBIRD_TEST_NATIVE_ACT") != "1",
    reason="Requires exact pinned CPU ACT environment",
)
@pytest.mark.parametrize("bits", [8, 4])
def test_actual_packed_http_replay(native_act_source, tmp_path, bits):
    from firebird_quant.native_application import run_job as quantize
    from firebird_quant.native_package import inventory

    source = native_act_source
    converted = quantize(
        {
            "schema_version": 1,
            "job_id": "generated",
            "operation": "policy.quantize",
            "source": {
                "path": str(source),
                "files": inventory(source),
                "manifest_sha256": None,
                "artifact_id": "generated",
                "artifact_manifest_sha256": "a" * 64,
            },
            "output_dir": str(tmp_path / "quantized"),
            "native_quantization": {"format": "firebird_quant", "bits": bits, "group_size": 64},
            "timeout_seconds": 120,
        }
    )
    source = Path(converted["artifact"]["path"]) / "policy"
    info = inspect_policy(source)
    doc = corpus(
        tmp_path / "observations", json.loads((source / "config.json").read_bytes()), count=2
    )
    job = {
        "schema_version": 1,
        "job_id": "generated-replay",
        "operation": "policy.run",
        "source": {
            "path": str(source),
            "files": info["files"],
            "model_id": info["model_id"],
            "artifact_id": "generated-packed",
            "artifact_manifest_sha256": "a" * 64,
        },
        "observations": {
            "path": str(tmp_path / "observations"),
            "manifest_sha256": sha(canonical(doc)),
        },
        "output_dir": str(tmp_path / "output"),
        "timeout_seconds": 120,
    }
    result = run_job(job)
    assert result["report"]["reset_repeat_exact"] and result["report"]["server_closed"]
    assert (
        result["report"]["task_success"] is None and result["report"]["quality_verified"] is False
    )
    artifact = Path(result["artifact"]["path"])
    proof = json.loads((artifact / "predictions.json").read_bytes())
    assert len(proof["records"]) == 2
    assert all(
        len(row["actions"]) == 100 and all(len(action) == 6 for action in row["actions"])
        for row in proof["records"]
    )
    assert inventory(source) == info["files"]
    for name, expected in json.loads((artifact / "manifest.json").read_bytes())["files"].items():
        assert sha((artifact / name).read_bytes()) == expected


def valid_result(job):
    from firebird_quant.native_package import RUNTIME

    doc, _ = validate(job)
    row = doc["samples"][0]
    actions = [[0.0] * 6 for _ in range(100)]
    result = {
        "schema_version": 1,
        "model_id": job["source"]["model_id"],
        "versions": dict(RUNTIME),
        "device": "cpu",
        "mode": "independent_observation_replay",
        "server_closed": True,
        "external_network_disabled": True,
        "floating_master_reads_blocked": True,
        "task_success": None,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "isaac_runtime_verified": False,
        "records": [
            {
                "sample_index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp_seconds": 0,
                "input_rgb_sha256": row["image"]["sha256"],
                "input_state_sha256": sha(canonical(row["state"])),
                "actions": actions,
                "reset_repeat_exact": True,
                "requests": [{"seconds": 0.1, "actions_sha256": sha(canonical(actions))}] * 2,
            }
        ],
    }
    return doc, result


@pytest.mark.parametrize(
    "fn",
    [
        lambda r: r.update(quality_verified=True),
        lambda r: r.update(server_closed=False),
        lambda r: r.update(device="cuda"),
        lambda r: r.update(task_success=True),
        lambda r: r["records"][0].update(frame_index=1),
        lambda r: r["records"][0].update(sample_index=False),
        lambda r: r["records"][0].update(actions=[[0.0] * 6]),
        lambda r: r["records"][0].update(reset_repeat_exact=False),
        lambda r: r["records"][0]["requests"][0].update(actions_sha256="0" * 64),
    ],
)
def test_child_result_cannot_change_input_or_claim_quality(job, fn):
    from sim_worker.rollout.native_replay_contracts import validate_result

    doc, result = valid_result(job)
    validate_result(result, job["source"], doc)
    fn(result)
    with pytest.raises(ValueError):
        validate_result(result, job["source"], doc)


def test_durability_failure_never_publishes(tmp_path):
    from sim_worker.rollout import native_replay_contracts as contract

    stage, destination = tmp_path / "stage", tmp_path / "published"
    stage.mkdir()
    (stage / "result").write_bytes(b"retained")
    with (
        patch.object(contract, "sync_directory", side_effect=OSError("sync failed")),
        pytest.raises(OSError),
    ):
        contract.publish(stage, destination)
    assert not destination.exists()


def test_prepare_selection_limits_before_runtime_import():
    from sim_worker.rollout.native_replay_prepare import prepare

    request = {
        "schema_version": 1,
        "source": {
            "path": "/private/tmp/model",
            "files": {},
            "model_id": "sha256:" + "a" * 64,
            "artifact_id": "x",
            "artifact_manifest_sha256": "b" * 64,
        },
        "dataset_snapshot": {},
        "selection": [],
        "semantics": {},
        "output_dir": "/private/tmp/nonexistent-replay-output-limit",
    }
    with patch("importlib.metadata.version") as version, pytest.raises(ValueError, match="Select"):
        prepare(request)
    version.assert_not_called()


@pytest.mark.parametrize("relative", ["native-run/extra.json", "native-run"])
def test_cli_result_cannot_mutate_published_artifact(job, tmp_path, monkeypatch, relative):
    from sim_worker.rollout import native_replay

    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical(job))
    destination = Path(job["output_dir"]) / relative
    monkeypatch.setattr(sys, "argv", ["native_replay", str(request_path), str(destination)])
    with patch.object(native_replay, "run_job") as child, pytest.raises(SystemExit):
        native_replay.main()
    child.assert_not_called()
    assert not Path(job["output_dir"]).exists()


@pytest.mark.parametrize("inside", [True, False])
def test_prepare_cli_result_never_added_to_observation_inventory(tmp_path, monkeypatch, inside):
    from sim_worker.rollout import native_replay_prepare

    output = tmp_path / "observations"
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical({"output_dir": str(output)}))
    destination = output / "extra.json" if inside else output
    monkeypatch.setattr(sys, "argv", ["native_replay_prepare", str(request_path), str(destination)])
    with patch.object(native_replay_prepare, "prepare") as child, pytest.raises(SystemExit):
        native_replay_prepare.main()
    child.assert_not_called()
    assert not output.exists()
