"""CPU guard tests with synthetic bytes; these are never model/robot evidence."""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from policykit import application
from policykit import spatial_application as app
from policykit import spatial_protocol as protocol
from policykit.spatial_fixture import load_arrays, validate_arrays
from policykit.worker import atomic_json, sha256


def raw_arrays():
    return {
        "observation.images.image": np.zeros((1, 3, 360, 360), dtype=np.float32),
        "observation.images.image2": np.ones((1, 3, 360, 360), dtype=np.float32),
        "observation.state": np.zeros((1, 8), dtype=np.float32),
        "task": np.array(["synthetic test instruction"]),
        "noise": np.zeros((1, 50, 32), dtype=np.float32),
    }


def asset_bundle(tmp_path, monkeypatch, *, reference=False):
    root = tmp_path / "assets"
    root.mkdir()
    for prefix, names in (("policy", protocol.POLICY_FILES), ("backbone", protocol.BACKBONE_FILES)):
        (root / prefix).mkdir()
        for name in names:
            (root / prefix / name).write_text("{}")
    atomic_json(
        root / "policy/config.json",
        {
            "type": "smolvla",
            "chunk_size": 50,
            "n_action_steps": 50,
            "max_action_dim": 32,
            "max_state_dim": 32,
            "num_steps": 10,
            "output_features": {"action": {"shape": [7]}},
            "input_features": {"observation.state": {"shape": [6]}},
        },
    )
    for name in ("policy_preprocessor", "policy_postprocessor"):
        state = next(x for x in protocol.POLICY_FILES if x.startswith(name + "_step_"))
        atomic_json(root / "policy" / (name + ".json"), {"steps": [{"state_file": state}]})
    (root / "fixtures").mkdir()
    np.savez_compressed(root / "fixtures/a.npz", **raw_arrays())
    if reference:
        (root / protocol.REFERENCE_FILE).write_bytes(b"synthetic native weight fixture")
        monkeypatch.setattr(protocol, "WEIGHTS_SHA256", sha256(root / protocol.REFERENCE_FILE))
    record = {
        "schema_version": 1,
        "checkpoint": {
            "repo_id": protocol.MODEL,
            "revision": protocol.REVISION,
            "weights_sha256": protocol.WEIGHTS_SHA256,
        },
        "backbone": {"repo_id": protocol.BACKBONE, "revision": protocol.BACKBONE_REVISION},
        "floating_gguf_sha256": "a" * 64,
        "reference_weights": reference,
        "fixtures": ["fixtures/a.npz"],
        "files": {
            p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()
        },
    }
    atomic_json(root / "spatial-assets.json", record)
    return root, record


def evaluation():
    return {
        "mode": "libero",
        "suite": "libero_spatial",
        "task_ids": None,
        "initial_states": [0],
        "final_states": [2],
        "seed": 42,
        "steps": 280,
        "warmups": 1,
        "repetitions": 2,
        "parity_limits": {"profile": "synthetic-only", "max_rmse": 0.1, "max_abs_error": 0.5},
    }


def report_for(record, *, backend="cpp", final=True):
    spec, digest = protocol.protocol(evaluation(), record, {"synthetic": True}, final)
    return {
        "protocol": spec,
        "protocol_sha256": digest,
        "target_identity": {"gpu_uuid": "GPU-synthetic", "name": "fake", "driver_version": "fake"},
        "episodes": [
            {
                **row,
                "status": "episode_complete",
                "task_success": True,
                "steps": 40,
                "benchmark_horizon": 280,
            }
            for row in spec["episodes"]
        ],
        "requested_episodes": 10,
        "complete_episodes": 10,
        "success_rate": 1.0,
        "runtime": {"backend": backend, "synthetic": True},
        "backend": backend,
        "backend_configuration": "cpp-bf16" if backend == "cpp" else "native-bf16",
        "p95_ms": 20,
        "peak_device_mib": 30,
        "memory_coverage": {"complete": True},
    }


def test_all_ten_tasks_and_phase_protocols_are_explicit(tmp_path, monkeypatch):
    _, record = asset_bundle(tmp_path, monkeypatch)
    initial, first = protocol.protocol(evaluation(), record, {"x": 1})
    final, second = protocol.protocol(evaluation(), record, {"x": 1}, True)
    assert initial["task_ids"] == list(range(10))
    assert len(initial["episodes"]) == 10 and initial["state_ids"] == [0]
    assert final["state_ids"] == [2] and first != second
    assert final["episodes"][0] == {
        "task_id": 0,
        "init_state_id": 2,
        "seed": 44,
        "noise_seed": 2042,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"task_ids": []},
        {"task_ids": [1, 1]},
        {"task_ids": [True]},
        {"task_ids": [10]},
        {"initial_states": [0, 0]},
        {"final_states": [0]},
        {"final_states": [1000]},
        {"seed": -1},
        {"suite": "libero_object"},
    ],
)
def test_protocol_rejects_invalid_or_overlapping_episodes(change):
    with pytest.raises(ValueError):
        protocol.episode_ids({**evaluation(), **change}, True)


def test_copy_strips_native_weights_but_preserves_comparable_protocol(tmp_path, monkeypatch):
    source, record = asset_bundle(tmp_path, monkeypatch, reference=True)
    target = tmp_path / "copy"
    target.mkdir()
    copied = protocol.copy_assets(source, target)
    assert not (target / protocol.REFERENCE_FILE).exists()
    assert copied["reference_weights"] is False
    assert protocol.protocol(evaluation(), record, {}) == protocol.protocol(
        evaluation(), copied, {}
    )
    with pytest.raises(ValueError, match="absent"):
        protocol.verify_assets(target, require_reference=True)


@pytest.mark.parametrize(
    "failure",
    ["missing", "modified", "extra", "symlink", "pipeline", "dimension", "hash", "duplicate"],
)
def test_asset_inventory_fails_closed(tmp_path, monkeypatch, failure):
    source, record = asset_bundle(tmp_path, monkeypatch)
    if failure == "missing":
        (source / "backbone/tokenizer.json").unlink()
    elif failure == "modified":
        (source / "backbone/tokenizer.json").write_text("different")
    elif failure == "extra":
        (source / "policy/model.safetensors").write_text("unlisted weights")
    elif failure == "symlink":
        (source / "policy/unused").symlink_to(tmp_path, target_is_directory=True)
    elif failure == "pipeline":
        path = source / "policy/policy_preprocessor.json"
        atomic_json(path, {"steps": [{"state_file": "../wrong.safetensors"}]})
        record["files"]["policy/policy_preprocessor.json"] = sha256(path)
    elif failure == "dimension":
        path = source / "policy/config.json"
        config = json.loads(path.read_text())
        config["num_steps"] = 9
        atomic_json(path, config)
        record["files"]["policy/config.json"] = sha256(path)
    elif failure == "hash":
        record["floating_gguf_sha256"] = "wrong"
    else:
        shutil.copyfile(source / "fixtures/a.npz", source / "fixtures/b.npz")
        record["fixtures"].append("fixtures/b.npz")
        record["files"]["fixtures/b.npz"] = record["files"]["fixtures/a.npz"]
    atomic_json(source / "spatial-assets.json", record)
    with pytest.raises(ValueError):
        protocol.verify_assets(source)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "policy//config.json",
        "policy/./config.json",
        "policy\\config.json",
    ],
)
def test_asset_path_cannot_escape_bundle(tmp_path, name):
    with pytest.raises(ValueError):
        protocol.relative_file(tmp_path, name)


def test_offline_bundle_reloads_in_fresh_process_without_source_or_cache(tmp_path, monkeypatch):
    source, _ = asset_bundle(tmp_path, monkeypatch)
    target = tmp_path / "package"
    target.mkdir()
    protocol.copy_assets(source, target)
    shutil.rmtree(source)
    output = tmp_path / "process"
    output.mkdir()
    monkeypatch.setenv("HF_HOME", "/must-not-use-old-cache")
    env = app.offline_environment(output, tmp_path)
    worker = Path(application.__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from policykit.spatial_protocol import verify_assets; "
        "verify_assets(sys.argv[2]); import os; print(os.getpid())"
    )
    child = subprocess.run(
        [sys.executable, "-c", code, str(worker), str(target)],
        cwd=output,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert int(child.stdout) != os.getpid()
    assert env["HF_HOME"] != "/must-not-use-old-cache"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert not list(Path(env["HF_HOME"]).iterdir())


@pytest.mark.parametrize("change", ["shape", "nan", "dtype", "image-range", "task", "extra"])
def test_raw_fixture_validation_rejects_unsafe_inputs(change):
    values = raw_arrays()
    if change == "shape":
        values["noise"] = values["noise"][:, :49]
    elif change == "nan":
        values["noise"][0, 0, 0] = np.nan
    elif change == "dtype":
        values["noise"] = values["noise"].astype(np.float64)
    elif change == "image-range":
        values["observation.images.image"][0, 0, 0, 0] = 1.1
    elif change == "task":
        values["task"] = np.array(["x"], dtype=object)
    else:
        values["extra"] = np.array([1])
    with pytest.raises(ValueError):
        validate_arrays(values)


def test_fixture_archive_rejects_extra_fields_and_decodes_valid_arrays(tmp_path):
    path = tmp_path / "fixture.npz"
    np.savez_compressed(path, **raw_arrays())
    assert load_arrays(path)["noise"].shape == (1, 50, 32)
    np.savez_compressed(path, **raw_arrays(), other=np.ones(1))
    with pytest.raises(ValueError, match="inventory"):
        load_arrays(path)


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "hash", "shape", "nan", "bool", "nondeterministic"]
)
def test_full_action_parity_evidence_is_required(tmp_path, monkeypatch, failure):
    _, assets = asset_bundle(tmp_path, monkeypatch)
    result = {
        "fixture_actions_deterministic": True,
        "fixture_actions": [
            {
                "fixture_sha256": assets["files"]["fixtures/a.npz"],
                "actions": [[0.0] * 7 for _ in range(50)],
            }
        ],
    }
    assert app.fixture_actions(result, assets)
    if failure == "missing":
        result["fixture_actions"] = []
    elif failure == "duplicate":
        result["fixture_actions"] *= 2
    elif failure == "hash":
        result["fixture_actions"][0]["fixture_sha256"] = "wrong"
    elif failure == "shape":
        result["fixture_actions"][0]["actions"].pop()
    elif failure in {"nan", "bool"}:
        result["fixture_actions"][0]["actions"][0][0] = float("nan") if failure == "nan" else True
    else:
        result["fixture_actions_deterministic"] = False
    with pytest.raises(ValueError):
        app.fixture_actions(result, assets)


@pytest.mark.parametrize(
    "failure", ["target", "protocol", "state", "truncated", "success", "coverage", "count"]
)
def test_comparisons_reject_noncomparable_or_incomplete_evidence(tmp_path, monkeypatch, failure):
    _, record = asset_bundle(tmp_path, monkeypatch)
    result = report_for(record)
    reference = report_for(record, backend="native-bf16")
    assert app.paired(result, reference, evaluation())
    if failure == "target":
        reference["target_identity"]["gpu_uuid"] = "GPU-other"
    elif failure == "protocol":
        reference["protocol"]["steps"] = 279
    elif failure == "state":
        reference["episodes"][0]["init_state_id"] = 1
    elif failure == "truncated":
        reference["episodes"][0]["task_success"] = False
        reference["success_rate"] = 0.9
    elif failure == "success":
        reference["success_rate"] = float("nan")
    elif failure == "coverage":
        reference["memory_coverage"]["complete"] = False
    else:
        reference["requested_episodes"] = 9
    assert not app.paired(result, reference, evaluation())


@pytest.mark.parametrize(
    "text,expected",
    [
        ("GPU-one, 11, 10\nGPU-one, 12, 20\nGPU-one, 13, 8000\nGPU-two, 11, 500", 30),
        ("GPU-one, 13, 8000", None),
        ("GPU-one, 11, N/A", None),
        ("GPU-one, 11, nan", None),
        ("GPU-one, 11, 0", None),
        ("GPU-one, 11, -5", None),
    ],
)
def test_memory_never_counts_other_processes_or_invalid_samples(text, expected):
    assert app.parse_memory(text, {11, 12}, "GPU-one") == expected


def test_process_telemetry_covers_each_stage_and_reaps_timeout(tmp_path, monkeypatch):
    pytest.importorskip("psutil")

    def sample(*args, **kwargs):
        return SimpleNamespace(stdout="fixture", returncode=0)

    monkeypatch.setattr(app.subprocess, "run", sample)
    monkeypatch.setattr(app, "parse_memory", lambda text, pids, gpu: 100.0 if pids else None)
    code = """
import json, os, pathlib, time
p = pathlib.Path('.')
(p/'pid').write_text(str(os.getpid()))
for name in ('reload', 'timing', 'rollout-0-2'):
    temp=p/'stage.tmp'; temp.write_text(json.dumps({'stage':name})); temp.replace(p/'stage.json')
    while True:
        ack = p/'sampled.json'
        if ack.exists() and json.loads(ack.read_text())['stage'] == name:
            break
        time.sleep(.01)
"""
    stages = ["reload", "timing", "rollout-0-2"]
    result = app.measure_child(
        [sys.executable, "-c", code],
        tmp_path,
        os.environ.copy(),
        stages,
        {"gpu_uuid": "GPU-test"},
        10,
    )
    assert set(result) == set(stages)
    assert all(row["gpu_samples"] > 0 for row in result.values())
    (tmp_path / "stage.json").unlink()
    with pytest.raises(TimeoutError):
        app.measure_child(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            os.environ.copy(),
            stages,
            {"gpu_uuid": "GPU-test"},
            0.2,
        )


def test_package_strips_reference_and_allows_reference_to_exceed_budget(tmp_path, monkeypatch):
    source, assets = asset_bundle(tmp_path, monkeypatch, reference=True)
    (source / "model.gguf").write_bytes(b"synthetic, not actual GGUF")
    application.publish(
        source,
        {
            "task": "libero_spatial",
            "precision": "float",
            "native_model_sha256": protocol.WEIGHTS_SHA256,
            "model_sha256": sha256(source / "model.gguf"),
        },
        "fixture",
    )
    output = tmp_path / "output"
    output.mkdir()
    result = report_for(assets)
    native = report_for(assets, backend="native-bf16")
    control = copy.deepcopy(result)
    control.update(peak_device_mib=9000, p95_ms=1000)
    job = {
        "job_id": "synthetic",
        "final": True,
        "output_dir": str(output),
        "artifact": {"path": str(source), "id": "source"},
        "parameters": {
            "evaluation": evaluation(),
            "limits": {
                "min_success_rate": 1,
                "max_success_drop": 0,
                "max_p95_ms": 100,
                "max_peak_device_mib": 100,
            },
        },
        "prior_reports": [
            {**native, "stage": "final-reference"},
            {**control, "stage": "final-control"},
            {**result, "stage": "final-evaluation"},
        ],
    }

    def reload(_, path):
        assert path == output / "package-pending" and path != source
        assert not (path / protocol.REFERENCE_FILE).exists()
        protocol.verify_assets(path)
        return result

    monkeypatch.setattr(application, "evaluate_package", reload)
    response = application.run_policy(job)
    package = Path(response["artifact"]["path"])
    manifest = application.verify(package)
    assert manifest["metadata"]["native_model_sha256"] == protocol.WEIGHTS_SHA256
    assert not (package / protocol.REFERENCE_FILE).exists()
    assert manifest["metadata"]["deployment_verified"] is True


@pytest.mark.parametrize("failure", [None, "state", "normalization"])
def test_spatial_gguf_must_preserve_all_eight_normalized_coordinates(
    tmp_path, monkeypatch, failure
):
    import gguf
    from safetensors.numpy import save_file

    root, _ = asset_bundle(tmp_path, monkeypatch)
    writer = gguf.GGUFWriter(root / "model.gguf", "smolvla")
    for key, value in {
        "chunk_size": 50,
        "max_action_dim": 32,
        "real_action_dim": 7,
        "image_size": 512,
        "max_state_dim": 32,
        "num_steps": 10,
        "real_state_dim": 6 if failure == "state" else 8,
    }.items():
        writer.add_uint32("smolvla." + key, value)
    for filename, prefix, feature, size in (
        (
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "state",
            "observation.state",
            8,
        ),
        ("policy_postprocessor_step_0_unnormalizer_processor.safetensors", "action", "action", 7),
    ):
        saved = {
            feature + ".mean": np.zeros(size, dtype=np.float32),
            feature + ".std": np.ones(size, dtype=np.float32),
        }
        save_file(saved, root / "policy" / filename)
        for suffix in ("mean", "std"):
            value = saved[feature + "." + suffix].copy()
            if failure == "normalization" and prefix == "state":
                value[7] += 1
            writer.add_tensor(prefix + "_" + suffix, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    manifest = {"metadata": {"precision": "float", "action_dim": 7}}
    if failure:
        with pytest.raises(ValueError, match="Spatial state|normalization differs"):
            app.spatial_contract(root, manifest)
    else:
        assert app.spatial_contract(root, manifest)["action_length"] == 1600


@pytest.mark.parametrize("backend", ["native-bf16", "cpp"])
@pytest.mark.parametrize("failure", [None, "pid", "fixture", "episode", "truncated", "timing"])
def test_application_envelope_requires_complete_fresh_child_result(
    tmp_path, monkeypatch, backend, failure
):
    root, assets = asset_bundle(tmp_path, monkeypatch, reference=True)
    (root / "model.gguf").write_bytes(b"synthetic GGUF, parser mocked")
    application.publish(root, {"task": "libero_spatial", "precision": "float"}, "fixture")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(app, "spatial_contract", lambda *_: {})
    monkeypatch.setattr(app, "runtime_identity", lambda *_: {"synthetic": True})
    monkeypatch.setattr(app, "simulator_identity", lambda *_: {"synthetic": True})
    monkeypatch.setattr(app, "target_identity", lambda: {"gpu_uuid": "GPU-test"})

    def child(command, pending, env, stages, target, timeout):
        request = json.loads((pending / "request.json").read_text())
        assert request["source"] == str(root)
        assert env["HF_HUB_OFFLINE"] == "1" and env["HF_HOME"].startswith(str(pending))
        result = {
            "protocol_sha256": request["protocol_sha256"],
            "manifest_sha256": request["manifest_sha256"],
            "inference_runtime": {"actual_child": "synthetic"},
            "process_id": os.getpid() + 1,
            "episodes": [
                {
                    **row,
                    "status": "episode_complete",
                    "task_success": False,
                    "steps": 280,
                    "benchmark_horizon": 280,
                }
                for row in request["protocol"]["episodes"]
            ],
            "samples_ms": [10.0, 11.0],
            "fixture_actions_deterministic": True,
            "fixture_actions": [
                {
                    "fixture_sha256": assets["files"]["fixtures/a.npz"],
                    "actions": [[0.0] * 7 for _ in range(50)],
                }
            ],
        }
        if failure == "pid":
            result["process_id"] = os.getpid()
        elif failure == "fixture":
            result["fixture_actions"] = []
        elif failure == "episode":
            result["episodes"].pop()
        elif failure == "truncated":
            result["episodes"][0]["steps"] = 279
        elif failure == "timing":
            result["samples_ms"] = [0, 11]
        atomic_json(pending / "result.json", result)
        return {stage: {"gpu_samples": 2, "sampled_peak_device_used_mib": 100} for stage in stages}

    monkeypatch.setattr(app, "measure_child", child)
    job = {
        "job_id": "fixture",
        "output_dir": str(output),
        "artifact": {"path": str(root)},
        "evaluation_backend": backend,
        "runtime": {"device": "cuda", "simulator_lane": str(tmp_path)},
        "parameters": {"evaluation": evaluation()},
    }
    if failure:
        with pytest.raises(ValueError):
            application.evaluate_policy(job)
    else:
        report = application.evaluate_policy(job)["report"]
        assert report["complete_episodes"] == 10 and report["success_rate"] == 0
        assert report["source_native_model_sha256"] == protocol.WEIGHTS_SHA256
        assert report["memory_coverage"]["complete"] is True
        assert report["backend"] == backend and report["fresh_reload_verified"] is True
        if backend == "native-bf16":
            assert report["native_model_sha256"] == sha256(root / protocol.REFERENCE_FILE)


@pytest.mark.skipif(os.name != "posix", reason="Spatial Linux process group contract")
def test_outer_group_cancellation_kills_inference_and_server(tmp_path):
    import signal
    import time

    import psutil

    worker = Path(application.__file__).resolve().parents[1]
    child_code = """
import subprocess, sys, os, pathlib, time
server = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path('inference.pid').write_text(str(os.getpid()))
pathlib.Path('server.pid').write_text(str(server.pid))
time.sleep(60)
"""
    wrapper = tmp_path / "supervisor.py"
    wrapper.write_text(
        "import sys, os; from pathlib import Path; from types import SimpleNamespace\n"
        f"sys.path.insert(0, {str(worker)!r})\n"
        "from policykit import spatial_application as app\n"
        "app.subprocess.run = lambda *a, **k: SimpleNamespace(stdout='',returncode=0)\n"
        f"app.measure_child([sys.executable, '-c', {child_code!r}], Path.cwd(), "
        "os.environ.copy(), ['reload','timing'], {'gpu_uuid':'GPU-fixture'}, 60)\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(wrapper)],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pids = []
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "server.pid").exists():
            assert process.poll() is None, "Supervisor failed before child setup"
            if time.monotonic() > deadline:
                pytest.fail("Inference/server setup timed out")
            time.sleep(0.02)
        pids = [int((tmp_path / name).read_text()) for name in ("inference.pid", "server.pid")]
        assert all(psutil.pid_exists(pid) for pid in pids)
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        deadline = time.monotonic() + 5

        def executing(pid):
            try:
                return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False

        while any(executing(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(executing(pid) for pid in pids)
        assert not psutil.pid_exists(pids[0]), "Supervisor must reap its direct inference child"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="Spatial Linux process group contract")
def test_cancellation_during_spawn_is_not_lost_and_restores_signal_handlers(tmp_path, monkeypatch):
    import signal

    original = app.subprocess.Popen
    previous = signal.getsignal(signal.SIGTERM)
    children = []

    def interrupted_spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        # The OS child exists but Popen has not yet returned the handle to the supervisor.
        os.kill(os.getpid(), signal.SIGTERM)
        return child

    monkeypatch.setattr(app.subprocess, "Popen", interrupted_spawn)
    with pytest.raises(InterruptedError, match="process launch"):
        app.measure_child(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            os.environ.copy(),
            ["reload", "timing"],
            {"gpu_uuid": "GPU-test"},
            10,
        )
    assert children[0].poll() is not None
    assert signal.getsignal(signal.SIGTERM) == previous
