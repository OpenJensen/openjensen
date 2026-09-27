"""CPU acceptance regressions; synthetic files and reports are not model evidence."""

import copy
import json
import os
import platform
from pathlib import Path

import gguf
import numpy as np
import pytest
from policykit import application


def bundle(tmp_path, *, packed=True, chunk=50, matrices=112):
    source = tmp_path / "source"
    source.mkdir()
    writer = gguf.GGUFWriter(source / "model.gguf", "smolvla")
    for name, value in {
        "chunk_size": chunk,
        "max_action_dim": 32,
        "real_action_dim": 7,
        "image_size": 512,
    }.items():
        writer.add_uint32("smolvla." + name, value)
    types = {}
    names = [f"vlm.blk.{i // 7}.matrix{i % 7}.weight" for i in range(matrices)] + [
        "action_out_proj.weight"
    ]
    for name in names:
        value = np.ones((32, 32), dtype=np.float32)
        dtype = (
            gguf.GGMLQuantizationType.Q8_0
            if packed and name.startswith("vlm.")
            else gguf.GGMLQuantizationType.F32
        )
        if dtype != gguf.GGMLQuantizationType.F32:
            value = gguf.quants.quantize(value, dtype)
        writer.add_tensor(name, value, raw_dtype=dtype)
        types[name] = dtype.name
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if packed:
        (source / "conversion-manifest.json").write_text(
            json.dumps({"tensor_types": types, "quantized_tensors": names[:-1]})
        )
    metadata = {
        "task": "libero_object",
        "action_dim": 7,
        "precision": {"language": "Q8_0", "vision": None} if packed else "float",
    }
    (source / "tokenizer").mkdir()
    (source / "tokenizer/tokenizer.json").write_text("{}")
    application.publish(source, metadata, "synthetic fixture")
    return source


def engine_job(tmp_path, source):
    output = tmp_path / "output"
    output.mkdir()
    return {
        "schema_version": 1,
        "job_id": "fixture",
        "output_dir": str(output),
        "artifact": {"id": "fixture:source", "path": str(source), "format": "gguf"},
        "runtime": {"device": "cpu", "build": str(tmp_path / "build")},
        "parameters": {"evaluation": {"mode": "engine", "warmups": 1, "repetitions": 2}},
    }


def mock_engine(monkeypatch, *, chunk=50, packed=112, missing=None):
    def measurement(argv, log, runtime):
        if log.name == "reload.log":
            log.write_text(
                f"packed resident matrices: lm={packed} vision=0\naction_len={chunk * 32}\n"
                + "0\n" * (chunk * 32)
            )
        else:
            log.write_text("vla-bench: samples_ms=10,20\n")
        return {
            "sampled_peak_rss_mib": None if missing == log.name else 100,
            "memory_scope": "synthetic CPU fixture",
            "rss_samples": 2,
        }

    monkeypatch.setattr(application, "cpu_measure", measurement)
    monkeypatch.setattr(application, "runtime_identity", lambda _: {"fixture": True})


@pytest.mark.parametrize("chunk,matrices", [(50, 112), (8, 7), (50, 224)])
def test_engine_derives_action_length_and_packing_from_artifact(
    tmp_path, monkeypatch, chunk, matrices
):
    source = bundle(tmp_path, chunk=chunk, matrices=matrices)
    job = engine_job(tmp_path, source)
    mock_engine(monkeypatch, chunk=chunk, packed=matrices)
    report = application.engine(job)
    assert report["finite_action_values"] == chunk * 32
    assert report["artifact_contract"]["packed_matrices"] == {"language": matrices, "vision": 0}


@pytest.mark.parametrize("missing", ["reload.log", "timing.log"])
def test_engine_missing_one_required_memory_stage_stays_unqualified(tmp_path, monkeypatch, missing):
    job = engine_job(tmp_path, bundle(tmp_path, matrices=224))
    mock_engine(monkeypatch, packed=224, missing=missing)
    report = application.engine(job)
    assert report["peak_device_mib"] is None
    assert report["memory_coverage"]["complete"] is False


@pytest.mark.parametrize("peak,samples", [(None, 0), (123, 0), (-1, 3), (float("nan"), 3)])
def test_missing_rollout_telemetry_cannot_reuse_engine_peak(tmp_path, monkeypatch, peak, samples):
    job = engine_job(tmp_path, bundle(tmp_path))
    job["runtime"].update(device="cuda", simulator_lane=str(tmp_path))
    job["parameters"]["evaluation"] = {
        "mode": "libero",
        "task_id": 0,
        "initial_states": [0, 1],
        "final_states": [2],
        "seed": 42,
        "steps": 500,
    }
    reference = {
        "peak_device_mib": 100,
        "runtime": {"fixture": True},
        "measurements": {
            "reload": {"sampled_peak_device_used_mib": 100, "gpu_samples": 2},
            "timing": {"sampled_peak_device_used_mib": 100, "gpu_samples": 2},
        },
    }
    monkeypatch.setattr(application, "engine", lambda _: copy.deepcopy(reference))
    monkeypatch.setattr(application, "runtime_identity", lambda _: {"fixture": True})

    def measurement(argv, log, timeout):
        state = argv[argv.index("--init-state-id") + 1]
        target = (
            Path(job["output_dir"])
            / f"episodes/evaluation/candidate/task0-init{state}-seed42-steps500/result.json"
        )
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"status": "episode_complete", "task_success": True}))
        return {
            "sampled_peak_device_used_mib": peak if state == "0" else 200,
            "gpu_samples": samples if state == "0" else 3,
        }

    monkeypatch.setattr(application, "measure", measurement)
    report = application.evaluate_policy(job)["report"]
    assert report["complete_episodes"] == 2 and report["success_rate"] == 1
    assert report["peak_device_mib"] is None
    assert report["memory_coverage"]["complete"] is False


def runtime_fixture(tmp_path, monkeypatch):
    build = tmp_path / "build"
    (build / "tests").mkdir(parents=True)
    (build / "bin").mkdir()
    for name in ["vla-bench", "tests/vla_predict_check", "vla-server"]:
        (build / name).write_bytes(b"synthetic ELF fixture")
    library = build / "bin/libvla.so"
    library.write_bytes(b"original inference library")
    vendor = tmp_path / "vendor"
    (vendor / "eval/client").mkdir(parents=True)
    (vendor / "eval/client/adapter.py").write_text("# adapter one")
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    (worker / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        application.subprocess,
        "check_output",
        lambda argv, **kw: f"libvla.so => {library} (0x123)\n",
    )
    return {
        "device": "cpu",
        "build": str(build),
        "vendor": str(vendor),
        "worker_root": str(worker),
        "env": {},
    }, library


@pytest.mark.parametrize("changed", ["library", "adapter", "lock", "configuration"])
def test_runtime_identity_changes_when_execution_inputs_change(tmp_path, monkeypatch, changed):
    runtime, library = runtime_fixture(tmp_path, monkeypatch)
    before = application.runtime_identity(runtime)
    if changed == "library":
        library.write_bytes(b"different inference library")
    elif changed == "adapter":
        (Path(runtime["vendor"]) / "eval/client/adapter.py").write_text("# changed adapter")
    elif changed == "lock":
        (Path(runtime["worker_root"]) / "uv.lock").write_text("version = 2\n")
    else:
        runtime["env"]["OMP_NUM_THREADS"] = "8"
    assert application.runtime_identity(runtime) != before


def test_package_verification_uses_fresh_process_and_survives_unavailable_source(
    tmp_path, monkeypatch
):
    source = bundle(tmp_path, packed=False)
    job = engine_job(tmp_path, source)
    fixture_root = tmp_path / "fixture-worker"
    (fixture_root / "policykit").mkdir(parents=True)
    (fixture_root / "policykit/__init__.py").write_text("")
    (fixture_root / "policykit/application.py").write_text("""
import hashlib,json,os,sys
from pathlib import Path
request,result=map(Path,sys.argv[1:])
job=json.loads(request.read_text())
package=Path(job['artifact']['path'])
assert 'package' in package.name
assert not Path(os.environ['UNAVAILABLE_SOURCE']).exists()
manifest=json.loads((package/'manifest.json').read_text())
for name,expected in manifest['files'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected
report={'scope':'engine_diagnostics','runtime':{'fixture':True},'fresh_reload_verified':True,
        'worker_process_id':os.getpid(),'artifact_path':str(package.resolve()),
        'artifact_manifest_sha256':hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((package/'model.gguf').read_bytes()).hexdigest(),
        'success_rate':None,'complete_episodes':0,'p95_ms':10,'peak_device_mib':100}
result.write_text(json.dumps({'schema_version':1,'job_id':job['job_id'],'report':report}))
def main():
    return 0
""")
    monkeypatch.setenv("PYTHONPATH", str(fixture_root))
    monkeypatch.setenv("UNAVAILABLE_SOURCE", str(source))
    original = application.copied

    def copy_then_hide(origin, destination):
        value = original(origin, destination)
        Path(origin).rename(tmp_path / "unavailable-source")
        return value

    monkeypatch.setattr(application, "copied", copy_then_hide)
    monkeypatch.setattr(
        application, "evaluate_policy", lambda _: {"report": {"runtime": {"fixture": True}}}
    )
    response = application.run_policy(job)
    package = Path(response["artifact"]["path"])
    report = json.loads((package / "reload-verification.json").read_text())
    assert report["worker_process_id"] != os.getpid()
    assert report["artifact_path"] != str(source)
    assert application.verify(package)["metadata"]["deployment_verified"] is False
    assert (package / "tokenizer/tokenizer.json").is_file()


def test_rejects_native_count_that_disagrees_with_gguf(tmp_path, monkeypatch):
    job = engine_job(tmp_path, bundle(tmp_path, matrices=112))
    mock_engine(monkeypatch, packed=224)
    with pytest.raises(ValueError, match="resident packing"):
        application.engine(job)


@pytest.mark.parametrize("change", ["audit", "dimension", "precision"])
def test_rejects_false_bundle_metadata(tmp_path, monkeypatch, change):
    source = bundle(tmp_path)
    manifest = application.verify(source)
    if change == "audit":
        audit = json.loads((source / "conversion-manifest.json").read_text())
        audit["tensor_types"].pop(next(iter(audit["tensor_types"])))
        (source / "conversion-manifest.json").write_text(json.dumps(audit))
    elif change == "dimension":
        manifest["metadata"]["action_dim"] = 8
    else:
        manifest["metadata"]["precision"] = "float"
    application.publish(source, manifest["metadata"], "synthetic fixture")
    mock_engine(monkeypatch)
    with pytest.raises(ValueError, match="audit|action dimension|protected tensor"):
        application.engine(engine_job(tmp_path, source))


def test_complete_memory_samples_qualify_but_are_labeled_sampled():
    from policykit.acceptance import memory_coverage

    resource = {"sampled_peak_device_used_mib": 110, "gpu_samples": 2}
    result = memory_coverage(
        {"reload": resource, "timing": resource, "rollout-0": resource}, "cuda"
    )
    assert result["peak_device_mib"] == 110
    assert result["memory_coverage"]["complete"] is True
    assert "transients" in result["memory_coverage"]["sampling_limit"]


def test_unresolved_native_library_fails_closed(tmp_path, monkeypatch):
    runtime, _ = runtime_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        application.subprocess, "check_output", lambda *args, **kwargs: "libcuda.so => not found\n"
    )
    with pytest.raises(ValueError, match="Unresolved native linked library"):
        application.runtime_identity(runtime)


def test_unsupported_runtime_platform_is_explicit(tmp_path, monkeypatch):
    runtime, _ = runtime_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="Linux ELF"):
        application.runtime_identity(runtime)


def test_runtime_identity_never_exports_environment_secrets(tmp_path, monkeypatch):
    runtime, _ = runtime_fixture(tmp_path, monkeypatch)
    runtime["env"]["PRIVATE_TOKEN"] = "fixture-secret-do-not-export"
    assert "fixture-secret-do-not-export" not in json.dumps(application.runtime_identity(runtime))


def test_materialized_payload_changes_are_rejected(tmp_path, monkeypatch):
    source = bundle(tmp_path, packed=False)
    job = engine_job(tmp_path, source)

    def mutate(_, package):
        (package / "model.gguf").write_bytes(b"changed during verification")
        return {"runtime": {"fixture": True}}

    monkeypatch.setattr(application, "evaluate_package", mutate)
    with pytest.raises(ValueError, match="Artifact identity changed"):
        application.run_policy(job)
    assert not (Path(job["output_dir"]) / "package").exists()


def test_failed_fresh_worker_never_publishes(tmp_path, monkeypatch):
    source = bundle(tmp_path, packed=False)
    job = engine_job(tmp_path, source)
    from types import SimpleNamespace

    monkeypatch.setattr(
        application.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    with pytest.raises(ValueError, match="Fresh package worker failed"):
        application.run_policy(job)
    assert not (Path(job["output_dir"]) / "package").exists()


def test_diagnostic_telemetry_with_nonfinite_value_remains_json_serializable(tmp_path, monkeypatch):
    job = engine_job(tmp_path, bundle(tmp_path, packed=False))
    mock_engine(monkeypatch, packed=0)
    original = application.cpu_measure

    def nonfinite(*args):
        result = original(*args)
        result["sampled_peak_rss_mib"] = float("nan")
        return result

    monkeypatch.setattr(application, "cpu_measure", nonfinite)
    report = application.engine(job)
    assert report["peak_device_mib"] is None
    json.dumps(report, allow_nan=False)


def test_export_cannot_overwrite_any_tested_payload_file(tmp_path, monkeypatch):
    source = bundle(tmp_path, packed=False)
    manifest = application.verify(source)
    (source / "runtime-lock.json").write_text("{}")
    application.publish(source, manifest["metadata"], "synthetic fixture")
    job = engine_job(tmp_path, source)
    monkeypatch.setattr(application, "evaluate_package", lambda *_: {"runtime": {"fixture": True}})
    with pytest.raises(ValueError, match="reserved export"):
        application.run_policy(job)


def test_engine_honors_single_selected_camera_and_reports_real_action_preview(
    tmp_path, monkeypatch
):
    source = bundle(tmp_path)
    manifest = application.verify(source)
    application.publish(
        source,
        {**manifest["metadata"], "camera_keys": ["observation.images.front"]},
        "Single camera",
    )
    request = engine_job(tmp_path, source)
    mock_engine(monkeypatch)
    original = application.cpu_measure
    commands = []

    def measure(command, log, runtime):
        commands.append(command)
        return original(command, log, runtime)

    monkeypatch.setattr(application, "cpu_measure", measure)
    report = application.engine(request)
    assert commands[0][-2:] == ["", "1"]
    assert commands[1][commands[1].index("--images") + 1] == "1"
    assert report["synthetic_input"]["camera_count"] == 1
    assert report["action_preview"] == [0.0] * 7
    assert report["task_success"] is None
