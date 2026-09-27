"""Cloud Evaluate/Run protocol and source staging; fixtures never allocate GPUs."""

import json
from pathlib import Path

import pytest
from vla_platform.lifecycle import sky_runner
from vla_platform.lifecycle.sky_inference import stage_inference


@pytest.fixture
def pipeline(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "workers/vla_cpp"))
    from policykit import cloud_inference

    return cloud_inference


def job(tmp_path, pipeline, operation="policy.evaluate", kind="gguf"):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "model.gguf").write_bytes(b"Native protocol fixture, not model weights")
    artifact = pipeline.application.publish(
        source, {"architecture": "smolvla", "camera_keys": ["front"]}, "Policy", kind
    )
    return {
        "schema_version": 1,
        "job_id": "fixture",
        "operation": operation,
        "output_dir": str(output),
        "artifact": {"id": "original:operation", **artifact},
        "parameters": {"evaluation": {"mode": "engine", "suite": "libero_object"}},
        "runtime": {"device": "cuda"},
    }


def measured_report():
    return {
        "scope": "engine_diagnostics",
        "fresh_reload_verified": True,
        "runtime": {"device": "cuda", "fixture_only": True},
        "memory_coverage": {"complete": True},
        "finite_action_values": 1600,
        "samples_ms": [10.0, 20.0],
        "p50_ms": 15.0,
        "p95_ms": 19.5,
        "peak_device_mib": 1000.0,
        "success_rate": None,
        "complete_episodes": 0,
    }


@pytest.mark.parametrize(
    "operation,kind,method",
    [
        ("policy.evaluate", "gguf", "evaluate_policy"),
        ("policy.evaluate", "deployment_package", "evaluate_policy"),
        ("policy.run", "gguf", "run_policy"),
        ("policy.run", "deployment_package", "evaluate_package"),
    ],
)
def test_cloud_inference_uses_measured_engine_and_fresh_package_paths(
    tmp_path, monkeypatch, pipeline, operation, kind, method
):
    request = job(tmp_path, pipeline, operation, kind)
    calls = []

    def action(received, *args):
        calls.append((received, args))
        return measured_report() if method == "evaluate_package" else {"report": measured_report()}

    monkeypatch.setattr(pipeline.application, method, action)
    result = pipeline.inference_job(request)
    assert len(calls) == 1 and calls[0][0]["artifact"]["id"] == "original:operation"
    if method == "evaluate_package":
        assert calls[0][1] == (Path(request["artifact"]["path"]),)
    assert result["report"]["p95_ms"] == 19.5
    assert result["report"]["task_success"] is None
    assert result["report"]["deployment_verified"] is False
    assert result["report"]["package_reexecuted"] == (kind == "deployment_package")
    assert (Path(request["output_dir"]) / "inference-report.json").is_file()


@pytest.mark.parametrize(
    "fault",
    [
        {"runtime": {"device": "cpu"}},
        {"memory_coverage": {"complete": False}},
        {"fresh_reload_verified": False},
        {"samples_ms": []},
        {"finite_action_values": 0},
    ],
)
def test_cloud_inference_cannot_report_success_without_required_measurements(
    tmp_path, monkeypatch, pipeline, fault
):
    request = job(tmp_path, pipeline)
    monkeypatch.setattr(
        pipeline.application,
        "evaluate_policy",
        lambda _: {"report": {**measured_report(), **fault}},
    )
    with pytest.raises(ValueError, match="complete CUDA"):
        pipeline.inference_job(request)


def test_cloud_inference_rejects_incompatible_task_mode_before_execution(tmp_path, pipeline):
    request = job(tmp_path, pipeline)
    request["parameters"]["evaluation"]["mode"] = "libero"
    with pytest.raises(ValueError, match="simulator tasks"):
        pipeline.inference_job(request)


@pytest.mark.parametrize("accelerator,architecture", [("T4", "75"), ("L4", "89"), ("A100", "80")])
def test_cloud_inference_build_is_pinned_instrumented_and_has_worker_locks(
    tmp_path, accelerator, architecture
):
    worker = Path(__file__).parents[1] / "workers/smolvla_qlora"
    setup = "\n".join(stage_inference(tmp_path, worker, accelerator))
    assert "-DGGML_CUDA=ON" in setup and "-DGGML_CUDA=OFF" not in setup
    assert "-DCMAKE_CUDA_ARCHITECTURES=" + architecture in setup
    assert "--target vla-bench vla_predict_check" in setup
    assert "instrument_cuda_bench.py" in setup
    assert "cuda-nvcc-12-8" in setup and "sha256sum --check" in setup
    assert (tmp_path / "quantization-worker/uv.lock").is_file()
    assert (tmp_path / "quantization-worker/policykit/cloud_inference.py").is_file()
    assert not list(tmp_path.rglob("*.gguf"))


def test_prepared_evaluation_uses_inference_worker_without_training_dependencies(
    tmp_path, monkeypatch
):
    from test_sky_runner import target as target_fixture

    worker = Path(__file__).parents[1] / "workers/smolvla_qlora"
    monkeypatch.setattr(sky_runner, "worker_root", lambda: worker)
    target = target_fixture.__wrapped__()
    payload = {
        "schema_version": 1,
        "operation": "policy.evaluate",
        "job_id": "evaluation",
        "parameters": {"timeout_seconds": 300},
        "artifact": None,
    }
    task, _ = sky_runner.prepare(payload, tmp_path / "stage", target)
    task = json.loads(task.read_text())
    assert "requirements-smolvla-linux.txt" not in task["setup"]
    dispatch = json.loads((tmp_path / "stage/sky-bundle/dispatch.json").read_text())
    assert dispatch["worker_module"] == "policykit.cloud_inference"
    assert task["resources"]["memory"] == "16+"
