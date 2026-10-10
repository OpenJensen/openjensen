"""Checkpoint selection, conversion lineage and cloud source staging without GPU allocation."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from vla_platform.lifecycle.sky_quantization import CPPZMQ_SHA256, VENDOR_COMMIT, stage_quantization


@pytest.fixture
def pipeline(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "workers/vla_cpp"))
    from policykit import cloud_quantize

    monkeypatch.setattr(cloud_quantize, "describe_runtime", lambda vendor: {})
    return cloud_quantize


def checkpoint_job(tmp_path, pipeline, *, kind="training_checkpoint", architecture="smolvla"):
    source, output = tmp_path / "selected", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    checkpoint = source / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "recipe.json").write_text('{"steps": 12}')
    (checkpoint / "adapter.safetensors").write_bytes(b"selected step 12, not the latest step 24")
    pipeline.application.publish(checkpoint, {}, "checkpoint")
    info = pipeline.application.publish(
        source, {"architecture": architecture, "step": 12}, "Step 12", kind
    )
    return {
        "schema_version": 1,
        "job_id": "quantization-job",
        "operation": "policy.quantize",
        "output_dir": str(output),
        "runtime": {
            "vendor": str(tmp_path / "vendor"),
            "device": "cuda",
            "smoke_build": str(tmp_path / "native-smoke"),
        },
        "parameters": {"precision": {"language": "Q4_0", "vision": "Q8_0"}},
        "artifact": {"id": "training-job:checkpoint-000012", **info},
    }


def install_fake_converters(monkeypatch, pipeline, observed):
    package = ModuleType("firebird_vla")
    export = ModuleType("firebird_vla.export")

    def export_checkpoint(source, destination):
        observed["checkpoint"] = source
        observed["adapter"] = (source / "adapter.safetensors").read_bytes()
        destination.mkdir()
        (destination / "model.safetensors").write_bytes(b"floating intermediate")
        (destination / "policy_preprocessor.json").write_text("{}")
        (destination / "normalizer.safetensors").write_bytes(b"stats")

    export.export_checkpoint = export_checkpoint
    monkeypatch.setitem(sys.modules, "firebird_vla", package)
    monkeypatch.setitem(sys.modules, "firebird_vla.export", export)

    def import_policy(job):
        observed["converted"] = job["artifact"]["path"]
        destination = Path(job["output_dir"]) / "bundle"
        destination.mkdir()
        (destination / "model.gguf").write_bytes(b"float gguf")
        return {
            "artifact": pipeline.application.publish(
                destination, {"architecture": "smolvla", "precision": "float", "step": 12}, "float"
            )
        }

    def quantize_policy(job):
        observed["precision"] = job["parameters"]["precision"]
        observed["quantized_input"] = Path(job["artifact"]["path"])
        destination = Path(job["output_dir"]) / "conversion" / "bundle"
        destination.mkdir(parents=True)
        (destination / "model.gguf").write_bytes(b"packed")
        return {
            "artifact": pipeline.application.publish(
                destination, {"precision": observed["precision"]}, "Q4_0 + vision Q8"
            ),
            "report": {"scope": "conversion_only", "seconds": 0.1},
        }

    monkeypatch.setattr(pipeline.application, "import_policy", import_policy)
    monkeypatch.setattr(pipeline.application, "quantize_policy", quantize_policy)

    def inference(model, executable, output, **kwargs):
        observed.setdefault("inference_models", []).append(model.read_bytes())
        output.mkdir(parents=True, exist_ok=True)
        actions = output / "cpu-inference-smoke-actions.json"
        pipeline.atomic_json(
            actions, {"values": [0.125 if model.read_bytes() == b"packed" else 0.25] * 128}
        )
        report = {
            "scope": "synthetic_input_native_inference",
            "finite_action_values": 128,
            "backend": "cpu",
            "calls": 1,
            "synthetic_image_count": kwargs["cameras"],
            "image_size": 512,
            "action_chunk_size": 4,
            "max_action_dim": 32,
            "real_action_dim": 6,
            "input_sha256": "a" * 64,
            "executable_sha256": "b" * 64,
            "model_sha256": pipeline.sha256(model),
            "actions_sha256": pipeline.sha256(actions),
        }
        pipeline.atomic_json(output / "cpu-inference-smoke.json", report)
        return report

    monkeypatch.setattr(pipeline, "verify_cpu_inference", inference)


def test_selected_checkpoint_exports_quantizes_and_keeps_small_provenance(
    tmp_path, monkeypatch, pipeline
):
    observed = {}
    install_fake_converters(monkeypatch, pipeline, observed)
    job = checkpoint_job(tmp_path, pipeline)
    response = pipeline.quantize_checkpoint(job)
    bundle = Path(response["artifact"]["path"])
    manifest = pipeline.application.verify(bundle)
    assert observed["checkpoint"] == Path(job["artifact"]["path"]) / "checkpoint"
    assert observed["adapter"] == b"selected step 12, not the latest step 24"
    assert observed["precision"] == {"language": "Q4_0", "vision": "Q8_0"}
    assert not observed["quantized_input"].exists()
    assert not list(Path(job["output_dir"]).glob(".floating-*"))
    assert "processors/normalizer.safetensors" in manifest["files"]
    assert not list(bundle.rglob("model.safetensors"))
    assert manifest["metadata"]["native_inference_verified"] is True
    assert response["report"]["inference"]["finite_action_values"] == 128
    assert observed["inference_models"] == [b"float gguf", b"packed"]
    comparison = response["report"]["comparison"]
    assert comparison["action_rmse"] == comparison["action_max_abs_difference"] == 0.125
    assert comparison["coordinates"] == 24
    assert comparison["source_file_bytes"] == len(b"float gguf")
    assert comparison["quantized_file_bytes"] == len(b"packed")
    assert comparison["validation_loss"] is comparison["task_success"] is None
    assert "quantization-comparison.json" in manifest["files"]
    assert "reference-check/cpu-inference-smoke-actions.json" in manifest["files"]
    lineage = json.loads((bundle / "checkpoint-lineage.json").read_text())
    assert lineage["source_artifact_id"] == "training-job:checkpoint-000012"
    assert lineage["checkpoint_step"] == response["report"]["checkpoint_step"] == 12
    assert lineage["source_manifest_sha256"] == pipeline.sha256(
        Path(job["artifact"]["path"]) / "manifest.json"
    )


def test_quantization_failure_removes_floating_intermediates(tmp_path, monkeypatch, pipeline):
    observed = {}
    install_fake_converters(monkeypatch, pipeline, observed)
    job = checkpoint_job(tmp_path, pipeline)

    def fail(job):
        raise ValueError("precision audit failed")

    monkeypatch.setattr(pipeline.application, "quantize_policy", fail)
    with pytest.raises(ValueError, match="precision audit"):
        pipeline.quantize_checkpoint(job)
    assert not list(Path(job["output_dir"]).glob(".floating-*"))
    pipeline.application.verify(Path(job["artifact"]["path"]))


def test_quantization_refuses_corrupt_selected_checkpoint(tmp_path, pipeline):
    job = checkpoint_job(tmp_path, pipeline)
    (Path(job["artifact"]["path"]) / "checkpoint/adapter.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        pipeline.quantize_checkpoint(job)


def test_quantization_refuses_other_architectures_before_loading(tmp_path, pipeline):
    job = checkpoint_job(tmp_path, pipeline, architecture="act")
    with pytest.raises(ValueError, match="supports SmolVLA"):
        pipeline.quantize_checkpoint(job)


def test_failed_reference_never_publishes_comparison_and_retires_intermediates(
    tmp_path, monkeypatch, pipeline
):
    install_fake_converters(monkeypatch, pipeline, {})
    job = checkpoint_job(tmp_path, pipeline)

    def fail(*args, **kwargs):
        raise ValueError("Original prediction is non-finite")

    monkeypatch.setattr(pipeline, "verify_cpu_inference", fail)
    with pytest.raises(ValueError, match="Original prediction"):
        pipeline.quantize_checkpoint(job)
    assert not list(Path(job["output_dir"]).glob(".floating-*"))
    assert not (Path(job["output_dir"]) / "conversion").exists()
    pipeline.application.verify(Path(job["artifact"]["path"]))


def test_cloud_setup_contains_only_sources_and_pins_converter(tmp_path):
    root = Path(__file__).resolve().parents[1]
    setup = stage_quantization(tmp_path, root / "workers/smolvla_qlora")
    assert (tmp_path / "quantization-worker/policykit/cloud_quantize.py").is_file()
    assert not list(tmp_path.rglob("*.safetensors"))
    assert not list(tmp_path.rglob("*.pyc"))
    assert any(VENDOR_COMMIT in command for command in setup)
    assert any("--describe-runtime" in command for command in setup)
    packages = setup[0].split()
    assert {
        "build-essential",
        "cmake",
        "git",
        "curl",
        "pkg-config",
        "libprotobuf-dev",
        "protobuf-compiler",
        "libzmq3-dev",
        "libssl-dev",
        "libcurl4-openssl-dev",
    }.issubset(packages)
    assert "cppzmq-dev" not in packages
    assert any("cppzmq/v4.10.0/zmq.hpp" in command for command in setup)
    assert any(CPPZMQ_SHA256 in command and "sha256sum --check" in command for command in setup)
    assert any("--target vla_predict_check -j2" in command for command in setup)
