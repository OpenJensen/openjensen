"""Application envelopes and failure boundaries; stubbed probes are not ML evidence."""

import hashlib
import json
from pathlib import Path

import pytest
from test_training_source import digest, resign
from test_training_source import training_bundle as training_bundle

from firebird_act import application, export
from firebird_act.bundle import canonical, inventory, read_json, verify_export
from firebird_act.probe import FIXTURE_SEEDS, VERSIONS


def job(source: Path, output: Path) -> dict:
    return {
        "schema_version": 1,
        "job_id": "export-job",
        "operation": "policy.export",
        "output_dir": str(output),
        "artifact": {
            "id": "training-job:operation",
            "format": "training_checkpoint",
            "path": str(source),
            "manifest_sha256": digest(source / "manifest.json"),
        },
    }


@pytest.fixture
def stub_probes(monkeypatch):
    calls = []

    def probe(checkpoint, report, timeout, *, forbidden_sources=()):
        calls.append((checkpoint, timeout, forbidden_sources))
        return {
            "schema_version": 1,
            "device": "cpu",
            "dtype": "float32",
            "cpu_threads": 1,
            "network_disabled": True,
            "versions": dict(VERSIONS),
            "python": "3.12.14",
            "checkpoint_files": inventory(checkpoint),
            "fixtures": [
                {
                    "seed": seed,
                    "input_sha256": f"{seed:064x}",
                    "queue_and_reset_exact": True,
                    "chunk": [[0.0] * 6 for _ in range(100)],
                    "postprocessed": [[0.0] * 6 for _ in range(100)],
                }
                for seed in FIXTURE_SEEDS
            ],
        }

    monkeypatch.setattr(export, "run_probe", probe)
    return calls


def tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}


def test_export_envelope_preserves_training_and_exact_package(
    training_bundle, tmp_path, stub_probes
):
    before = tree(training_bundle)
    payload = job(training_bundle, tmp_path / "operation")
    response = application.run_job(payload)
    artifact = response["artifact"]
    assert artifact["format"] == "inference_export"
    output = Path(artifact["path"])
    manifest = read_json(output / "manifest.json")
    assert manifest["files"] == {k: v for k, v in tree(output).items() if k != "manifest.json"}
    assert verify_export(output / "policy")["inference_only"] is True
    metadata = manifest["metadata"]
    assert metadata["source_artifact_id"] == payload["artifact"]["id"]
    assert metadata["source_manifest_sha256"] == payload["artifact"]["manifest_sha256"]
    assert metadata["policy_manifest_sha256"] == digest(output / "policy/manifest.json")
    assert metadata["checkpoint_manifest_sha256"] == digest(
        training_bundle / "checkpoint/manifest.json"
    )
    assert metadata["checkpoint_step"] == 2
    assert metadata["inference_only"] is True
    assert metadata["training_resume_supported"] is False
    assert metadata["synthetic_parity_verified"] is True
    assert metadata["fresh_reload_verified"] is True
    assert metadata["task_success"] is None and metadata["calibration_verified"] is False
    assert response["report"]["scope"] == "ACT inference export and synthetic CPU parity"
    assert len(stub_probes) == 3
    assert stub_probes[-1][2]
    assert tree(training_bundle) == before
    assert not list(output.parent.glob(".act-application-*"))


@pytest.mark.parametrize(
    "field,value", [("operation", "policy.quantize"), ("schema_version", True)]
)
def test_rejects_wrong_protocol_before_probe(training_bundle, tmp_path, stub_probes, field, value):
    payload = job(training_bundle, tmp_path / "operation")
    payload[field] = value
    with pytest.raises(ValueError):
        application.run_job(payload)
    assert not stub_probes


def test_rejects_nontraining_artifact(training_bundle, tmp_path, stub_probes):
    payload = job(training_bundle, tmp_path / "operation")
    payload["artifact"]["format"] = "inference_export"
    with pytest.raises(ValueError, match="training checkpoint"):
        application.run_job(payload)
    assert not stub_probes


def test_admission_and_export_bind_same_bytes(training_bundle, tmp_path, monkeypatch, stub_probes):
    original = application.export_policy

    def changed(source, output, **kwargs):
        config = read_json(source / "train_config.json")
        (source / "train_config.json").write_bytes(canonical(config | {"changed": True}))
        resign(training_bundle)
        return original(source, output, **kwargs)

    monkeypatch.setattr(application, "export_policy", changed)
    output = tmp_path / "operation"
    with pytest.raises(ValueError, match="source|Source"):
        application.run_job(job(training_bundle, output))
    assert not (output / "inference-export").exists()
    assert not list(output.glob(".act-application-*"))


def test_resume_state_mutation_during_export_cannot_publish(
    training_bundle, tmp_path, monkeypatch, stub_probes
):
    original = application.export_policy

    def changed(source, output, **kwargs):
        receipt = original(source, output, **kwargs)
        (training_bundle / "checkpoint/training_state/optimizer_state.safetensors").write_bytes(
            b"changed"
        )
        return receipt

    monkeypatch.setattr(application, "export_policy", changed)
    output = tmp_path / "operation"
    with pytest.raises(ValueError):
        application.run_job(job(training_bundle, output))
    assert not (output / "inference-export").exists()


def test_missing_complete_reload_proof_cannot_publish(
    training_bundle, tmp_path, monkeypatch, stub_probes
):
    original = application.export_policy

    def missing(source, output, **kwargs):
        receipt = original(source, output, **kwargs)
        receipt["final_package_reload"]["exact_equal"] = False
        return receipt

    monkeypatch.setattr(application, "export_policy", missing)
    output = tmp_path / "operation"
    with pytest.raises(ValueError, match="reload"):
        application.run_job(job(training_bundle, output))
    assert not (output / "inference-export").exists()


def test_output_inside_training_bundle_rejected(training_bundle, stub_probes):
    before = tree(training_bundle)
    with pytest.raises(ValueError, match="outside"):
        application.run_job(job(training_bundle, training_bundle / "operation"))
    assert tree(training_bundle) == before and not stub_probes


def test_existing_output_not_replaced(training_bundle, tmp_path, stub_probes):
    output = tmp_path / "operation"
    (output / "inference-export").mkdir(parents=True)
    marker = output / "inference-export/user.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        application.run_job(job(training_bundle, output))
    assert marker.read_text() == "preserve" and not stub_probes


def test_raced_output_not_replaced(training_bundle, tmp_path, monkeypatch, stub_probes):
    original = application.publish_new_directory

    def race(staging, output):
        output.mkdir()
        original(staging, output)

    monkeypatch.setattr(application, "publish_new_directory", race)
    output = tmp_path / "operation"
    with pytest.raises(OSError):
        application.run_job(job(training_bundle, output))
    assert (output / "inference-export").is_dir()
    assert not list((output / "inference-export").iterdir())


def test_cli_error_is_bounded_and_preserves_source(training_bundle, tmp_path, monkeypatch):
    output = tmp_path / "operation"
    output.mkdir()
    request, result = output / "request.json", output / "result.json"
    request.write_bytes(canonical(job(training_bundle, output)))
    before = tree(training_bundle)

    def fail(*args, **kwargs):
        raise ValueError("failure " + "x" * 6000)

    monkeypatch.setattr(application, "export_policy", fail)
    assert application.main([str(request), str(result)]) == 1
    response = json.loads(result.read_text())
    assert response["schema_version"] == 1 and response["job_id"] == "export-job"
    assert len(response["error"]) <= 2000 and "artifact" not in response
    assert tree(training_bundle) == before


def test_cli_success_and_new_result_guard(training_bundle, tmp_path, stub_probes):
    output = tmp_path / "operation"
    output.mkdir()
    request, result = output / "request.json", output / "result.json"
    request.write_bytes(canonical(job(training_bundle, output)))
    assert application.main([str(request), str(result)]) == 0
    response = json.loads(result.read_text())
    assert (
        response["job_id"] == "export-job" and response["artifact"]["format"] == "inference_export"
    )
    before = hashlib.sha256(result.read_bytes()).hexdigest()
    assert application.main([str(request), str(result)]) == 1
    assert hashlib.sha256(result.read_bytes()).hexdigest() == before
    assert len(stub_probes) == 3


@pytest.mark.parametrize("size", [None, -1, True, 0])
def test_invalid_receipt_size_cannot_publish(
    training_bundle, tmp_path, monkeypatch, stub_probes, size
):
    original = application.export_policy

    def malformed(source, output, **kwargs):
        receipt = original(source, output, **kwargs)
        receipt["package_bytes"] = size
        return receipt

    monkeypatch.setattr(application, "export_policy", malformed)
    output = tmp_path / "operation"
    with pytest.raises(ValueError, match="size"):
        application.run_job(job(training_bundle, output))
    assert not (output / "inference-export").exists()


def test_local_dataset_snapshot_is_not_relabelled_as_huggingface(
    training_bundle, tmp_path, stub_probes
):
    manifest = read_json(training_bundle / "manifest.json")
    manifest["metadata"]["dataset"] = {
        "source": "local",
        "repo_id": "firebird/local-fixture",
        "revision": "sha256:" + "a" * 64,
    }
    (training_bundle / "manifest.json").write_bytes(canonical(manifest))
    resign(training_bundle)
    with pytest.raises(ValueError, match="dataset lineage"):
        application.run_job(job(training_bundle, tmp_path / "export"))
    assert not stub_probes
