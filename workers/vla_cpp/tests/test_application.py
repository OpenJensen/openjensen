"""Adapter acceptance checks with synthetic reports, not performance evidence."""

from pathlib import Path

import pytest
from policykit import application


def package_job(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.gguf").write_bytes(b"fixture, not a model")
    application.publish(source, {"precision": "float"}, "fixture")
    output = tmp_path / "output"
    output.mkdir()
    report = {
        "runtime": {"fixture": True},
        "success_rate": 0.5,
        "complete_episodes": 2,
        "p95_ms": 10,
        "peak_device_mib": 20,
    }
    job = {
        "job_id": "fixture",
        "final": True,
        "output_dir": str(output),
        "parameters": {
            "limits": {
                "min_success_rate": 0.5,
                "max_success_drop": 0,
                "max_p95_ms": 100,
                "max_peak_device_mib": 100,
            },
            "evaluation": {"final_states": [2, 3]},
        },
        "artifact": {"id": "fixture:source", "path": str(source)},
        "prior_reports": [{"stage": "final-reference", **report}],
    }
    return job, report


def test_package_rejects_quality_drop_even_when_absolute_limits_pass(tmp_path, monkeypatch):
    job, report = package_job(tmp_path)
    job["prior_reports"][0]["success_rate"] = 1
    monkeypatch.setattr(application, "evaluate_policy", lambda _: {"report": report})
    with pytest.raises(ValueError, match="paired reference"):
        application.run_policy(job)
    assert not (Path(job["output_dir"]) / "package").exists()


def test_package_preserves_exact_source_and_acceptance_evidence(tmp_path, monkeypatch):
    job, report = package_job(tmp_path)
    monkeypatch.setattr(application, "evaluate_policy", lambda _: {"report": report})
    response = application.run_policy(job)
    path = Path(response["artifact"]["path"])
    manifest = application.verify(path)
    assert (path / "model.gguf").read_bytes() == b"fixture, not a model"
    assert manifest["metadata"]["deployment_verified"] is True
    assert {"lineage.json", "workflow-evidence.json", "reload-verification.json"} <= set(
        manifest["files"]
    )
