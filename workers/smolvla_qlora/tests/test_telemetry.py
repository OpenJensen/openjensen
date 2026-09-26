import json
from types import SimpleNamespace

import pytest

from firebird_vla.telemetry import environment_report, report


def test_phase_reports_preserve_observed_steps_and_atomic_status(tmp_path, capsys):
    report(tmp_path, "training", "Optimizing", step=17, total_steps=100, train_loss=0.24)
    report(tmp_path, "validation", "Evaluating", step=17, total_steps=100)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["train_loss"] == 0.24
    assert [row["step"] for row in records] == [17, 17]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "running"
    assert status["phase"] == "validation"
    assert status["total_steps"] == 100
    assert status["timestamp"]
    # Bad telemetry cannot poison the persisted status or downstream JSON API.
    with pytest.raises(ValueError):
        report(
            tmp_path, "training", "Bad metric", step=18, total_steps=100, train_loss=float("nan")
        )
    assert json.loads((tmp_path / "status.json").read_text()) == status


def test_reload_pending_does_not_claim_run_completion(tmp_path, capsys):
    report(tmp_path, "verifying", "Reload pending", step=100, total_steps=100)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "running"
    assert status["phase"] == "verifying"
    report(tmp_path, "failed", "Reload mismatch", step=100, total_steps=100)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"


def test_environment_records_versions_and_source_without_host_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "never-publish-me")
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.6"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 90000, benchmark=False),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        ),
        cuda=SimpleNamespace(get_device_name=lambda index: "Test GPU"),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    environment = environment_report(fake_torch, "bfloat16")
    assert environment["gpu"] == "Test GPU"
    assert environment["deterministic_algorithms"] is False
    assert len(environment["source_files_sha256"]["train.py"]) == 64
    assert "torch" in environment["packages"]
    assert "never-publish-me" not in json.dumps(environment)
    assert "not guaranteed" in environment["reproducibility_note"]
