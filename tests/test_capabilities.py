import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform.api import create_app
from vla_platform.capabilities import registry
from vla_platform.contracts import Capability, CapabilityTarget
from vla_platform.settings import Settings


def by_operation(items):
    return {item.operation: item for item in items}


def target(item, operating_system, device="cpu"):
    return next(
        item
        for item in item.targets
        if item.operating_system == operating_system and item.device == device
    )


def test_planned_policy_operations_never_advertise_runnable():
    items = by_operation(registry(local_root=Path("configured")))
    assert len(items) == 7
    for operation in ["finetune", "distill", "quantize", "evaluate", "run"]:
        item = items[f"policy.{operation}"]
        assert item.status == "planned" and item.implementation == "planned"
        assert not item.runnable and item.backend is None
        assert len(item.targets) == 9
        assert all(row.support == "unsupported" for row in item.targets)
        assert all(
            row.evidence_state == "untested" and not row.evidence_refs for row in item.targets
        )


def test_evidence_is_scoped_to_os_device_and_source():
    items = by_operation(registry())
    local, hub = items["dataset.inspect.local"], items["dataset.inspect"]
    assert target(local, "windows").evidence_state == "fixture"
    assert "API-001.md" in target(local, "windows").evidence_refs[0]
    assert target(hub, "windows").support == "supported"
    assert target(hub, "windows").evidence_state == "live_source"
    assert "INT-001.md" in target(hub, "windows").evidence_refs[0]
    assert "f7f9a41" in target(hub, "windows").reason
    assert "No interactive-browser" in target(hub, "windows").reason
    assert target(hub, "macos").evidence_state == "live_source"
    assert target(local, "macos").evidence_state == "fixture"
    assert target(local, "linux").support == "supported"
    assert target(local, "linux").evidence_state == "fixture"
    assert "FND-003.md" in target(local, "linux").evidence_refs[0]
    assert "6319c09" in target(local, "linux").reason
    assert "unverified on Linux" in target(local, "linux").reason
    assert target(hub, "linux").support == "untested"
    assert target(hub, "linux").evidence_state == "untested"
    assert target(hub, "linux").evidence_refs == []
    for item in [local, hub]:
        for row in item.targets:
            if row.device != "cpu":
                assert row.support == "unsupported"
                assert row.evidence_state == "untested" and row.evidence_refs == []


@pytest.mark.parametrize("configured", [False, True])
def test_api_uses_registry_and_local_configuration(tmp_path, configured):
    root = tmp_path / "not-created" if configured else None
    with TestClient(create_app(Settings(data_dir=tmp_path / "state", local_root=root))) as client:
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        records = response.json()
        assert records == [item.model_dump() for item in registry(local_root=root)]
        local = by_operation([Capability.model_validate(item) for item in records])[
            "dataset.inspect.local"
        ]
        assert local.runnable is configured
        assert local.status == ("available" if configured else "planned")
        assert local.implementation == "registered"
        # Advertising a configured root must not create it or claim valid dataset inputs.
        if root:
            assert not root.exists()


def test_registration_without_a_run_remains_untested(monkeypatch):
    monkeypatch.setattr("vla_platform.capabilities.platform.system", lambda: "Linux")
    item = by_operation(registry())["dataset.inspect"]
    assert item.implementation == "registered" and item.runnable
    row = target(item, "linux")
    assert row.support == "untested" and row.evidence_state == "untested"
    assert row.evidence_refs == []


def test_unregistered_host_does_not_advertise_runnable(monkeypatch):
    monkeypatch.setattr("vla_platform.capabilities.platform.system", lambda: "UnknownOS")
    items = registry(local_root=Path("configured"))
    assert all(item.status == "planned" and not item.runnable for item in items)
    assert "not a registered target" in items[0].description


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "available"},
        {"runnable": True},
        {"implementation": "registered", "runnable": True, "status": "available"},
    ],
)
def test_planned_or_missing_backend_cannot_be_promoted(changes):
    item = by_operation(registry())["policy.finetune"].model_dump()
    item.update(changes)
    with pytest.raises(ValidationError):
        Capability.model_validate(item)


@pytest.mark.parametrize(
    "changes",
    [
        {"support": "supported"},
        {"evidence_state": "fixture"},
        {"evidence_refs": ["invented-run"]},
        {"support": "supported", "evidence_state": "live_source", "evidence_refs": [" "]},
        {"support": "unsupported", "evidence_state": "fixture", "evidence_refs": ["fixture"]},
        {"device": "unknown-accelerator"},
        {"operating_system": "unknown-os"},
    ],
)
def test_unproven_or_unknown_target_evidence_is_rejected(changes):
    row = {"operating_system": "linux", "device": "cpu", "reason": "No run recorded"}
    row.update(changes)
    with pytest.raises(ValidationError):
        CapabilityTarget.model_validate(row)


def test_duplicate_target_cannot_hide_contradictory_evidence():
    item = by_operation(registry())["dataset.inspect"].model_dump()
    item["targets"].append(item["targets"][0])
    with pytest.raises(ValidationError, match="unique"):
        Capability.model_validate(item)


def test_capability_query_has_no_ml_imports():
    code = (
        "import sys; from vla_platform.capabilities import registry; registry(); "
        "blocked = {'torch', 'lerobot', 'transformers', 'openvla', 'sky'}; "
        "assert not blocked.intersection(name.split('.')[0] for name in sys.modules); "
        "print('Capability query imported no ML or cloud modules.')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=True,
    )
    assert result.stdout.strip() == "Capability query imported no ML or cloud modules."
