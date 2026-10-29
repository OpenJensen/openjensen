"""Declarative target admission and persistence; never allocates compute."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from vla_platform.compute_targets import (
    ComputeTarget,
    ComputeTargets,
    TargetError,
    TargetResources,
)


def runtime(**changes):
    values = dict(
        id="worker",
        label="Registered worker",
        execution="native",
        provider="local",
        device="cpu",
        training_python="python",
        training_root="worker",
        worker_root="worker",
        conversion_python=None,
        conversion_image=None,
        evaluation_python=None,
        evaluation_image=None,
        native_quantization_only=False,
        native_replay_only=False,
        native_distillation_only=False,
        export_only=False,
        training_only=False,
        native_distillation_python=None,
        native_quantization_python=None,
        native_replay_python=None,
        act_export_python=None,
    )
    return SimpleNamespace(**{**values, **changes})


def catalog(**changes):
    entry = runtime(**changes)
    from vla_platform.lifecycle.runtime import RuntimeCatalog

    return RuntimeCatalog.model_construct(runtimes=[entry], sources=[])


def target(**changes):
    return ComputeTarget.model_validate(
        {
            "id": "gpu-server",
            "label": "GPU server",
            "runtime_id": "worker",
            **changes,
        }
    )


def remote(**changes):
    return target(
        **{
            "adapter": "skypilot",
            "infrastructure": "ssh",
            "location": "gpu-pool",
            "enabled": True,
            "resources": {"accelerator": "L4", "gpu_count": 1},
            **changes,
        }
    )


def test_fresh_registry_does_not_enable_or_create_targets(tmp_path):
    registry = ComputeTargets(tmp_path)
    public = registry.list(catalog())
    assert public.targets == [] and public.connection_refs == []
    assert not registry.path.exists()
    assert public.runtime_options[0].id == "worker"
    assert public.runtime_options[-1].id == "portable-policy"
    assert "policy.finetune" in public.runtime_options[0].operations


def test_multiple_targets_round_trip_and_are_private(tmp_path):
    registry = ComputeTargets(tmp_path)
    registry.save(target())
    registry.save(target(id="second-server", label="Second server"))
    saved = ComputeTargets(tmp_path).list(catalog())
    assert [item.definition.id for item in saved.targets] == ["gpu-server", "second-server"]
    assert all(item.status == "disabled" and not item.launchable for item in saved.targets)
    assert registry.path.stat().st_mode & 0o077 == 0
    assert json.loads(registry.path.read_text())["schema_version"] == 1
    registry.remove("gpu-server")
    assert [item.definition.id for item in registry.list(catalog()).targets] == ["second-server"]


def test_snapshot_is_immutable_and_bound_to_configuration(tmp_path):
    registry = ComputeTargets(tmp_path)
    registry.save(target(enabled=True))
    snapshot = registry.snapshot("gpu-server", "policy.finetune", catalog())
    registry.save(target(enabled=True, resources={"memory_gib": 16}))
    updated = registry.snapshot("gpu-server", "policy.finetune", catalog())
    assert snapshot.configuration_sha256 != updated.configuration_sha256
    assert snapshot.definition.resources.memory_gib == 4
    with pytest.raises(ValidationError):
        snapshot.definition.label = "Changed"


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "../unsafe"},
        {"runtime_id": "worker; command"},
        {"label": "Invalid\nlabel"},
        {"adapter": "local", "infrastructure": "gcp"},
        {"location": "unexpected-pool"},
        {"partition": "unexpected"},
        {"operations": ["policy.finetune", "policy.finetune"]},
        {"operations": ["module.install"]},
        {"env": {"TOKEN": "private"}},
        {"command": "untrusted"},
        {"resources": {"gpu_count": 1}},
        {"resources": {"cpus": True}},
        {"resources": {"concurrency": 0}},
    ],
)
def test_target_fields_reject_commands_secrets_and_invalid_limits(changes):
    with pytest.raises(ValidationError):
        target(**changes)


def test_missing_disabled_or_incompatible_target_never_yields_snapshot(tmp_path):
    registry = ComputeTargets(tmp_path)
    with pytest.raises(TargetError, match="not found"):
        registry.snapshot("missing", "policy.finetune", catalog())
    registry.save(target())
    with pytest.raises(TargetError, match="Enable"):
        registry.snapshot("gpu-server", "policy.finetune", catalog())
    registry.save(target(enabled=True, operations=("policy.finetune",)))
    with pytest.raises(TargetError, match="does not support"):
        registry.snapshot("gpu-server", "policy.quantize", catalog())
    assert (
        registry.public(target(enabled=True), catalog(id="another-worker")).status
        == "setup_required"
    )


def test_operation_allowlist_cannot_expand_runtime_capability(tmp_path):
    registry = ComputeTargets(tmp_path)
    entry = target(enabled=True, operations=("policy.quantize",))
    registry.save(entry)
    state = registry.public(entry, catalog(training_only=True))
    assert state.operations == [] and not state.launchable
    assert "does not support" in state.message


def test_remote_specialized_workers_and_cpu_only_recipe_are_unavailable(tmp_path, monkeypatch):
    registry = ComputeTargets(tmp_path)
    monkeypatch.setattr(registry, "sky_executable", lambda: "sky")
    assert not registry.public(remote(), catalog(native_quantization_only=True)).launchable
    assert not registry.public(remote(resources=TargetResources()), catalog()).launchable
    registry.save(remote())
    state = registry.public(registry.get("gpu-server"), catalog())
    assert state.launchable
    assert state.operations == [
        "policy.evaluate",
        "policy.finetune",
        "policy.quantize",
        "policy.run",
    ]


def test_connection_reference_is_resolved_privately_and_fingerprinted(tmp_path):
    registry = ComputeTargets(tmp_path)
    entry = remote(connection_ref="shared-compute")
    registry.connection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "connections": {
                    "shared-compute": {
                        "sky_api_endpoint": "https://compute.example.invalid",
                        "workspace": "research",
                    },
                },
            }
        )
    )
    first = registry.fingerprint(entry)
    connection = registry.connection(entry)
    assert connection["workspace"] == "research"
    assert "compute.example.invalid" not in registry.list(catalog()).model_dump_json()
    document = json.loads(registry.connection_path.read_text())
    document["connections"]["shared-compute"]["workspace"] = "other"
    registry.connection_path.write_text(json.dumps(document))
    assert first != registry.fingerprint(entry)


@pytest.mark.parametrize(
    "config",
    [
        {"sky_api_endpoint": "https://name:password@example.invalid"},
        {"sky_api_endpoint": "https://example.invalid/?token=private"},
        {"sky_api_endpoint": "https://example.invalid", "token": "private"},
        {"sky_api_endpoint": "https://[invalid"},
        {"sky_api_endpoint": "https://example.invalid:99999"},
        {"sky_api_endpoint": "https://example.invalid", "workspace": "bad;command"},
    ],
)
def test_connection_references_cannot_embed_credentials_or_commands(tmp_path, config):
    registry = ComputeTargets(tmp_path)
    registry.connection_path.write_text(
        json.dumps({"schema_version": 1, "connections": {"compute": config}})
    )
    with pytest.raises(TargetError):
        registry.connection(remote(connection_ref="compute"))


def test_corrupt_or_symlink_registry_is_not_silently_replaced(tmp_path):
    registry = ComputeTargets(tmp_path)
    registry.path.write_text("not JSON")
    with pytest.raises(TargetError):
        registry.save(target())
    assert registry.path.read_text() == "not JSON"
    registry.path.unlink()
    saved = tmp_path / "original.json"
    saved.write_text("private")
    registry.path.symlink_to(saved)
    with pytest.raises(TargetError):
        registry.save(target())
    assert saved.read_text() == "private"


def test_check_does_not_run_work_and_settings_changes_revoke_check(tmp_path):
    registry = ComputeTargets(tmp_path)
    registry.save(target(enabled=True))
    checked = asyncio.run(registry.check("gpu-server", catalog()))
    assert checked.status == "configured" and checked.checked_at
    snapshot = registry.snapshot("gpu-server", "policy.finetune", catalog(), require_checked=True)
    assert snapshot.configuration_sha256 == checked.sha256
    registry.save(target(enabled=True, label="Renamed server"))
    assert registry.public(registry.get("gpu-server"), catalog()).checked_at is None
    with pytest.raises(TargetError, match="Check this compute"):
        registry.snapshot("gpu-server", "policy.finetune", catalog(), require_checked=True)


@pytest.mark.parametrize(
    "value,enabled",
    [
        (["GCP", "SSH"], True),
        ({"enabled_clouds": ["gcp"]}, True),
        ({"gcp": {"enabled": True}}, True),
        ({"gcp": {"enabled": False}}, False),
        ({"message": "gcp enabled"}, False),
        ([], False),
    ],
)
def test_provider_check_requires_explicit_machine_readable_access(value, enabled):
    assert ComputeTargets._enabled_infrastructure(value, "gcp") is enabled


def test_local_cuda_recipe_requires_explicit_gpu_permission(tmp_path):
    registry = ComputeTargets(tmp_path)
    cpu_only = target(enabled=True)
    state = registry.public(cpu_only, catalog(device="cuda"))
    assert not state.launchable and "requires one GPU" in state.message
    with_gpu = target(enabled=True, resources={"accelerator": "L4", "gpu_count": 1})
    assert registry.public(with_gpu, catalog(device="cuda")).launchable


@pytest.mark.parametrize(
    "record",
    [
        {"status": "queued"},
        {"status": "running"},
        {"status": "cancelled", "cancellation": {"cleanup": "pending", "finished_at": None}},
        {"status": "cancelled", "cancellation": {"cleanup": "failed", "finished_at": "saved"}},
    ],
)
def test_active_target_edits_and_removal_preserve_configuration(tmp_path, record):
    from fastapi import HTTPException
    from vla_platform import compute_targets_api as api

    registry = ComputeTargets(tmp_path)
    registry.save(target(enabled=True))
    rows = [{**record, "execution": {"target_id": "gpu-server"}}]

    class Connection:
        async def execute(self, statement):
            return SimpleNamespace(scalars=lambda: rows)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    execution = SimpleNamespace(
        settings=SimpleNamespace(data_dir=tmp_path),
        lifecycle=SimpleNamespace(catalog=catalog()),
        storage=SimpleNamespace(engine=SimpleNamespace(connect=Connection)),
        lock=asyncio.Lock(),
        target_slots={"gpu-server": object()},
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(execution=execution)))

    async def attempt():
        for action in (
            api.save("gpu-server", target(enabled=True, label="Changed"), request),
            api.remove("gpu-server", request),
        ):
            with pytest.raises(HTTPException) as error:
                await action
            assert error.value.status_code == 409
        assert registry.get("gpu-server").label == "GPU server"
        rows.clear()
        await api.save("gpu-server", target(enabled=True, label="Changed"), request)
        assert "gpu-server" not in execution.target_slots
        assert registry.get("gpu-server").label == "Changed"

    asyncio.run(attempt())


def test_shipped_remote_recipe_does_not_require_local_ml_environment(tmp_path, monkeypatch):
    from vla_platform.compute.recipes import portable_runtimes
    from vla_platform.lifecycle.runtime import RuntimeCatalog, command

    registry = ComputeTargets(tmp_path)
    monkeypatch.setattr(registry, "sky_executable", lambda: "sky")
    empty = RuntimeCatalog(runtimes=[], sources=[])
    options = registry.list(empty).runtime_options
    assert [item.id for item in options] == ["portable-policy"]
    assert options[0].device == "cuda"
    remote_target = remote(runtime_id="portable-policy")
    registry.save(remote_target)
    snapshot = registry.snapshot(remote_target.id, "policy.finetune", empty)
    assert snapshot.definition.runtime_id == "portable-policy"
    local_target = target(
        enabled=True, runtime_id="portable-policy", resources={"accelerator": "L4", "gpu_count": 1}
    )
    assert not registry.public(local_target, empty).launchable
    with pytest.raises(ValueError, match="SkyPilot"):
        command(
            portable_runtimes()[0],
            tmp_path / "request",
            tmp_path / "result",
            tmp_path,
            "owned",
            True,
            False,
            False,
        )


def test_disabled_targets_still_explain_failed_configuration_checks(tmp_path):
    registry = ComputeTargets(tmp_path)
    entry = target(runtime_id="missing")
    registry.save(entry)
    state = asyncio.run(registry.check(entry.id, catalog()))
    assert state.status == "disabled" and not state.launchable
    assert "Register the selected worker" in state.message
