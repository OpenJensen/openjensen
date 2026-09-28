"""Policy/runtime admission using tiny local markers; no SDK, models or cloud calls."""

import asyncio
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from vla_platform.lifecycle import isaac_runner as runner
from vla_platform.lifecycle import simulation
from vla_platform.lifecycle.contracts import PolicyArtifact, SimulationTarget

CPU_FILES = (
    "firebird_quant/src/firebird_quant/native_consumer.py",
    "firebird_quant/src/firebird_quant/native_package.py",
    "firebird_quant/src/firebird_quant/codec.py",
    "firebird_quant/src/firebird_quant/model.py",
    "firebird_quant/src/firebird_quant/state.py",
    "act_optimizer/src/firebird_act/bundle.py",
    "act_optimizer/src/firebird_act/probe.py",
    "act_optimizer/src/firebird_act/control_schema.py",
    "skypilot/remote/policy_cpu_setup.sh",
    "skypilot/remote/policy_cpu_run.sh",
    "skypilot/remote/policy-cpu.requirements.txt",
)
CPU_CLAIMS = {
    "policy_runtime": "packed-act-cpu",
    "policy_device": "cpu",
    "model_format": "firebird_quant",
}
CUDA_CLAIMS = {
    "policy_runtime": "lerobot-cuda",
    "policy_device": "cuda",
    "model_format": "safetensors",
}


@pytest.fixture
def profile(tmp_path):
    root = tmp_path / "runner"
    sky = root / "workers/skypilot"
    sky.mkdir(parents=True)
    (root / "workers/isaac_sim").mkdir()
    for name in (
        "rollout_launch.py",
        "rollout_sdk.py",
        "sky.sh",
        "launch-rollout.sh",
        "config.yaml",
        "rollout.local.yaml",
    ):
        (sky / name).write_text("# inert source marker\n")
    key = tmp_path / "credential.json"
    key.write_text("not a credential; disposable fixture")
    task = sky / "rollout.local.yaml"
    return runner.SimulationProfile(
        "cup-fixture",
        "Disposable cup profile",
        root,
        task,
        "fixture-project",
        key,
        "gs://fixture-results/experiments",
        runner._sha(task),
        True,
    )


@pytest.fixture
def packed_profile(profile):
    for name in CPU_FILES:
        path = profile.runner_root / "workers" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# inert CPU dependency marker\n")
    return replace(profile, policy_runtime="packed-act-cpu")


@pytest.fixture
def checkpoint():
    return {
        "policy_type": "act",
        "model_id": "sha256:" + "a" * 64,
        "camera_key": "observation.images.front",
        "width": 64,
        "height": 64,
        "state_dim": 6,
        "action_dim": 6,
        "chunk_size": 8,
        "action_steps": 3,
    }


def profile_record(profile):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(profile).items()
    }


def load_profile(tmp_path, record):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [record]}))
    return runner.load_profiles(path)


def artifact(*, format="inference_export", metadata=None, path="jobs/source/artifact"):
    return PolicyArtifact(
        id="source:operation",
        project_id="project-a",
        job_id="source",
        label="Inert policy fixture",
        format=format,
        path=path,
        manifest_sha256="b" * 64,
        file_bytes=0,
        metadata={"architecture": "act", **(metadata or {})},
    )


def test_omitted_runtime_retains_legacy_cuda_identity_and_local_path(profile, tmp_path):
    record = profile_record(profile)
    record.pop("policy_runtime")
    (legacy,) = load_profile(tmp_path, record)
    (explicit,) = load_profile(tmp_path, record | {"policy_runtime": "lerobot-cuda"})
    assert legacy == explicit == profile
    assert legacy.identity_hash() == explicit.identity_hash()
    assert legacy.policy_pythonpath == str(profile.runner_root / "workers/isaac_sim")
    public = legacy.public()
    assert public["architectures"] == ["act", "smolvla"]
    assert public["accelerators"] == ["L4", "H100"]
    assert public["policy_formats"] == ["safetensors"]
    assert public["policy_device"] == "cuda"


def test_packed_profile_has_explicit_cpu_capability_and_only_fixed_local_paths(
    packed_profile, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-unrelated-secret")
    public = packed_profile.public()
    assert public["policy_runtime"] == "packed-act-cpu"
    assert public["policy_device"] == "cpu"
    assert public["architectures"] == ["act"]
    assert public["accelerators"] == ["L4"]
    assert public["policy_formats"] == ["firebird_quant"]
    assert public["task_success"] is None and public["experimental"] is True
    assert str(packed_profile.credential_file) not in json.dumps(public)
    env = packed_profile.environment()
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(packed_profile.runner_root / "workers" / name)
        for name in ("isaac_sim", "firebird_quant/src", "act_optimizer/src")
    ]
    assert "OPENROUTER_API_KEY" not in env


@pytest.mark.parametrize("mode", [None, True, [], {}, "cuda", "packed-act-cuda"])
def test_profile_runtime_must_be_a_known_string(profile, tmp_path, mode):
    with pytest.raises(ValueError, match="policy runtime"):
        replace(profile, policy_runtime=mode)
    with pytest.raises(ValueError, match="policy runtime"):
        load_profile(tmp_path, profile_record(profile) | {"policy_runtime": mode})


@pytest.mark.parametrize("name", CPU_FILES)
def test_packed_identity_binds_every_cpu_dependency(packed_profile, name):
    before = packed_profile.identity_hash()
    path = packed_profile.runner_root / "workers" / name
    path.write_text("# changed fixed CPU dependency\n")
    assert packed_profile.identity_hash() != before


@pytest.mark.parametrize("name", CPU_FILES)
def test_packed_profile_requires_each_fixed_cpu_dependency(packed_profile, name):
    (packed_profile.runner_root / "workers" / name).unlink()
    with pytest.raises(ValueError, match="missing"):
        packed_profile.identity_hash()


def test_packed_dependency_symlink_is_not_an_admitted_source(packed_profile, tmp_path):
    path = packed_profile.runner_root / "workers" / CPU_FILES[0]
    outside = tmp_path / "borrowed.py"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="nonregular"):
        packed_profile.identity_hash()


def test_packed_dependencies_do_not_redefine_legacy_cuda_source_scope(packed_profile):
    legacy = replace(packed_profile, policy_runtime="lerobot-cuda")
    before = legacy.identity_hash()
    path = packed_profile.runner_root / "workers/firebird_quant/src/firebird_quant/codec.py"
    path.write_text("# changed packed-only decoder\n")
    assert legacy.identity_hash() == before
    assert packed_profile.identity_hash() != before


@pytest.mark.parametrize("family", ["act", "smolvla"])
@pytest.mark.parametrize("explicit_format", [False, True])
def test_legacy_and_explicit_float_checkpoints_remain_cuda_admissible(
    profile, checkpoint, family, explicit_format
):
    value = checkpoint | {"policy_type": family}
    if explicit_format:
        value["model_format"] = "safetensors"
    assert runner.require_policy_runtime(profile, value) == "lerobot-cuda"
    accepted = runner.admit(profile, value)
    assert accepted["model_id"] == checkpoint["model_id"]
    assert accepted["task_success"] is None and accepted["calibration_verified"] is False


def test_packed_act_admission_reports_cpu_without_quality_claim(packed_profile, checkpoint):
    value = checkpoint | {"model_format": "firebird_quant"}
    accepted = runner.admit(packed_profile, value)
    assert {key: accepted[key] for key in CPU_CLAIMS} == CPU_CLAIMS
    assert accepted["model_id"] == checkpoint["model_id"]
    assert accepted["profile_sha256"] == packed_profile.identity_hash()
    assert accepted["task_success"] is None and accepted["calibration_verified"] is False


@pytest.mark.parametrize(
    "mode,family,encoding",
    [
        ("lerobot-cuda", "act", "firebird_quant"),
        ("packed-act-cpu", "act", "safetensors"),
        ("packed-act-cpu", "smolvla", "firebird_quant"),
        ("lerobot-cuda", "openvla", "safetensors"),
        ("lerobot-cuda", "act", None),
        ("lerobot-cuda", "act", []),
        ("packed-act-cpu", "act", {}),
        ("packed-act-cpu", "act", "gguf"),
    ],
)
def test_runtime_gate_refuses_unknown_family_and_wrong_or_malformed_encoding(
    packed_profile, checkpoint, mode, family, encoding
):
    profile = replace(packed_profile, policy_runtime=mode)
    with pytest.raises(ValueError):
        runner.require_policy_runtime(
            profile, checkpoint | {"policy_type": family, "model_format": encoding}
        )


@pytest.mark.parametrize("value", [None, [], "checkpoint", 1])
def test_runtime_gate_refuses_non_object_checkpoint(profile, value):
    with pytest.raises(ValueError):
        runner.require_policy_runtime(profile, value)


@pytest.mark.parametrize("family", [None, True, [], {}])
def test_runtime_gate_refuses_malformed_policy_family(profile, checkpoint, family):
    with pytest.raises(ValueError):
        runner.require_policy_runtime(profile, checkpoint | {"policy_type": family})


@pytest.mark.parametrize("mode", [None, [], {}, "unknown"])
def test_runtime_gate_does_not_treat_invalid_profile_as_cuda(checkpoint, mode):
    with pytest.raises(ValueError):
        runner.require_policy_runtime(SimpleNamespace(policy_runtime=mode), checkpoint)


def test_historical_target_defaults_and_explicit_packed_target():
    record = {
        "profile_id": "cup-fixture",
        "profile_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
    }
    legacy = SimulationTarget.model_validate(record)
    assert legacy.policy_runtime == "lerobot-cuda"
    assert legacy.accelerators == ["L4", "H100"]
    packed = SimulationTarget.model_validate(
        record | {"policy_runtime": "packed-act-cpu", "accelerators": ["L4"]}
    )
    assert packed.policy_runtime == "packed-act-cpu" and packed.accelerators == ["L4"]
    assert SimulationTarget.model_validate(packed.model_dump()) == packed


@pytest.mark.parametrize(
    "mode,resources",
    [
        ("lerobot-cuda", ["L4"]),
        ("lerobot-cuda", ["H100", "L4"]),
        ("packed-act-cpu", ["L4", "H100"]),
        ("packed-act-cpu", []),
        ("packed-act-cpu", ["L4", "L4"]),
        ("packed-act-cpu", ["H100"]),
    ],
)
def test_target_runtime_cannot_disagree_with_accepted_resources(mode, resources):
    with pytest.raises(ValidationError, match="resources"):
        SimulationTarget(
            profile_id="cup-fixture",
            profile_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            policy_runtime=mode,
            accelerators=resources,
        )


@pytest.mark.parametrize(
    "format,metadata,expected",
    [
        ("native_checkpoint", {}, "safetensors"),
        ("inference_export", {"format": "safetensors"}, "safetensors"),
        ("native_quantized", {}, "firebird_quant"),
        (
            "inference_export",
            {"format": "firebird_quant", "checkpoint": {"model_format": "firebird_quant"}},
            "firebird_quant",
        ),
    ],
)
def test_artifact_encoding_recognizes_legacy_and_packed_exports(format, metadata, expected):
    assert simulation.artifact_model_format(artifact(format=format, metadata=metadata)) == expected


@pytest.mark.parametrize("claim", [None, True, [], {}, "gguf"])
def test_malformed_checkpoint_encoding_claim_is_rejected(claim):
    value = artifact(metadata={"checkpoint": {"model_format": claim}})
    with pytest.raises(ValueError, match="encoding"):
        simulation.artifact_model_format(value)


@pytest.mark.parametrize(
    "format,metadata",
    [
        ("native_quantized", {"format": "safetensors"}),
        ("native_quantized", {"checkpoint": {"model_format": "safetensors"}}),
        (
            "inference_export",
            {"format": "safetensors", "checkpoint": {"model_format": "firebird_quant"}},
        ),
    ],
)
def test_contradictory_artifact_encodings_do_not_select_a_runtime(format, metadata):
    with pytest.raises(ValueError, match="contradict"):
        simulation.artifact_model_format(artifact(format=format, metadata=metadata))


@pytest.mark.parametrize(
    "mode,format,metadata,allowed",
    [
        ("lerobot-cuda", "native_checkpoint", {}, True),
        ("lerobot-cuda", "inference_export", {"architecture": "smolvla"}, True),
        ("lerobot-cuda", "native_quantized", {}, False),
        ("packed-act-cpu", "native_quantized", {}, True),
        ("packed-act-cpu", "inference_export", {"format": "firebird_quant"}, True),
        ("packed-act-cpu", "inference_export", {}, False),
        ("packed-act-cpu", "native_quantized", {"architecture": "smolvla"}, False),
    ],
)
def test_source_admission_matches_policy_encoding_before_bundle_read(
    tmp_path, packed_profile, mode, format, metadata, allowed
):
    lifecycle = SimpleNamespace(settings=SimpleNamespace(data_dir=tmp_path / "workspace"))
    value = artifact(format=format, metadata=metadata)
    profile = replace(packed_profile, policy_runtime=mode)
    if allowed:
        assert simulation.source_directory(lifecycle, value, profile) == (
            lifecycle.settings.data_dir / value.path
        )
    else:
        with pytest.raises(ValueError, match="compatible"):
            simulation.source_directory(lifecycle, value, profile)


def test_packed_admission_preserves_owned_path_and_remote_package_guards(tmp_path, packed_profile):
    data = tmp_path / "workspace"
    lifecycle = SimpleNamespace(settings=SimpleNamespace(data_dir=data))
    with pytest.raises(ValueError, match="application job directory"):
        simulation.source_directory(
            lifecycle, artifact(format="native_quantized", path="../outside"), packed_profile
        )
    value = artifact(format="native_quantized")
    directory = data / value.path
    directory.mkdir(parents=True)
    (directory / "remote.json").write_text("{}")
    with pytest.raises(ValueError, match="remote checkpoint"):
        simulation.source_directory(lifecycle, value, packed_profile)


@pytest.mark.parametrize("mode", ["lerobot-cuda", "packed-act-cpu"])
def test_actual_run_stops_after_inspection_on_encoding_mismatch(
    tmp_path, packed_profile, checkpoint, monkeypatch, mode
):
    profile = replace(packed_profile, policy_runtime=mode)
    opposite = "firebird_quant" if mode == "lerobot-cuda" else "safetensors"
    inspected = checkpoint | {"model_format": opposite}
    command = AsyncMock(return_value=(0, json.dumps(inspected).encode()))
    event = AsyncMock()
    monkeypatch.setattr(runner, "_command", command)
    directory = tmp_path / "owned-job"
    with pytest.raises(ValueError, match="encoding"):
        asyncio.run(runner.run(profile, tmp_path / "model", directory, event, 30))
    command.assert_awaited_once()
    assert command.await_args.args[1][:2] == [str(profile.python), "-c"]
    assert command.await_args.kwargs["capture"] is True
    event.assert_not_awaited()
    assert not (directory / "request.json").exists()
    assert not (directory / "receipts").exists()


def test_changed_packed_source_refuses_run_before_any_subprocess(
    tmp_path, packed_profile, monkeypatch
):
    accepted = packed_profile.identity_hash()
    path = packed_profile.runner_root / "workers" / CPU_FILES[0]
    path.write_text("# decoder changed after request acceptance\n")
    command = AsyncMock(side_effect=AssertionError("No process may start"))
    monkeypatch.setattr(runner, "_command", command)
    with pytest.raises(ValueError, match="changed after"):
        asyncio.run(
            runner.run(
                packed_profile,
                tmp_path / "model",
                tmp_path / "owned-job",
                AsyncMock(),
                30,
                expected_profile_sha256=accepted,
            )
        )
    command.assert_not_awaited()


def test_packed_source_change_during_inspection_does_not_reach_dispatch(
    tmp_path, packed_profile, checkpoint, monkeypatch
):
    accepted = packed_profile.identity_hash()
    path = packed_profile.runner_root / "workers" / CPU_FILES[0]

    async def inspect_only(*args, **kwargs):
        path.write_text("# changed while local checkpoint inspection returned\n")
        return 0, json.dumps(checkpoint | {"model_format": "firebird_quant"}).encode()

    command = AsyncMock(side_effect=inspect_only)
    event = AsyncMock()
    monkeypatch.setattr(runner, "_command", command)
    directory = tmp_path / "owned-job"
    with pytest.raises(ValueError, match="changed during admission"):
        asyncio.run(
            runner.run(
                packed_profile,
                tmp_path / "model",
                directory,
                event,
                30,
                expected_profile_sha256=accepted,
            )
        )
    command.assert_awaited_once()
    event.assert_not_awaited()
    assert not (directory / "request.json").exists()
    assert not (directory / "receipts").exists()


def context_record(profile):
    return {
        "schema_version": 1,
        "group_name": "isaac-act-test-bbbbbbbb",
        "rollout_id": "b" * 32,
        "model_id": "sha256:" + "a" * 64,
        "results_prefix": profile.results_uri + "/" + "b" * 32,
    }


def saved_context(tmp_path, value):
    receipts = tmp_path / "owned-context/receipts"
    receipts.mkdir(parents=True)
    (receipts / "launch-context.json").write_text(json.dumps(value))
    return receipts.parent


@pytest.mark.parametrize("explicit", [False, True])
def test_cuda_context_retains_legacy_absence_and_accepts_complete_claims(
    tmp_path, profile, explicit
):
    value = context_record(profile) | (CUDA_CLAIMS if explicit else {})
    assert runner._context(profile, saved_context(tmp_path, value)) == value


def test_packed_context_requires_and_retains_complete_runtime_claims(tmp_path, packed_profile):
    value = context_record(packed_profile) | CPU_CLAIMS
    assert runner._context(packed_profile, saved_context(tmp_path, value)) == value


@pytest.mark.parametrize("mode", ["lerobot-cuda", "packed-act-cpu"])
@pytest.mark.parametrize("missing", list(CPU_CLAIMS))
def test_context_partial_runtime_claims_fail_closed(tmp_path, packed_profile, mode, missing):
    profile = replace(packed_profile, policy_runtime=mode)
    value = context_record(profile) | (CPU_CLAIMS if mode == "packed-act-cpu" else CUDA_CLAIMS)
    value.pop(missing)
    with pytest.raises(ValueError):
        runner._context(profile, saved_context(tmp_path, value))


@pytest.mark.parametrize(
    "changes",
    [
        {},
        CUDA_CLAIMS,
        CPU_CLAIMS | {"policy_device": "cuda"},
        CPU_CLAIMS | {"model_format": "safetensors"},
        CPU_CLAIMS | {"policy_runtime": []},
        CPU_CLAIMS | {"policy_device": None},
        CPU_CLAIMS | {"model_format": {}},
    ],
)
def test_packed_context_rejects_omitted_wrong_and_malformed_claims(
    tmp_path, packed_profile, changes
):
    with pytest.raises(ValueError):
        runner._context(
            packed_profile, saved_context(tmp_path, context_record(packed_profile) | changes)
        )


def test_cuda_context_cannot_recover_a_packed_runtime(tmp_path, profile):
    with pytest.raises(ValueError):
        runner._context(profile, saved_context(tmp_path, context_record(profile) | CPU_CLAIMS))


@pytest.mark.parametrize("mode", ["lerobot-cuda", "packed-act-cpu"])
@pytest.mark.parametrize("archive", [False, True])
def test_resolver_keeps_owned_cwd_and_fixed_runtime_source_paths(
    tmp_path, packed_profile, monkeypatch, mode, archive
):
    profile = replace(packed_profile, policy_runtime=mode)
    process = SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0))
    spawn = AsyncMock(return_value=process)
    stop = AsyncMock()
    lifecycle = SimpleNamespace(act_spawn_owned=spawn, act_stop_owned=stop)
    source, destination = tmp_path / "input", tmp_path / "output"
    receipt = tmp_path / "receipt.json"
    expected = (destination, {"fixture": "no model execution"})
    monkeypatch.setattr(simulation, "check_receipt", lambda output, record: expected)
    assert (
        asyncio.run(
            simulation.resolve(lifecycle, profile, source, destination, receipt, archive=archive)
        )
        == expected
    )
    spawn.assert_awaited_once()
    args, kwargs = spawn.await_args
    assert args[:3] == (str(profile.python), "-m", "sim_worker.rollout.checkpoint_package")
    assert args[3:9] == (
        "--source",
        str(source),
        "--output-dir",
        str(destination),
        "--json-output",
        str(receipt),
    )
    assert args[9:] == (("--archive",) if archive else ())
    assert kwargs["cwd"] == profile.runner_root / "workers/isaac_sim"
    assert kwargs["env"]["PYTHONPATH"] == profile.policy_pythonpath
    assert kwargs.get("start_new_session", False) is (os.name != "nt")
    stop.assert_not_awaited()
