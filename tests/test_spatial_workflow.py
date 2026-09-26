"""Spatial software gates with synthetic workers; no GPU/model quality claims."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_lifecycle import project, submit, wait
from vla_platform.api import create_app
from vla_platform.lifecycle.contracts import Evaluation, PolicyRequest
from vla_platform.lifecycle.runtime import Runtime, Source, command
from vla_platform.settings import Settings


@pytest.fixture
def spatial(tmp_path):
    config = tmp_path / "runtimes.json"
    config.write_text(
        json.dumps(
            {
                "runtimes": [
                    {
                        "id": "fixture",
                        "label": "Spatial fixture",
                        "python": sys.executable,
                        "evaluation_python": sys.executable,
                        "device": "cuda",
                        "worker_root": str(Path(__file__).parent / "fixtures/spatial_worker"),
                        "vendor": str(tmp_path),
                        "build": str(tmp_path),
                        "simulator_lane": str(tmp_path),
                    }
                ],
                "sources": [
                    {
                        "id": "source",
                        "label": "Synthetic Spatial source",
                        "path": str(tmp_path / "source"),
                        "sha256": "0" * 64,
                        "task": "libero_spatial",
                        "evaluation_bundle": str(tmp_path / "assets"),
                        "evaluation_bundle_sha256": "1" * 64,
                    }
                ],
            }
        )
    )
    return Settings(data_dir=tmp_path / "workspace", runtime_config=config)


def evaluation(**changes):
    return {
        "mode": "libero",
        "suite": "libero_spatial",
        "task_ids": [0, 2],
        "initial_states": [0, 1],
        "final_states": [2, 3],
        "parity_limits": {"profile": "synthetic-exact-only", "max_rmse": 0, "max_abs_error": 0},
        **changes,
    }


def scenario(settings, changes):
    catalog = json.loads(settings.runtime_config.read_text())
    catalog["runtimes"][0]["label"] = "scenario:" + json.dumps(changes)
    settings.runtime_config.write_text(json.dumps(catalog))


def run(client, **fields):
    return wait(
        client, submit(client, project(client), evaluation=evaluation(), limits={}, **fields)
    )


def test_spatial_defaults_and_explicit_tolerance_contract():
    spatial = Evaluation(suite="libero_spatial", mode="libero")
    assert spatial.task_ids == list(range(10)) and spatial.steps == 280
    assert Evaluation().suite == "libero_object" and Evaluation().steps == 500
    for fields in (
        {"suite": "libero_object", "task_ids": [0]},
        {"suite": "libero_spatial", "task_ids": [1, 1]},
        {"suite": "libero_spatial", "task_ids": [10]},
        {"suite": "libero_spatial", "steps": 281},
    ):
        with pytest.raises(ValidationError):
            Evaluation(mode="libero", **fields)
    with pytest.raises(ValidationError, match="explicit parity"):
        PolicyRequest(
            operation="policy.workflow",
            runtime_id="fixture",
            source_id="source",
            evaluation={"suite": "libero_spatial", "mode": "libero"},
        )


def test_spatial_source_requires_pinned_asset_pair():
    fields = {"id": "source", "label": "fixture", "path": "/model", "sha256": "0" * 64}
    with pytest.raises(ValidationError, match="pinned local"):
        Source(**fields, task="libero_spatial")
    with pytest.raises(ValidationError, match="supplied together"):
        Source(**fields, evaluation_bundle="/assets")


def test_evaluation_environment_uses_only_the_fixed_worker_module(tmp_path):
    runtime = Runtime(
        id="fixture",
        label="fixture",
        python="conversion-python",
        worker_root="/worker",
        evaluation_python="evaluation-python",
        evaluation_image="eval-image",
        vendor="/vendor",
        build="/build",
    )
    argv, cwd, _ = command(
        runtime, tmp_path / "job", tmp_path / "result", tmp_path, "fixture", evaluation=True
    )
    assert argv[-6:] == [
        "evaluation-python",
        "eval-image",
        "-m",
        "policykit.application",
        str(tmp_path / "job"),
        str(tmp_path / "result"),
    ]
    assert cwd is None


def test_spatial_uses_native_and_cpp_controls_and_exact_episode_counts(spatial):
    with TestClient(create_app(spatial)) as client:
        job = run(client)
        assert job["status"] == "succeeded", job
        result = job["result"]
        assert result["decision"] == "validated", result
        reports = {report["stage"]: report for report in result["reports"]}
        assert reports["native-reference"]["runtime"] != reports["baseline"]["runtime"]
        assert reports["native-reference"]["complete_episodes"] == 4
        assert reports["final-control"]["complete_episodes"] == 4
        assert reports["final-reference"]["backend"] == "native-bf16"
        assert reports["final-control"]["backend"] == "cpp"
        assert reports["float-parity"]["passed"] is True
        assert result["selected_artifact_id"].endswith(":package-and-reload")
        package = result["artifacts"][-1]
        assert package["metadata"]["precision"]["language"] == "Q8_0"
        assert all(artifact["metadata"]["fixture_only"] for artifact in result["artifacts"])


@pytest.mark.parametrize(
    "fault",
    [
        {"parity": True},
        {"fields": {"fixture_actions": []}},
        {"fields": {"fixture_actions_deterministic": False}},
        {"fields": {"protocol_sha256": "0" * 64}},
        {"fields": {"model_sha256": "0" * 64}},
        {"fields": {"memory_coverage": {"complete": False}}},
        {"duplicate_episode": True},
    ],
)
def test_bad_floating_control_blocks_compression(spatial, fault):
    scenario(spatial, {"baseline": fault})
    with TestClient(create_app(spatial)) as client:
        job = run(client)
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == "no_feasible_candidate"
        assert not any(
            report["stage"].startswith("quantize") for report in job["result"]["reports"]
        )


@pytest.mark.parametrize(
    "stage,fault",
    [
        ("final-reference", {"fields": {"runtime": {"family": "changed"}}}),
        (
            "final-control",
            {
                "fields": {
                    "target_identity": {
                        "gpu_uuid": "different",
                        "name": "synthetic GPU",
                        "driver_version": "fixture",
                    }
                }
            },
        ),
        ("final-evaluation", {"failures": 1}),
        ("final-evaluation", {"duplicate_episode": True}),
        ("package-and-reload", {"fields": {"model_sha256": "0" * 64}}),
        ("package-and-reload", {"fields": {"protocol_sha256": "0" * 64}}),
        ("package-and-reload", {"fields": {"fresh_reload_verified": False}}),
        ("package-and-reload", {"package_model": True}),
        ("package-and-reload", {"fields": {"memory_coverage": {"complete": False}}}),
    ],
)
def test_final_failure_does_not_select_replacement_or_register_package(spatial, stage, fault):
    scenario(spatial, {stage: fault})
    with TestClient(create_app(spatial)) as client:
        job = run(client, candidates=[{"language": "Q8_0"}, {"language": "Q4_0"}])
        assert job["status"] == "succeeded", job
        assert job["result"]["decision"] == "no_feasible_candidate", job
        assert job["result"]["selected_artifact_id"] is None
        assert not any(
            artifact["format"] == "deployment_package" for artifact in job["result"]["artifacts"]
        )
        assert (
            sum(report["stage"] == "final-evaluation" for report in job["result"]["reports"]) == 1
        )


def test_quality_drop_is_checked_against_native_even_when_cpp_float_is_weaker(spatial):
    scenario(spatial, {"baseline": {"failures": 2}, "evaluate-0": {"failures": 1}})
    with TestClient(create_app(spatial)) as client:
        job = wait(
            client,
            submit(
                client, project(client), evaluation=evaluation(), limits={"min_success_rate": 0}
            ),
        )
        assert job["result"]["decision"] == "no_feasible_candidate"
        assert not any(report["stage"] == "final-reference" for report in job["result"]["reports"])


def test_cpp_control_quality_cannot_be_borrowed_from_weaker_native_reference(spatial):
    scenario(
        spatial,
        {
            "native-reference": {"failures": 2},
            "baseline": {"fields": {"peak_device_mib": 20}},
            "evaluate-0": {"failures": 1},
        },
    )
    with TestClient(create_app(spatial)) as client:
        job = wait(
            client,
            submit(
                client,
                project(client),
                evaluation=evaluation(),
                limits={"min_success_rate": 0, "max_peak_device_mib": 15},
            ),
        )
        assert job["result"]["decision"] == "no_feasible_candidate"
        assert not any(report["stage"] == "final-reference" for report in job["result"]["reports"])


def test_reference_resource_budgets_do_not_disqualify_fitting_candidate(spatial):
    scenario(
        spatial,
        {
            "native-reference": {"fields": {"peak_device_mib": 30, "p95_ms": 30}},
            "baseline": {"fields": {"peak_device_mib": 30, "p95_ms": 30}},
            "final-reference": {"fields": {"peak_device_mib": 30, "p95_ms": 30}},
            "final-control": {"fields": {"peak_device_mib": 30, "p95_ms": 30}},
        },
    )
    with TestClient(create_app(spatial)) as client:
        job = wait(
            client,
            submit(
                client,
                project(client),
                evaluation=evaluation(),
                limits={"max_peak_device_mib": 15, "max_p95_ms": 15},
            ),
        )
        assert job["result"]["decision"] == "validated", job


@pytest.mark.parametrize("alter_extra_asset", [False, True])
def test_float_package_allows_only_reference_weight_removal(tmp_path, alter_extra_asset):
    import hashlib
    from types import SimpleNamespace

    from vla_platform.lifecycle.spatial import validate_package_receipt

    def encoded(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    source, package = tmp_path / "source", tmp_path / "package"
    source.mkdir()
    package.mkdir()
    native_hash = digest(b"synthetic native weights")
    assets = {
        "schema_version": 1,
        "reference_weights": True,
        "files": {
            "policy/model.safetensors": native_hash,
            "tokenizer.json": digest(b"pinned tokenizer"),
        },
    }
    source_files = {
        "model.gguf": digest(b"synthetic GGUF"),
        "policy/model.safetensors": native_hash,
        "tokenizer.json": digest(b"pinned tokenizer"),
        "spatial-assets.json": digest(encoded(assets)),
    }
    metadata = {"precision": "float", "native_model_sha256": native_hash}
    source_manifest = {"schema_version": 1, "metadata": metadata, "files": source_files}
    (source / "manifest.json").write_bytes(encoded(source_manifest))
    (source / "spatial-assets.json").write_bytes(encoded(assets))
    chosen = SimpleNamespace(
        path="source", metadata=metadata, manifest_sha256=digest(encoded(source_manifest))
    )
    # Emulate the documented worker transformation, then its fresh-process receipt.
    assets["files"].pop("policy/model.safetensors")
    assets["reference_weights"] = False
    files = {
        name: value for name, value in source_files.items() if name != "policy/model.safetensors"
    }
    files["spatial-assets.json"] = digest(encoded(assets))
    if alter_extra_asset:
        files["tokenizer.json"] = digest(b"unapproved tokenizer change")
    tested_manifest = {"schema_version": 1, "metadata": metadata, "files": files}
    receipt = {"manifest_sha256": digest(encoded(tested_manifest)), "files": files}
    (package / "tested-payload.json").write_bytes(encoded(receipt))
    package_manifest = {
        **tested_manifest,
        "files": {**files, "tested-payload.json": digest(encoded(receipt))},
    }
    (package / "manifest.json").write_bytes(encoded(package_manifest))
    packaged = SimpleNamespace(path="package", metadata={**metadata, "deployment_verified": True})
    lifecycle = SimpleNamespace(settings=SimpleNamespace(data_dir=tmp_path))
    report = {"artifact_manifest_sha256": receipt["manifest_sha256"]}
    if alter_extra_asset:
        with pytest.raises(ValueError, match="exact tested candidate"):
            validate_package_receipt(lifecycle, packaged, chosen, report)
    else:
        validate_package_receipt(lifecycle, packaged, chosen, report)


@pytest.mark.parametrize("changes", [{"mode": "engine"}, {"steps": 1}, {"steps": 279}])
def test_unsupported_spatial_protocol_is_rejected_before_job_creation(spatial, changes):
    with TestClient(create_app(spatial)) as client:
        pid = project(client)
        response = client.post(
            f"/api/v1/projects/{pid}/policy-jobs",
            json={
                "operation": "policy.workflow",
                "runtime_id": "fixture",
                "source_id": "source",
                "evaluation": evaluation(**changes),
            },
        )
        assert response.status_code == 422, response.text
        assert client.get(f"/api/v1/projects/{pid}/jobs").json() == []


@pytest.mark.parametrize("steps", [0, 281])
def test_report_episode_steps_must_be_executed_within_requested_horizon(spatial, steps):
    scenario(spatial, {"baseline": {"episode_steps": steps}})
    with TestClient(create_app(spatial)) as client:
        job = run(client)
        assert job["result"]["decision"] == "no_feasible_candidate", job
        assert not any(x["stage"].startswith("quantize") for x in job["result"]["reports"])


@pytest.mark.parametrize("fault", ["wrong_fixture", "missing_fixture", "wrong_asset_inventory"])
def test_shared_bad_parity_inputs_cannot_qualify_even_when_controls_agree(spatial, fault):
    scenario(spatial, {stage: {fault: True} for stage in ("native-reference", "baseline")})
    with TestClient(create_app(spatial)) as client:
        job = run(client)
        assert job["result"]["decision"] == "no_feasible_candidate", job
        assert not any(x["stage"].startswith("quantize") for x in job["result"]["reports"])


@pytest.mark.parametrize("stage", ["evaluate-0", "final-evaluation", "package-and-reload"])
def test_every_candidate_and_package_must_use_all_pinned_fixtures(spatial, stage):
    # The float control exceeds the limit, so it cannot hide a rejected packed candidate.
    scenario(
        spatial, {"baseline": {"fields": {"peak_device_mib": 30}}, stage: {"missing_fixture": True}}
    )
    with TestClient(create_app(spatial)) as client:
        job = wait(
            client,
            submit(
                client, project(client), evaluation=evaluation(), limits={"max_peak_device_mib": 15}
            ),
        )
        assert job["result"]["decision"] == "no_feasible_candidate", job
        assert job["result"]["selected_artifact_id"] is None
        assert not any(x["format"] == "deployment_package" for x in job["result"]["artifacts"])


def test_export_cannot_append_untested_runtime_assets(spatial):
    scenario(spatial, {"package-and-reload": {"extra_package_asset": True}})
    with TestClient(create_app(spatial)) as client:
        job = run(client)
        assert job["result"]["decision"] == "no_feasible_candidate", job
        assert job["result"]["selected_artifact_id"] is None
        assert not any(x["format"] == "deployment_package" for x in job["result"]["artifacts"])
