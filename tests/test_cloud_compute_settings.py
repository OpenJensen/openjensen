"""Cloud compute checks use fixtures and never provision or query real credentials."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform import cloud_compute_catalog as catalog
from vla_platform.api import create_app
from vla_platform.compute_settings import ComputePreferences, ComputeSettings, ComputeSettingsUpdate
from vla_platform.lifecycle.runtime import RuntimeCatalog, command
from vla_platform.settings import Settings

GCP_CONFIG = {"project_id": "robotics-demo", "region": "us-central1"}
ADC_TOKEN = b"secret-adc-token-that-must-never-be-persisted"
SKY_ENDPOINT = "http://127.0.0.1:46580"


def save_connection(directory, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cloud-connections.json").write_text(
        json.dumps({"version": 1, "providers": {"gcp": {**GCP_CONFIG, **overrides}}})
    )


def offerings(*gpu_ids, region="us-central1"):
    return {
        gpu_id: [
            {
                "cloud": "GCP",
                "accelerator_name": gpu_id,
                "accelerator_count": 1,
                "instance_type": catalog.GCP_GPU_BY_ID[gpu_id].instance_type,
                "region": region,
            }
        ]
        for gpu_id in gpu_ids
    }


@pytest.fixture
def probes(monkeypatch):
    calls = []
    monkeypatch.setattr(catalog, "sky_executable", lambda: "/fixture/sky")
    monkeypatch.setattr(catalog, "sky_python", lambda _: "/fixture/python")
    monkeypatch.setattr("vla_platform.compute_settings.shutil.which", lambda _: "/fixture/gcloud")

    async def run(executable, args, **kwargs):
        calls.append((executable, args))
        if executable == "/fixture/python":
            if len(args) > 2 and args[1] == catalog._WORKSPACE_SCRIPT:
                assert args[4] == "verify"
                return (
                    "FIREBIRD_SKY_WORKSPACE="
                    + json.dumps(
                        {"status": "ready", "workspace": args[2], "endpoint": SKY_ENDPOINT}
                    )
                ).encode()
            assert args[0] == "-c" and "google.cloud.storage" in args[1]
            return b""
        if args[:3] == ["auth", "application-default", "print-access-token"]:
            return ADC_TOKEN
        if args[:2] == ["services", "list"]:
            return json.dumps(
                [{"config": {"name": name}} for name in catalog.REQUIRED_GCP_SERVICES]
            ).encode()
        if args[:2] == ["gpus", "list"]:
            assert args == [
                "gpus",
                "list",
                "--all",
                "--infra",
                "gcp/us-central1",
                "--output",
                "json",
            ]
            return json.dumps(offerings("L4", "T4", "A100")).encode()
        raise AssertionError("Unexpected setup command")

    monkeypatch.setattr(catalog, "run_readonly", run)
    return calls, run


def test_existing_local_settings_preserved_without_inventing_cloud_connection(tmp_path):
    (tmp_path / "compute-settings.json").write_text(
        json.dumps({"version": 1, "local": {"enabled": False, "label": "Lab"}})
    )
    compute = ComputeSettings(tmp_path)
    assert compute.preferences().local.label == "Lab"
    assert compute.preferences().local.enabled is False
    assert compute.preferences().gcp.model_dump() == {
        "enabled": True,
        "default_gpu": "A100",
        "disk_size_gb": 200,
        "idle_minutes": 10,
    }
    assert compute.cloud_runtimes() == []


def test_settings_patch_preserves_other_provider_and_persists(tmp_path, probes):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        client.put("/api/v1/compute-settings", json={"local": {"enabled": False, "label": "Lab"}})
        response = client.put(
            "/api/v1/compute-settings",
            json={
                "gcp": {
                    "enabled": True,
                    "default_gpu": "L4",
                    "disk_size_gb": 300,
                    "idle_minutes": 5,
                }
            },
        )
        assert response.status_code == 200
        assert response.json()["local"] == {"enabled": False, "label": "Lab"}
        configured = response.json()["gcp"]
        client.put(
            "/api/v1/compute-settings", json={"local": {"enabled": True, "label": "Desktop"}}
        )
        assert client.get("/api/v1/compute-settings").json()["gcp"] == configured
    assert ComputeSettings(tmp_path).preferences().gcp.model_dump() == configured
    assert probes[0] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"gcp": {"default_gpu": "RTXPRO6000"}},
        {"gcp": {"default_gpu": "A100-80GB"}},
        {"gcp": {"default_gpu": "H100"}},
        {"gcp": {"default_gpu": "A100; shell-command"}},
        {"gcp": {"enabled": "true"}},
        {"gcp": {"disk_size_gb": 99}},
        {"gcp": {"disk_size_gb": 2001}},
        {"gcp": {"disk_size_gb": 200.5}},
        {"gcp": {"idle_minutes": 0}},
        {"gcp": {"idle_minutes": 61}},
        {"gcp": {"project_id": "different-project"}},
        {"gcp": {"run": "arbitrary-command"}},
    ],
)
def test_cloud_settings_only_accept_reviewed_resource_parameters(payload):
    with pytest.raises(ValidationError):
        ComputeSettingsUpdate.model_validate(payload)


def test_listing_does_not_verify_credentials_or_claim_ready(tmp_path, probes):
    save_connection(tmp_path)
    compute = ComputeSettings(tmp_path)
    compute.update(ComputeSettingsUpdate.model_validate({"gcp": {"enabled": True}}))
    response = compute.public(RuntimeCatalog())
    assert response.gcp_status.status == "unchecked"
    assert response.gcp_status.configured is True
    assert response.gcp_status.skypilot_installed is True
    assert all(runtime.enabled and runtime.launchable for runtime in response.runtimes)
    assert all(runtime.needs_preparation for runtime in response.runtimes)
    assert [gpu.id for gpu in response.gpu_options if gpu.supported] == ["L4", "T4", "A100"]
    assert not any(gpu.available for gpu in response.gpu_options)
    assert probes[0] == []
    with pytest.raises(ValueError, match="prepared automatically"):
        compute.cloud_target("skypilot-gcp-A100")


def test_user_triggered_check_exposes_verified_cloud_targets_without_provisioning(tmp_path, probes):
    save_connection(tmp_path)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        client.put("/api/v1/compute-settings", json={"gcp": {"enabled": True}})
        response = client.post("/api/v1/compute-settings/gcp/check")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["gcp_status"]["status"] == "ready"
        assert data["gcp_status"]["setup_commands"] == []
        assert data["gcp_status"]["project_id"] == GCP_CONFIG["project_id"]
        assert data["gcp_status"]["region"] == GCP_CONFIG["region"]
        assert data["gcp_status"]["checked_at"]
        enabled = [runtime for runtime in data["runtimes"] if runtime["enabled"]]
        assert [runtime["id"] for runtime in enabled] == [
            "skypilot-gcp-L4",
            "skypilot-gcp-T4",
            "skypilot-gcp-A100",
        ]
        assert all(runtime["execution"] == "skypilot" for runtime in enabled)
        options = client.get("/api/v1/policy-options").json()
        smolvla = next(model for model in options["training_models"] if model["id"] == "smolvla")
        assert smolvla["runtime_ids"] == [runtime["id"] for runtime in enabled]
        assert ADC_TOKEN.decode() not in response.text
        assert ADC_TOKEN.decode() not in (tmp_path / "compute-settings.json").read_text()
        compute = client.app.state.execution.lifecycle.compute
        target = compute.cloud_target("skypilot-gcp-A100")
        assert target == {
            "project_id": "robotics-demo",
            "workspace": catalog.sky_workspace_name("robotics-demo"),
            "sky_api_endpoint": SKY_ENDPOINT,
            "region": "us-central1",
            "accelerator": "A100",
            "gpu_count": 1,
            "instance_type": "a2-highgpu-1g",
            "disk_size_gb": 200,
            "idle_minutes": 10,
        }
        with pytest.raises(ValueError, match="Unknown"):
            compute.cloud_target("skypilot-gcp-RTXPRO6000")
    assert len(probes[0]) == 5
    assert not any(
        args[0] in {"launch", "check", "exec"} or "enable" in args for _, args in probes[0]
    )
    # Authentication and catalog checks must be repeated after an application restart.
    assert ComputeSettings(tmp_path).gcp_status().status == "unchecked"


def test_disabled_cloud_remains_disabled_after_successful_setup_check(tmp_path, probes):
    save_connection(tmp_path)
    compute = ComputeSettings(tmp_path)
    compute.update(ComputeSettingsUpdate.model_validate({"gcp": {"enabled": False}}))
    assert asyncio.run(compute.check_gcp()).status == "ready"
    assert not any(option.available for option in compute.gpu_options())
    assert not any(compute.enabled(runtime) for runtime in compute.cloud_runtimes())
    with pytest.raises(ValueError, match="Enable Google Cloud"):
        compute.cloud_target("skypilot-gcp-L4")


@pytest.mark.parametrize("failure", ["dependencies", "adc", "services", "catalog"])
def test_setup_failures_never_expose_cli_output_or_enable_targets(
    tmp_path, probes, monkeypatch, failure
):
    save_connection(tmp_path)
    compute = ComputeSettings(tmp_path)
    compute.update(ComputeSettingsUpdate.model_validate({"gcp": {"enabled": True}}))
    original = probes[1]

    async def failed(executable, args, **kwargs):
        if (
            (failure == "dependencies" and args[0] == "-c")
            or (failure == "adc" and args[0] == "auth")
            or (failure == "services" and args[0] == "services")
        ):
            raise catalog.SetupCheckError("safe check failure")
        if failure == "catalog" and args[0] == "gpus":
            return b"not-json-secret-token"
        return await original(executable, args, **kwargs)

    monkeypatch.setattr(catalog, "run_readonly", failed)
    state = asyncio.run(compute.check_gcp())
    assert state.status == "setup_required"
    assert "secret-token" not in state.message
    commands = state.setup_commands
    if failure == "dependencies":
        assert len(commands) == 1 and "install" in commands[0]
    elif failure == "adc":
        assert commands == ["gcloud auth application-default login"]
    elif failure == "services":
        assert commands == [
            f"sky check --workspace {catalog.sky_workspace_name('robotics-demo')} gcp"
        ]
    else:
        assert commands == []
    assert not any(option.available for option in compute.gpu_options())


def test_failed_recheck_and_changed_region_revoke_previous_readiness(tmp_path, probes, monkeypatch):
    save_connection(tmp_path)
    compute = ComputeSettings(tmp_path)
    compute.update(ComputeSettingsUpdate.model_validate({"gcp": {"enabled": True}}))
    assert asyncio.run(compute.check_gcp()).status == "ready"
    assert compute.cloud_target("skypilot-gcp-L4")["region"] == "us-central1"
    save_connection(tmp_path, region="us-west1")
    assert compute.gcp_status().status == "unchecked"
    with pytest.raises(ValueError, match="prepared automatically"):
        compute.cloud_target("skypilot-gcp-L4")
    save_connection(tmp_path)

    async def failed(*args, **kwargs):
        raise catalog.SetupCheckError("safe setup error")

    monkeypatch.setattr(catalog, "run_readonly", failed)
    assert asyncio.run(compute.check_gcp()).status == "setup_required"
    assert not any(option.available for option in compute.gpu_options())


def test_only_verified_single_gpu_instance_in_selected_region_is_offered():
    document = offerings("A100", "T4", "L4")
    document["A100"][0]["accelerator_count"] = 8
    document["T4"][0]["region"] = "us-east4"
    document["L4"][0]["instance_type"] = "n1-highmem-8"
    assert catalog.regional_gpu_offerings(json.dumps(document).encode(), "us-central1") == set()
    assert catalog.regional_gpu_offerings(
        json.dumps(offerings("A100", "L4")).encode(), "us-central1"
    ) == {"A100", "L4"}


def test_reconnecting_same_cloud_project_requires_new_check(tmp_path, probes):
    save_connection(tmp_path)
    compute = ComputeSettings(tmp_path)
    assert asyncio.run(compute.check_gcp()).status == "ready"
    saved = tmp_path / "cloud-connections.json"
    saved.unlink()
    assert compute.gcp_status().status == "unchecked"
    save_connection(tmp_path)
    assert compute.gcp_status().status == "unchecked"


def test_removed_default_gpu_migrates_without_revoking_local_opt_in(tmp_path):
    path = tmp_path / "compute-settings.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "local": {"enabled": True, "label": "User enabled desktop"},
                "gcp": {"enabled": True, "default_gpu": "A100-80GB"},
            }
        )
    )
    preferences = ComputeSettings(tmp_path).preferences()
    assert preferences.local.enabled is True
    assert preferences.gcp.default_gpu == "A100"
    assert json.loads(path.read_text())["gcp"]["default_gpu"] == "A100"


def test_cloud_runtime_cannot_fall_back_to_native_execution(tmp_path):
    compute = ComputeSettings(tmp_path)
    runtime = compute.runtime("skypilot-gcp-A100")
    with pytest.raises(ValueError, match="cloud runner"):
        command(runtime, Path("request"), Path("output"), tmp_path, "container", training=True)
    config = tmp_path / "runtimes.json"
    config.write_text(RuntimeCatalog(runtimes=[runtime]).model_dump_json())
    with pytest.raises(ValueError, match="managed by compute settings"):
        RuntimeCatalog.load(config)


def test_import_check_uses_sky_environment_and_never_executes_shell_wrapper(tmp_path, monkeypatch):
    python = tmp_path / "python3"
    python.write_text("")
    python.chmod(0o700)
    sky = tmp_path / "sky"
    sky.write_text(f"#!{python}\n")
    assert catalog.sky_python(str(sky)) == str(python)
    sky.write_text("#!/bin/sh\nexec arbitrary-shell\n")
    assert catalog.sky_python(str(sky)) is None


def test_atomic_update_cannot_publish_failed_cloud_settings(tmp_path, monkeypatch):
    compute = ComputeSettings(tmp_path)
    previous = compute.preferences()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("vla_platform.compute_settings.os.replace", fail)
    with pytest.raises(OSError):
        compute.update(ComputePreferences.model_validate({"gcp": {"enabled": True}}))
    assert compute.preferences() == previous


@pytest.mark.parametrize(
    "mode,existing,status,create_count",
    [
        ("verify", None, "missing", 0),
        ("verify", "robotics-demo", "ready", 0),
        ("verify", "different-project", "mismatch", 0),
        ("prepare", None, "ready", 1),
        ("prepare", "robotics-demo", "ready", 0),
        ("prepare", "different-project", "mismatch", 0),
    ],
)
def test_workspace_script_only_creates_missing_scoped_entry(
    monkeypatch, capsys, mode, existing, status, create_count
):
    name = catalog.sky_workspace_name("robotics-demo")
    untouched = {
        "default": {"gcp": {"project_id": "existing-project"}},
        "robotics-team": {"ssh": {"allowed_node_pools": ["xbox-360"]}},
    }
    workspaces = copy.deepcopy(untouched)
    if existing:
        workspaces[name] = {"gcp": {"project_id": existing}}
    sky = ModuleType("sky")
    client = ModuleType("sky.client")
    sdk = ModuleType("sky.client.sdk")
    server = ModuleType("sky.server")
    common = ModuleType("sky.server.common")
    requests = []
    checks = []

    def request(method, path, **kwargs):
        assert method == "POST" and path == "/workspaces/create"
        body = kwargs["json"]
        assert body == {"workspace_name": name, "config": {"gcp": {"project_id": "robotics-demo"}}}
        assert name not in workspaces
        requests.append(body)
        workspaces[name] = copy.deepcopy(body["config"])
        return "created"

    sky.get = lambda value: value
    sky.workspaces = ModuleType("sky.workspaces")  # SkyPilot 0.13 exposes a module here.
    sdk.workspaces = lambda: copy.deepcopy(workspaces)

    def check(*, infra_list, verbose, workspace):
        assert infra_list == ("gcp",) and verbose is False and workspace == name
        checks.append(workspace)
        return {name: {"GCP": ["compute", "storage"]}}

    sdk.check = check
    client.sdk = sdk
    common.check_server_healthy_or_start = lambda function: function
    common.make_authenticated_request = request
    common.get_request_id = lambda response: response
    common.get_server_url = lambda: SKY_ENDPOINT
    server.common = common
    for key, module in (
        ("sky", sky),
        ("sky.client", client),
        ("sky.client.sdk", sdk),
        ("sky.server", server),
        ("sky.server.common", common),
    ):
        monkeypatch.setitem(sys.modules, key, module)
    monkeypatch.setattr(sys, "argv", ["check.py", name, "robotics-demo", mode])
    exec(catalog._WORKSPACE_SCRIPT, {})
    result = json.loads(capsys.readouterr().out.split("FIREBIRD_SKY_WORKSPACE=")[1])
    assert result == {
        "status": status,
        "workspace": name,
        "endpoint": SKY_ENDPOINT,
        "reason": None,
        "error_type": None,
    }
    assert len(requests) == create_count
    assert len(checks) == int(mode == "prepare" and status == "ready")
    assert {key: workspaces[key] for key in untouched} == untouched


@pytest.mark.parametrize("status", ["missing", "mismatch", "error"])
def test_workspace_verification_fails_closed_without_mutation(monkeypatch, status):
    monkeypatch.setattr(catalog, "sky_python", lambda _: "/fixture/python")
    calls = []

    async def run(executable, args, **kwargs):
        calls.append(args)
        assert args[-1] == "verify"
        return (
            "FIREBIRD_SKY_WORKSPACE="
            + json.dumps({"status": status, "workspace": args[2], "endpoint": SKY_ENDPOINT})
        ).encode()

    monkeypatch.setattr(catalog, "run_readonly", run)
    with pytest.raises(catalog.SetupCheckError):
        asyncio.run(catalog.verify_sky_workspace("/fixture/sky", "robotics-demo"))
    assert len(calls) == 1


def test_prepare_endpoint_then_readonly_recheck_returns_pinned_workspace(
    tmp_path, probes, monkeypatch
):
    save_connection(tmp_path)
    prepare_calls = []

    async def prepare(executable, args, **kwargs):
        assert executable == "/fixture/python"
        assert args[-1] == "prepare"
        prepare_calls.append(args)
        return (
            "FIREBIRD_SKY_WORKSPACE="
            + json.dumps({"status": "ready", "workspace": args[2], "endpoint": SKY_ENDPOINT})
        ).encode()

    monkeypatch.setattr(catalog, "run_setup_command", prepare)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        client.put("/api/v1/compute-settings", json={"gcp": {"enabled": False}})
        result = client.post("/api/v1/compute-settings/gcp/prepare").json()
        assert result["gcp_status"]["status"] == "ready"
        assert result["gcp_status"]["workspace"] == catalog.sky_workspace_name("robotics-demo")
        assert result["gcp_status"]["sky_api_endpoint"] == SKY_ENDPOINT
        assert result["gcp"]["enabled"] is False
    assert len(prepare_calls) == 1
    assert len(probes[0]) == 5


def test_readonly_check_requires_prepared_workspace(tmp_path, probes, monkeypatch):
    save_connection(tmp_path)

    async def missing(*args, **kwargs):
        raise catalog.SetupCheckError("Prepare a SkyPilot workspace in Settings.")

    monkeypatch.setattr(catalog, "verify_sky_target", missing)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        response = client.post("/api/v1/compute-settings/gcp/check").json()
        assert response["gcp_status"]["status"] == "setup_required"
        assert "Prepare" in response["gcp_status"]["message"]
        assert response["gcp_status"]["setup_commands"] == []
        assert not any(option["available"] for option in response["gpu_options"])


def test_failed_preparation_survives_restart_without_blocking_retry_selection(
    tmp_path, probes, monkeypatch
):
    save_connection(tmp_path)
    compute = ComputeSettings(tmp_path)

    async def failed(*args, **kwargs):
        raise catalog.SetupCheckError("Permission was denied for this workspace.")

    monkeypatch.setattr(catalog, "prepare_sky_target", failed)
    state = asyncio.run(compute.prepare_gcp())
    assert state.status == "setup_required"
    restarted = ComputeSettings(tmp_path)
    assert restarted.gcp_status().message == state.message
    assert all(option.launchable and not option.available for option in restarted.gpu_options())
    assert asyncio.run(restarted.check_gcp()).status == "ready"
    assert not (tmp_path / "compute-last-failure.json").exists()
    assert ComputeSettings(tmp_path).gcp_status().status == "unchecked"


def test_configured_endpoint_probe_does_not_start_or_query_the_server(monkeypatch):
    monkeypatch.setattr(catalog, "sky_python", lambda _: "/fixture/python")

    async def local_only(executable, args, **kwargs):
        assert executable == "/fixture/python"
        assert 15 <= kwargs["timeout"] <= 30
        assert args[0] == "-c"
        assert "common.get_server_url()" in args[1]
        assert "check_server_healthy_or_start" not in args[1]
        assert "sdk." not in args[1] and "sky.get" not in args[1]
        return ("FIREBIRD_SKY_ENDPOINT=" + json.dumps(SKY_ENDPOINT)).encode()

    monkeypatch.setattr(catalog, "run_readonly", local_only)
    assert asyncio.run(catalog.configured_sky_endpoint("/fixture/sky")) == SKY_ENDPOINT


def test_saved_endpoint_is_forced_during_workspace_verification(monkeypatch):
    monkeypatch.setattr(catalog, "sky_python", lambda _: "/fixture/python")

    async def verify(executable, args, **kwargs):
        assert kwargs["env_overrides"] == {"SKYPILOT_API_SERVER_ENDPOINT": SKY_ENDPOINT}
        return (
            "FIREBIRD_SKY_WORKSPACE="
            + json.dumps({"status": "ready", "workspace": args[2], "endpoint": SKY_ENDPOINT})
        ).encode()

    monkeypatch.setattr(catalog, "run_readonly", verify)
    assert asyncio.run(
        catalog.verify_sky_target("/fixture/sky", "robotics-demo", SKY_ENDPOINT)
    ) == {
        "workspace": catalog.sky_workspace_name("robotics-demo"),
        "sky_api_endpoint": SKY_ENDPOINT,
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://different-server.example",
        "https://user:secret@example.test",
        "https://example.test?token=secret",
        "https://example.test#secret",
        "file:///private/file",
        "http://localhost\n/",
        "https://example.test:invalid",
        None,
    ],
)
def test_endpoint_drift_and_credentials_in_workspace_response_are_rejected(monkeypatch, endpoint):
    monkeypatch.setattr(catalog, "sky_python", lambda _: "/fixture/python")

    async def verify(executable, args, **kwargs):
        return (
            "FIREBIRD_SKY_WORKSPACE="
            + json.dumps({"status": "ready", "workspace": args[2], "endpoint": endpoint})
        ).encode()

    monkeypatch.setattr(catalog, "run_readonly", verify)
    with pytest.raises(catalog.SetupCheckError) as error:
        asyncio.run(catalog.verify_sky_target("/fixture/sky", "robotics-demo", SKY_ENDPOINT))
    assert "secret" not in str(error.value)
