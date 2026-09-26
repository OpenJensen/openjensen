"""Cloud settings checks use fixtures only; no host credentials or cloud calls."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.cloud_connections import (
    CloudConnections,
    GcpConnectionConfig,
    _CheckFailed,
    run_cloud_cli,
)
from vla_platform.settings import Settings

GCP = {"project_id": "robot-project", "region": "us-central1"}
GCP_ACCOUNT = [{"account": "operator@example.test", "credential": "must-not-leak"}]
GCP_PROJECT = {"projectId": "robot-project", "lifecycleState": "ACTIVE"}


@pytest.fixture
def cloud_cli(monkeypatch):
    command = AsyncMock(side_effect=AssertionError("Unexpected cloud CLI invocation"))
    monkeypatch.setattr("vla_platform.cloud_connections.run_cloud_cli", command)
    return command


@pytest.fixture
def cloud_client(tmp_path, cloud_cli):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        yield client


def test_list_does_not_inspect_credentials_or_call_clouds(cloud_client, cloud_cli):
    response = cloud_client.get("/api/v1/cloud-connections")
    assert response.status_code == 200
    providers = response.json()["providers"]
    assert [p["provider"] for p in providers] == ["gcp"]
    assert all(p["status"] == "disconnected" and p["config"] is None for p in providers)
    cloud_cli.assert_not_awaited()


def test_gcp_verifies_active_identity_and_project_then_saves_selection(
    cloud_client, cloud_cli, tmp_path
):
    cloud_cli.side_effect = [GCP_ACCOUNT, GCP_PROJECT]
    response = cloud_client.post("/api/v1/cloud-connections/gcp/connect", json=GCP)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "connected"
    assert result["config"] == GCP
    assert result["identity"] == {
        "account": "operator@example.test",
    }
    assert result["checked_at"]
    assert "must-not-leak" not in response.text
    assert cloud_cli.await_args_list[0].args == (
        "gcloud",
        ["auth", "list", "--filter=status:ACTIVE", "--format=json(account)", "--quiet"],
    )
    assert cloud_cli.await_args_list[1].args == (
        "gcloud",
        ["projects", "describe", "robot-project", "--format=json", "--quiet"],
    )
    saved = tmp_path / "cloud-connections.json"
    assert json.loads(saved.read_text()) == {"version": 1, "providers": {"gcp": GCP}}
    assert saved.stat().st_mode & 0o077 == 0
    assert cloud_client.get("/api/v1/cloud-connections").json()["providers"][0] == result


def test_connect_enables_cloud_training_without_resetting_resource_preferences(
    cloud_client, cloud_cli
):
    cloud_client.put(
        "/api/v1/compute-settings",
        json={
            "gcp": {"enabled": False, "default_gpu": "L4", "disk_size_gb": 300, "idle_minutes": 5}
        },
    )
    cloud_cli.side_effect = [GCP_ACCOUNT, GCP_PROJECT]
    assert (
        cloud_client.post("/api/v1/cloud-connections/gcp/connect", json=GCP).json()["status"]
        == "connected"
    )
    compute = cloud_client.get("/api/v1/compute-settings").json()
    assert compute["gcp"] == {
        "enabled": True,
        "default_gpu": "L4",
        "disk_size_gb": 300,
        "idle_minutes": 5,
    }
    assert compute["local"]["enabled"] is False
    assert compute["gcp_status"]["status"] == "unchecked"
    assert all(
        runtime["launchable"] and runtime["needs_preparation"] for runtime in compute["runtimes"]
    )
    assert cloud_cli.await_count == 2  # Connecting does not prepare or provision a GPU.


@pytest.mark.parametrize(
    "provider,payload",
    [
        ("gcp", {**GCP, "service_account_json": "secret"}),
        ("gcp", {**GCP, "project_id": "--bad"}),
        ("gcp", {**GCP, "region": "us-central1;bad"}),
        ("gcp", {"profile": "default", "region": "us-east-1"}),
        ("aws", GCP),
        ("azure", GCP),
    ],
)
def test_invalid_and_secret_input_rejected_before_cli(cloud_client, cloud_cli, provider, payload):
    response = cloud_client.post(f"/api/v1/cloud-connections/{provider}/connect", json=payload)
    assert response.status_code == 422
    cloud_cli.assert_not_awaited()


@pytest.mark.parametrize("action", ["connect", "recheck", "disconnect"])
def test_removed_aws_provider_is_rejected_before_cli(cloud_client, cloud_cli, action):
    response = cloud_client.post(
        f"/api/v1/cloud-connections/aws/{action}",
        json={"profile": "default", "region": "us-east-1"} if action == "connect" else None,
    )
    assert response.status_code == 422
    cloud_cli.assert_not_awaited()


@pytest.mark.parametrize(
    "replies,expected",
    [
        ([[], GCP_PROJECT], "setup_required"),
        ([GCP_ACCOUNT, {**GCP_PROJECT, "lifecycleState": "DELETE_REQUESTED"}], "setup_required"),
        ([GCP_ACCOUNT, {**GCP_PROJECT, "projectId": "different-project"}], "setup_required"),
        ([[{"account": "bad\nidentity"}], GCP_PROJECT], "error"),
    ],
)
def test_gcp_failure_never_persists_or_claims_connection(
    cloud_client, cloud_cli, tmp_path, replies, expected
):
    cloud_cli.side_effect = replies
    result = cloud_client.post("/api/v1/cloud-connections/gcp/connect", json=GCP).json()
    assert result["status"] == expected
    assert result["identity"] is None
    assert not (tmp_path / "cloud-connections.json").exists()
    assert (
        cloud_client.get("/api/v1/cloud-connections").json()["providers"][0]["status"]
        == "disconnected"
    )


def test_failed_edit_preserves_active_and_saved_connection(cloud_client, cloud_cli, tmp_path):
    cloud_cli.side_effect = [GCP_ACCOUNT, GCP_PROJECT]
    original = cloud_client.post("/api/v1/cloud-connections/gcp/connect", json=GCP).json()
    saved = (tmp_path / "cloud-connections.json").read_text()
    cloud_cli.side_effect = _CheckFailed("setup_required", "Sign in again.")
    failed = cloud_client.post(
        "/api/v1/cloud-connections/gcp/connect", json={**GCP, "project_id": "other-project"}
    ).json()
    assert failed["status"] == "setup_required"
    assert failed["config"]["project_id"] == "other-project"
    assert cloud_client.get("/api/v1/cloud-connections").json()["providers"][0] == original
    assert (tmp_path / "cloud-connections.json").read_text() == saved


def test_recheck_removes_stale_verified_identity(cloud_client, cloud_cli, tmp_path):
    cloud_cli.side_effect = [GCP_ACCOUNT, GCP_PROJECT]
    cloud_client.post("/api/v1/cloud-connections/gcp/connect", json=GCP)
    cloud_cli.side_effect = _CheckFailed("setup_required", "Sign in again.")
    result = cloud_client.post("/api/v1/cloud-connections/gcp/recheck").json()
    assert result["status"] == "setup_required"
    assert result["identity"] is None
    assert result["config"] == GCP
    assert cloud_client.get("/api/v1/cloud-connections").json()["providers"][0] == result
    assert json.loads((tmp_path / "cloud-connections.json").read_text())["providers"]["gcp"] == GCP


def test_disconnect_only_removes_workspace_selection(cloud_client, cloud_cli, tmp_path):
    cloud_cli.side_effect = [GCP_ACCOUNT, GCP_PROJECT]
    cloud_client.post("/api/v1/cloud-connections/gcp/connect", json=GCP)
    cloud_cli.reset_mock()
    result = cloud_client.post("/api/v1/cloud-connections/gcp/disconnect").json()
    assert result["status"] == "disconnected"
    assert result["config"] is None
    assert json.loads((tmp_path / "cloud-connections.json").read_text())["providers"] == {}
    cloud_cli.assert_not_awaited()


def test_saved_connections_need_recheck_after_restart(tmp_path, cloud_cli):
    cloud_cli.side_effect = [GCP_ACCOUNT, GCP_PROJECT]
    asyncio.run(CloudConnections(tmp_path).connect("gcp", GcpConnectionConfig(**GCP)))
    cloud_cli.reset_mock()
    restarted = CloudConnections(tmp_path).list().providers[0]
    assert restarted.status == "unverified"
    assert restarted.config.model_dump() == GCP
    assert restarted.identity is None
    cloud_cli.assert_not_awaited()


@pytest.mark.parametrize(
    "obsolete_record", [{"profile": "old-profile", "region": "us-east-1"}, "invalid"]
)
def test_saved_aws_record_is_ignored_without_losing_gcp(tmp_path, cloud_cli, obsolete_record):
    (tmp_path / "cloud-connections.json").write_text(
        json.dumps({"version": 1, "providers": {"gcp": GCP, "aws": obsolete_record}})
    )
    providers = CloudConnections(tmp_path).list().providers
    assert len(providers) == 1
    assert providers[0].provider == "gcp"
    assert providers[0].status == "unverified"
    assert providers[0].config.model_dump() == GCP
    cloud_cli.assert_not_awaited()


def test_corrupt_file_keeps_app_available_without_guessing_connection(tmp_path, cloud_cli):
    (tmp_path / "cloud-connections.json").write_text("not json")
    assert all(p.status == "error" for p in CloudConnections(tmp_path).list().providers)
    cloud_cli.assert_not_awaited()


def test_atomic_save_failure_does_not_report_connected(tmp_path, cloud_cli, monkeypatch):
    cloud_cli.side_effect = [GCP_ACCOUNT, GCP_PROJECT]
    service = CloudConnections(tmp_path)
    monkeypatch.setattr("vla_platform.cloud_connections.os.replace", Mock(side_effect=OSError))
    result = asyncio.run(service.connect("gcp", GcpConnectionConfig(**GCP)))
    assert result.status == "error"
    assert result.identity is None
    assert service.list().providers[0].status == "disconnected"
    assert not list(tmp_path.glob(".cloud-connections-*"))


def test_cross_origin_cannot_check_or_disconnect_accounts(cloud_client, cloud_cli):
    for action in ("connect", "recheck", "disconnect"):
        response = cloud_client.post(
            f"/api/v1/cloud-connections/gcp/{action}",
            json=GCP if action == "connect" else None,
            headers={"Origin": "https://untrusted.example"},
        )
        assert response.status_code == 403
    cloud_cli.assert_not_awaited()


def test_missing_cli_is_setup_required_and_does_not_execute(monkeypatch):
    monkeypatch.setattr("vla_platform.cloud_connections.shutil.which", lambda _: None)
    spawn = AsyncMock()
    monkeypatch.setattr("vla_platform.cloud_connections.asyncio.create_subprocess_exec", spawn)
    with pytest.raises(_CheckFailed, match="Install Google Cloud CLI") as exc:
        asyncio.run(run_cloud_cli("gcloud", ["auth", "list"]))
    assert exc.value.status == "setup_required"
    spawn.assert_not_awaited()


def test_cli_uses_argv_and_does_not_return_sensitive_diagnostics(monkeypatch):
    monkeypatch.setattr("vla_platform.cloud_connections.shutil.which", lambda _: "/fixture/gcloud")
    process = Mock(returncode=1, communicate=AsyncMock(return_value=(b"", b"SECRET TOKEN")))
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr("vla_platform.cloud_connections.asyncio.create_subprocess_exec", spawn)
    with pytest.raises(_CheckFailed) as exc:
        asyncio.run(run_cloud_cli("gcloud", ["auth", "list"]))
    assert "SECRET" not in str(exc.value)
    assert spawn.await_args.args == ("/fixture/gcloud", "auth", "list")
    assert spawn.await_args.kwargs["env"]["CLOUDSDK_CORE_DISABLE_PROMPTS"] == "1"
    assert "shell" not in spawn.await_args.kwargs


@pytest.mark.parametrize("parent_returncode", [None, 0])
def test_cli_timeout_kills_process_and_returns_short_failure(monkeypatch, parent_returncode):
    monkeypatch.setattr("vla_platform.cloud_connections.shutil.which", lambda _: "/fixture/gcloud")
    process = Mock(returncode=parent_returncode, communicate=AsyncMock(return_value=(b"", b"")))
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr("vla_platform.cloud_connections.asyncio.create_subprocess_exec", spawn)
    kill_group = Mock()
    monkeypatch.setattr("vla_platform.cloud_connections.os.killpg", kill_group)

    async def timeout(awaitable, *, timeout):
        await awaitable
        if timeout == 15:
            raise TimeoutError
        assert timeout == 3

    monkeypatch.setattr("vla_platform.cloud_connections.asyncio.wait_for", timeout)
    with pytest.raises(_CheckFailed, match="timed out"):
        asyncio.run(run_cloud_cli("gcloud", ["auth", "list"]))
    kill_group.assert_called_once()


def test_provider_cli_invalid_json_is_not_returned(monkeypatch):
    monkeypatch.setattr("vla_platform.cloud_connections.shutil.which", lambda _: "/fixture/gcloud")
    process = Mock(returncode=0, communicate=AsyncMock(return_value=(b"PRIVATE BAD JSON", b"")))
    monkeypatch.setattr(
        "vla_platform.cloud_connections.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    )
    with pytest.raises(_CheckFailed, match="invalid response") as exc:
        asyncio.run(run_cloud_cli("gcloud", ["auth", "list"]))
    assert "PRIVATE" not in str(exc.value)
