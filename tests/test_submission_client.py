"""Explicit CLI keys and read-only reconciliation, with no automatic mutation retry."""

import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner
from vla_platform import cli, cli_client
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.tui_client import ApiClient, ApiError


def accepted():
    return Job(
        id="saved",
        project_id="p",
        request=IntakeRequest(source="local", path="data"),
        created_at=now(),
        updated_at=now(),
    ).model_dump()


def test_transport_keeps_key_on_single_lost_response(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadError("fixture lost acknowledgment")

    api = ApiClient("http://127.0.0.1:9999", transport=httpx.MockTransport(handle))
    monkeypatch.setattr(cli_client, "client", lambda: api)
    with pytest.raises(ApiError, match="sent once"):
        asyncio.run(
            cli_client.request_json(
                "POST",
                "/projects/p/intakes",
                {"source": "local", "path": "data"},
                idempotency_key="saved-key",
            )
        )
    assert len(calls) == 1 and calls[0].headers["idempotency-key"] == "saved-key"
    assert api.http.is_closed


def test_cli_lookup_only_gets_original_scoped_job(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=accepted())

    api = ApiClient("http://127.0.0.1:9999", transport=httpx.MockTransport(handle))
    monkeypatch.setattr(cli_client, "client", lambda: api)
    result = CliRunner().invoke(
        cli.app, ["jobs", "submission", "p", "dataset.inspect", "saved-key"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "saved"
    assert len(calls) == 1 and calls[0].method == "GET"
    assert str(calls[0].url).endswith("/projects/p/submissions/saved-key?operation=dataset.inspect")
    assert api.http.is_closed


@pytest.mark.parametrize("command", ["inspect", "policy", "augmentation"])
def test_cli_exposes_explicit_key_without_changing_recipe(tmp_path, monkeypatch, command):
    calls = []

    async def receive(method, path, payload, **kwargs):
        calls.append((method, path, payload, kwargs))
        return {"fixture": True}

    monkeypatch.setattr(cli_client, "request_json", receive)
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"fixture":"original"}')
    args = (
        ["inspect", "p", "--path", "data"]
        if command == "inspect"
        else [command, "submit", "p", str(recipe)]
    )
    result = CliRunner().invoke(cli.app, [*args, "--idempotency-key", "saved-key"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0][0] == "POST"
    assert calls[0][3] == {"idempotency_key": "saved-key"}
    assert "idempotency_key" not in calls[0][2]


@pytest.mark.parametrize("project,kind", [("foreign", "dataset.inspect"), ("p", "policy.run")])
def test_lookup_rejects_wrong_scope(monkeypatch, project, kind):
    value = accepted()
    value.update(project_id=project, kind=kind)
    api = ApiClient(
        "http://127.0.0.1:9999",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=value)),
    )
    monkeypatch.setattr(cli_client, "client", lambda: api)
    with pytest.raises(ApiError):
        asyncio.run(cli_client.submission_job("p", "dataset.inspect", "key"))
    assert api.http.is_closed


def test_invalid_key_fails_before_network(monkeypatch):
    def unexpected(_):
        pytest.fail("No request for invalid local key")

    api = ApiClient("http://127.0.0.1:9999", transport=httpx.MockTransport(unexpected))
    monkeypatch.setattr(cli_client, "client", lambda: api)
    with pytest.raises(ApiError, match="Idempotency key"):
        asyncio.run(
            cli_client.request_json("POST", "/projects/p/intakes", {}, idempotency_key="bad key")
        )
    assert api.http.is_closed
