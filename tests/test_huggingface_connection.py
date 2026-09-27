"""Explicit HF credentials never appear in API results, jobs, logs, or error details."""

import asyncio
import json
import os
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from vla_platform.api import create_app
from vla_platform.huggingface_connection import (
    WHOAMI_URL,
    CredentialError,
    HuggingFaceConnection,
    HuggingFaceTokenInput,
    verify_huggingface_token,
)
from vla_platform.settings import Settings

TOKEN = "hf_ExampleFakeTokenForTestsOnly123456789"
REPLACEMENT = "hf_ReplacementFakeTokenForTestsOnly987654321"


@pytest.fixture
def identity(monkeypatch):
    verify = AsyncMock(return_value="robotic-user")
    monkeypatch.setattr("vla_platform.huggingface_connection.verify_huggingface_token", verify)
    return verify


def test_no_token_needed_for_public_models_or_inherited_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    connection = HuggingFaceConnection(tmp_path)
    assert connection.token() is None
    assert connection.status().configured is False
    assert not connection.path.exists()


def test_token_saved_privately_and_status_only_exposes_mask(tmp_path, identity):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        response = client.put("/api/v1/huggingface-connection", json={"token": TOKEN})
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        result = response.json()
        assert result["configured"] is True
        assert result["username"] == "robotic-user"
        assert result["token_hint"] == "••••" + TOKEN[-4:]
        assert result["checked_at"]
        assert TOKEN not in response.text
        assert "token" not in result
        assert TOKEN not in repr(HuggingFaceTokenInput(token=TOKEN))
        assert app.state.huggingface_connection.token() == TOKEN
        assert client.get("/api/v1/huggingface-connection").json() == result
        assert TOKEN not in client.get("/api/v1/policy-options").text
    saved = tmp_path / "huggingface-credential.json"
    if os.name == "posix":
        assert saved.stat().st_mode & 0o077 == 0
    assert json.loads(saved.read_text())["token"] == TOKEN
    # Tokens are loaded only from this known application credential file.
    restored = HuggingFaceConnection(tmp_path)
    assert restored.token() == TOKEN
    assert restored.status().model_dump() == result
    identity.assert_awaited_once_with(TOKEN)


@pytest.mark.parametrize(
    "payload",
    [
        {"token": "bad-token-secret"},
        {"token": "hf_bad\nheader-secret"},
        {"token": TOKEN + "x" * 600},
        {"token": {"hidden": TOKEN}},
        {"token": [TOKEN]},
        {"unexpected": TOKEN},
        {"token": TOKEN, "extra": REPLACEMENT},
    ],
)
def test_invalid_input_never_echoes_submitted_secrets(tmp_path, identity, payload):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        response = client.put("/api/v1/huggingface-connection", json=payload)
        assert response.status_code == 422
        assert TOKEN not in response.text
        assert REPLACEMENT not in response.text
        assert "bad-token-secret" not in response.text
        assert "header-secret" not in response.text
    identity.assert_not_awaited()
    assert not (tmp_path / "huggingface-credential.json").exists()


def test_failed_replacement_preserves_previous_credential(tmp_path, identity):
    connection = HuggingFaceConnection(tmp_path)
    original = asyncio.run(connection.save(SecretStr(TOKEN)))
    saved = connection.path.read_bytes()
    identity.side_effect = CredentialError("Hugging Face rejected this token.")
    with pytest.raises(CredentialError):
        asyncio.run(connection.save(SecretStr(REPLACEMENT)))
    assert connection.token() == TOKEN
    assert connection.status() == original
    assert connection.path.read_bytes() == saved


def test_failed_atomic_write_preserves_saved_token_and_cleans_temporary_file(
    tmp_path, identity, monkeypatch
):
    connection = HuggingFaceConnection(tmp_path)
    original = asyncio.run(connection.save(SecretStr(TOKEN)))

    def failed(*args):
        raise OSError("fixture write failure")

    monkeypatch.setattr("vla_platform.huggingface_connection.os.replace", failed)
    with pytest.raises(OSError):
        asyncio.run(connection.save(SecretStr(REPLACEMENT)))
    assert connection.token() == TOKEN
    assert connection.status() == original
    assert HuggingFaceConnection(tmp_path).token() == TOKEN
    assert not list(tmp_path.glob(".huggingface-credential-*"))


def test_disconnect_deletes_only_app_credential(tmp_path, identity):
    unrelated = tmp_path / "other-token.json"
    unrelated.write_text("untouched")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        client.post("/api/v1/huggingface-connection", json={"token": TOKEN})
        response = client.delete("/api/v1/huggingface-connection")
        assert response.status_code == 200
        assert response.json()["configured"] is False
        assert response.json()["token_hint"] is None
        assert TOKEN not in response.text
        assert client.delete("/api/v1/huggingface-connection").status_code == 200
    assert unrelated.read_text() == "untouched"
    assert not (tmp_path / "huggingface-credential.json").exists()
    assert HuggingFaceConnection(tmp_path).token() is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are not Windows ACLs")
def test_nonprivate_saved_file_never_loaded(tmp_path, identity):
    connection = HuggingFaceConnection(tmp_path)
    asyncio.run(connection.save(SecretStr(TOKEN)))
    connection.path.chmod(0o644)
    assert HuggingFaceConnection(tmp_path).token() is None


def test_symlink_saved_file_never_loaded_or_target_deleted(tmp_path, identity):
    connection = HuggingFaceConnection(tmp_path)
    asyncio.run(connection.save(SecretStr(TOKEN)))
    original = tmp_path / "private-original.json"
    connection.path.rename(original)
    original.chmod(0o600)
    connection.path.symlink_to(original)
    unsafe = HuggingFaceConnection(tmp_path)
    assert unsafe.token() is None
    asyncio.run(unsafe.disconnect())
    assert original.exists()


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("vla_platform.huggingface_connection.httpx.AsyncClient", client)


def test_official_identity_verification_uses_authorization_header_only(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == WHOAMI_URL
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.content == b"" and not request.url.query
        return httpx.Response(200, json={"name": "robotic-user", "other_secret": TOKEN})

    mock_http(monkeypatch, handler)
    assert asyncio.run(verify_huggingface_token(TOKEN)) == "robotic-user"
    assert len(calls) == 1


@pytest.mark.parametrize("status", [301, 302, 401, 403, 429, 500])
def test_provider_errors_and_redirects_never_expose_token(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text=TOKEN, headers={"location": "https://untrusted.example"})

    mock_http(monkeypatch, handler)
    with pytest.raises(CredentialError) as error:
        asyncio.run(verify_huggingface_token(TOKEN))
    assert TOKEN not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("name", [TOKEN, "bad\nname", {"token": TOKEN}, None])
def test_malformed_provider_identity_never_becomes_public_status(monkeypatch, name):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"name": name}))
    with pytest.raises(CredentialError) as error:
        asyncio.run(verify_huggingface_token(TOKEN))
    assert TOKEN not in str(error.value)


def test_external_origin_cannot_save_token_and_local_delete_preflight_allowed(tmp_path, identity):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        response = client.put(
            "/api/v1/huggingface-connection",
            headers={"origin": "https://untrusted.example"},
            json={"token": TOKEN},
        )
        assert response.status_code == 403
        response = client.options(
            "/api/v1/huggingface-connection",
            headers={"origin": "http://localhost:3000", "access-control-request-method": "DELETE"},
        )
        assert response.status_code == 200
    identity.assert_not_awaited()
