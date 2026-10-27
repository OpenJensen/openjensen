"""Credential lifecycle without network calls, provider charges or secret readbacks."""

import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from vla_platform.api import create_app
from vla_platform.api_connections import ApiConnections, ApiKeyError
from vla_platform.augmentation.service import Augmentation
from vla_platform.settings import Settings

KEY = "test-provider-key-private-1234567890"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "JENSEN_MODULES", "FIREBIRD_MODULES"):
        monkeypatch.delenv(key, raising=False)


def test_saved_key_is_private_and_status_never_returns_it(tmp_path):
    service = ApiConnections(tmp_path)
    assert not service.provider("gemini").configured
    result = service.save("gemini", SecretStr(KEY))
    assert result.configured and result.source == "saved"
    assert KEY not in service.status().model_dump_json()
    assert ApiConnections(tmp_path).key("gemini") == KEY
    assert os.stat(tmp_path / "api-credentials/gemini.json").st_mode & 0o077 == 0
    assert not service.disconnect("gemini").configured


def test_saved_key_overrides_environment_and_removing_reveals_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    service = ApiConnections(tmp_path)
    assert service.provider("openai").source == "environment"
    other = KEY + "-replacement"
    service.save("openai", SecretStr(other))
    assert service.key("openai") == other
    assert service.disconnect("openai").source == "environment"
    assert service.key("openai") == KEY


def test_key_write_requires_installed_integration_and_survives_restart(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        missing = client.put("/api/v1/api-connections/openai", json={"api_key": KEY})
        assert missing.status_code == 409 and KEY not in missing.text
        assert client.post("/api/v1/modules/openai/install").status_code == 200
        result = client.put("/api/v1/api-connections/openai", json={"api_key": KEY})
        assert result.status_code == 200 and result.json()["configured"]
        assert result.headers["cache-control"] == "no-store" and KEY not in result.text
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        response = client.get("/api/v1/api-connections")
        assert response.headers["cache-control"] == "no-store"
        assert KEY not in response.text and any(
            item["configured"] for item in response.json()["providers"]
        )
        assert client.delete("/api/v1/modules/openai").status_code == 200
        assert ApiConnections(tmp_path).key("openai") == KEY
        assert client.delete("/api/v1/api-connections/openai").status_code == 409
        assert client.post("/api/v1/modules/openai/install").status_code == 200
        removed = client.delete("/api/v1/api-connections/openai")
        assert removed.status_code == 200 and not removed.json()["configured"]


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": KEY + "\ninvalid"},
        {"api_key": KEY, "unexpected": KEY},
        {"api_key": {"private": KEY}},
        {"api_key": KEY * 200},
    ],
)
def test_validation_responses_never_echo_input(tmp_path, payload):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        result = client.put("/api/v1/api-connections/openai", json=payload)
        assert result.status_code == 422 and KEY not in result.text


def test_untrusted_saved_files_never_become_credentials(tmp_path):
    folder = tmp_path / "api-credentials"
    folder.mkdir()
    target = tmp_path / "external-key"
    target.write_text(json.dumps({"version": 1, "api_key": KEY}))
    (folder / "gemini.json").symlink_to(target)
    service = ApiConnections(tmp_path)
    assert not service.provider("gemini").configured
    with pytest.raises((ApiKeyError, OSError)):
        service.key("gemini")
    with pytest.raises(ApiKeyError):
        service.save("gemini", SecretStr(KEY))
    assert KEY not in service.status().model_dump_json()
    assert target.read_text() == json.dumps({"version": 1, "api_key": KEY})


def test_saved_gemini_key_is_consumed_by_augmentation_without_restart(tmp_path, monkeypatch):
    import vla_platform.augmentation.service as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/available/tool")
    service = ApiConnections(tmp_path)
    augmentation = Augmentation(SimpleNamespace(settings=Settings(data_dir=tmp_path)))
    assert not augmentation.options().configured
    service.save("gemini", SecretStr(KEY))
    assert augmentation.options().configured
    assert augmentation.options().auth_mode == "gemini_api_key"
    assert augmentation.api_key() == KEY
    assert KEY not in augmentation.options().model_dump_json()
    service.disconnect("gemini")
    assert not augmentation.options().configured


@pytest.mark.parametrize("value", [None, 123, [], {"key": KEY}])
def test_malformed_optional_credential_never_prevents_core_startup(tmp_path, value):
    folder = tmp_path / "api-credentials"
    folder.mkdir()
    path = folder / "gemini.json"
    path.write_text(json.dumps({"version": 1, "api_key": value}))
    path.chmod(0o600)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/modules").status_code == 200
        result = client.get("/api/v1/api-connections")
        assert result.status_code == 200 and KEY not in result.text
        assert not next(item for item in result.json()["providers"] if item["id"] == "gemini")[
            "configured"
        ]


def test_uninstalling_gemini_keeps_key_but_disables_augmentation_auth(tmp_path, monkeypatch):
    import vla_platform.augmentation.service as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/available/tool")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        client.post("/api/v1/modules/gemini/install").raise_for_status()
        client.put("/api/v1/api-connections/gemini", json={"api_key": KEY}).raise_for_status()
        augmentation = client.app.state.execution.augmentation
        assert augmentation.options().configured and augmentation.api_key() == KEY
        client.delete("/api/v1/modules/gemini").raise_for_status()
        assert ApiConnections(tmp_path).key("gemini") == KEY
        assert not augmentation.options().configured and augmentation.api_key() is None
        assert "from Modules" in augmentation.options().setup_message
