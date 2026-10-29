"""Model discovery is explicit and never performs inference or exposes credentials."""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from vla_platform.agent_control import model_catalog
from vla_platform.agent_control.settings import Profiles
from vla_platform.api import create_app
from vla_platform.api_connections import ApiConnections
from vla_platform.settings import Settings


def transport(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        model_catalog.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_models_are_listed_only_from_fixed_openai_endpoint(monkeypatch):
    key = "private-example-openai-credential"
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://api.openai.com/v1/models"
        assert request.method == "GET"
        assert request.headers["Authorization"] == "Bearer " + key
        assert not request.content
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-6-astra"},
                    {"id": "gpt-6.1-sol"},
                    {"id": "gpt-6-astra"},
                    {"id": "bad name"},
                    {"id": key},
                ]
            },
        )

    transport(monkeypatch, handler)
    assert asyncio.run(model_catalog.list_openai_models(key)) == ["gpt-6-astra", "gpt-6.1-sol"]
    assert len(requests) == 1


@pytest.mark.parametrize(
    "status, body",
    [
        (401, b"private-example-key"),
        (302, b""),
        (200, b"not json"),
        (200, b'{"data":null}'),
        (200, b"x" * (1024 * 1024 + 1)),
    ],
)
def test_failed_discovery_is_bounded_and_sanitized(monkeypatch, status, body):
    transport(monkeypatch, lambda request: httpx.Response(status, content=body))
    with pytest.raises(model_catalog.ModelCatalogError) as error:
        asyncio.run(model_catalog.list_openai_models("private-example-key"))
    assert "private-example-key" not in str(error.value)
    assert "model ID manually" in str(error.value)


def test_catalog_route_requires_saved_key_and_is_not_called_when_listing_profiles(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []

    async def listed(key):
        calls.append(key)
        return ["gpt-6-astra"]

    monkeypatch.setattr(model_catalog, "list_openai_models", listed)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/api/v1/agent/providers").status_code == 200
        assert not calls
        response = client.get("/api/v1/agent/openai/models")
        assert response.status_code == 409
        monkeypatch.setattr(Profiles, "integration_installed", lambda self: True)
        response = client.get("/api/v1/agent/openai/models")
        assert response.status_code == 422
        assert not calls
        key = "private-example-openai-credential"
        ApiConnections(tmp_path).save("openai", SecretStr(key))
        response = client.get("/api/v1/agent/openai/models")
        assert response.status_code == 200
        assert response.json() == ["gpt-6-astra"]
        assert key not in response.text
        assert calls == [key]
