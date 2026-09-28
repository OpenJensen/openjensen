"""Local HTTP admission and cache policy, without sockets or dispatched workers."""

import httpx
import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.settings import Settings


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as value:
        yield value


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Origin": "https://attacker.invalid"}, 403),
        ({"Origin": "null"}, 403),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Origin": "", "Sec-Fetch-Site": "cross-site"}, 403),
        ({"Host": "attacker.invalid", "Origin": "http://attacker.invalid"}, 400),
        ({"Origin": "http://localhost:3000", "Sec-Fetch-Site": "cross-site"}, 201),
        ({"Origin": "http://127.0.0.1:8000"}, 201),
        ({"Host": "localhost:8123", "Origin": "http://localhost:8123"}, 201),
        ({"Sec-Fetch-Site": "same-origin"}, 201),
        ({"Sec-Fetch-Site": "same-site"}, 201),
        ({"Sec-Fetch-Site": "none"}, 201),
        ({}, 201),
    ],
)
def test_browser_boundary_precedes_mutation_and_preserves_cli(client, headers, status):
    response = client.post("/api/v1/projects", json={"name": "Boundary"}, headers=headers)
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    projects = client.get("/api/v1/projects").json()
    assert len(projects) == (1 if status == 201 else 0)
    if status == 201:
        assert projects == [response.json()]


def test_originless_cross_site_read_is_refused(client):
    client.post("/api/v1/projects", json={"name": "Private project"})
    response = client.get("/api/v1/projects", headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Origin is not allowed for the local application"}
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "method,path,payload,status",
    [
        ("GET", "/health", None, 200),
        ("GET", "/projects", None, 200),
        ("GET", "/jobs/missing", None, 404),
        ("GET", "/projects/missing/jobs", None, 404),
        ("POST", "/jobs/missing/cancel", None, 404),
        ("POST", "/projects", {"name": " "}, 422),
        ("GET", "/not-a-route", None, 404),
        ("DELETE", "/health", None, 405),
    ],
)
def test_api_success_and_handled_errors_are_not_cacheable(client, method, path, payload, status):
    response = client.request(method, "/api/v1" + path, json=payload)
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"


def test_cors_preflight_keeps_current_allowed_headers(client):
    response = client.options(
        "/api/v1/projects",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,idempotency-key",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "idempotency-key" in response.headers["access-control-allow-headers"].lower()
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "error,status", [(httpx.ConnectError("offline"), 502), (TimeoutError(), 504)]
)
def test_upstream_resolution_errors_are_uncached_without_a_job(client, monkeypatch, error, status):
    async def fail(_):
        raise error

    monkeypatch.setattr("vla_platform.datasets.cache.resolve_hub_revision", fail)
    project = client.post("/api/v1/projects", json={"name": "Failed resolution"}).json()
    response = client.post(
        f"/api/v1/projects/{project['id']}/intakes", json={"repo_id": "fixture/data"}
    )
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    assert client.get(f"/api/v1/projects/{project['id']}/jobs").json() == []


def test_saved_submission_missing_receipt_keeps_key_and_no_store(client):
    project = client.post("/api/v1/projects", json={"name": "Recovery"}).json()
    key = "boundary-request-12345678"
    response = client.get(
        f"/api/v1/projects/{project['id']}/submissions/{key}",
        params={"operation": "dataset.inspect"},
    )
    assert response.status_code == 404
    assert response.headers["idempotency-key"] == key
    assert response.headers["cache-control"] == "no-store"


def test_api_cache_policy_does_not_change_static_assets(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "asset.txt").write_text("fixture asset")
    with TestClient(create_app(Settings(data_dir=tmp_path / "state", static_dir=static))) as client:
        response = client.get("/asset.txt")
        assert response.status_code == 200
        assert response.text == "fixture asset"
        assert response.headers.get("cache-control") != "no-store"
        assert "etag" in response.headers
