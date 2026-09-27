"""Public gateway boundary tests; no cloud resources or real network are used."""

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient

SPEC = importlib.util.spec_from_file_location(
    "firebird_public_gateway", Path(__file__).resolve().parents[1] / "deploy/xbox/public_gateway.py"
)
gateway = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gateway
SPEC.loader.exec_module(gateway)
ORIGIN = "https://firebird.example.ts.net"
PASSWORD = "fixture-only-not-a-real-password"
AUTH = "Basic " + base64.b64encode(f"firebird:{PASSWORD}".encode()).decode()


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def config(tmp_path):
    static = tmp_path / "web"
    (static / "docs").mkdir(parents=True)
    (static / "_next").mkdir()
    (static / "index.html").write_text("<h1>Firebird</h1>")
    (static / "docs/index.html").write_text("<h1>API reference</h1>")
    (static / "_next/app.js").write_text("fixture static asset")
    value = {
        "public_origin": ORIGIN,
        "static_dir": str(static),
        "username": "firebird",
        "password_salt_hex": "ab" * 16,
        "password_hash_hex": hashlib.pbkdf2_hmac(
            "sha256", PASSWORD.encode(), b"\xab" * 16, 210000
        ).hex(),
        "password_iterations": 210000,
        "preview_origins": ["http://127.0.0.1:18097"],
    }
    return gateway.GatewayConfig.from_value(value), value


@pytest.mark.parametrize(
    "path",
    [
        "/firebird/",
        "/firebird/docs/",
        "/firebird/_next/app.js",
        "/firebird/api/v1/health",
        "/firebird/openapi.json",
        "/outside",
    ],
)
def test_every_path_requires_auth_before_static_or_upstream(config, path):
    def forbidden(_request):
        raise AssertionError("Unauthenticated requests must not reach the upstream")

    with TestClient(
        gateway.PublicGateway(config[0], transport=httpx.MockTransport(forbidden)), base_url=ORIGIN
    ) as client:
        response = client.get(path)
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Basic")
        assert response.headers["cache-control"] == "no-store"
        response = client.head(path)
        assert response.status_code == 401 and response.content == b""


def test_public_demo_serves_real_history_metrics_and_assets_without_login(config):
    seen = []

    def upstream(request):
        seen.append(request.url.path)
        return httpx.Response(
            200,
            stream=Chunks(b'{"value": "saved backend result"}'),
            headers={"content-type": "application/json"},
        )

    public = gateway.GatewayConfig.from_value({**config[1], "public_readonly": True})
    with TestClient(
        gateway.PublicGateway(public, transport=httpx.MockTransport(upstream)), base_url=ORIGIN
    ) as client:
        for path in [
            "/",
            "/docs/",
            "/_next/app.js",
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/projects",
            "/api/v1/policy-options",
            "/api/v1/capabilities",
            "/api/v1/projects/project-1/jobs",
            "/api/v1/projects/project-1/artifacts",
            "/api/v1/jobs/job-1",
            "/api/v1/jobs/job-1/training",
            "/api/v1/jobs/job-1/events",
            "/api/v1/jobs/job-1/episodes/0",
        ]:
            response = client.get("/firebird" + path)
            assert response.status_code == 200, path
            assert "www-authenticate" not in response.headers
            assert response.headers["cache-control"] == "no-store"
        assert client.get("/firebird/api/v1/jobs/job-1/training").json() == {
            "value": "saved backend result"
        }
        assert "/api/v1/jobs/job-1/training" in seen


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/v1/projects/p/policy-jobs"),
        ("POST", "/api/v1/jobs/j/cancel"),
        ("POST", "/api/v1/projects"),
        ("PUT", "/api/v1/huggingface-connection"),
        ("DELETE", "/api/v1/cloud-connections/gcp"),
        ("GET", "/api/v1/cloud-connections"),
        ("GET", "/api/v1/cloud-runs"),
        ("GET", "/api/v1/compute-settings"),
        ("GET", "/api/v1/huggingface-connection"),
        ("GET", "/api/v1/projects/p/artifacts/a/download"),
        ("GET", "/api/v1/jobs/j/training/reproducibility"),
        ("GET", "/api/v1/teaching/state"),
        ("GET", "/api/v1/unknown-future-route"),
    ],
)
def test_public_demo_blocks_writes_accounts_downloads_and_unknown_apis(config, method, path):
    def forbidden(_request):
        raise AssertionError("Protected requests must not reach the backend")

    public = gateway.GatewayConfig.from_value({**config[1], "public_readonly": True})
    with TestClient(
        gateway.PublicGateway(public, transport=httpx.MockTransport(forbidden)), base_url=ORIGIN
    ) as client:
        response = client.request(method, "/firebird" + path, headers={"Origin": ORIGIN})
        assert response.status_code == 403
        assert "read-only" in response.json()["detail"]
        assert "www-authenticate" not in response.headers


def test_public_demo_preserves_authenticated_owner_writes_and_origin_checks(config):
    received = []

    def upstream(request):
        received.append(request)
        return httpx.Response(
            200, stream=Chunks(b'{"owner": true}'), headers={"content-type": "application/json"}
        )

    public = gateway.GatewayConfig.from_value({**config[1], "public_readonly": True})
    with TestClient(
        gateway.PublicGateway(public, transport=httpx.MockTransport(upstream)),
        base_url=ORIGIN,
        headers={"Authorization": AUTH},
    ) as client:
        assert client.get("/firebird/api/v1/cloud-connections").status_code == 200
        assert client.post("/firebird/api/v1/projects").status_code == 403
        assert (
            client.post("/firebird/api/v1/projects", headers={"Origin": ORIGIN}).status_code == 200
        )
        assert len(received) == 2
        assert all("authorization" not in request.headers for request in received)


@pytest.mark.parametrize("value", ["true", 1, None, []])
def test_public_access_requires_explicit_boolean(config, value):
    with pytest.raises(ValueError, match="Invalid gateway configuration"):
        gateway.GatewayConfig.from_value({**config[1], "public_readonly": value})


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer wrong",
        "Basic !!!",
        "Basic YQ==",
        "Basic " + base64.b64encode(b"firebird:wrong").decode(),
        "Basic " + base64.b64encode(f"wrong:{PASSWORD}".encode()).decode(),
    ],
)
def test_malformed_or_wrong_auth_rejected_without_details(config, authorization):
    with TestClient(gateway.PublicGateway(config[0]), base_url=ORIGIN) as client:
        response = client.get("/firebird/", headers={"Authorization": authorization})
        assert response.status_code == 401
        assert response.text == "Authentication required"
        assert PASSWORD not in response.text


def test_static_prefix_redirects_assets_and_path_containment(config, tmp_path):
    (tmp_path / "private.txt").write_text("must never be served")
    (config[0].static_dir / "escape.txt").symlink_to(tmp_path / "private.txt")
    with TestClient(
        gateway.PublicGateway(config[0]), base_url=ORIGIN, headers={"Authorization": AUTH}
    ) as client:
        assert client.get("/firebird/").text == "<h1>Firebird</h1>"
        assert client.get("/firebird/docs/").text == "<h1>API reference</h1>"
        assert client.get("/firebird/_next/app.js").text == "fixture static asset"
        assert (
            client.get("/firebird/docs", follow_redirects=False).headers["location"]
            == "/firebird/docs/"
        )
        assert client.get("/firebird", follow_redirects=False).headers["location"] == "/firebird/"
        for path in ("/outside", "/firebird/escape.txt", "/firebird/%2e%2e/private.txt"):
            assert client.get(path).status_code == 404
        assert (
            client.get(
                "/firebird/",
                headers={"Host": "evil.example", "X-Forwarded-Host": "firebird.example.ts.net"},
            ).status_code
            == 400
        )


def test_http_funnel_hop_with_public_host_never_redirects_to_http_or_loopback(config):
    with TestClient(
        gateway.PublicGateway(config[0]),
        base_url="http://127.0.0.1:8097",
        headers={"Authorization": AUTH, "Host": "firebird.example.ts.net"},
    ) as client:
        response = client.get("/firebird/docs", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/firebird/docs/"


def test_cached_valid_auth_bypasses_slow_invalid_verification_and_excess_attempts_do_not_queue(
    config, monkeypatch
):
    async def exercise():
        app = gateway.PublicGateway(config[0])
        valid = {"headers": [(b"authorization", AUTH.encode())]}
        invalid = {"headers": [(b"authorization", b"Basic " + base64.b64encode(b"firebird:wrong"))]}
        assert await app.authenticated(valid)
        started, release = asyncio.Event(), asyncio.Event()
        slow_calls = 0

        async def slow_verification(_function, *_args):
            nonlocal slow_calls
            slow_calls += 1
            started.set()
            await release.wait()
            return b"\x00" * 32

        monkeypatch.setattr(gateway.asyncio, "to_thread", slow_verification)
        pending = asyncio.create_task(app.authenticated(invalid))
        try:
            await asyncio.wait_for(started.wait(), 1)
            assert await asyncio.wait_for(app.authenticated(valid), 1)
            assert not await asyncio.wait_for(app.authenticated(invalid), 1)
            assert slow_calls == 1
            assert not pending.done()
        finally:
            release.set()
            assert not await pending

    asyncio.run(exercise())


def test_proxy_origin_boundary_request_sanitization_queries_and_internal_redirect(config):
    received = []

    async def upstream(request):
        received.append((request, await request.aread()))
        return httpx.Response(
            307,
            headers={
                "Location": "http://127.0.0.1:8096/api/v1/result?part=2",
                "Connection": "x-private",
                "X-Private": "remove",
                "Set-Cookie": "private=remove",
                "Content-Length": "0",
            },
            stream=Chunks(),
        )

    with TestClient(
        gateway.PublicGateway(config[0], transport=httpx.MockTransport(upstream)),
        base_url=ORIGIN,
        headers={"Authorization": AUTH},
    ) as client:
        for headers in (
            {},
            {"Origin": "https://evil.example"},
            {"Origin": ORIGIN, "Sec-Fetch-Site": "cross-site"},
        ):
            assert client.post("/firebird/api/v1/jobs", headers=headers).status_code == 403
        assert not received
        response = client.post(
            "/firebird/api/v1/jobs?name=a%2Fb&x=1",
            content=b"streamed-input",
            headers={
                "Origin": ORIGIN,
                "Content-Type": "application/json",
                "Cookie": "other-app=private",
                "Connection": "x-private",
                "X-Private": "remove",
                "Forwarded": "host=evil.example",
                "X-Forwarded-Host": "evil.example",
            },
            follow_redirects=False,
        )
        assert response.status_code == 307
        assert response.headers["location"] == "/firebird/api/v1/result?part=2"
        assert "set-cookie" not in response.headers and "x-private" not in response.headers
        request, body = received[-1]
        assert str(request.url) == "http://127.0.0.1:8096/api/v1/jobs?name=a%2Fb&x=1"
        assert request.headers["host"] == "127.0.0.1:8096"
        assert body == b"streamed-input"
        for key in (
            "authorization",
            "cookie",
            "origin",
            "forwarded",
            "x-forwarded-host",
            "x-private",
        ):
            assert key not in request.headers
        assert client.get("/firebird/openapi.json", follow_redirects=False).status_code == 307
        assert received[-1][0].url.path == "/openapi.json"
        assert (
            client.post(
                "/firebird/api/v1/jobs",
                headers={"Origin": "http://127.0.0.1:18097"},
                follow_redirects=False,
            ).status_code
            == 307
        )


def test_proxy_response_streams_each_chunk_before_requesting_next_and_closes(config):
    delivered = []

    class VerifiedStream(Chunks):
        async def __aiter__(self):
            yield b"first"
            assert delivered == [b"first"]
            yield b"second"

    stream = VerifiedStream()

    async def upstream(_request):
        return httpx.Response(
            206,
            headers={
                "Content-Range": "bytes 0-10/20",
                "Content-Length": "11",
                "Content-Type": "video/mp4",
            },
            stream=stream,
        )

    async def exercise():
        app = gateway.PublicGateway(config[0], transport=httpx.MockTransport(upstream))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": "/firebird/api/v1/video",
            "raw_path": b"/firebird/api/v1/video",
            "query_string": b"",
            "root_path": "",
            "server": ("firebird.example.ts.net", 443),
            "headers": [
                (b"host", b"firebird.example.ts.net"),
                (b"authorization", AUTH.encode()),
                (b"range", b"bytes=0-10"),
            ],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                assert message["status"] == 206
            elif message.get("body"):
                delivered.append(message["body"])

        async with app.lifespan(None):
            await app(scope, receive, send)

    asyncio.run(exercise())
    assert delivered == [b"first", b"second"]
    assert stream.closed


@pytest.mark.skipif(os.name != "posix", reason="Xbox gateway configuration uses POSIX permissions")
def test_config_requires_private_file_and_rejects_arbitrary_preview_origin(config, tmp_path):
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(config[1]))
    path.chmod(0o600)
    assert gateway.GatewayConfig.load(path).username == "firebird"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        gateway.GatewayConfig.load(path)
    path.chmod(0o600)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(OSError):
        gateway.GatewayConfig.load(alias)
    with pytest.raises(ValueError):
        gateway.GatewayConfig.from_value({**config[1], "preview_origins": ["https://evil.example"]})
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "field,value",
    [
        ("public_origin", 42),
        ("static_dir", []),
        ("password_iterations", True),
        ("password_hash_hex", None),
        ("preview_origins", {}),
        ("preview_origins", [{}]),
    ],
)
def test_malformed_config_field_types_fail_closed(config, field, value):
    with pytest.raises(ValueError, match="Invalid gateway configuration"):
        gateway.GatewayConfig.from_value({**config[1], field: value})


def test_gateway_configuration_fails_closed_without_posix_permissions(tmp_path, monkeypatch):
    # No filesystem call is available on this stand-in: failure must precede open.
    monkeypatch.setattr(gateway, "os", SimpleNamespace(name="nt"))
    with pytest.raises(ValueError, match="requires POSIX"):
        gateway.GatewayConfig.load(tmp_path / "gateway.json")


def test_gateway_rejects_arbitrary_preview_origin_on_every_platform(config):
    with pytest.raises(ValueError, match="Invalid gateway configuration"):
        gateway.GatewayConfig.from_value({**config[1], "preview_origins": ["https://evil.example"]})


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO boundary")
def test_gateway_rejects_fifo_without_blocking_startup(tmp_path):
    fifo = tmp_path / "gateway.json"
    os.mkfifo(fifo, 0o600)
    code = """
import runpy, sys
config = runpy.run_path(sys.argv[1])["GatewayConfig"]
try:
    config.load(sys.argv[2])
except ValueError as exc:
    assert "regular file" in str(exc)
    print("rejected fifo")
else:
    raise AssertionError("FIFO accepted as a credential file")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, gateway.__file__, str(fifo)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rejected fifo"
