"""Private settings and bounded real loopback intelligence relay, no paid inference."""

import asyncio
import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vla_platform import teaching_intelligence_api as api
from vla_platform import teaching_settings as settings
from vla_platform.teaching_api import router

KEY = "fixture-private-openrouter-key"
PAYLOAD = dict(
    request_id="request-1",
    session_id="session",
    episode_id=None,
    expected_revision=2,
    prompt="What should I review?",
    consent=True,
)
CONTEXT = dict(session_id="session", episode_id=None, revision=2)
RESULT = dict(
    schema_version=1,
    request_id="request-1",
    kind="decision",
    advisory_only=True,
    current_at_return=True,
    choices=[
        dict(id="review_instruction", description="Review instruction"),
        dict(id="inspect_camera", description="Inspect camera"),
    ],
    proposal=dict(
        choice="review_instruction",
        confidence=0.7,
        probabilities=dict(review_instruction=0.8, inspect_camera=0.2),
        receipt=dict(
            context=CONTEXT,
            requested_model="typesafe/jev-1.13",
            returned_model="typesafe/jev-1.13",
            request_sha256="a" * 64,
            response_id=None,
            reported_cost_usd=None,
            elapsed_seconds=0.01,
            frame=None,
            generated_proposal=True,
        ),
    ),
)


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    token = tmp_path / "control-token"
    token.write_text("x" * 32)
    token.chmod(0o600)
    record = dict(reply=copy.deepcopy(RESULT), status=200, calls=[])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def handle_request(self):
            assert self.headers["Authorization"] == "Bearer " + "x" * 32
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None
            record["calls"].append((self.path, json.loads(body) if body else None))
            raw = record.get("raw") or json.dumps(record["reply"]).encode()
            self.send_response(record["status"])
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = handle_request
        do_POST = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("FIREBIRD_TEACHING_VOICE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("FIREBIRD_TEACHING_CONTROL_TOKEN_FILE", str(token))
    app = FastAPI()
    app.state.execution = SimpleNamespace(settings=SimpleNamespace(data_dir=tmp_path))
    app.include_router(router)
    with TestClient(app) as client:
        yield client, record, tmp_path
    server.shutdown()
    server.server_close()
    thread.join(2)


def test_settings_private_atomic_replace_reload_and_delete_only_managed_file(app_client):
    client, record, path = app_client
    live = path / "livekit.env"
    live.write_text("private unchanged")
    endpoint = "/api/v1/teaching/intelligence/settings"
    assert not client.get(endpoint).json()["saved"]
    response = client.put(endpoint, json=dict(openrouter_api_key=KEY, voice_model="provider/chat"))
    assert response.status_code == 200 and response.json()["saved"]
    assert KEY not in response.text and "openrouter_api_key" not in response.text
    assert response.headers["cache-control"] == "no-store"
    saved = settings.read(path / settings.FILENAME)
    assert saved["openrouter_api_key"] == KEY
    if os.name == "posix":
        assert (path / settings.FILENAME).stat().st_mode & 0o077 == 0
    assert client.get(endpoint).json()["revision"] == saved["revision"]
    assert not client.delete(endpoint).json()["saved"]
    assert live.read_text() == "private unchanged" and not record["calls"]


@pytest.mark.parametrize(
    "payload",
    [
        dict(openrouter_api_key=KEY, voice_model="typesafe/jev-1.13"),
        dict(openrouter_api_key=KEY, voice_model="provider/" + KEY),
        dict(openrouter_api_key="tiny", voice_model="provider/model"),
        dict(openrouter_api_key=KEY, voice_model="provider/model", url="https://other"),
    ],
)
def test_invalid_credentials_not_echoed_even_in_validation(app_client, payload):
    client, record, path = app_client
    response = client.put("/api/v1/teaching/intelligence/settings", json=payload)
    assert response.status_code == 422 and KEY not in response.text and not record["calls"]
    assert not (path / settings.FILENAME).exists()


def test_oversized_body_rejected_before_parse_or_broker(app_client):
    client, record, _ = app_client
    response = client.post("/api/v1/teaching/intelligence/decision", content=b"x" * 16385)
    assert response.status_code == 413 and not record["calls"]


def test_real_loopback_receipt_is_context_bound_no_retry(app_client):
    client, record, _ = app_client
    response = client.post("/api/v1/teaching/intelligence/decision", json=PAYLOAD)
    assert response.status_code == 200 and response.json() == RESULT
    assert record["calls"] == [("/intelligence/decision", PAYLOAD)]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(request_id="other"),
        lambda v: v.update(schema_version=True),
        lambda v: v.update(current_at_return=1),
        lambda v: v["proposal"]["receipt"]["context"].update(revision=3),
        lambda v: v["proposal"]["receipt"].update(returned_model="other/model"),
        lambda v: v["proposal"].update(
            probabilities={"review_instruction": 1.0, "inspect_camera": 1.0}
        ),
        lambda v: v["proposal"].update(choice="start_robot"),
        lambda v: v["proposal"]["receipt"].update(reported_cost_usd=float("inf")),
    ],
)
def test_malformed_or_mismatched_proposal_withheld(app_client, mutate):
    client, record, _ = app_client
    mutate(record["reply"])
    response = client.post("/api/v1/teaching/intelligence/decision", json=PAYLOAD)
    assert (
        response.status_code == 503
        and sum(path == "/intelligence/decision" for path, _ in record["calls"]) == 1
    )


@pytest.mark.parametrize(
    "code,status",
    [("stale_context", 409), ("stale_frame", 409), ("cancelled", 409), ("rate_limited", 503)],
)
def test_provider_failures_are_sanitized_not_retried(app_client, code, status):
    client, record, _ = app_client
    record.update(status=503, reply={"code": code, "error": KEY})
    response = client.post("/api/v1/teaching/intelligence/decision", json=PAYLOAD)
    assert (
        response.status_code == status
        and KEY not in response.text
        and sum(path == "/intelligence/decision" for path, _ in record["calls"]) == 1
    )


def test_missing_or_malformed_status_does_not_claim_connected(app_client):
    client, record, _ = app_client
    for path in ("/voice/status", "/intelligence/status"):
        value = client.get("/api/v1/teaching" + path).json()
        assert value["broker_reachable"] is False and value["provider_access_verified"] is False


@pytest.mark.parametrize("kind", ["symlink", "fifo", "oversize", "duplicate", "permissions"])
def test_settings_reader_failure_is_bounded_and_secret_free(tmp_path, kind):
    path = tmp_path / settings.FILENAME
    if kind in ("fifo", "permissions") and os.name != "posix":
        pytest.skip("POSIX file contract")
    path.write_text(KEY)
    path.chmod(0o600)
    if kind == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "missing")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    elif kind == "oversize":
        path.write_bytes(b"x" * 8193)
    elif kind == "permissions":
        path.chmod(0o644)
    else:
        path.write_text('{"schema_version":1,"schema_version":1}')
    value = settings.status(tmp_path)
    assert not value["saved"] and KEY not in json.dumps(value)


def test_repeated_cancellation_waits_for_cancel_delivery_and_owned_cleanup(monkeypatch):
    async def check():
        started, cancel_started, release, stopped = [asyncio.Event() for _ in range(4)]
        calls = []

        async def broker(path, payload=None):
            calls.append(path)
            if path.endswith("/cancel"):
                cancel_started.set()
                await release.wait()
                return {}
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        monkeypatch.setattr(api, "broker", broker)

        async def disconnected():
            return False

        request = SimpleNamespace(is_disconnected=disconnected)
        task = asyncio.create_task(api.advise("decision", api.AdviceRequest(**PAYLOAD), request))
        await started.wait()
        task.cancel()
        await cancel_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set() and calls == ["/intelligence/decision", "/intelligence/cancel"]

    asyncio.run(check())


def test_browser_disconnect_requests_cancel_without_second_provider_attempt(monkeypatch):
    async def check():
        started = asyncio.Event()
        calls = []

        async def broker(path, payload=None):
            calls.append(path)
            if path.endswith("/cancel"):
                return {}
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(api, "broker", broker)

        async def disconnected():
            await started.wait()
            return True

        with pytest.raises(asyncio.CancelledError):
            await api.advise(
                "decision",
                api.AdviceRequest(**PAYLOAD),
                SimpleNamespace(is_disconnected=disconnected),
            )
        assert calls == ["/intelligence/decision", "/intelligence/cancel"]

    asyncio.run(check())


def test_actual_app_to_broker_to_provider_adapter_uses_saved_file_and_no_controls(
    tmp_path, monkeypatch
):
    """Real two-hop loopback; only the external HTTPS provider is replaced by transport."""
    import sys
    from pathlib import Path

    import httpx

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "workers/teaching"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "workers/isaac_sim"))
    from firebird_teaching import intelligence_config
    from firebird_teaching.intelligence import IntelligenceService
    from firebird_teaching.providers import Providers
    from firebird_teaching.voice_join import JoinBroker, serve

    settings.save(
        tmp_path,
        settings.IntelligenceSettingsInput(openrouter_api_key=KEY, voice_model="provider/chat"),
    )
    calls = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(
                {
                    "model": "typesafe/jev-1.13",
                    "answers": {
                        "selection": {
                            "type": "choice",
                            "choice": "review_instruction",
                            "confidence": 0.7,
                            "probabilities": {"review_instruction": 0.8, "inspect_camera": 0.2},
                        }
                    },
                }
            ).encode()

    async def transport(request):
        calls.append(str(request.url))
        if request.url.host == "openrouter.ai":
            return httpx.Response(
                200, headers={"Content-Type": "application/json"}, stream=Stream()
            )
        assert request.url.path == "/state" and request.method == "GET"

        class StateStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield json.dumps(
                    {
                        **CONTEXT,
                        "mode": "idle",
                        "steps": 0,
                        "instruction": "Generated scene",
                        "sim_time": 0,
                    }
                ).encode()

        return httpx.Response(200, stream=StateStream())

    transport = httpx.MockTransport(transport)
    control = SimpleNamespace(origin="http://127.0.0.1:1", token="x" * 32)
    service = IntelligenceService(
        control,
        transport=transport,
        provider_factory=lambda cfg: Providers(cfg, transport=transport),
        config_loader=lambda: intelligence_config.load_config(
            {intelligence_config.CONFIG_ENV: str(tmp_path / settings.FILENAME)}
        ),
    )
    broker = JoinBroker(control)
    broker.intelligence = service
    server = serve(broker, "x" * 32, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "control-token"
    token.write_text("x" * 32)
    token.chmod(0o600)
    monkeypatch.setenv("FIREBIRD_TEACHING_CONTROL_TOKEN_FILE", str(token))
    monkeypatch.setenv("FIREBIRD_TEACHING_VOICE_URL", f"http://127.0.0.1:{server.server_port}")
    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            status = client.get("/api/v1/teaching/intelligence/status").json()
            assert status["configured"] and status["credential_source"] == "managed_file"
            assert status["configuration_revision"] == settings.status(tmp_path)["revision"]
            result = client.post("/api/v1/teaching/intelligence/decision", json=PAYLOAD)
            assert (
                result.status_code == 200
                and result.json()["proposal"]["choice"] == "review_instruction"
            )
            assert KEY not in result.text
            assert (
                client.post("/api/v1/teaching/intelligence/decision", json=PAYLOAD).status_code
                == 409
            )
        assert sum("openrouter.ai" in url for url in calls) == 1
        assert not any("/commands" in url for url in calls)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        # This app test deliberately imports optional worker modules without installing them.
        # Other core tests should not depend on that import side effect.
        for name in list(sys.modules):
            if name.startswith("firebird_teaching"):
                sys.modules.pop(name)
