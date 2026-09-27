"""Real loopback relay tests; no simulator, microphone or cloud inference claims."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vla_platform.teaching_api import router

STATE = {
    "mode": "idle",
    "episode_id": None,
    "revision": 0,
    "session_id": "session-1",
    "instruction": "Move gripper",
    "sim_time": 0.0,
    "joints": ["gripper"],
}
RECEIPT = {
    "command_id": "command-1",
    "status": "queued",
    "request_sha256": "a" * 64,
    "received_monotonic_ns": 123,
}


@pytest.fixture
def relay_app(tmp_path, monkeypatch):
    token = "private-test-token-" + "a" * 32
    path = tmp_path / "token"
    path.write_text(token)
    path.chmod(0o600)
    record = {"reply": STATE.copy(), "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self):
            record["requests"].append((self.path, self.headers.get("Authorization")))
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                record["body"] = json.loads(self.rfile.read(length))
            if record.get("slow"):
                self.send_response(200)
                self.send_header("Content-Length", "1002")
                self.end_headers()
                try:
                    for _ in range(1000):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.02)
                    self.wfile.write(b"{}")
                except OSError:
                    pass
                return
            raw = json.dumps(record["reply"]).encode()
            self.send_response(record["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = respond
        do_POST = respond

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("FIREBIRD_TEACHING_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("FIREBIRD_TEACHING_VOICE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("FIREBIRD_TEACHING_CONTROL_TOKEN_FILE", str(path))
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client, record, path, token
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_authenticated_state_sanitizes_private_fields(relay_app):
    client, record, _, token = relay_app
    record["reply"].update(secret="never-return", fault="private/path/key.json")
    response = client.get("/api/v1/teaching/state")
    assert response.status_code == 200 and response.json()["connected"]
    assert "private/path" not in response.text and "never-return" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert record["requests"] == [("/state", "Bearer " + token)]


@pytest.mark.parametrize(
    "reply", [{"mode": "idle", "revision": "wrong"}, {"mode": "running", "revision": -1}, []]
)
def test_invalid_state_is_disconnected(relay_app, reply):
    client, record, *_ = relay_app
    record["reply"] = reply
    assert client.get("/api/v1/teaching/state").json()["connected"] is False


@pytest.mark.parametrize(
    "origin", ["https://example.com", "http://localhost:8768", "http://127.0.0.1:8768/path"]
)
def test_external_or_ambiguous_origin_never_contacted(relay_app, monkeypatch, origin):
    client, record, *_ = relay_app
    monkeypatch.setenv("FIREBIRD_TEACHING_URL", origin)
    assert not client.get("/api/v1/teaching/state").json()["connected"]
    assert not record["requests"]


def test_command_identity_and_arguments_are_bound(relay_app):
    client, record, *_ = relay_app
    payload = {
        "command_id": "command-1",
        "session_id": "session-1",
        "episode_id": None,
        "expected_revision": 0,
        "operation": "correct",
        "arguments": {"joint": "gripper", "delta_rad": 0.05},
    }
    record["reply"] = RECEIPT.copy()
    response = client.post("/api/v1/teaching/commands", json=payload)
    assert response.status_code == 202 and record["body"] == payload
    record["reply"]["command_id"] = "unrelated"
    assert client.post("/api/v1/teaching/commands", json=payload).status_code == 503
    payload["arguments"]["delta_rad"] = 1
    assert client.post("/api/v1/teaching/commands", json=payload).status_code == 422


def test_rejection_never_exposes_worker_error(relay_app):
    client, record, *_ = relay_app
    record["reply"] = {**RECEIPT, "status": "rejected", "error": "private-token"}
    response = client.get("/api/v1/teaching/commands/command-1")
    assert response.json()["status"] == "rejected" and "private-token" not in response.text


@pytest.mark.parametrize("status", [302, 401, 500])
def test_upstream_error_is_bounded_and_sanitized(relay_app, status):
    client, record, *_ = relay_app
    record.update(status=status, reply={"detail": "private-provider-token"})
    response = client.get("/api/v1/teaching/frame")
    assert response.status_code in (409, 503)
    assert "private-provider-token" not in response.text


def test_invalid_voice_response_fails_closed(relay_app):
    client, record, *_ = relay_app
    record["reply"] = {"status": "room_created", "session_id": "session-1"}
    assert (
        client.post("/api/v1/teaching/voice/join", json={"session_id": "session-1"}).status_code
        == 503
    )
    record["reply"].update(
        room="firebird-teaching-" + "a" * 32,
        url="wss://example.livekit.cloud",
        token="a" * 50,
        operator_identity="operator-" + "b" * 32,
        expires_in_seconds=300,
        agent_name="firebird-teaching",
    )
    result = client.post("/api/v1/teaching/voice/join", json={"session_id": "session-1"})
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    assert (
        client.post("/api/v1/teaching/voice/join", json={"session_id": "other"}).status_code == 503
    )


def test_frame_size_and_encoding_are_validated(relay_app):
    client, record, *_ = relay_app
    record["reply"] = {"available": False}
    assert client.get("/api/v1/teaching/frame").json() == {"available": False}
    record["reply"] = {
        "width": 1,
        "height": 1,
        "rgb_base64": "AAAA",
        "step": 0,
        "episode_id": "episode-1",
        "sim_time": 0.01,
        "published_monotonic_ns": 1,
    }
    assert client.get("/api/v1/teaching/frame").status_code == 200
    record["reply"]["width"] = 20
    assert client.get("/api/v1/teaching/frame").status_code == 503


def test_private_token_required(relay_app):
    client, record, path, _ = relay_app
    path.unlink()
    assert not client.get("/api/v1/teaching/state").json()["connected"]
    assert not record["requests"]


def test_nonfinite_overflow_is_rejected(relay_app):
    client, record, *_ = relay_app
    record["reply"] = {**STATE, "sim_time": float("inf")}
    assert not client.get("/api/v1/teaching/state").json()["connected"]
    from vla_platform.teaching_api import finite_float

    with pytest.raises(ValueError):
        finite_float("1e309")


def test_malformed_receipt_timestamp_is_rejected(relay_app):
    client, record, *_ = relay_app
    record["reply"] = {**RECEIPT, "received_monotonic_ns": {"private": "unexpected"}}
    assert client.get("/api/v1/teaching/commands/command-1").status_code == 503


def test_slow_trickle_obeys_total_deadline(relay_app, monkeypatch):
    from vla_platform import teaching_api

    client, record, *_ = relay_app
    record["slow"] = True
    monkeypatch.setattr(teaching_api, "RELAY_DEADLINE", 0.15)
    start = time.monotonic()
    response = client.get("/api/v1/teaching/state")
    assert response.json()["connected"] is False
    assert time.monotonic() - start < 1.5
