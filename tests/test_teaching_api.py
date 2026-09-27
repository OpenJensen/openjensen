"""Real loopback relay tests; no simulator, microphone or cloud inference claims."""

import base64
import hashlib
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


def frame_fixture():
    # Same exact v1 envelope as the generated teaching worker, with unrelated
    # remote and application monotonic clock epochs.
    rgb = bytes([200, 20, 30] * 4)
    context = {"session_id": "session-1", "revision": 7, "active_episode_id": None, "mode": "idle"}
    return {
        "schema_version": 1,
        "available": True,
        **context,
        "current_context": dict(context),
        "episode_id": "preview-first",
        "step": 0,
        "sim_time": 0.0,
        "camera_key": "observation.images.front",
        "camera_prim": "/World/front",
        "observation_received_monotonic_ns": 2**60,
        "published_monotonic_ns": 2**60 + 100,
        "source_age_ns": 1000,
        "width": 2,
        "height": 2,
        "rgb_base64": base64.b64encode(rgb).decode(),
        "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
        "joints": ["gripper"],
        "state_rad": [0.25],
        "units": "rad",
    }


def test_atomic_frame_relay_preserves_identity_pixels_and_conservative_age(relay_app):
    client, record, _, token = relay_app
    record["reply"] = frame_fixture()
    result = client.get("/api/v1/teaching/frame?session_id=session-1")
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    data = result.json()
    assert {k: data[k] for k in record["reply"]} == record["reply"]
    assert data["capture_id"] == str(2**60)
    assert 1000 <= data["relay_age_ns"] <= 5_000_000_000
    assert record["requests"] == [("/frame", "Bearer " + token)]
    assert client.get("/api/v1/teaching/frame?session_id=other").status_code == 503
    # Optional query preserves existing direct clients, with atomic validation.
    assert client.get("/api/v1/teaching/frame").status_code == 200
    assert client.get("/api/v1/teaching/frame?session_id=bad%0A").status_code == 422


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"available": 1},
        {"width": True},
        {"height": 1921},
        {"rgb_base64": "AAAA"},
        {"rgb_sha256": "a" * 64},
        {"units": "degrees"},
        {"camera_prim": "/World/front\n"},
        {"camera_key": "unexpected"},
        {
            "current_context": {
                "session_id": "session-1",
                "revision": 8,
                "active_episode_id": None,
                "mode": "idle",
            }
        },
        {
            "current_context": {
                "session_id": "new",
                "revision": 7,
                "active_episode_id": None,
                "mode": "idle",
            }
        },
        {"episode_id": "recording-not-preview"},
        {"joints": ["gripper", "gripper"], "state_rad": [0.0, 0.0]},
        {"state_rad": []},
        {"state_rad": [True]},
        {"state_rad": [float("inf")]},
        {"step": -1},
        {"source_age_ns": 5_000_000_001},
        {"source_age_ns": 99},
        {"observation_received_monotonic_ns": 2**60 + 101},
        {"private_path": "/secret"},
    ],
)
def test_invalid_atomic_frames_fail_before_browser_render(relay_app, changes):
    client, record, *_ = relay_app
    record["reply"] = frame_fixture() | changes
    result = client.get("/api/v1/teaching/frame?session_id=session-1")
    assert result.status_code == 503
    assert "secret" not in result.text


def test_unavailable_and_legacy_frames_cannot_look_live(relay_app):
    client, record, *_ = relay_app
    record["reply"] = {"schema_version": 1, "available": False}
    assert client.get("/api/v1/teaching/frame").json() == record["reply"]
    record["reply"] = {"available": False}
    assert client.get("/api/v1/teaching/frame").status_code == 503
    record["reply"] = {
        k: v
        for k, v in frame_fixture().items()
        if k
        in {
            "episode_id",
            "step",
            "sim_time",
            "published_monotonic_ns",
            "width",
            "height",
            "rgb_base64",
        }
    }
    assert client.get("/api/v1/teaching/frame").status_code == 503


def test_relay_elapsed_counts_toward_frame_age():
    from vla_platform.teaching_api import public_frame, unique_object

    with pytest.raises(ValueError, match="Stale"):
        public_frame(frame_fixture(), session_id="session-1", elapsed_ns=5_000_000_000)
    with pytest.raises(ValueError, match="Duplicate"):
        unique_object([("session_id", "old"), ("session_id", "new")])


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


def test_frame_relay_accepts_actual_worker_generated_envelope():
    import importlib.util
    import sys
    from pathlib import Path

    from vla_platform.teaching_api import public_frame

    name = "firebird_test_atomic_frame"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "workers/teaching/firebird_teaching/frame_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        capture = module.ObservationSnapshot(
            session_id="real-protocol",
            revision=3,
            active_episode_id=None,
            mode="idle",
            episode_id="preview-generated",
            step=0,
            sim_time=0.0,
            camera_key="observation.images.front",
            camera_prim="/World/front",
            observation_received_monotonic_ns=2**60,
            width=2,
            height=2,
            rgb=bytes(range(12)),
            joints=("gripper",),
            state_rad=(0.2,),
        )
        source = capture.payload(capture.context, 2**60 + 100, 2**60 + 200)
        result = public_frame(source, session_id="real-protocol", elapsed_ns=300)
        assert {key: result[key] for key in source} == source
        assert result["relay_age_ns"] == 500
        assert result["capture_id"] == str(2**60)
        for unavailable in (
            {"schema_version": True, "available": False},
            {"schema_version": 1, "available": 0},
        ):
            with pytest.raises(ValueError):
                public_frame(unavailable, session_id="real-protocol", elapsed_ns=0)
    finally:
        del sys.modules[name]
