"""Offline full provider adapter + fixed executor transport; no network inference."""

import asyncio
import json
import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from firebird_teaching import intelligence as mod
from firebird_teaching import intelligence_config as config
from firebird_teaching.frame_snapshot import ObservationSnapshot
from firebird_teaching.providers import ProviderError, Providers


class Stream(httpx.AsyncByteStream):
    def __init__(self, value):
        self.value = json.dumps(value).encode()

    async def __aiter__(self):
        yield self.value


def response(status, *, json):
    return httpx.Response(status, headers={"Content-Type": "application/json"}, stream=Stream(json))


KEY = "fixture-openrouter-key-never-real"
PAYLOAD = dict(
    request_id="request-1",
    session_id="session",
    episode_id=None,
    expected_revision=2,
    prompt="What should the operator review?",
    consent=True,
)
STATE = dict(
    session_id="session",
    episode_id=None,
    revision=2,
    mode="idle",
    steps=0,
    sim_time=0.0,
    instruction="Review the generated scene",
)


def setup(reply=None):
    seen = []
    state = STATE.copy()
    frame = ObservationSnapshot(
        "session",
        2,
        None,
        "idle",
        "preview-one",
        0,
        0.0,
        "observation.images.front",
        "/World/Camera",
        time.monotonic_ns(),
        2,
        2,
        bytes([1, 2, 3]) * 4,
        ("joint",),
        (0.0,),
    )
    frames = [frame]

    async def transport(request):
        seen.append(request)
        if request.url.host == "127.0.0.1":
            assert request.method == "GET"
            assert request.headers["Authorization"] == "Bearer control-token"
            if request.url.path == "/state":
                return response(200, json=state)
            assert request.url.path == "/frame"
            value = frames[0]
            return response(
                200,
                json=value.payload(
                    value.context, value.observation_received_monotonic_ns, time.monotonic_ns()
                ),
            )
        assert request.url.host == "openrouter.ai"
        if reply:
            return await reply(request, state, frames)
        if request.url.path.endswith("/decisions"):
            return response(
                200,
                json={
                    "model": "typesafe/jev-1.13",
                    "answers": {
                        "selection": {
                            "type": "choice",
                            "choice": "review_instruction",
                            "confidence": 0.7,
                            "probabilities": {"review_instruction": 0.8, "inspect_camera": 0.2},
                        }
                    },
                },
            )
        return response(
            200,
            json={
                "model": "perceptron/perceptron-mk1.5",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Generated pixels only.",
                                    "uncertain": True,
                                    "observations": [],
                                }
                            )
                        },
                    }
                ],
            },
        )

    mock = httpx.MockTransport(transport)
    service = mod.IntelligenceService(
        SimpleNamespace(origin="http://127.0.0.1:1", token="control-token"),
        config_loader=lambda: config.IntelligenceConfig(
            KEY, "provider/chat", "environment", "environment", None
        ),
        provider_factory=lambda settings: Providers(settings, transport=mock),
        transport=mock,
    )
    return service, seen, state, frames


@pytest.mark.parametrize("kind", ["decision", "perception"])
def test_real_adapters_return_bound_advisory_only_receipt(kind):
    service, seen, _, _ = setup()
    result = service.run(kind, PAYLOAD)
    assert result["kind"] == kind and result["advisory_only"] is True
    assert result["proposal"]["receipt"]["context"] == dict(
        session_id="session", episode_id=None, revision=2
    )
    assert KEY not in json.dumps(result)
    assert len([r for r in seen if r.method == "POST"]) == 1
    assert all("/commands" not in str(r.url) for r in seen)
    with pytest.raises(ProviderError, match="already attempted"):
        service.run(kind, PAYLOAD)
    assert service.status()["busy"] is False


@pytest.mark.parametrize(
    "changes",
    [
        dict(consent=1),
        dict(prompt=" " * 513 + "x"),
        dict(expected_revision=True),
        dict(url="https://other"),
        dict(session_id="session\n"),
    ],
)
def test_bad_requests_never_reach_provider(changes):
    service, seen, _, _ = setup()
    with pytest.raises(ProviderError):
        service.run("decision", {**PAYLOAD, **changes})
    assert not seen


def test_context_change_withholds_result_after_exactly_one_provider_request():
    async def reply(request, state, frames):
        state["revision"] = 3
        return response(
            200,
            json={
                "model": "typesafe/jev-1.13",
                "answers": {
                    "selection": {
                        "type": "choice",
                        "choice": "review_instruction",
                        "confidence": 0.7,
                        "probabilities": {"review_instruction": 0.8, "inspect_camera": 0.2},
                    }
                },
            },
        )

    service, seen, _, _ = setup(reply)
    with pytest.raises(ProviderError) as error:
        service.run("decision", PAYLOAD)
    assert error.value.code == "stale_context"
    assert sum(r.method == "POST" for r in seen) == 1


def test_new_capture_same_context_is_not_promoted_as_current():
    async def reply(request, state, frames):
        frames[0] = replace(frames[0], step=1)
        return response(
            200,
            json={
                "model": "perceptron/perceptron-mk1.5",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"summary": "Old pixels", "uncertain": True, "observations": []}
                            )
                        },
                    }
                ],
            },
        )

    service, seen, _, _ = setup(reply)
    with pytest.raises(ProviderError) as error:
        service.run("perception", PAYLOAD)
    assert error.value.code == "stale_frame"
    assert sum(r.method == "POST" for r in seen) == 1


def test_expired_frame_fails_before_provider():
    service, seen, _, frames = setup()
    frames[0] = replace(
        frames[0], observation_received_monotonic_ns=time.monotonic_ns() - 6_000_000_000
    )
    with pytest.raises(ProviderError):
        service.run("perception", PAYLOAD)
    assert not any(r.method == "POST" for r in seen)


def test_explicit_cancel_busy_and_wrong_session_do_not_create_second_request():
    entered, stopped = threading.Event(), threading.Event()

    async def reply(*_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    service, seen, _, _ = setup(reply)
    errors = []

    def run():
        try:
            service.run("decision", PAYLOAD)
        except ProviderError as e:
            errors.append(e.code)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(3)
        assert (
            service.cancel(dict(request_id="request-1", session_id="other"))["status"]
            == "not_active"
        )
        with pytest.raises(ProviderError) as error:
            service.run("decision", {**PAYLOAD, "request_id": "request-2"})
        assert error.value.code == "busy"
        assert (
            service.cancel(dict(request_id="request-1", session_id="session"))["status"]
            == "cancellation_requested"
        )
        thread.join(3)
        assert not thread.is_alive() and stopped.is_set() and errors == ["cancelled"]
        assert sum(r.method == "POST" for r in seen) == 1
    finally:
        service.cancel(dict(request_id="request-1", session_id="session"))
        thread.join(3)


def test_whole_deadline_releases_busy_and_does_not_retry(monkeypatch):
    async def reply(*_):
        await asyncio.Event().wait()

    service, seen, _, _ = setup(reply)
    monkeypatch.setattr(mod, "DEADLINE", 0.03)
    with pytest.raises(ProviderError) as error:
        service.run("decision", PAYLOAD)
    assert error.value.code == "timeout" and not service.status()["busy"]
    assert sum(r.method == "POST" for r in seen) == 1


def saved(tmp_path):
    path = tmp_path / "config.json"
    value = dict(
        schema_version=1, revision="a" * 32, openrouter_api_key=KEY, voice_model="provider/chat"
    )
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path, value


def test_config_explicit_opt_in_environment_precedence_and_no_secret_repr(tmp_path, monkeypatch):
    path, _ = saved(tmp_path)
    loaded = config.load_config({config.CONFIG_ENV: str(path)})
    assert loaded.key == KEY and loaded.key_source == "managed_file" and KEY not in repr(loaded)
    other = "fixture-other-secret-key"
    loaded = config.load_config(
        {
            config.CONFIG_ENV: str(path),
            "OPENROUTER_API_KEY": other,
            "FIREBIRD_VOICE_MODEL": "other/chat",
        }
    )
    assert loaded.key == other and loaded.voice_model == "other/chat" and loaded.revision is None
    with pytest.raises(ValueError):
        config.load_config({})
    with pytest.raises(ValueError):
        config.load_config({"OPENROUTER_API_KEY": KEY, "FIREBIRD_VOICE_MODEL": "provider/" + KEY})
    monkeypatch.setenv(config.CONFIG_ENV, str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("FIREBIRD_VOICE_MODEL", "")
    assert config.voice_environment()["OPENROUTER_API_KEY"] == KEY
    assert os.environ["OPENROUTER_API_KEY"] == ""


@pytest.mark.parametrize(
    "malformation", ["symlink", "ancestor", "oversize", "duplicate", "bool", "permissions", "fifo"]
)
def test_private_file_rejects_unsafe_sources(tmp_path, malformation):
    path, value = saved(tmp_path)
    if malformation in ("fifo", "permissions") and os.name != "posix":
        pytest.skip("POSIX filesystem contract")
    if malformation == "symlink":
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    elif malformation == "ancestor":
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        path = link / path.name
    elif malformation == "oversize":
        path.write_bytes(b"x" * 8193)
    elif malformation == "duplicate":
        path.write_text('{"schema_version":1,"schema_version":1}')
    elif malformation == "bool":
        value["schema_version"] = True
        path.write_text(json.dumps(value))
    elif malformation == "permissions":
        path.chmod(0o644)
    else:
        path.unlink()
        os.mkfifo(path, 0o600)
    started = time.monotonic()
    with pytest.raises((ValueError, OSError)):
        config.read_saved(path)
    assert time.monotonic() - started < 1


def test_actual_broker_auth_origin_body_limits_and_fixed_routes():
    from firebird_teaching.voice_join import JoinBroker, serve

    service, seen, _, _ = setup()
    broker = JoinBroker(service.control)
    broker.intelligence = service
    server = serve(broker, "x" * 32, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False, timeout=3
        ) as client:
            assert client.get("/intelligence/status").status_code == 401
            headers = {"Authorization": "Bearer " + "x" * 32}
            assert (
                client.get(
                    "/intelligence/status", headers={**headers, "Origin": "http://localhost"}
                ).status_code
                == 401
            )
            status = client.get("/intelligence/status", headers=headers)
            assert status.status_code == 200 and status.json()["provider_access_verified"] is False
            assert KEY not in status.text and not seen
            assert client.get("/arbitrary", headers=headers).status_code == 404
            assert (
                client.post(
                    "/intelligence/decision", headers=headers, content=b"x" * 8193
                ).status_code
                == 400
            )
            reply = client.post("/intelligence/decision", headers=headers, json=PAYLOAD)
            assert reply.status_code == 200 and reply.json()["advisory_only"] is True
            assert (
                client.post("/intelligence/decision", headers=headers, json=PAYLOAD).status_code
                == 409
            )
            assert sum(r.method == "POST" for r in seen) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_cancellation_acknowledged_after_last_await_cannot_return_advice(monkeypatch):
    service, _, _, _ = setup()
    original = service._run

    async def final_window(kind, payload):
        result = await original(kind, payload)
        assert (
            service.cancel(
                {"request_id": payload["request_id"], "session_id": payload["session_id"]}
            )["status"]
            == "cancellation_requested"
        )
        return result

    monkeypatch.setattr(service, "_run", final_window)
    with pytest.raises(ProviderError) as error:
        service.run("decision", PAYLOAD)
    assert error.value.code == "cancelled"
