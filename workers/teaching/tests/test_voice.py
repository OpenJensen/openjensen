import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from firebird_teaching.credentials import control_token
from firebird_teaching.livekit_agent import (
    SessionClient,
    VoiceSettings,
    build_agent,
    validate_dispatch,
)
from firebird_teaching.voice_join import JoinBroker


@pytest.fixture
def voice_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("FIREBIRD_", "LIVEKIT_", "OPENROUTER_")):
            monkeypatch.delenv(name)
    data = {
        "FIREBIRD_TEACHING_CONTROL_URL": "http://127.0.0.1:8768",
        "FIREBIRD_TEACHING_CONTROL_TOKEN": "x" * 32,
        "FIREBIRD_VOICE_MODEL": "provider/explicit-model",
        "OPENROUTER_API_KEY": "private-openrouter-key",
        "LIVEKIT_URL": "wss://fixture.livekit.cloud",
        "LIVEKIT_API_KEY": "private-key",
        "LIVEKIT_API_SECRET": "private-secret-long-enough-for-sha256",
    }
    for name, value in data.items():
        monkeypatch.setenv(name, value)
    return data


def test_openrouter_default_and_missing_key_no_fallback(voice_env, monkeypatch):
    cfg = VoiceSettings.from_env()
    assert cfg.mode == "openrouter" and cfg.model == "provider/explicit-model"
    assert "private" not in repr(cfg) and "x" * 32 not in repr(cfg)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        VoiceSettings.from_env()


def test_decision_provider_is_not_generic_chat(voice_env, monkeypatch):
    monkeypatch.setenv("FIREBIRD_VOICE_MODEL", "typesafe/jev-1.13")
    with pytest.raises(ValueError, match="typed decision"):
        VoiceSettings.from_env()


@pytest.mark.parametrize(
    "room,meta",
    [
        ("vizi-teleop-dev", {"operator_identity": "operator-a", "teaching_session_id": "session"}),
        (
            "firebird-teaching-test",
            {"operator_identity": "stranger", "teaching_session_id": "session"},
        ),
        (
            "firebird-teaching-test",
            {
                "operator_identity": "operator-a",
                "teaching_session_id": "session",
                "commands": ["move"],
            },
        ),
    ],
)
def test_room_scope_rejects_existing_pocs_and_bad_dispatch(room, meta):
    with pytest.raises(ValueError):
        validate_dispatch(room, json.dumps(meta))


def test_exact_room_scope():
    data = {"operator_identity": "operator-a", "teaching_session_id": "session"}
    assert validate_dispatch("firebird-teaching-test", json.dumps(data)) == data


def test_actual_livekit_tools_use_only_typed_executor():
    class Client:
        def __init__(self):
            self.calls = []

        def request(self, path):
            return {"mode": "idle", "joints": ["joint_0"]}

        def command(self, operation, arguments):
            self.calls.append((operation, arguments))
            return {"status": "queued"}

    client = Client()
    agent = build_agent(client)
    assert len(agent.tools) == 8

    async def check():
        assert (await agent.correct_joint("joint_0", 0.1))["status"] == "queued"
        assert (await agent.pause_demonstration())["status"] == "queued"

    asyncio.run(check())
    assert client.calls == [("correct", {"joint": "joint_0", "delta_rad": 0.1}), ("pause", {})]


def test_session_restart_invalidates_room(monkeypatch):
    monkeypatch.setattr(
        "firebird_teaching.control.Client.request", lambda *args, **kwargs: {"session_id": "new"}
    )
    client = SessionClient("http://127.0.0.1:8768", "x" * 32, "old")
    with pytest.raises(ValueError, match="changed"):
        client.request("/commands", {})


def test_livekit_token_scope_and_room_cleanup(voice_env, monkeypatch):
    import jwt
    from livekit import api

    events = []

    class Service:
        def __init__(self, **kwargs):
            self.room = self
            self.agent_dispatch = self

        async def create_room(self, request):
            events.append(("create", request))

        async def create_dispatch(self, request):
            events.append(("dispatch", request))

        async def delete_room(self, request):
            events.append(("delete", request))

        async def aclose(self):
            events.append(("close", None))

    monkeypatch.setattr(api, "LiveKitAPI", Service)
    client = SimpleNamespace(request=lambda path: {"session_id": "session", "mode": "idle"})
    clock = [0]
    broker = JoinBroker(client, clock=lambda: clock[0])
    result = broker.join({"session_id": "session"})
    claims = jwt.decode(
        result["token"],
        voice_env["LIVEKIT_API_SECRET"],
        algorithms=["HS256"],
        options={"verify_aud": False},
    )
    assert claims["video"]["room"] == result["room"] and result["room"].startswith(
        "firebird-teaching-"
    )
    assert (
        claims["video"]["canPublishSources"] == ["microphone"]
        and claims["video"]["canPublishData"] is False
    )
    assert not claims["video"].get("roomAdmin") and claims["exp"] - claims["nbf"] == 300
    create = events[0][1]
    assert create.max_participants == 2 and create.empty_timeout == 60
    dispatch = events[1][1]
    assert dispatch.agent_name == "firebird-teaching"
    assert json.loads(dispatch.metadata) == {
        "operator_identity": result["operator_identity"],
        "teaching_session_id": "session",
    }
    assert broker.join({"session_id": "session"})["token"] == result["token"]
    clock[0] = 301
    with pytest.raises(ValueError, match="expired"):
        broker.join({"session_id": "session"})


def test_broker_does_not_issue_for_faulted_or_wrong_executor():
    client = SimpleNamespace(request=lambda path: {"session_id": "session", "mode": "faulted"})
    with pytest.raises(ValueError, match="not ready"):
        JoinBroker(client).join({"session_id": "session"})
    client = SimpleNamespace(request=lambda path: {"session_id": "other", "mode": "idle"})
    with pytest.raises(ValueError, match="session changed"):
        JoinBroker(client).join({"session_id": "session"})


def test_private_token_file_and_nonregular_rejection(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREBIRD_TEACHING_CONTROL_TOKEN", raising=False)
    path = tmp_path / "token"
    path.write_text("t" * 32)
    path.chmod(0o600)
    monkeypatch.setenv("FIREBIRD_TEACHING_CONTROL_TOKEN_FILE", str(path))
    assert control_token() == "t" * 32
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        control_token()
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        control_token()


def test_broker_missing_model_key_is_explicit_without_secret_values(voice_env, monkeypatch):
    from firebird_teaching.livekit_agent import VoiceConfigurationError

    monkeypatch.delenv("OPENROUTER_API_KEY")
    client = SimpleNamespace(request=lambda path: {"session_id": "session", "mode": "idle"})
    with pytest.raises(VoiceConfigurationError) as error:
        JoinBroker(client).join({"session_id": "session"})
    assert "OPENROUTER_API_KEY" in str(error.value)
    assert "private" not in str(error.value)
