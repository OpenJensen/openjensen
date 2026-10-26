"""Real planner-process boundary and private-profile regressions without paid requests."""

import asyncio
import json
import os
import sys

import pytest
from vla_platform.agent_control import runtime
from vla_platform.agent_control.contracts import ProviderInput
from vla_platform.agent_control.settings import ProfileError, Profiles


def profile(**changes):
    return {
        "provider": "mock",
        "model": "deterministic-mock-v1",
        "reasoning_effort": "low",
        "api_key": None,
        **changes,
    }


def payload(**changes):
    return {
        "operation": "plan",
        "task": "shoulder_pan +0.03 rad",
        "observation": {"observation_id": "obs-1", "joints": {"shoulder_pan": 0.0}},
        "joint_names": ["shoulder_pan"],
        "timeout_seconds": 5,
        **changes,
    }


def test_actual_mock_subprocess_and_final_receipt_turn():
    async def scenario():
        result = await runtime.run(profile(), payload())
        assert result["status"] == "proposed"
        assert result["arguments"] == {"joint": "shoulder_pan", "delta_rad": 0.03}
        assert result["model"] == "deterministic-mock-v1"
        final = await runtime.run(
            profile(),
            payload(operation="continue", continuation={}, receipt={"status": "replayed"}),
        )
        assert final["status"] == "complete"
        assert final["usage"]["total_tokens"] == 0
        assert final["continuation"] is None

    asyncio.run(scenario())


def test_actual_runtime_rejects_unadmitted_image_without_exposing_credentials():
    key = "private-example-credential"
    with pytest.raises(runtime.RuntimeError) as caught:
        asyncio.run(
            runtime.run(
                profile(provider="openai_responses", model="gpt-6-astra", api_key=key),
                payload(observation={"image_data_url": "https://invalid.example/image"}),
            )
        )
    # Image validation runs before any external request.
    assert key not in str(caught.value)


def test_runtime_invalid_request_is_sanitized():
    key = "private-example-credential"
    with pytest.raises(runtime.RuntimeError) as caught:
        asyncio.run(runtime.run(profile(api_key=key), payload(unsupported=key)))
    assert key not in str(caught.value)


def test_runtime_cancellation_reaps_owned_process_and_filters_environment(tmp_path, monkeypatch):
    observed = tmp_path / "observed.json"
    executable = tmp_path / "planner-stub"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "record = {'pid':os.getpid(), 'environment':dict(os.environ), 'argv':sys.argv}\n"
        f"open({str(observed)!r}, 'w').write(json.dumps(record))\n"
        "time.sleep(30)\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("ROBOT_LAB_BEARER_TOKEN", "must-not-leave-core")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-global-key")
    monkeypatch.setenv("OPENJENSEN_AGENT_API_KEY", "unrelated-agent-key")
    monkeypatch.setattr(runtime.sys, "executable", str(executable))

    async def scenario():
        task = asyncio.create_task(
            runtime.run(profile(provider="openai_responses", api_key="profile-only-key"), payload())
        )
        for _ in range(100):
            if observed.is_file():
                break
            await asyncio.sleep(0.01)
        assert observed.is_file()
        record = json.loads(observed.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(record["pid"], 0)
        env = record["environment"]
        assert "ROBOT_LAB_BEARER_TOKEN" not in env
        assert "OPENAI_API_KEY" not in env
        assert env["OPENJENSEN_AGENT_API_KEY"] == "profile-only-key"
        assert "profile-only-key" not in json.dumps(record["argv"])

    asyncio.run(scenario())


def test_profiles_are_private_and_public_status_contains_no_key(tmp_path):
    profiles = Profiles(tmp_path)
    key = "private-example-openai-credential"
    status = profiles.save("astra", ProviderInput(provider="openai_responses", api_key=key))
    assert key not in status.model_dump_json()
    assert not status.checked
    assert profiles.path.stat().st_mode & 0o777 == 0o600
    revision = status.revision
    checked = profiles.checked("astra", revision, ["text", "images", "tools", "tool_continuation"])
    assert checked.checked
    replaced = profiles.save(
        "astra", ProviderInput(provider="openai_responses", model="gpt-6-astra")
    )
    assert not replaced.checked
    assert replaced.revision != revision
    assert profiles.get("astra")["api_key"] == key
    with pytest.raises(ProfileError):
        profiles.checked("astra", revision, ["text"])
    assert profiles.get("mock")["model"] == "deterministic-mock-v1"


def test_profiles_reject_symlinks_and_invalid_identities(tmp_path):
    profiles = Profiles(tmp_path)
    with pytest.raises(ProfileError):
        profiles.save("../escape", ProviderInput(provider="mock"))
    target = tmp_path / "unrelated.json"
    target.write_text("{}")
    profiles.path.symlink_to(target)
    with pytest.raises(ProfileError):
        profiles.read()
    with pytest.raises(ProfileError):
        profiles.save(
            "astra",
            ProviderInput(provider="openai_responses", api_key="private-example-credential"),
        )
    assert target.read_text() == "{}"


def test_mock_profiles_never_keep_unused_credentials(tmp_path):
    profiles = Profiles(tmp_path)
    status = profiles.save(
        "offline", ProviderInput(provider="mock", api_key="unused-private-credential")
    )
    assert status.model == "deterministic-mock-v1"
    assert profiles.get("offline")["api_key"] is None
    assert "unused-private-credential" not in profiles.path.read_text()
