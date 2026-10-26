"""Agent relay admission must preserve original host observation and request identity."""

import asyncio
import base64
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from vla_platform import teaching_api
from vla_platform.teaching_sessions.service import TeachingSessions


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def request(operation="observe"):
    return {
        "command_id": "command_fixture",
        "session_id": "executor_fixture",
        "agent_session_id": "agent_fixture",
        "authority_epoch": 1,
        "arguments": {},
    }


def state():
    return {
        "mode": "running",
        "revision": 2,
        "episode_id": "episode_fixture",
        "session_id": "executor_fixture",
        "state_rad": [0.0],
        "joints": ["joint_0"],
        "agent_owner": "agent_fixture",
        "authority_epoch": 1,
        "agent_clock_policy": "stepped",
        "agent_capabilities": {
            "schema_version": 1,
            "named_joint_correction": True,
            "stepped_capture": True,
            "physical_execution": False,
        },
    }


def observation():
    # Valid fixture PNG envelope; image decoding stays the isolated provider's concern.
    png = b"\x89PNG\r\n\x1a\nfixture"
    result = {
        "schema_version": 1,
        "observation_id": "observation_fixture",
        "target_session_id": "executor_fixture",
        "agent_session_id": "agent_fixture",
        "episode_id": "episode_fixture",
        "authority_epoch": 1,
        "revision": 2,
        "embodiment_manifest_id": "a" * 64,
        "scene_sha256": "b" * 64,
        "calibration_id": None,
        "clock_policy": "stepped",
        "captured_monotonic_ns": 1,
        "valid_until_monotonic_ns": 30_000_000_001,
        "source_age_ms": 0,
        "step": 0,
        "sim_time": 0.0,
        "joints": ["joint_0"],
        "state_rad": [0.0],
        "units": "rad",
        "image_data_url": "data:image/png;base64," + base64.b64encode(png).decode(),
        "image_sha256": hashlib.sha256(png).hexdigest(),
    }
    result["observation_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def receipt(body=None):
    body = body or request()
    return {
        "command_id": body["command_id"],
        "status": "acknowledged",
        "request_sha256": hashlib.sha256(
            canonical({"operation": "observe", "payload": body})
        ).hexdigest(),
        "state": state(),
        "observation": observation(),
        "observation_age_ms": 25,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"units": "degrees"},
        {"state_rad": [float("nan")]},
        {"state_rad": []},
        {"target_session_id": "wrong"},
        {"observation_sha256": "f" * 64},
        {"image_sha256": "f" * 64},
        {"secret": "must-not-forward"},
        {"valid_until_monotonic_ns": 60_000_000_001},
    ],
)
def test_agent_observation_cannot_change_binding(change):
    with pytest.raises(ValueError):
        teaching_api.public_agent_observation(observation() | change, session_id="executor_fixture")


def test_only_known_agent_evidence_is_forwarded():
    raw = receipt() | {"control_token": "not-public"}
    raw["state"]["secret"] = "not-public"
    result = teaching_api.public_agent_receipt(raw, session_id="executor_fixture")
    assert "control_token" not in result and "secret" not in result["state"]
    assert result["observation"] == observation() and result["observation_age_ms"] == 25
    assert result["state"]["agent_capabilities"]["stepped_capture"]


def test_configured_agent_relay_verifies_request_hash(monkeypatch):
    async def run():
        async def relay(path, payload=None, **kwargs):
            assert kwargs["maximum"] == 16 * 1024**2
            return receipt(payload)

        monkeypatch.setattr(teaching_api, "relay", relay)
        result = await teaching_api.agent_relay("/agent/observe", request())
        assert result["observation"]["revision"] == 2

        async def bad_relay(*args, **kwargs):
            return receipt() | {"request_sha256": "f" * 64}

        monkeypatch.setattr(teaching_api, "relay", bad_relay)
        with pytest.raises(HTTPException):
            await teaching_api.agent_relay("/agent/observe", request())

    asyncio.run(run())


def test_managed_agent_relay_rejects_cross_project_and_executor(monkeypatch):
    async def run():
        class Execution:
            async def get(self, _):
                return SimpleNamespace(
                    id="job_fixture",
                    project_id="project_fixture",
                    kind="teaching.capture",
                    status="running",
                )

        service = TeachingSessions(Execution())
        service.live["job_fixture"] = SimpleNamespace(
            session_id="executor_fixture",
            stop=asyncio.Event(),
            process=SimpleNamespace(returncode=None),
        )

        async def upstream(live, path, body, maximum):
            return receipt(body)

        monkeypatch.setattr(service, "upstream", upstream)
        result = await service.agent_relay(
            "project_fixture", "job_fixture", "/agent/observe", request()
        )
        assert result["state"]["authority_epoch"] == 1
        with pytest.raises(ValueError):
            await service.agent_relay("other_project", "job_fixture", "/agent/observe", request())
        with pytest.raises(ValueError):
            await service.agent_relay(
                "project_fixture",
                "job_fixture",
                "/agent/observe",
                request() | {"session_id": "other_executor"},
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "path,body",
    [
        ("http://elsewhere.invalid", request()),
        ("/agent/observe", request() | {"url": "http://elsewhere.invalid"}),
        ("/agent/observe", request() | {"authority_epoch": True}),
        ("/agent/observe", request() | {"arguments": {"shell": "false"}}),
        (
            "/agent/acquire",
            request() | {"authority_epoch": 0, "arguments": {"clock_policy": "hardware"}},
        ),
    ],
)
def test_private_agent_adapter_is_closed(path, body):
    with pytest.raises(ValueError):
        teaching_api.validate_agent_request(path, body)


@pytest.mark.parametrize("damage", [None, "actor", "proposal", "missing"])
def test_agent_promotion_binds_exact_per_frame_provenance(tmp_path, damage):
    import threading
    import time
    from pathlib import Path

    from test_recording_preparation import converted_fixture
    from vla_platform.datasets.recordings import checked_conversion

    job, sources, output, response, inventory = converted_fixture(tmp_path)
    source = Path(sources[0]["path"])
    meta_path = source / "session.json"
    meta = json.loads(meta_path.read_bytes()) | {
        "schema_version": 2,
        "environment_kind": "simulation",
    }
    meta_path.write_bytes(canonical(meta))
    episode_id = sources[0]["episodes"][0]["episode_id"]
    rows = [
        {
            "step": step,
            "actor": "llm_agent",
            "environment_kind": "simulation",
            "agent_session_id": "agent_fixture",
            "proposal_id": "proposal_fixture",
            "execution_id": "execution_fixture",
            "provider_model": "mock-v1",
            "observation_id": "observation_fixture",
            "observation_sha256": "a" * 64,
            "authority_epoch": 1,
            "clock_policy": "stepped",
            "skill_manifest_version": 1,
        }
        for step in range(2)
    ]
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    (source / episode_id / "trajectory.jsonl").write_bytes(raw)
    receipt_path = source / episode_id / "episode.json"
    receipt = json.loads(receipt_path.read_bytes()) | {
        "schema_version": 2,
        "trajectory_sha256": hashlib.sha256(raw).hexdigest(),
    }
    receipt_path.write_bytes(canonical(receipt))
    selected = job.request.recordings.captures[0]
    selected.session_sha256 = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    selected.episodes[0].receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    sources[0]["session_sha256"] = selected.session_sha256
    sources[0]["episodes"][0]["receipt_sha256"] = selected.episodes[0].receipt_sha256
    provenance_path = output / "meta/firebird-demonstrations.json"
    provenance = json.loads(provenance_path.read_bytes())
    provenance["sources"][0]["source_files"] = inventory(source)
    provenance["episodes"][0].update(
        {
            "source_trajectory_sha256": receipt["trajectory_sha256"],
            "capture_schema_version": 2,
            "environment_kind": "simulation",
            "actors": ["llm_agent"],
            "action_provenance": rows,
        }
    )
    if damage == "actor":
        provenance["episodes"][0]["action_provenance"][0]["actor"] = "human_operator"
    elif damage == "proposal":
        provenance["episodes"][0]["action_provenance"][0]["proposal_id"] = "invented"
    elif damage == "missing":
        del provenance["episodes"][0]["action_provenance"]
    provenance_path.write_bytes(canonical(provenance))
    response["files"] = inventory(output)
    response["selection_sha256"] = hashlib.sha256(
        canonical(job.request.recordings.model_dump())
    ).hexdigest()

    def check():
        return checked_conversion(
            response, job, sources, output, deadline=time.monotonic() + 5, stop=threading.Event()
        )

    if damage is None:
        assert check()[1].task_success_verified is False
    else:
        with pytest.raises(ValueError, match="provenance"):
            check()
