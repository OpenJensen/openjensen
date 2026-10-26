"""Protocol acceptance with deterministic simulator fixtures, never physical motion."""

import dataclasses
import hashlib
import json

import pytest
from conftest import act

from jensen_teaching.agent import AgentCommand
from jensen_teaching.contracts import canonical
from jensen_teaching.control import Mailbox
from jensen_teaching.dataset import inspect_capture


def agent_command(session, operation, arguments=None, *, epoch=None, ident=None):
    import uuid

    return AgentCommand.parse(
        operation,
        {
            "command_id": ident or uuid.uuid4().hex,
            "session_id": session.journal.session_id,
            "agent_session_id": "agent_fixture",
            "authority_epoch": (0 if operation == "acquire" else session.agent.epoch)
            if epoch is None
            else epoch,
            "arguments": arguments or {},
        },
    )


def start(session, clock="stepped"):
    act(session, "task", {"instruction": "Move joint_0 by 0.03 radians"})
    act(session, "start")
    session.agent.execute(
        agent_command(
            session,
            "acquire",
            {
                "clock_policy": clock,
                "expected_revision": session.revision,
                "episode_id": session.episode_id,
            },
        )
    )
    return session.agent.execute(agent_command(session, "observe"))["observation"]


def correction(session, observation, **changes):
    return agent_command(
        session,
        "execute",
        {
            "observation_id": observation["observation_id"],
            "observation_sha256": observation["observation_sha256"],
            "joint": "joint_0",
            "delta_rad": 0.03,
            "proposal_id": "proposal_fixture",
            "provider_model": "mock-v1",
            **changes,
        },
    )


def payload(command):
    result = dataclasses.asdict(command)
    del result["operation"]
    return result


def test_stepped_fresh_capture_and_one_actual_step_receipt(session):
    original = session.sim.observe
    captures = []

    def measured():
        result = original()
        captures.append(result)
        return result

    session.sim.observe = measured
    observation = start(session)
    assert len(captures) == 2  # start and a genuinely new owner-thread observe
    digest = observation.pop("observation_sha256")
    assert hashlib.sha256(canonical(observation)).hexdigest() == digest
    observation["observation_sha256"] = digest
    assert observation["image_data_url"].startswith("data:image/png;base64,")
    session.tick()
    assert session.sim.counter == 0
    box = Mailbox()
    box.publish(session)
    cmd = correction(session, observation)
    body = payload(cmd)
    first = box.submit(body, agent_operation="execute")
    assert box.submit(body, agent_operation="execute") == first
    box.drain(session)
    assert box.get(cmd.command_id)["status"] == "executing"
    session.tick()
    box.publish(session)
    final = box.get(cmd.command_id)
    assert final["status"] == "acknowledged"
    assert final["execution"]["status"] == "completed"
    assert final["execution"]["applied_target_rad"][0] == pytest.approx(0.03)
    assert final["execution"]["next_state_rad"][0] == pytest.approx(0.03)
    assert final["execution"]["task_success_verified"] is False
    session.tick()
    assert session.sim.counter == 1
    assert box.submit(body, agent_operation="execute")["status"] == "acknowledged"
    assert len(session.sim.applications) == 1
    with pytest.raises(ValueError, match="reused"):
        box.submit(
            body | {"arguments": body["arguments"] | {"delta_rad": 0.02}}, agent_operation="execute"
        )


@pytest.mark.parametrize(
    "damage", ["epoch", "observation", "hash", "revision", "age", "drift", "joint", "single_flight"]
)
def test_observation_and_authority_reject_without_motion(session, damage):
    observed = start(session)
    cmd = correction(session, observed)
    if damage == "epoch":
        cmd = dataclasses.replace(cmd, authority_epoch=cmd.authority_epoch + 1)
    elif damage == "observation":
        cmd = correction(session, observed, observation_id="invented")
    elif damage == "hash":
        cmd = correction(session, observed, observation_sha256="f" * 64)
    elif damage == "revision":
        session.revision += 1
    elif damage == "age":
        session.agent.clock_ns = lambda: observed["valid_until_monotonic_ns"]
    elif damage == "drift":
        session.observation = dataclasses.replace(session.observation, state=(0.04,) * 6)
    elif damage == "joint":
        cmd = correction(session, observed, joint="invented")
    else:
        session.agent.execute(cmd)
    with pytest.raises(ValueError):
        session.agent.execute(cmd)
    assert not session.sim.applications


@pytest.mark.parametrize("delta", [True, float("nan"), float("inf"), 0, 0.150001])
def test_agent_values_strict(session, delta):
    observed = start(session)
    with pytest.raises(ValueError):
        correction(session, observed, delta_rad=delta)
    assert not session.sim.applications


def test_manual_takeover_and_late_result(session):
    observed = start(session)
    old_epoch = session.agent.epoch
    late = correction(session, observed)
    act(session, "pause")
    assert session.agent.owner is None and session.agent.epoch > old_epoch
    act(session, "start")
    with pytest.raises(ValueError, match="ownership"):
        session.agent.execute(late)
    session.tick()
    assert session.sim.state[0] == 0


def test_continuous_contract_tolerates_tick_and_rejects_drift(session):
    observed = start(session, "continuous")
    session.tick()
    assert session.sim.counter == 1 and session.revision == observed["revision"]
    session.agent.execute(correction(session, observed))
    session.tick()
    assert session.sim.counter == 2 and session.sim.state[0] == pytest.approx(0.03)


def test_agent_capture_v2_truthful_applied_action_and_actor(session):
    observed = start(session)
    ident = session.episode_id
    session.agent.execute(correction(session, observed, delta_rad=0.15))
    session.tick()
    act(session, "finish")
    meta, episodes, _ = inspect_capture(session.journal.root, [ident])
    assert meta["schema_version"] == 2 and meta["environment_kind"] == "simulation"
    row = episodes[0]["rows"][0]
    assert row["actor"] == "llm_agent" and row["action_source"] == "agent_joint_position"
    assert row["requested_target_rad"][0] == 0.15
    assert row["applied_target_rad"][0] == pytest.approx(2 / 30)
    assert row["next_state_rad"][0] == row["applied_target_rad"][0]
    assert row["observation_sha256"] == observed["observation_sha256"]
    assert episodes[0]["receipt"]["outcome"] == "unknown"


def test_legacy_capture_parser_still_rejects_agent_source(session):
    act(session, "task", {"instruction": "legacy human"})
    act(session, "start")
    session.tick()
    ident = session.episode_id
    act(session, "finish")
    root = session.journal.root
    meta = json.loads((root / "session.json").read_text())
    meta["schema_version"] = 1
    (root / "session.json").write_bytes(canonical(meta))
    receipt_path = root / ident / "episode.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["schema_version"] = 1
    receipt_path.write_bytes(canonical(receipt))
    inspect_capture(root, [ident])
    events_path = root / ident / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[0]["source"] = "agent_command"
    raw = b"".join(canonical(event) + b"\n" for event in events)
    events_path.write_bytes(raw)
    receipt["events"] = events
    receipt["events_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt_path.write_bytes(canonical(receipt))
    with pytest.raises(ValueError, match="legacy supervision"):
        inspect_capture(root, [ident])


def test_expired_owner_pauses_and_cannot_renew_from_observe(session):
    observed = start(session)
    session.agent.clock_ns = lambda: observed["valid_until_monotonic_ns"] + 1
    session.tick()
    assert session.mode == "paused" and session.agent.owner is None
    with pytest.raises(ValueError):
        session.agent.execute(agent_command(session, "observe"))
    assert not session.sim.applications


def test_queued_acquire_cannot_cross_manual_pause_and_resume(session):
    act(session, "task", {"instruction": "bounded correction"})
    act(session, "start")
    acquire = agent_command(
        session,
        "acquire",
        {
            "clock_policy": "stepped",
            "expected_revision": session.revision,
            "episode_id": session.episode_id,
        },
    )
    act(session, "pause")
    act(session, "start")
    with pytest.raises(ValueError, match="context"):
        session.agent.execute(acquire)
    assert session.agent.owner is None
