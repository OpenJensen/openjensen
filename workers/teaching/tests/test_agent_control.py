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
    session.agent.execute(
        correction(
            session,
            observed,
            delta_rad=0.15,
            prompt_version="agent-joint-v1",
            prompt_sha256="d" * 64,
        )
    )
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
    assert row["prompt_version"] == "agent-joint-v1" and row["prompt_sha256"] == "d" * 64
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


def segment(session, observation, targets=None, duration=0.1):
    return agent_command(
        session,
        "segment",
        {
            "observation_id": observation["observation_id"],
            "observation_sha256": observation["observation_sha256"],
            "targets": targets or {"joint_0": 0.15},
            "duration_seconds": duration,
            "proposal_id": "segment_proposal",
            "provider_model": "scripted-native-v1",
        },
    )


def test_segment_runs_local_ticks_and_acknowledges_only_the_last_tick(session):
    observed = start(session)
    box = Mailbox()
    box.publish(session)
    command = segment(session, observed)
    box.submit(payload(command), agent_operation="segment")
    box.drain(session)
    assert box.get(command.command_id)["status"] == "executing"
    for tick in range(3):
        session.tick()
        box.publish(session)
        receipt = box.get(command.command_id)
        assert receipt["status"] == ("acknowledged" if tick == 2 else "executing")
    evidence = receipt["execution"]
    assert evidence["executed_ticks"] == 3
    assert evidence["simulation_elapsed_seconds"] == pytest.approx(0.1)
    assert evidence["requested_target_rad"][0] == 0.15
    assert evidence["applied_target_rad"][0] == pytest.approx(0.15)
    assert evidence["speed_limit_clipped"][0] is True
    assert evidence["next_state_rad"][0] == pytest.approx(0.15)
    session.tick()
    assert len(session.sim.applications) == 3
    assert box.submit(payload(command), agent_operation="segment")["status"] == "acknowledged"
    assert len(session.sim.applications) == 3


def test_named_hold_segment_is_local_bounded_wait(session):
    observed = start(session)
    command = segment(session, observed, targets={"joint_0": 0.0}, duration=0.2)
    result = session.agent.execute(command)
    for _ in range(6):
        session.tick()
    assert result["execution"]["status"] == "completed"
    assert result["execution"]["executed_ticks"] == 6
    assert session.sim.state == (0.0,) * 6
    assert len(session.sim.applications) == 6


def test_segment_manual_takeover_retains_partial_evidence_without_more_motion(session):
    observed = start(session)
    box = Mailbox()
    box.publish(session)
    command = segment(session, observed, duration=0.2)
    box.submit(payload(command), agent_operation="segment")
    box.drain(session)
    session.tick()
    box.publish(session)
    act(session, "pause")
    box.publish(session)
    receipt = box.get(command.command_id)
    assert receipt["status"] == "rejected" and receipt["execution"]["status"] == "stopped"
    assert receipt["execution"]["executed_ticks"] == 1
    assert receipt["execution"]["next_state_rad"][0] == pytest.approx(2 / 30)
    session.tick()
    assert len(session.sim.applications) == 1


def test_segment_release_interrupts_in_flight_local_action(session):
    observed = start(session)
    command = segment(session, observed, duration=0.2)
    result = session.agent.execute(command)
    session.tick()
    session.agent.execute(agent_command(session, "release"))
    session.tick()
    assert result["execution"]["status"] == "stopped" and session.mode == "paused"
    assert len(session.sim.applications) == 1


@pytest.mark.parametrize("damage", ["delta", "duration", "ticks", "names", "zero_targets"])
def test_segment_contract_rejects_outside_local_envelope(session, damage):
    observed = start(session)
    with pytest.raises(ValueError):
        if damage == "delta":
            session.agent.execute(segment(session, observed, targets={"joint_0": 0.15001}))
        elif damage == "duration":
            segment(session, observed, duration=2.01)
        elif damage == "ticks":
            session.settings = dataclasses.replace(
                session.settings, sim=dataclasses.replace(session.settings.sim, fps=60)
            )
            session.agent.execute(segment(session, observed, duration=1.1))
        elif damage == "names":
            session.agent.execute(segment(session, observed, targets={"invented": 0.03}))
        else:
            body = payload(segment(session, observed))
            body["arguments"]["targets"] = {}
            AgentCommand.parse("segment", body)
    assert not session.sim.applications


def test_heartbeat_renews_only_current_epoch_within_fixed_arming_budget(session):
    clock = {"now": 1_000_000_000}
    session.agent.clock_ns = lambda: clock["now"]
    act(session, "task", {"instruction": "abstract fixture goal"})
    act(session, "start")
    session.agent.execute(
        agent_command(
            session,
            "acquire",
            {
                "clock_policy": "stepped",
                "expected_revision": session.revision,
                "episode_id": session.episode_id,
                "max_session_seconds": 120,
            },
        )
    )
    observed = session.agent.execute(agent_command(session, "observe"))["observation"]
    initial_expiry = observed["valid_until_monotonic_ns"]
    clock["now"] += 25_000_000_000
    session.agent.execute(agent_command(session, "heartbeat"))
    assert session.agent.lease_deadline_ns == clock["now"] + 30_000_000_000
    assert observed["valid_until_monotonic_ns"] == initial_expiry
    clock["now"] += 10_000_000_000
    with pytest.raises(ValueError, match="expired"):
        session.agent.execute(correction(session, observed))
    with pytest.raises(ValueError, match="ownership"):
        session.agent.execute(agent_command(session, "heartbeat", epoch=session.agent.epoch + 1))
    clock["now"] = session.agent.lease_deadline_ns
    with pytest.raises(ValueError, match="ownership"):
        session.agent.execute(agent_command(session, "heartbeat"))
    session.tick()
    assert session.agent.owner is None and session.mode == "paused"


def test_heartbeat_does_not_extend_past_arming_cap_or_reacquire_after_takeover(session):
    clock = {"now": 1_000_000_000}
    session.agent.clock_ns = lambda: clock["now"]
    act(session, "task", {"instruction": "bounded fixture"})
    act(session, "start")
    session.agent.execute(
        agent_command(
            session,
            "acquire",
            {
                "clock_policy": "stepped",
                "expected_revision": session.revision,
                "episode_id": session.episode_id,
                "max_session_seconds": 45,
            },
        )
    )
    clock["now"] += 25_000_000_000
    session.agent.execute(agent_command(session, "heartbeat"))
    assert session.agent.lease_deadline_ns == session.agent.arm_deadline_ns
    old_epoch = session.agent.epoch
    act(session, "pause")
    act(session, "start")
    with pytest.raises(ValueError):
        session.agent.execute(agent_command(session, "heartbeat", epoch=old_epoch))
    assert session.agent.owner is None


def test_segment_capture_preserves_skill_event_and_applied_actions(session):
    observed = start(session)
    ident = session.episode_id
    session.agent.execute(segment(session, observed))
    session.tick()
    session.tick()
    session.tick()
    act(session, "finish")
    _, episodes, _ = inspect_capture(session.journal.root, [ident])
    rows = episodes[0]["rows"]
    assert len(rows) == 3 and all(r["skill_manifest_version"] == 2 for r in rows)
    assert rows[0]["applied_target_rad"][0] == pytest.approx(2 / 30)
    assert rows[-1]["applied_target_rad"][0] == pytest.approx(0.15)
    event = next(e for e in episodes[0]["receipt"]["events"] if e["source"] == "agent_command")
    assert event["operation"] == "segment" and event["arguments"] == {
        "targets": {"joint_0": 0.15},
        "duration_seconds": 0.1,
    }


def test_extended_snapshot_stays_frozen_and_keeps_original_capture_age(session):
    clock = {"now": 1_000_000_000}
    session.agent.clock_ns = lambda: clock["now"]
    act(session, "task", {"instruction": "Pick up the configured object"})
    act(session, "start")
    session.agent.execute(
        agent_command(
            session,
            "acquire",
            {
                "clock_policy": "stepped",
                "expected_revision": session.revision,
                "episode_id": session.episode_id,
                "max_session_seconds": 120,
                "planning_snapshot_seconds": 120,
            },
        )
    )
    observed = session.agent.execute(agent_command(session, "observe"))["observation"]
    assert observed["scene_frozen_for_planning"] is True
    assert observed["snapshot_policy"] == "frozen_step_v1"
    assert observed["joint_limits"] == {f"joint_{i}": [-0.2, 0.2] for i in range(6)}
    for _ in range(4):
        clock["now"] += 20_000_000_000
        session.agent.execute(agent_command(session, "heartbeat"))
        session.tick()
    assert session.sim.counter == 0 and session.observation.sim_time == 0
    assert observed["captured_monotonic_ns"] == 1_000_000_000
    assert observed["valid_until_monotonic_ns"] == 121_000_000_000
    result = session.agent.execute(correction(session, observed))
    session.tick()
    assert result["execution"]["status"] == "completed"
    assert session.sim.counter == 1


def test_frozen_admission_recaptures_real_state_instead_of_refreshing_old_binding(session):
    observed = start(session)
    # Simulate an out-of-band mutation without updating the session cache.
    session.sim.state = (0.04,) * 6
    with pytest.raises(ValueError, match="drift"):
        session.agent.execute(correction(session, observed))
    assert not session.sim.applications


def test_continuous_snapshot_keeps_short_age_and_rejects_extended_policy(session):
    observed = start(session, "continuous")
    assert observed["scene_frozen_for_planning"] is False
    assert observed["valid_until_monotonic_ns"] - observed["captured_monotonic_ns"] == 2_000_000_000
    body = payload(
        agent_command(
            session,
            "acquire",
            {
                "clock_policy": "continuous",
                "expected_revision": session.revision,
                "episode_id": session.episode_id,
            },
        )
    )
    body["arguments"]["planning_snapshot_seconds"] = 120
    with pytest.raises(ValueError, match="frozen"):
        AgentCommand.parse("acquire", body)


@pytest.mark.parametrize(
    "metadata",
    [
        {"prompt_version": "agent-goal-v1"},
        {"prompt_sha256": "d" * 64},
        {"prompt_version": "bad version", "prompt_sha256": "d" * 64},
        {"prompt_version": "agent-goal-v1", "prompt_sha256": "not-a-hash"},
    ],
)
def test_invalid_prompt_provenance_never_dispatches(session, metadata):
    observed = start(session)
    with pytest.raises(ValueError, match="prompt provenance"):
        correction(session, observed, **metadata)
    assert not session.sim.applications
