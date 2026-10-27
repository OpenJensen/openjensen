"""The advertised native segment envelope fits its finite local tick budget."""

import dataclasses
import hashlib

import pytest
from conftest import SimulationFixture, act, settings

from jensen_teaching.agent import AgentCommand
from jensen_teaching.contracts import canonical
from jensen_teaching.journal import Journal
from jensen_teaching.session import Session


def invoke(session, operation, arguments):
    return session.agent.execute(
        AgentCommand.parse(
            operation,
            {
                "command_id": operation + "_fixture",
                "session_id": session.journal.session_id,
                "agent_session_id": "agent_fixture",
                "authority_epoch": 0 if operation == "acquire" else session.agent.epoch,
                "arguments": arguments,
            },
        )
    )


@pytest.mark.parametrize("fps", [30, 60])
def test_maximum_advertised_segment_finishes_exactly_sixty_ticks(tmp_path, fps):
    cfg = settings()
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, fps=fps), max_steps=90)
    simulation = SimulationFixture(cfg.sim)
    session = Session(cfg, simulation, Journal(tmp_path / "capture", cfg))
    try:
        act(session, "task", {"instruction": "Bounded local hold"})
        act(session, "start")
        invoke(
            session,
            "acquire",
            {
                "clock_policy": "stepped",
                "expected_revision": session.revision,
                "episode_id": session.episode_id,
            },
        )
        observed = invoke(session, "observe", {})["observation"]
        limit = min(2.0, 60 / fps)
        assert observed["action_limits"] == {
            "min_duration_seconds": 0.1,
            "max_duration_seconds": limit,
            "max_ticks": 60,
        }
        immutable = {key: value for key, value in observed.items() if key != "observation_sha256"}
        assert hashlib.sha256(canonical(immutable)).hexdigest() == observed["observation_sha256"]
        arguments = {
            "observation_id": observed["observation_id"],
            "observation_sha256": observed["observation_sha256"],
            "targets": {"joint_0": 0.0},
            "duration_seconds": limit,
            "proposal_id": "proposal_fixture",
            "provider_model": "fixture-v1",
        }
        receipt = invoke(session, "segment", arguments)["execution"]
        for _ in range(60):
            session.tick()
        assert receipt["status"] == "completed" and receipt["executed_ticks"] == 60
        assert receipt["simulation_elapsed_seconds"] == pytest.approx(limit)
        assert len(simulation.applications) == 60
    finally:
        session.close()
