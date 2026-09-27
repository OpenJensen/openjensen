"""Generated deterministic simulator fixtures: these never demonstrate task success."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Existing Isaac interfaces are reused without importing the NVIDIA runtime.
root = Path(__file__).resolve().parents[3]
isaac = root / "workers/isaac_sim"
if not isaac.exists():
    isaac = Path(os.environ.get("FIREBIRD_ISAAC_SOURCE", str(isaac)))
sys.path.insert(0, str(isaac))

from sim_worker.rollout.contracts import Frame, Observation, SimSpec  # noqa: E402

from firebird_teaching.contracts import Command, Settings  # noqa: E402
from firebird_teaching.journal import Journal  # noqa: E402
from firebird_teaching.session import Session  # noqa: E402


class SimulationFixture:
    def __init__(self, spec):
        self.spec = spec
        self.counter = 0
        self.state = (0.0,) * len(spec.joints)
        self.joint_limits = tuple((-0.2, 0.2) for _ in spec.joints)
        self.reset_count = 0
        self.applications = []
        self.episode = None

    def reset(self, episode_id):
        self.episode = episode_id
        self.counter = 0
        self.state = (0.0,) * len(self.spec.joints)
        self.reset_count += 1

    def observe(self):
        return Observation(
            self.episode,
            self.counter,
            self.counter / self.spec.fps,
            self.state,
            Frame(
                self.spec.width,
                self.spec.height,
                bytes([self.counter % 255, 64, 128]) * self.spec.width * self.spec.height,
            ),
        )

    def apply(self, action):
        self.target = tuple(action)
        self.applications.append(tuple(action))

    def step(self):
        self.counter += 1
        self.state = self.target


def settings():
    return Settings(
        SimSpec(
            "/generated/fixture.usda",
            "/World/Camera",
            "/World/Robot",
            tuple(f"joint_{i}" for i in range(6)),
            32,
            32,
            30,
            120,
        ),
        "0" * 64,
        "generated-fixture-scene",
        30,
        1024**2,
        "synthetic",
    )


def make_session(path):
    cfg = settings()
    sim = SimulationFixture(cfg.sim)
    journal = Journal(path, cfg)
    return Session(cfg, sim, journal)


def command(session, operation, args=None, ident=None):
    import uuid

    return Command.parse(
        {
            "command_id": ident or uuid.uuid4().hex,
            "session_id": session.journal.session_id,
            "episode_id": session.episode_id,
            "expected_revision": session.revision,
            "operation": operation,
            "arguments": args or {},
        }
    )


def act(session, operation, args=None):
    return session.execute(command(session, operation, args))


def record_fixture(path, episodes=2, frames=6):
    session = make_session(path)
    ids = []
    act(session, "task", {"instruction": "Generated fixture: joint correction, outcome unknown"})
    for episode in range(episodes):
        act(session, "start")
        ids.append(session.episode_id)
        for n in range(frames):
            if n == 1:
                act(session, "correct", {"joint": "joint_0", "delta_rad": 0.15})
            session.tick()
        if episode == 1:
            act(session, "mark_failure", {"reason": "Generated operator failure annotation"})
        act(session, "finish")
    session.close()
    return ids


@pytest.fixture
def session(tmp_path):
    s = make_session(tmp_path / "capture")
    yield s
    s.journal.abort()
