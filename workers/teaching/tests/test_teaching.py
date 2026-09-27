import dataclasses
import hashlib
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from conftest import act, command, record_fixture

from firebird_teaching.contracts import Command, canonical, decode
from firebird_teaching.control import Client, Mailbox, server
from firebird_teaching.dataset import convert, inspect_capture


def test_applied_action_pause_reset_and_lineage(session):
    act(session, "task", {"instruction": "Move one named joint"})
    act(session, "start")
    ident = session.episode_id
    act(session, "correct", {"joint": "joint_0", "delta_rad": 0.15})
    session.tick()
    row = json.loads((session.journal.root / ident / "trajectory.jsonl").read_text())
    assert row["requested_target_rad"][0] == 0.15
    assert row["applied_target_rad"][0] == pytest.approx(2 / 30)
    assert row["state_rad"][0] == 0 and row["next_state_rad"][0] == pytest.approx(2 / 30)
    assert row["speed_limit_clipped"][0]
    act(session, "pause")
    session.tick()
    assert session.sim.counter == 1
    act(session, "start")
    session.tick()
    assert session.sim.state[0] == pytest.approx(2 / 30)
    act(session, "reset")
    assert session.mode == "idle" and session.sim.reset_count == 2
    meta, episodes, _ = inspect_capture(session.journal.root, [ident])
    assert meta["origin"] == "synthetic" and episodes[0]["receipt"]["outcome"] == "unknown"


@pytest.mark.parametrize(
    "change",
    [
        {"operation": []},
        {"operation": "execute_python"},
        {"expected_revision": True},
        {"arguments": {"joint": "joint_0", "delta_rad": True}},
        {"arguments": {"joint": "joint_0", "delta_rad": float("nan")}},
        {"arguments": {"joint": "joint_0", "delta_rad": 0.151}},
        {"arguments": {"joint": "joint_0", "delta_rad": 0}},
        {"arguments": {"joint": "joint_0", "delta_rad": 0.1, "shell": "oops"}},
        {"episode_id": "../escape"},
    ],
)
def test_bounded_command_admission(session, change):
    data = dataclasses.asdict(command(session, "correct", {"joint": "joint_0", "delta_rad": 0.1}))
    with pytest.raises(ValueError):
        Command.parse(data | change)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":NaN}', b"[]", b"{}" * 9000])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_owner_thread_and_stale_revision(session):
    c = command(session, "task", {"instruction": "test"})
    errors = []

    def wrong_thread():
        try:
            session.execute(c)
        except RuntimeError as error:
            errors.append(str(error))

    t = threading.Thread(target=wrong_thread)
    t.start()
    t.join()
    assert errors and session.instruction == ""
    session.execute(c)
    with pytest.raises(ValueError, match="Stale"):
        session.execute(c)


def test_capture_fault_retains_torn_episode(session):
    act(session, "task", {"instruction": "test"})
    act(session, "start")
    ident = session.episode_id
    session.tick()
    session.sim.step = lambda: (_ for _ in ()).throw(RuntimeError("sim failed"))
    with pytest.raises(RuntimeError):
        session.tick()
    assert session.mode == "faulted"
    assert (session.journal.root / ident / "frames/000000.rgb").exists()
    assert (session.journal.root / ident / "events.jsonl").read_bytes()
    assert not (session.journal.root / ident / "episode.json").exists()
    with pytest.raises(ValueError):
        inspect_capture(session.journal.root, [ident])


def test_bad_observation_prevents_admission(session):
    act(session, "task", {"instruction": "test"})
    act(session, "start")
    original = session.sim.observe
    session.sim.observe = lambda: dataclasses.replace(original(), sim_time=123)
    with pytest.raises(ValueError, match="clock"):
        session.tick()
    assert session.mode == "faulted"


def test_mailbox_expiry_duplicate_and_priority_pause(session):
    act(session, "task", {"instruction": "test"})
    act(session, "start")
    clock = [0.0]
    box = Mailbox(clock=lambda: clock[0])
    box.publish(session)
    correction = dataclasses.asdict(
        command(session, "correct", {"joint": "joint_0", "delta_rad": 0.1})
    )
    receipt = box.submit(correction)
    assert box.submit(correction) == receipt
    with pytest.raises(ValueError, match="reused"):
        box.submit(correction | {"operation": "pause", "arguments": {}})
    pause = dataclasses.asdict(command(session, "pause"))
    box.submit(pause)
    box.drain(session)
    session.tick()
    assert session.mode == "paused" and not session.sim.applications
    box.drain(session)
    assert box.get(correction["command_id"])["status"] == "rejected"
    start = dataclasses.asdict(command(session, "start"))
    box.submit(start)
    clock[0] = 3
    box.drain(session)
    assert "expired" in box.get(start["command_id"])["error"]
    assert session.mode == "paused"


def test_loopback_transport_does_not_execute_in_network_thread(session):
    token = "x" * 32
    box = Mailbox()
    box.publish(session)
    http = server(box, token)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{http.server_port}"
    client = Client(origin, token)
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(origin + "/state")
        assert error.value.code == 401
        with pytest.raises(HTTPError):
            urlopen(
                Request(
                    origin + "/state",
                    headers={
                        "Authorization": "Bearer " + token,
                        "Origin": "https://elsewhere.invalid",
                    },
                )
            )
        payload = dataclasses.asdict(
            command(session, "task", {"instruction": "a real queued command"})
        )
        receipt = client.request("/commands", payload)
        assert receipt["status"] == "queued" and session.instruction == ""
        box.drain(session)
        box.publish(session)
        assert client.request("/commands/" + payload["command_id"])["status"] == "acknowledged"
        assert client.request("/state")["instruction"] == "a real queued command"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:90",
        "http://example.com:90",
        "http://localhost:90",
        "http://127.0.0.1:90/x",
        "http://secret@127.0.0.1:90",
    ],
)
def test_client_local_origin_only(url):
    with pytest.raises(ValueError):
        Client(url, "x" * 32)


@pytest.mark.parametrize(
    "damage", ["frame", "trajectory", "events", "symlink", "orphan", "false_success", "unfinalized"]
)
def test_capture_rejects_damage(tmp_path, damage):
    root = tmp_path / "capture"
    ids = record_fixture(root, 1)
    episode = root / ids[0]
    if damage == "frame":
        (episode / "frames/000000.rgb").write_bytes(b"x")
    elif damage == "trajectory":
        path = episode / "trajectory.jsonl"
        path.write_bytes(path.read_bytes()[:-2])
    elif damage == "events":
        path = episode / "events.jsonl"
        path.write_bytes(path.read_bytes()[:-2])
    elif damage == "symlink":
        (root / "link").symlink_to(episode / "episode.json")
    elif damage == "orphan":
        (episode / "frames/999999.rgb").write_bytes(b"x")
    elif damage == "false_success":
        path = episode / "episode.json"
        data = json.loads(path.read_text())
        data["outcome"] = "success"
        path.write_bytes(canonical(data))
    elif damage == "unfinalized":
        (episode / "episode.json").unlink()
    with pytest.raises(ValueError):
        inspect_capture(root, ids)


def test_genuine_lerobot_writer_roundtrip(tmp_path):
    root = tmp_path / "capture"
    ids = record_fixture(root)
    before = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "dataset"
    result = convert(root, output, ids)
    assert (
        result["readback_verified"]
        and result["frames"] == 12
        and not result["task_success_verified"]
    )
    assert (output / "meta/tasks.parquet").exists()
    lineage = json.loads((output / "meta/firebird-lineage.json").read_text())
    assert len(lineage["episodes"]) == 2 and {e["origin"] for e in lineage["episodes"]} == {
        "synthetic"
    }
    assert len({e["lineage_group"] for e in lineage["episodes"]}) == 1
    demonstrations = json.loads((output / "meta/firebird-demonstrations.json").read_text())
    assert [e["outcome"] for e in demonstrations["episodes"]] == [
        "unknown",
        "operator_reported_failure",
    ]
    assert before == {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        convert(root, output, ids)
    # A corrupt local writer output must never invoke metadata or media Hub fallback.
    from firebird_teaching.dataset import read_local_dataset

    (output / "meta/tasks.parquet").unlink()
    with pytest.raises(ValueError, match="must never download"):
        read_local_dataset("local/teaching", output)


def test_correction_acknowledgement_waits_for_applied_and_durable_frame(session):
    act(session, "task", {"instruction": "test"})
    act(session, "start")
    box = Mailbox()
    box.publish(session)
    payload = dataclasses.asdict(
        command(session, "correct", {"joint": "joint_0", "delta_rad": 0.1})
    )
    box.submit(payload)
    box.drain(session)
    box.publish(session)
    assert box.get(payload["command_id"])["status"] == "executing"
    assert not session.sim.applications
    session.tick()
    box.publish(session)
    receipt = box.get(payload["command_id"])
    assert receipt["status"] == "acknowledged" and receipt["applied_step"] == 0
    assert receipt["state"]["last_applied_command_id"] == payload["command_id"]


def test_bad_step_never_acknowledges_correction(session):
    act(session, "task", {"instruction": "test"})
    act(session, "start")
    box = Mailbox()
    box.publish(session)
    payload = dataclasses.asdict(
        command(session, "correct", {"joint": "joint_0", "delta_rad": 0.1})
    )
    box.submit(payload)
    box.drain(session)
    session.sim.step = lambda: (_ for _ in ()).throw(RuntimeError("step failed"))
    with pytest.raises(RuntimeError):
        session.tick()
    box.publish(session)
    assert box.get(payload["command_id"])["status"] == "rejected"
    assert "unverified" in box.get(payload["command_id"])["error"]
    assert box.state["mode"] == "faulted"


def test_idle_readiness_requires_real_scene_and_preview(session):
    session.prepare()
    assert session.mode == "idle" and session.episode_id is None
    assert session.observation is not None and session.observation.step == 0
    assert session.sim.counter == 0 and not session.sim.applications
    assert session.journal.current is None


def test_idle_readiness_scene_failure_stays_faulted(session):
    session.sim.reset = lambda episode: (_ for _ in ()).throw(ValueError("camera missing"))
    with pytest.raises(ValueError, match="camera missing"):
        session.prepare()
    assert session.mode == "faulted"


def test_executor_restart_rejects_command_atomically_even_if_idle_revision_matches(
    session, tmp_path
):
    from conftest import make_session

    old = command(session, "task", {"instruction": "old room command"})
    restarted = make_session(tmp_path / "restarted")
    assert restarted.episode_id == session.episode_id and restarted.revision == session.revision
    try:
        with pytest.raises(ValueError, match="Executor session changed"):
            restarted.execute(old)
        assert restarted.instruction == "" and restarted.revision == 0
    finally:
        restarted.journal.abort()


def test_lineage_identifier_matches_dataset_admission(tmp_path):
    root = tmp_path / "capture"
    ids = record_fixture(root, 1)
    path = root / "session.json"
    data = json.loads(path.read_text())
    data["lineage_group"] = "invalid spaces"
    path.write_bytes(canonical(data))
    with pytest.raises(ValueError, match="lineage group"):
        inspect_capture(root, ids)


def test_metadata_content_is_bound_to_initial_inventory(tmp_path, monkeypatch):
    from firebird_teaching import dataset

    root = tmp_path / "capture"
    ids = record_fixture(root, 1)
    original = dataset._inventory
    calls = [0]

    def changing_inventory(path):
        result = original(path)
        calls[0] += 1
        if calls[0] == 1:
            meta = root / "session.json"
            value = json.loads(meta.read_text())
            value["lineage_group"] = "changed-group"
            meta.write_bytes(canonical(value))
        return result

    monkeypatch.setattr(dataset, "_inventory", changing_inventory)
    with pytest.raises(ValueError, match="initial inventory"):
        inspect_capture(root, ids)


@pytest.mark.parametrize(
    "damage", ["future_command", "nonmotion_command", "clock", "discontinuous"]
)
def test_self_consistent_hashes_cannot_forge_motion_lineage(tmp_path, damage):
    root = tmp_path / "capture"
    ids = record_fixture(root, 1)
    folder = root / ids[0]
    receipt = json.loads((folder / "episode.json").read_text())
    rows = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
    if damage == "future_command":
        rows[0]["command_id"] = next(
            e["command_id"] for e in receipt["events"] if e["operation"] == "correct"
        )
    elif damage == "nonmotion_command":
        rows[0]["command_id"] = receipt["events"][-1]["command_id"]
    elif damage == "clock":
        rows[0]["applied_monotonic_ns"] = 1
    else:
        rows[1]["state_rad"][0] = 0.9
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    (folder / "trajectory.jsonl").write_bytes(raw)
    receipt["trajectory_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["bytes"] = len(raw) + len(rows) * 32 * 32 * 3
    (folder / "episode.json").write_bytes(canonical(receipt))
    with pytest.raises(ValueError, match="provenance|future|discontinuous"):
        inspect_capture(root, ids)


def test_inventory_growth_read_is_bounded(tmp_path, monkeypatch):
    from firebird_teaching import dataset

    root = tmp_path / "capture"
    root.mkdir()
    target = root / "file"
    target.write_bytes(b"a")
    original = dataset._open_regular

    class GrowingStream:
        def __init__(self, path):
            self.stream = original(path)
            self.grown = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, n):
            if not self.grown:
                self.grown = True
                with target.open("ab") as output:
                    output.write(b"b" * 100)
            return self.stream.read(n)

    monkeypatch.setattr(dataset, "_open_regular", GrowingStream)
    with pytest.raises(ValueError, match="changed during inventory"):
        dataset._inventory(root)


def test_ancestor_symlink_cannot_redirect_capture_read(tmp_path):
    from firebird_teaching.dataset import _read

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_bytes(b"not a capture")
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        _read(linked / "file", 1024)


def test_directory_sync_failure_never_acknowledges_an_action(session, monkeypatch):
    from firebird_teaching import journal

    act(session, "task", {"instruction": "test"})
    act(session, "start")
    box = Mailbox()
    box.publish(session)
    payload = dataclasses.asdict(
        command(session, "correct", {"joint": "joint_0", "delta_rad": 0.1})
    )
    box.submit(payload)
    box.drain(session)

    def fail_sync(path):
        raise OSError("directory sync failed")

    monkeypatch.setattr(journal, "sync_directory", fail_sync)
    with pytest.raises(OSError, match="directory sync"):
        session.tick()
    box.publish(session)
    assert box.get(payload["command_id"])["status"] == "rejected"
    assert session.last_applied_command_id is None
    assert not list(session.journal.root.glob("*/episode.json"))


@pytest.mark.parametrize("magnitude", [1e300, 1e30])
def test_float32_overflow_and_nonfinite_stats_never_publish(tmp_path, magnitude):
    root = tmp_path / "capture"
    ids = record_fixture(root, 1)
    folder = root / ids[0]
    rows = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
    for row in rows:
        row["state_rad"] = [magnitude] * 6
        row["next_state_rad"] = [magnitude] * 6
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    (folder / "trajectory.jsonl").write_bytes(raw)
    receipt = json.loads((folder / "episode.json").read_text())
    receipt["trajectory_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["bytes"] = len(raw) + len(rows) * 32 * 32 * 3
    (folder / "episode.json").write_bytes(canonical(receipt))
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="float32|Nonfinite|finite numeric"):
        convert(root, output, ids)
    assert not output.exists()
    assert (folder / "trajectory.jsonl").read_bytes() == raw
