"""Generated simulator fixtures only: atomic transport, not hardware timing/quality."""

import copy
import dataclasses
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import act, command, make_session

from firebird_teaching.control import Client, Mailbox, server
from firebird_teaching.frame_snapshot import admit_frame
from firebird_teaching.providers import ProviderError, Providers, ProviderSettings


def published(session):
    box = Mailbox()
    box.publish(session)
    return box, box.frame_payload()


def admit(payload, **kwargs):
    return admit_frame(
        payload,
        expected_session_id=kwargs.pop("expected_session_id", payload["session_id"]),
        request_started_monotonic_ns=kwargs.pop("request_started_monotonic_ns", 10),
        received_monotonic_ns=kwargs.pop("received_monotonic_ns", 20),
        **kwargs,
    )


def test_preview_rgb_joints_and_context_are_one_immutable_observation(session):
    session.prepare()
    box, payload = published(session)
    original = copy.deepcopy(payload)
    frame = admit(payload).observation
    assert frame.rgb == session.observation.frame.rgb
    assert frame.state_rad == session.observation.state
    assert frame.joints == session.settings.sim.joints
    assert frame.active_episode_id is None and frame.episode_id.startswith("preview-")
    assert payload["units"] == "rad" and payload["camera_prim"] == session.settings.sim.camera
    payload["joints"][0] = "changed"
    payload["state_rad"][0] = 999
    payload["current_context"]["revision"] = 900
    again = box.frame_payload()
    for key in ("joints", "state_rad", "current_context", "rgb_sha256"):
        assert again[key] == original[key]
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.revision = 900


def test_commands_cannot_relabel_old_pixels_or_add_motion(session, monkeypatch):
    session.prepare()
    box, before = published(session)
    original_observe = session.sim.observe
    calls = []
    monkeypatch.setattr(session.sim, "observe", lambda: calls.append(1) or original_observe())
    act(session, "task", {"instruction": "Generated fixture"})
    box.publish(session)
    after = box.frame_payload()
    assert not calls and not session.sim.applications and session.journal.count == 0
    assert after["revision"] == before["revision"]
    assert after["current_context"]["revision"] == before["revision"] + 1
    assert after["observation_received_monotonic_ns"] == before["observation_received_monotonic_ns"]
    with pytest.raises(ValueError, match="context changed"):
        admit(after)
    act(session, "start")
    box.publish(session)
    running = box.frame_payload()
    admit(running)
    assert running["revision"] == session.revision
    assert running["active_episode_id"] == running["episode_id"]
    assert len(calls) == 1
    session.tick()
    box.publish(session)
    moved = box.frame_payload()
    assert admit(moved).observation.step == 1
    assert moved["rgb_sha256"] != running["rgb_sha256"]
    count, rows, applications = len(calls), session.journal.count, len(session.sim.applications)
    monkeypatch.setattr(
        session.sim, "observe", lambda: (_ for _ in ()).throw(RuntimeError("blocked"))
    )
    result = act(session, "pause")
    session.tick()
    box.publish(session)
    paused = box.frame_payload()
    assert result["mode"] == "paused" and session.fault is None
    assert len(calls) == count and session.journal.count == rows
    assert len(session.sim.applications) == applications
    assert paused["mode"] == "running" and paused["current_context"]["mode"] == "paused"
    assert paused["rgb_sha256"] == moved["rgb_sha256"]
    with pytest.raises(ValueError, match="context changed"):
        admit(paused)
    monkeypatch.setattr(session.sim, "observe", original_observe)
    act(session, "start")
    box.publish(session)
    with pytest.raises(ValueError, match="context changed"):
        admit(box.frame_payload())
    session.tick()
    box.publish(session)
    assert admit(box.frame_payload()).observation.revision == session.revision


def test_reset_restart_fault_and_close_cannot_reuse_context(session, tmp_path, monkeypatch):
    session.prepare()
    box, before = published(session)
    act(session, "reset")
    box.publish(session)
    reset = box.frame_payload()
    assert reset["episode_id"] != before["episode_id"]
    assert admit(reset).observation.revision == session.revision
    other = make_session(tmp_path / "other")
    try:
        other.prepare()
        box.publish(other)
        with pytest.raises(ValueError, match="context changed"):
            admit(box.frame_payload(), expected_session_id=before["session_id"])
    finally:
        other.close()
    act(session, "task", {"instruction": "Generated fixture"})
    act(session, "start")
    box.publish(session)
    old = box.frame_payload()
    monkeypatch.setattr(
        session.journal, "append", lambda *_: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        session.tick()
    box.publish(session)
    faulted = box.frame_payload()
    assert faulted["rgb_sha256"] == old["rgb_sha256"]
    assert faulted["current_context"]["mode"] == "faulted"
    with pytest.raises(ValueError, match="context changed"):
        admit(faulted)
    session.close()
    box.publish(session)
    with pytest.raises(ValueError, match="context changed"):
        admit(box.frame_payload())


def test_source_age_does_not_reset_on_republish_or_poll(session, monkeypatch):
    import firebird_teaching.control as control

    session.prepare()
    acquired = session.frame_snapshot.observation_received_monotonic_ns
    clock = [acquired + 100]
    monkeypatch.setattr(control.time, "monotonic_ns", lambda: clock[0])
    box, first = published(session)
    clock[0] += 4_000_000_000
    box.publish(session)
    second = box.frame_payload()
    assert second["source_age_ns"] == 4_000_000_100
    assert second["observation_received_monotonic_ns"] == first["observation_received_monotonic_ns"]
    assert admit(second).source_age_at_receipt_ns == 4_000_000_110
    clock[0] += 2_000_000_000
    with pytest.raises(ValueError, match="stale"):
        admit(box.frame_payload())


def test_remote_clock_epochs_are_not_compared_and_provider_retains_age(session):
    session.prepare()
    _, payload = published(session)
    payload["observation_received_monotonic_ns"] = 9_000_000_000_000
    payload["published_monotonic_ns"] = 9_000_000_000_100
    payload["source_age_ns"] = 4_000_000_000
    admitted = admit(payload, request_started_monotonic_ns=100, received_monotonic_ns=200)
    source = admitted.provider_input()
    assert source.received_monotonic_ns == 200
    assert source.source_age_at_receipt_ns == 4_000_000_100
    clock = [200]
    api = Providers(ProviderSettings("generated-key-not-real"), clock_ns=lambda: clock[0])
    api._fresh(source, lambda: source.identity)
    clock[0] += 1_000_000_001
    with pytest.raises(ProviderError, match="stale"):
        api._fresh(source, lambda: source.identity)
    clock[0] = 199
    with pytest.raises(ProviderError, match="stale"):
        api._fresh(source, lambda: source.identity)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("available", False),
        ("units", "degrees"),
        ("session_id", "../bad"),
        ("revision", True),
        ("step", -1),
        ("step", True),
        ("sim_time", float("inf")),
        ("sim_time", -1),
        ("sim_time", 10**1000),
        ("mode", "faulted"),
        ("episode_id", "bad-preview"),
        ("active_episode_id", "mismatch"),
        ("camera_key", "another"),
        ("camera_prim", "/bad path"),
        ("observation_received_monotonic_ns", -1),
        ("published_monotonic_ns", 0),
        ("source_age_ns", -1),
        ("source_age_ns", 30_000_000_001),
        ("width", 1921),
        ("height", 0),
        ("width", True),
        ("rgb_base64", "invalid"),
        ("rgb_sha256", "0" * 64),
        ("joints", ["repeat", "repeat"]),
        ("joints", ["../bad"]),
        ("joints", ["a" * 97]),
        ("joints", []),
        ("joints", "wrong"),
        ("state_rad", [float("nan")] * 6),
        ("state_rad", [True] * 6),
        ("state_rad", [0]),
        ("state_rad", "wrong"),
    ],
)
def test_invalid_envelope_rejected(session, key, value):
    session.prepare()
    _, payload = published(session)
    payload[key] = value
    with pytest.raises(ValueError):
        admit(payload)


def test_schema_timing_and_encoding_rejections(session):
    session.prepare()
    _, payload = published(session)
    for change in ({"extra": 1}, {"current_context": {}}, {"current_context": []}):
        with pytest.raises(ValueError):
            admit(payload | change)
    missing = dict(payload)
    del missing["joints"]
    with pytest.raises(ValueError):
        admit(missing)
    for kwargs in (
        {"max_age_ns": 0},
        {"max_age_ns": 30_000_000_001},
        {"request_started_monotonic_ns": 30},
        {"received_monotonic_ns": True},
    ):
        with pytest.raises(ValueError):
            admit(payload, **kwargs)
    with pytest.raises(ValueError, match="source timing"):
        admit(payload | {"source_age_ns": 0})
    with pytest.raises(ValueError, match="encoding"):
        admit(payload | {"rgb_base64": "!" * len(payload["rgb_base64"])})
    with pytest.raises(ValueError):
        admit_frame(
            {"available": False},
            expected_session_id="generated",
            request_started_monotonic_ns=0,
            received_monotonic_ns=1,
        )


def test_frozen_snapshot_invalid_construction_and_clock(session):
    session.prepare()
    source = session.frame_snapshot
    with pytest.raises(ValueError):
        dataclasses.replace(source, rgb=bytearray(source.rgb))
    with pytest.raises(ValueError):
        dataclasses.replace(source, joints=list(source.joints))
    with pytest.raises(ValueError):
        dataclasses.replace(source, state_rad=list(source.state_rad))
    with pytest.raises(ValueError):
        source.payload(
            source.context,
            source.observation_received_monotonic_ns - 1,
            source.observation_received_monotonic_ns,
        )
    with pytest.raises(ValueError):
        source.payload(
            source.context,
            source.observation_received_monotonic_ns,
            source.observation_received_monotonic_ns - 1,
        )


def test_frame_http_roundtrip_and_owner_thread(session):
    session.prepare()
    box, _ = published(session)
    token = "x" * 32
    http = server(box, token)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        client = Client(f"http://127.0.0.1:{http.server_port}", token)
        started = time.monotonic_ns()
        payload = client.request("/frame")
        received = time.monotonic_ns()
        admitted = admit(
            payload, request_started_monotonic_ns=started, received_monotonic_ns=received
        )
        assert admitted.observation.rgb == session.observation.frame.rgb
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="simulator thread"):
                pool.submit(box.publish, session).result(timeout=2)
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)


def test_racing_reader_gets_complete_source_snapshots(tmp_path):
    box = Mailbox()
    ready = threading.Event()
    done = threading.Event()
    errors = []

    def owner():
        try:
            for index in range(3):
                s = make_session(tmp_path / f"capture-{index}")
                try:
                    act(s, "task", {"instruction": "Generated racing fixture"})
                    act(s, "start")
                    for _ in range(8):
                        s.tick()
                        box.publish(s)
                        ready.set()
                        time.sleep(0.001)
                finally:
                    s.close()
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=owner)
    thread.start()
    assert ready.wait(timeout=3)
    observed = 0
    try:
        while not done.is_set():
            payload = box.frame_payload()
            frame = admit(payload).observation
            assert frame.rgb == bytes([frame.step % 255, 64, 128]) * frame.width * frame.height
            assert frame.state_rad == (0.0,) * 6
            assert payload["rgb_sha256"] == hashlib.sha256(frame.rgb).hexdigest()
            observed += 1
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors and observed > 0


def test_empty_and_finished_frame_are_unavailable(session):
    box, payload = published(session)
    assert payload == {"available": False, "schema_version": 1}
    act(session, "task", {"instruction": "Generated fixture"})
    act(session, "start")
    session.tick()
    act(session, "finish")
    box.publish(session)
    assert box.frame_payload() == payload


def test_command_context_published_before_ack_without_waiting_for_tick(session):
    session.prepare()
    box, before = published(session)
    request = dataclasses.asdict(command(session, "task", {"instruction": "Generated fixture"}))
    box.submit(request)
    box.drain(session)
    assert box.get(request["command_id"])["status"] == "acknowledged"
    after = box.frame_payload()
    assert after["revision"] == before["revision"]
    assert after["current_context"]["revision"] == session.revision
    with pytest.raises(ValueError, match="context changed"):
        admit(after)
    assert not session.sim.applications and session.journal.count == 0


def test_direct_context_validation_rejects_modes_and_episode_mismatch(session):
    session.prepare()
    source = session.frame_snapshot
    for change in (
        {"mode": "unknown"},
        {"mode": "closed"},
        {"active_episode_id": "different", "mode": "running"},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(source, **change)
