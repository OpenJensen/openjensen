"""Generated raw captures and real disposable CPU processes, never Isaac acceptance."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import act

from firebird_teaching.dataset import inspect_capture
from firebird_teaching.managed import capture_size, verify_capture


def test_verifier_preserves_existing_applied_action_and_provenance(session, tmp_path):
    act(session, "task", {"instruction": "Generated software capture"})
    act(session, "start")
    episode = session.episode_id
    act(session, "correct", {"joint": "joint_0", "delta_rad": 0.15})
    session.tick()
    act(session, "finish")
    session.close()
    root = session.journal.root
    original = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    target = tmp_path / "verified.json"
    verify_capture(root, target, 1024**2)
    receipt = json.loads(target.read_text())
    meta, selected, inventory = inspect_capture(root, [episode])
    assert receipt["origin"] == "synthetic"
    assert receipt["session_id"] == meta["session_id"]
    assert receipt["lineage_group"] == meta["lineage_group"]
    assert receipt["inventory"] == inventory
    row = selected[0]["rows"][0]
    assert row["requested_target_rad"] != row["applied_target_rad"]
    assert row["applied_target_rad"][0] == pytest.approx(2 / 30)
    assert original == {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_verifier_does_not_ignore_an_unfinished_episode(session, tmp_path):
    act(session, "task", {"instruction": "Generated incomplete capture"})
    act(session, "start")
    session.tick()
    act(session, "finish")
    act(session, "start")
    session.tick()
    session.journal.abort()
    receipt = tmp_path / "no-receipt.json"
    with pytest.raises((ValueError, OSError)):
        verify_capture(session.journal.root, receipt, 1024**2)
    assert not receipt.exists()


def test_capture_budget_and_link_refusal(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "frame").write_bytes(b"1234")
    with pytest.raises(ValueError, match="byte"):
        capture_size(root, 3)
    (root / "linked").symlink_to(root / "frame")
    with pytest.raises(ValueError, match="node"):
        capture_size(root, 100)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group supervision")
@pytest.mark.parametrize("action", ["stop", "parent_eof", "cancel"])
def test_real_owned_group_stops_reaps_and_retains_reason(tmp_path, action):
    """The child is explicitly a generated CPU fixture, not a substitute Isaac backend."""
    import fcntl

    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "Path(sys.argv[1]).write_text(str(grandchild.pid))\n"
        "while True: time.sleep(.02)\n"
    )
    ready = tmp_path / "ready"
    terminal = tmp_path / "terminal.json"
    lease_path = tmp_path / "lease"
    lease = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = (
        "import sys; from pathlib import Path; from firebird_teaching.managed import supervise; "
        "supervise([sys.executable, sys.argv[1], sys.argv[2]], Path(sys.argv[3]), "
        "Path(sys.argv[4]), Path(sys.argv[5]), 20, 1048576, int(sys.argv[6]))"
    )
    worker = Path(__file__).resolve().parents[1]
    environment = os.environ | {
        "PYTHONPATH": str(worker) + os.pathsep + str(worker.parent / "isaac_sim"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(child),
            str(ready),
            str(tmp_path / "capture"),
            str(terminal),
            str(tmp_path / "log"),
            str(lease),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        pass_fds=(lease,),
        env=environment,
    )
    os.close(lease)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), (
            process.stderr.read() if process.poll() is not None else "child not ready"
        )
        contender = os.open(lease_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
        if action == "stop":
            process.stdin.write(b"stop\n")
            process.stdin.flush()
        elif action == "parent_eof":
            process.stdin.close()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=15) == -signal.SIGKILL
        value = json.loads(terminal.read_text())
        assert (
            value["reason"]
            == {"stop": "requested_stop", "parent_eof": "parent_lost", "cancel": "cancelled"}[
                action
            ]
        )
        assert value["group_id"] == process.pid
        # Descendants can briefly remain zombies until init reaps them; no running
        # worker is acceptable, and the application has a stricter group-absence gate.
        probe = subprocess.run(
            ["ps", "-o", "stat=", "-p", ready.read_text()],
            text=True,
            capture_output=True,
            check=False,
        )
        assert not probe.stdout.strip() or probe.stdout.strip().startswith("Z")
        contender = os.open(lease_path, os.O_RDWR)
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stderr:
            process.stderr.close()
