"""Small stdlib/fake-observer checks; never load Torch or launch the native chain."""

import ctypes
import importlib.util
import os
import signal
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/control_chain_supervisor.py"
spec = importlib.util.spec_from_file_location("chain_supervisor_checks", SOURCE)
supervisor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervisor
spec.loader.exec_module(supervisor)


def test_fixed_chain_reuses_one_saved_student_and_both_packed_outputs(tmp_path):
    repo, output = tmp_path / "repo", tmp_path / "owner"
    stages = supervisor.stages(repo, "/pinned/python", output)
    assert [name for name, _ in stages] == [
        "generate",
        "distill",
        "verify-distill",
        "quant-8",
        "http-8",
        "quant-4",
        "http-4",
        "verify-final",
    ]
    root = output / "chain"
    assert stages[0][1][-2:] == ["generate", str(root)]
    assert stages[1][1] == [
        "/pinned/python",
        "-m",
        "firebird_distill.application",
        str(root / "distill-request.json"),
        str(root / "distill-result.json"),
    ]
    for bits, offset in ((8, 3), (4, 5)):
        quant, http = stages[offset][1], stages[offset + 1][1]
        assert quant[-2:] == [
            str(root / f"quant-{bits}-request.json"),
            str(root / f"quant-{bits}-result.json"),
        ]
        assert http[2:5] == [
            str(root / f"quant-{bits}/native-quantized/policy"),
            str(root / f"quant-{bits}/native-quantized/verification.json"),
            str(root / f"http-{bits}.json"),
        ]
        assert http[-7:] == [
            "--packed",
            "--forbid",
            str(root / "teacher"),
            "--forbid",
            str(root / "corpus"),
            "--forbid",
            str(root / "operation/distilled-policy/policy"),
        ]


def test_environment_does_not_inherit_user_paths_or_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/private/credential")
    monkeypatch.setenv("PYTHONPATH", "/other/checkout")
    monkeypatch.setenv("HOME", "/private/user")
    env = supervisor.clean_environment(tmp_path / "repo", tmp_path / "owner")
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
    assert "/other/checkout" not in env.values()
    assert env["HOME"] == str(tmp_path / "owner/home")
    assert env["TMPDIR"] == str(tmp_path / "owner/tmp")
    assert env["PYTHONPATH"].split(os.pathsep)[-1] == str(tmp_path / "repo/workers/isaac_sim")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
    ):
        assert env[key] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_new_owner_refuses_existing_or_linked_paths(tmp_path):
    root = tmp_path / "owner"
    supervisor.new_owner(root)
    assert sorted(p.name for p in root.iterdir()) == ["home", "logs", "tmp", "workdir"]
    assert not (root / "chain").exists()
    with pytest.raises(FileExistsError):
        supervisor.new_owner(root)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(supervisor.Refused, match="links"):
        supervisor.new_owner(link / "new")


def test_disk_accounting_includes_nested_temporary_snapshots(tmp_path):
    nested = tmp_path / "tmp/worker/snapshot"
    nested.mkdir(parents=True)
    (nested / "weights").write_bytes(b"x" * 8192)
    expected = sum(p.stat().st_blocks * 512 for p in [tmp_path, *tmp_path.rglob("*")])
    assert supervisor.allocated_bytes(tmp_path) == expected
    (nested / "link").symlink_to(nested / "weights")
    with pytest.raises(supervisor.Refused, match="link"):
        supervisor.allocated_bytes(tmp_path)


def test_disk_accounting_refuses_external_hardlinks(tmp_path):
    (tmp_path / "original").write_bytes(b"source")
    os.link(tmp_path / "original", tmp_path / "linked")
    with pytest.raises(supervisor.Refused, match="inventory"):
        supervisor.allocated_bytes(tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("owned_rss_bytes", supervisor.LIMITS["owned_rss_bytes"] + 1),
        ("allocated_output_bytes", supervisor.LIMITS["allocated_output_bytes"] + 1),
        ("free_disk_bytes", supervisor.LIMITS["free_disk_bytes"] - 1),
        ("swap_bytes", supervisor.LIMITS["swap_growth_bytes"] + 1),
    ],
)
def test_each_resource_limit_aborts_without_retry(key, value):
    sample = {
        "owned_rss_bytes": 1,
        "allocated_output_bytes": 1,
        "free_disk_bytes": supervisor.LIMITS["free_disk_bytes"],
        "swap_bytes": 0,
    }
    sample[key] = value
    with pytest.raises(supervisor.Refused):
        supervisor.resource_check(sample, 0, 1)


def test_work_deadline_is_not_reset_per_stage():
    sample = {
        "owned_rss_bytes": 1,
        "allocated_output_bytes": 1,
        "free_disk_bytes": supervisor.LIMITS["free_disk_bytes"],
        "swap_bytes": 0,
    }
    with pytest.raises(supervisor.Refused, match="deadline"):
        supervisor.resource_check(sample, 0, 600)


def test_fixed_ps_and_swap_parsers():
    assert supervisor.process_rows(" 31 1 31 42 501 S\n") == {31: (1, 31, 42 * 1024, 501, "S")}
    assert supervisor.swap_bytes("total = 2048.00M used = 128.00M free = 1920.00M") == 128 * 1024**2
    with pytest.raises(supervisor.Refused):
        supervisor.process_rows("31 1 31 unavailable S")
    with pytest.raises(supervisor.Refused):
        supervisor.swap_bytes("permission denied")


def test_bsd_kernel_layout_matches_sdk_64_bit_record():
    assert ctypes.sizeof(supervisor.BsdInfo) == 136
    assert supervisor.BsdInfo.pid.offset == 12
    assert supervisor.BsdInfo.ppid.offset == 16
    assert supervisor.BsdInfo.pgid.offset == 100
    assert supervisor.BsdInfo.seconds.offset == 120
    assert supervisor.BsdInfo.micros.offset == 128
    assert ctypes.sizeof(supervisor.VnodeStat) == 136
    assert supervisor.VnodeStat.inode.offset == 8
    assert ctypes.sizeof(supervisor.VnodeInfo) == 152
    assert ctypes.sizeof(supervisor.VnodePaths) == 2352


class Observer:
    def __init__(self, processes):
        self.processes = {p.pid: p for p in processes}
        self.tags = {}

    def rows(self):
        return {p.pid: (p.parent, p.group, p.rss, p.uid, "S") for p in self.processes.values()}

    def process(self, pid, rss=0):
        return self.processes.get(pid)

    def cwd_identity(self, pid):
        return self.tags.get(pid, (0, 0))


def proc(pid, parent, group, birth=1):
    return supervisor.Process(pid, (123, birth), parent, group, 4096, os.getuid())


def test_nested_new_session_is_retained_after_parent_exit():
    parent, child, grandchild = (
        proc(51001, 1, 51001),
        proc(51002, 51001, 51002),
        proc(51003, 51002, 51003),
    )
    observer = Observer([parent, child, grandchild])
    tracker = supervisor.Tracker(observer)
    tracker.register(parent)
    assert set(tracker.refresh()) == {51001, 51002, 51003}
    observer.processes = {51003: proc(51003, 1, 51003)}
    assert set(tracker.refresh()) == {51003}
    assert tracker.groups == {51001, 51002, 51003}
    assert len(tracker.history) == 3


def test_pid_reuse_prevents_signalling_successor(monkeypatch):
    original = proc(51001, 1, 51001)
    observer = Observer([original])
    tracker = supervisor.Tracker(observer)
    tracker.register(original)
    observer.processes[51001] = proc(51001, 1, 51001, birth=2)
    signals = []
    monkeypatch.setattr(os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(supervisor.Refused, match="reused"):
        tracker.signal_groups(signal.SIGKILL)
    assert signals == []


def test_leader_absence_does_not_prove_group_absence(monkeypatch):
    tracker = supervisor.Tracker(Observer([]))
    tracker.register(proc(51001, 1, 51001))
    monkeypatch.setattr(os, "killpg", lambda *_: None)
    with pytest.raises(supervisor.Refused, match="without a verified"):
        tracker.gone()


def test_permission_denial_never_means_cleanup_success(monkeypatch):
    tracker = supervisor.Tracker(Observer([]))
    tracker.register(proc(51001, 1, 51001))

    def denied(*_):
        raise PermissionError("observation denied")

    monkeypatch.setattr(os, "killpg", denied)
    with pytest.raises(supervisor.Refused, match="not kernel-confirmed"):
        tracker.gone()


def test_kernel_group_disappearance_confirms_cleanup(monkeypatch):
    tracker = supervisor.Tracker(Observer([]))
    tracker.register(proc(51001, 1, 51001))

    def absent(*_):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(os, "killpg", absent)
    assert tracker.gone()


def test_private_cwd_finds_orphan_before_first_ancestry_sample(tmp_path):
    orphan = proc(51002, 1, 51002)
    observer = Observer([orphan])
    tracker = supervisor.Tracker(observer, tmp_path)
    tracker.register(proc(51001, 1, 51001))  # Parent died before observing child.
    observer.tags[orphan.pid] = tracker.tag
    assert set(tracker.refresh()) == {orphan.pid}
    assert tracker.known[orphan.pid].birth == orphan.birth
    assert tracker.groups == {51001, 51002}


def test_wrong_cwd_tag_does_not_adopt_an_unrelated_process(tmp_path):
    other = proc(51002, 1, 51002)
    observer = Observer([other])
    tracker = supervisor.Tracker(observer, tmp_path)
    observer.tags[other.pid] = (tracker.tag[0], tracker.tag[1] + 1)
    assert tracker.refresh() == {}
    assert tracker.known == {}


def test_replaced_private_directory_refuses_ownership(tmp_path):
    tag = tmp_path / "workdir"
    tag.mkdir()
    tracker = supervisor.Tracker(Observer([]), tag)
    tag.rename(tmp_path / "preserved")
    tag.mkdir()
    with pytest.raises(supervisor.Refused, match="inode changed"):
        tracker.refresh()


def test_cwd_enrollment_rechecks_kernel_birth(monkeypatch, tmp_path):
    original = proc(51002, 1, 51002)
    observer = Observer([original])
    tracker = supervisor.Tracker(observer, tmp_path)
    observer.tags[original.pid] = tracker.tag
    sequence = iter([original, proc(51002, 1, 51002, birth=2)])
    monkeypatch.setattr(observer, "process", lambda *_: next(sequence))
    with pytest.raises(supervisor.Refused, match="birth changed"):
        tracker.refresh()
    assert not tracker.known


def test_denied_cwd_observation_never_proves_absence(monkeypatch, tmp_path):
    observer = Observer([proc(51002, 1, 51002)])
    tracker = supervisor.Tracker(observer, tmp_path)

    def denied(_pid):
        raise PermissionError("cwd observation denied")

    monkeypatch.setattr(observer, "cwd_identity", denied)
    with pytest.raises(PermissionError, match="denied"):
        tracker.gone()


@pytest.mark.parametrize("timed_out", [False, True])
def test_observation_failure_still_escalates_known_nested_sessions(monkeypatch, timed_out):
    parent, child = proc(51001, 1, 51001), proc(51002, 51001, 51002)
    observer = Observer([parent, child])
    tracker = supervisor.Tracker(observer)
    tracker.register(parent)
    tracker.register(child)

    def denied():
        if timed_out:
            raise supervisor.subprocess.TimeoutExpired("fixed observer", 2)
        raise supervisor.Refused("global observation failed")

    monkeypatch.setattr(observer, "rows", denied)
    calls = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append(("pid", pid, sig)))

    def kill_group(group, sig):
        calls.append(("group", group, sig))
        if sig == signal.SIGKILL:
            observer.processes.pop(group, None)

    monkeypatch.setattr(os, "killpg", kill_group)
    clock = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration)
    )

    class Child:
        pid = parent.pid

        def poll(self):
            return None if self.pid in observer.processes else 0

        def wait(self, timeout):
            assert self.pid not in observer.processes
            return 0

    owner = supervisor.Supervisor.__new__(supervisor.Supervisor)
    owner.observer, owner.tracker, owner.deadline = observer, tracker, 600
    with pytest.raises(supervisor.Refused, match="unverified"):
        owner.cleanup(Child())
    assert ("pid", parent.pid, signal.SIGTERM) == calls[0]
    assert ("group", parent.group, signal.SIGKILL) in calls
    assert ("group", child.group, signal.SIGKILL) in calls
    assert not observer.processes
    assert clock[0] < 15


def test_observer_command_timeout_is_a_fail_closed_observation(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise supervisor.subprocess.TimeoutExpired("fixed observer", 2)

    monkeypatch.setattr(supervisor.subprocess, "run", timeout)
    with pytest.raises(supervisor.Refused, match="unavailable"):
        supervisor.system_output(["/bin/ps"])


def test_fallback_skips_reused_pid_but_signals_other_verified_owner(monkeypatch):
    first, second = proc(51001, 1, 51001), proc(51002, 1, 51002)
    observer = Observer([first, second])
    tracker = supervisor.Tracker(observer)
    tracker.register(first)
    tracker.register(second)
    observer.processes[first.pid] = proc(first.pid, 1, first.group, birth=2)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda *args: calls.append(args))
    errors = tracker.signal_known(signal.SIGKILL)
    assert len(errors) == 1 and "reused" in errors[0]
    assert calls == [(second.group, signal.SIGKILL)]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"status":"passed","x":NaN}',
        b'{"status":"passed","x":1e309}',
        b'{"status":"failed","status":"passed"}',
    ],
)
def test_final_receipt_rejects_nonfinite_and_duplicate_json(raw):
    with pytest.raises(supervisor.Refused):
        supervisor.finite_json(raw)


def test_source_inventory_binds_provenance_but_ignores_isaac_environment(tmp_path):
    for name in supervisor.SOURCE_DIRS:
        (tmp_path / name).mkdir(parents=True)
    for name in supervisor.EXTRA_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixed input\n")
    provenance = tmp_path / supervisor.SOURCE_DIRS[0] / "provenance.py"
    provenance.write_text("# before\n")
    outside = tmp_path / "workers/isaac_sim/.venv/private.py"
    outside.parent.mkdir()
    outside.write_text("# excluded environment\n")
    before = supervisor.source_identity(tmp_path)
    assert not any(".venv" in path for path in before)
    provenance.write_text("# after\n")
    assert supervisor.source_identity(tmp_path) != before


def test_dry_plan_creates_no_directory_or_process(monkeypatch, tmp_path, capsys):
    output = tmp_path / "new"
    monkeypatch.setattr(
        sys, "argv", ["supervisor", "--python", "/existing/python", "--output", str(output)]
    )
    monkeypatch.setattr(
        supervisor.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("unexpected spawn")
    )
    assert supervisor.main() == 0
    assert not output.exists()
    assert '"execute": false' in capsys.readouterr().out
