"""Real stdlib children exercise the startup owner's descendant boundary, without ML."""

import json
import os
import signal
import subprocess
import sys
import time

import measure_startup as startup
import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Owned POSIX process groups")

CHILD = r"""
import os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(30)  # Emergency fixture lifetime; the tested owner must kill this earlier.
"""
LEADER = r"""
import json, os, pathlib, subprocess, sys, time
class Runtime:
    def __init__(self):
        self.child = subprocess.Popen([sys.executable, '-I', '-c', sys.argv[1], sys.argv[2]])
        deadline = time.monotonic() + 5
        while not pathlib.Path(sys.argv[2]).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('fixture descendant did not become ready')
            time.sleep(.01)
        if sys.argv[3] == 'constructor-failure':
            raise RuntimeError('generated constructor failure after child spawn')
runtime = Runtime()
pathlib.Path(sys.argv[4], 'first-action.json').write_text(json.dumps({
    'status': 'passed', 'child_pid': os.getpid(), 'process_to_first_action_s': .001
}))
"""


def group_gone(process):
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False  # Darwin transient exit is not disappearance evidence.
    return False


def await_gone(process, seconds=2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if group_gone(process):
            return True
        time.sleep(0.01)
    return group_gone(process)


@pytest.mark.parametrize("mode", ["returned", "constructor-failure"])
def test_parent_cleans_term_ignoring_descendant_after_leader_exit(tmp_path, monkeypatch, mode):
    real_popen = subprocess.Popen
    children = []
    ready, output = tmp_path / "descendant.pid", tmp_path / "output"

    def launch(_command, **kwargs):
        child = real_popen(
            [sys.executable, "-I", "-c", LEADER, CHILD, str(ready), mode, str(output)],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(startup.subprocess, "Popen", launch)
    monkeypatch.setattr(startup, "CLEANUP_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_startup.py",
            "--backend",
            "generated-no-model",
            "--root",
            str(tmp_path),
            "--fixture",
            str(tmp_path / "unused"),
            "--output",
            str(output),
        ],
    )
    try:
        if mode == "constructor-failure":
            with pytest.raises(RuntimeError, match="Startup worker failed"):
                startup.main()
        else:
            startup.main()
        assert ready.exists(), "descendant barrier must prove it installed SIGTERM ignore"
        assert children[0].returncode is not None, "leader must be reaped"
        assert await_gone(children[0]), "TERM-ignoring descendant survived leader exit"
        if mode == "constructor-failure":
            assert not (output / "result.json").exists()
        else:
            assert json.loads((output / "result.json").read_text())["child_pid"] == children[0].pid
    finally:
        # Fail-first test cleanup owns only its created process group.
        for child in children:
            if not group_gone(child):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    assert await_gone(child), "cannot confirm fixture cleanup"
            child.wait(timeout=5)
            assert await_gone(child), "fixture group was not reaped"


def test_cleanup_permission_failure_is_not_reported_as_success(monkeypatch):
    class Process:
        pid = 999999

        def poll(self):
            return 0

    def denied(*_):
        raise PermissionError("generated group inspection denied")

    monkeypatch.setattr(startup.os, "killpg", denied)
    with pytest.raises(PermissionError):
        startup.cleanup_group(Process())
