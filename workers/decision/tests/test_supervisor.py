import os
import signal
import subprocess
import sys
import time

import pytest

from firebird_decision import supervisor
from firebird_decision.contracts import DecisionError


def test_real_child_result_and_stderr_is_not_reflected():
    assert supervisor.run_owned([sys.executable, "-c", 'print("done")'], timeout=2) == b"done\n"
    with pytest.raises(DecisionError) as caught:
        supervisor.run_owned(
            [
                sys.executable,
                "-c",
                'import sys;print("secret/private/token",file=sys.stderr);sys.exit(1)',
            ],
            timeout=2,
        )
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "code,timeout,match",
    [
        ("import time;time.sleep(30)", 0.15, "deadline"),
        ('import sys;sys.stdout.write("x"*1000000);sys.stdout.flush()', 2, "bounded"),
        ('import sys;sys.stderr.write("x"*1000000);sys.stderr.flush()', 2, "bounded"),
    ],
)
def test_deadline_and_output_limit_kill_and_reap(code, timeout, match, monkeypatch):
    children = []
    popen = subprocess.Popen

    def track(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", track)
    before = time.monotonic()
    with pytest.raises(DecisionError, match=match):
        supervisor.run_owned([sys.executable, "-c", code], timeout=timeout)
    assert time.monotonic() - before < 4
    assert children[0].poll() is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(children[0].pid, os.WNOHANG)


def test_signal_during_creation_and_repeated_cleanup_reaps_child(monkeypatch):
    children = []
    popen = subprocess.Popen
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def create_then_signal(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        wait = child.wait

        def interrupt_wait(*args, **kwargs):
            signal.raise_signal(signal.SIGTERM)
            return wait(*args, **kwargs)

        child.wait = interrupt_wait
        signal.raise_signal(signal.SIGINT)
        return child

    monkeypatch.setattr(subprocess, "Popen", create_then_signal)
    with pytest.raises(DecisionError, match="interrupted"):
        supervisor.run_owned([sys.executable, "-c", "import time;time.sleep(30)"], timeout=2)
    assert children[0].poll() is not None
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())


def test_offline_guard_rejects_python_network_and_subprocess_operations():
    script = """
import sys
from firebird_decision.scorer import offline_guard
from firebird_decision.contracts import DecisionError
offline_guard()
for event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
    try:
        sys.audit(event, None)
    except DecisionError:
        continue
    raise AssertionError(event)
print('blocked')
"""
    assert supervisor.run_owned([sys.executable, "-c", script], timeout=2) == b"blocked\n"


def test_child_error_code_is_actionable_without_reflecting_private_text():
    code = """
import json, sys
print(json.dumps({'schema_version':1,'code':'token_limit','error':'secret/private/token'}),file=sys.stderr)
sys.exit(2)
"""
    with pytest.raises(DecisionError, match="512 tokens") as caught:
        supervisor.run_owned([sys.executable, "-c", code], timeout=2)
    assert "secret" not in str(caught.value)
    assert caught.value.code == "token_limit"
