"""Real process regressions for bounded native quantization ownership."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "act_optimizer" / "src"))
from firebird_quant.native_application import _environment, _Owner

pytestmark = pytest.mark.skipif(os.name != "posix", reason="This CPU owner supports POSIX only")


def stopped(pid):
    # A killed grandchild may briefly be an init-owned zombie; it cannot execute.
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=2
    )
    return result.returncode != 0 or result.stdout.strip().startswith("Z")


def wait_stopped(pid):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if stopped(pid):
            return
        time.sleep(0.03)
    pytest.fail("Owned descendant is still executing")


@pytest.mark.parametrize("exit_code", [0, 7])
def test_all_leader_exits_reap_group_descendant(tmp_path, exit_code):
    pidfile = tmp_path / "child"
    code = (
        "import subprocess,sys,pathlib;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid));sys.exit(int(sys.argv[2]))"
    )
    with (tmp_path / "log").open("wb") as log, _Owner(5) as owner:
        if exit_code:
            with pytest.raises(ValueError):
                owner.run(
                    [sys.executable, "-c", code, str(pidfile), str(exit_code)], _environment(), log
                )
        else:
            owner.run(
                [sys.executable, "-c", code, str(pidfile), str(exit_code)], _environment(), log
            )
    wait_stopped(int(pidfile.read_text()))


def test_deadline_kills_ignoring_child(tmp_path):
    pidfile = tmp_path / "child"
    code = (
        "import os,signal,time,sys,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)"
    )
    started = time.monotonic()
    with (tmp_path / "log").open("wb") as log, _Owner(0.3) as owner:
        with pytest.raises(TimeoutError):
            owner.run([sys.executable, "-c", code, str(pidfile)], _environment(), log)
    assert time.monotonic() - started < 4
    wait_stopped(int(pidfile.read_text()))


def test_spawn_window_signal_does_not_lose_child(tmp_path, monkeypatch):
    created = []
    original = subprocess.Popen
    with (tmp_path / "log").open("wb") as log, _Owner(5) as owner:

        def popen(*args, **kwargs):
            process = original(*args, **kwargs)
            created.append(process.pid)
            owner._signal(signal.SIGTERM, None)
            owner._signal(signal.SIGTERM, None)
            return process

        monkeypatch.setattr(subprocess, "Popen", popen)
        with pytest.raises(InterruptedError):
            owner.run([sys.executable, "-c", "import time;time.sleep(30)"], _environment(), log)
    monkeypatch.setattr(subprocess, "Popen", original)
    wait_stopped(created[0])


def test_real_repeated_signals_restore_previous_handlers(tmp_path):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    # The owned child sends both signals only after its parent has registered it.
    code = (
        "import os,signal,time,sys;time.sleep(.1);os.kill(int(sys.argv[1]),signal.SIGTERM);"
        "os.kill(int(sys.argv[1]),signal.SIGINT);time.sleep(30)"
    )
    with (tmp_path / "log").open("wb") as log, _Owner(5) as owner:
        with pytest.raises(InterruptedError):
            owner.run([sys.executable, "-c", code, str(os.getpid())], _environment(), log)
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_no_credentials_inherit(monkeypatch):
    for key in [
        "OPENROUTER_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
    ]:
        monkeypatch.setenv(key, "fixture-secret")
    environment = _environment()
    assert "fixture-secret" not in environment.values()
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_disappearing_group_permission_error_requires_kernel_confirmation(monkeypatch):
    from firebird_quant.native_application import _signal_group

    calls = []

    def kill(group, sig):
        calls.append((group, sig))
        if len(calls) < 3:
            raise PermissionError("disappearing fixture group")
        raise ProcessLookupError("gone")

    monkeypatch.setattr(os, "killpg", kill)
    _signal_group(123, signal.SIGKILL)
    assert calls == [(123, signal.SIGKILL), (123, 0), (123, 0)]


@pytest.mark.parametrize("probe_allowed", [False, True])
def test_permission_uncertainty_is_never_suppressed(monkeypatch, probe_allowed):
    from firebird_quant.native_application import _signal_group

    def kill(_group, sig):
        if sig == 0 and probe_allowed:
            return
        raise PermissionError("cleanup is not verified")

    monkeypatch.setattr(os, "killpg", kill)
    with pytest.raises(PermissionError, match="not verified"):
        _signal_group(123, signal.SIGKILL)


def test_bounded_offline_failure_diagnostic(tmp_path):
    from firebird_quant.native_application import _probe

    class FailingOwner:
        def run(self, command, environment, log):
            log.write(b"A" * 5000 + b"meaningful last error")
            raise ValueError("worker failed")

    with pytest.raises(ValueError) as error:
        _probe(FailingOwner(), "reload", tmp_path, tmp_path / "result")
    assert str(error.value).endswith("meaningful last error")
    assert len(str(error.value)) < 4200
