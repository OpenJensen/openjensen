import os
import signal
import sys

import pytest
from firebird_distill.application import Owner, environment


def test_environment_never_inherits_credentials(monkeypatch):
    for name in ("GOOGLE_APPLICATION_CREDENTIALS", "OPENROUTER_API_KEY", "HF_TOKEN", "HOME"):
        monkeypatch.setenv(name, "private")
        assert name not in environment([])
    assert environment([])["CUDA_VISIBLE_DEVICES"] == ""


def test_timeout_reaps_owned_child(monkeypatch):
    import subprocess

    original, children = subprocess.Popen, []

    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", capture)
    with pytest.raises(TimeoutError):
        with Owner(0.3) as owner:
            owner.run([sys.executable, "-c", "import time;time.sleep(30)"], environment([]))
    assert children and children[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)


def test_signal_in_spawn_window_is_not_lost(monkeypatch, tmp_path):
    import subprocess

    original = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        signal.raise_signal(signal.SIGTERM)
        return child

    handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(InterruptedError):
        with Owner(10) as owner:
            owner.run([sys.executable, "-c", "import time;time.sleep(30)"], environment([]))
    assert children[0].poll() is not None
    assert signal.getsignal(signal.SIGTERM) == handler


def test_successful_leader_does_not_leave_grandchild(tmp_path):
    import subprocess

    pidfile = tmp_path / "descendant"
    child = "import time;time.sleep(30)"
    command = (
        "import subprocess,pathlib,sys;"
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))"
    )
    with Owner(10) as owner:
        owner.run([sys.executable, "-c", command], environment([]))
    pid = int(pidfile.read_text())
    observed = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5
    )
    assert observed.returncode != 0 or observed.stdout.strip().startswith("Z")


def test_transient_cleanup_permission_error_requires_kernel_confirmation(monkeypatch):
    from firebird_distill.application import signal_owned_group

    observed = []

    def kill(pid, sig):
        observed.append((pid, sig))
        if sig:
            raise PermissionError("transient")
        raise ProcessLookupError("gone")

    monkeypatch.setattr(os, "killpg", kill)
    signal_owned_group(12345, signal.SIGTERM)
    assert observed == [(12345, signal.SIGTERM), (12345, 0)]


def test_real_cleanup_permission_failure_remains_visible(monkeypatch):
    from firebird_distill.application import signal_owned_group

    def kill(pid, sig):
        if sig:
            raise PermissionError("still alive")

    monkeypatch.setattr(os, "killpg", kill)
    with pytest.raises(PermissionError, match="still alive"):
        signal_owned_group(12345, signal.SIGTERM)
