"""Harness ownership tests use disposable normal-interpreter children, not a frozen proof."""

import errno
import os
import select
import signal
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from test_sidecar import module

harness = module("verify_frozen")


def test_clean_environment_has_no_ambient_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/private/source")
    monkeypatch.setenv("FIREBIRD_RUNTIME_CONFIG", "/private/runtime.json")
    monkeypatch.setenv("HTTPS_PROXY", "http://not-used.invalid")
    value = harness.clean_environment(tmp_path)
    assert set(value) == {"HOME", "TMPDIR", "PATH"}
    assert not Path(value["PATH"]).exists()


def test_cleanup_reaps_descendant_after_leader_exit(tmp_path):
    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "stdout=subprocess.DEVNULL); print(p.pid,flush=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0]
        descendant = int(process.stdout.readline())
        assert process.wait(timeout=10) == 0
        harness.cleanup_group(process)
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
    finally:
        harness.cleanup_group(process)
        process.stdout.close()


@pytest.mark.parametrize("interrupt", [signal.SIGINT, signal.SIGTERM, signal.SIGALRM])
def test_spawn_interruption_registers_and_reaps_owned_child(monkeypatch, interrupt):
    real_popen = subprocess.Popen
    observed = []

    def interrupted(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed.append(process)
        signal.raise_signal(interrupt)
        return process

    monkeypatch.setattr(harness.subprocess, "Popen", interrupted)
    with pytest.raises(InterruptedError):
        harness.spawn_owned(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
    assert len(observed) == 1 and observed[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(observed[0].pid, 0)


def test_transient_permission_requires_kernel_confirmation(monkeypatch):
    class Process:
        pid = 12345

        def poll(self):
            return 0

    observed = []

    def disappearing(pid, sig):
        observed.append((pid, sig))
        if len(observed) == 1:
            raise PermissionError("exiting")
        raise ProcessLookupError("gone")

    monkeypatch.setattr(harness.os, "killpg", disappearing)
    assert harness.group_exists(Process()) is False
    assert observed == [(12345, 0), (12345, 0)]


def test_persistent_permission_is_not_cleanup_success(monkeypatch):
    class Process:
        pid = 12345

        def poll(self):
            return 0

    def denied(pid, sig):
        raise PermissionError("unknown")

    monkeypatch.setattr(harness.os, "killpg", denied)
    with pytest.raises(PermissionError, match="unknown"):
        harness.group_exists(Process())


@pytest.mark.parametrize("result", ["http_error", "timeout", "success", "refused"])
def test_listener_shutdown_requires_connection_refusal(monkeypatch, result):
    def request(port, path):
        if result == "http_error":
            raise urllib.error.HTTPError("http://127.0.0.1/", 500, "error", {}, None)
        if result == "timeout":
            raise urllib.error.URLError(TimeoutError("not established"))
        if result == "refused":
            raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        return 200, b"{}"

    monkeypatch.setattr(harness, "request", request)
    if result == "refused":
        harness.assert_listener_closed(12345)
    else:
        with pytest.raises(AssertionError):
            harness.assert_listener_closed(12345)
