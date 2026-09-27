"""Real remote-bootstrap ownership regressions using local sleeping processes."""

import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from vla_platform.lifecycle import sky_bootstrap


@pytest.fixture
def bootstrap_source():
    return Path(sky_bootstrap.__file__).resolve()


def test_bootstrap_remains_standalone_python311(bootstrap_source):
    ast.parse(bootstrap_source.read_text(), feature_version=(3, 11))


def wait_for(path, process):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert process.poll() is None, "Bootstrap exited before the worker started"
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {path.name}")
        time.sleep(0.01)


def alive(pid):
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5
    )
    # An adopted zombie is no longer executing; init owns its final reap.
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="The remote SkyPilot bootstrap runs on Linux")
@pytest.mark.parametrize(
    "mode",
    ["timeout", "term", "int", "spawn_window", "repeated", "leader_success", "leader_failure"],
)
def test_timeout_and_interrupt_stop_owned_tree_only(tmp_path, bootstrap_source, mode):
    grandchild = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path('grandchild.pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    worker = (
        "import os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        "Path('worker.pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    if mode.startswith("leader_"):
        worker = worker.rsplit("time.sleep(60)", 1)[0] + (
            "\nwhile not Path('grandchild.pid').exists(): time.sleep(0.01)\n"
            + f"raise SystemExit({0 if mode == 'leader_success' else 7})"
        )
    launcher = f"""
import importlib.util, json, os, signal, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("bootstrap", {str(bootstrap_source)!r})
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
original = {{s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}}
if {mode!r} == "spawn_window":
    actual_spawn = b.subprocess.Popen
    def spawn(*args, **kwargs):
        child = actual_spawn(*args, **kwargs)
        Path("spawn-window").touch()
        deadline = time.monotonic() + 5
        while not Path("release-spawn").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Spawn test was not released")
            time.sleep(0.01)
        return child
    b.subprocess.Popen = spawn
code = b.run_worker([sys.executable, "-c", {worker!r}], env=os.environ.copy(),
                    timeout={1 if mode == "timeout" else 30}, terminate_grace=0.3)
assert original == {{s: signal.getsignal(s) for s in original}}
Path("outcome.json").write_text(json.dumps({{"code": code}}))
"""
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    process = subprocess.Popen(
        [sys.executable, "-c", launcher], cwd=tmp_path, start_new_session=True
    )
    owned = []
    try:
        wait_for(tmp_path / "worker.pid", process)
        wait_for(tmp_path / "grandchild.pid", process)
        owned = [int((tmp_path / name).read_text()) for name in ("worker.pid", "grandchild.pid")]
        if mode not in {"timeout", "leader_success", "leader_failure"}:
            if mode == "spawn_window":
                assert (tmp_path / "spawn-window").exists()
            number = signal.SIGINT if mode == "int" else signal.SIGTERM
            os.kill(process.pid, number)
            if mode == "spawn_window":
                (tmp_path / "release-spawn").touch()
            if mode == "repeated":
                for _ in range(5):
                    time.sleep(0.02)
                    os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        assert process.returncode == 0
        code = json.loads((tmp_path / "outcome.json").read_text())["code"]
        expected = {"timeout": 124, "leader_success": 0, "leader_failure": 7}
        assert code == (expected[mode] if mode in expected else 128 + number)
        deadline = time.monotonic() + 3
        while any(alive(pid) for pid in owned) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(alive(pid) for pid in owned)
        assert unrelated.poll() is None
    finally:
        for pid in owned:
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="The remote SkyPilot bootstrap runs on Linux")
@pytest.mark.parametrize("code", [0, 7])
def test_worker_exit_and_handlers_preserved(code):
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    assert (
        sky_bootstrap.run_worker(
            [sys.executable, "-c", f"raise SystemExit({code})"], env=os.environ.copy(), timeout=5
        )
        == code
    )
    assert handlers == {s: signal.getsignal(s) for s in handlers}
