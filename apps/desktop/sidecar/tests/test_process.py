"""Owned normal-interpreter processes and disposable workspaces; no native bundle claim."""

import hashlib
import json
import os
import select
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from test_sidecar import ROOT, SIDE, resource_tree

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX pipe harness; Windows packaging unverified"
)
NONCE = "b" * 32


def metadata(root):
    path = root / "fixture/meta/info.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "test_fixture",
                "total_episodes": 2,
                "total_frames": 20,
                "fps": 10,
                "features": {
                    "action": {"dtype": "float32", "shape": [6]},
                    "observation.state": {"dtype": "float32", "shape": [6]},
                },
            }
        )
    )
    return path


class Child:
    def __init__(self, root, resources, data, *, local=None):
        self.root = root
        self.log = (root / f"child-{time.monotonic_ns()}.log").open("wb")
        command = [
            sys.executable,
            str(SIDE / "entrypoint.py"),
            "serve",
            "--resources",
            str(resources),
            "--data-dir",
            str(data),
        ]
        if local:
            command += ["--local-root", str(local)]
        fake_worker = root / "fake-decision/src/firebird_decision/__main__.py"
        fake_worker.parent.mkdir(parents=True, exist_ok=True)
        fake_worker.write_text("raise AssertionError('must never run')")
        env = {
            "FIREBIRD_DECISION_PYTHON": sys.executable,
            "FIREBIRD_DECISION_ROOT": str(root / "fake-decision"),
            "FIREBIRD_DECISION_MODEL_DIR": str(root),
            "FIREBIRD_DECISION_ACCEPT_LICENSE": "CC-BY-NC-SA-4.0",
            "GEMINI_API_KEY": "not-a-real-secret",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT / "packages/core/src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            # These must not override explicit desktop paths.
            "FIREBIRD_DATA_DIR": str(root / "wrong-data"),
            "FIREBIRD_WEB_DIR": str(root / "wrong-static"),
            "FIREBIRD_RUNTIME_CONFIG": str(root / "missing-runtimes.json"),
        }
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=env,
            cwd=root,
        )
        self.pending = bytearray()

    def send(self, command="start", nonce=NONCE):
        self.process.stdin.write(
            json.dumps({"schema_version": 1, "command": command, "nonce": nonce}).encode() + b"\n"
        )
        self.process.stdin.flush()

    def record(self, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if b"\n" in self.pending:
                line, _, self.pending = self.pending.partition(b"\n")
                return json.loads(line)
            if select.select(
                [self.process.stdout], [], [], min(0.1, max(0, deadline - time.monotonic()))
            )[0]:
                block = os.read(self.process.stdout.fileno(), 4097)
                if not block:
                    raise AssertionError(
                        f"Child exited without response: {Path(self.log.name).read_text()}"
                    )
                self.pending.extend(block)
                assert len(self.pending) <= 8192
        raise AssertionError(f"Child response timeout: {Path(self.log.name).read_text()}")

    def finish(self, code=0):
        assert self.process.wait(timeout=15) == code, Path(self.log.name).read_text()

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process.stdout.close()
        self.log.close()


@contextmanager
def child(root, resources, data, **kwargs):
    owned = Child(root, resources, data, **kwargs)
    try:
        yield owned
    finally:
        owned.close()


def client_for(ready):
    assert ready["event"] == "ready" and ready["nonce"] == NONCE
    assert ready["host"] == "127.0.0.1" and 0 < ready["port"] < 65536
    assert len(ready["resources_sha256"]) == len(ready["workspace_id"]) == 64
    return httpx.Client(base_url=f"http://127.0.0.1:{ready['port']}", timeout=5, trust_env=False)


def test_http_intake_shutdown_restart_preserves_workspace_and_static(tmp_path):
    resources, data, datasets = tmp_path / "resources", tmp_path / "data", tmp_path / "datasets"
    resource_tree(resources)
    info = metadata(datasets)
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [resources / "resources.json", resources / "web/index.html", info]
    }
    with child(tmp_path, resources, data, local=datasets) as owned:
        owned.send()
        with client_for(owned.record()) as api:
            assert api.get("/api/v1/health").json()["status"] == "ok"
            assert api.get("/api/v1/decision/status").json()["configured"] is False
            assert api.get("/api/v1/augmentation-options").json()["auth_mode"] == "unconfigured"
            assert api.get("/").content == (resources / "web/index.html").read_bytes()
            response = api.post("/api/v1/projects", json={"name": "Sidecar fixture"})
            assert response.status_code == 201
            project = response.json()
            response = api.post(
                f"/api/v1/projects/{project['id']}/intakes",
                json={"source": "local", "path": "fixture"},
            )
            assert response.status_code == 202, response.text
            job = response.json()
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                job = api.get(f"/api/v1/jobs/{job['id']}").json()
                if job["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.05)
            assert job["status"] == "succeeded", job
            assert job["result"]["total_frames"] == 20
        owned.send("shutdown")
        assert owned.record()["reason"] == "shutdown"
        owned.finish()
    assert not (tmp_path / "wrong-data").exists()
    with child(tmp_path, resources, data, local=datasets) as owned:
        owned.send()
        with client_for(owned.record()) as api:
            assert api.get("/api/v1/projects").json() == [project]
            assert api.get(f"/api/v1/jobs/{job['id']}").json() == job
        owned.process.stdin.close()
        assert owned.record()["reason"] == "parent_eof"
        owned.finish()
    assert before == {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in before}


def test_second_child_cannot_take_workspace_lock_or_stop_first(tmp_path):
    resources, data = tmp_path / "resources", tmp_path / "data"
    resource_tree(resources)
    with child(tmp_path, resources, data) as first:
        first.send()
        with client_for(first.record()) as api:
            with child(tmp_path, resources, data) as second:
                second.send()
                second.finish(code=1)
                assert second.process.stdout.read() == b""
            assert api.get("/api/v1/health").status_code == 200
        first.send("shutdown")
        first.record()
        first.finish()


@pytest.mark.parametrize("fault", ["wrong-nonce", "oversize", "malformed"])
def test_invalid_control_shuts_down_owned_server(tmp_path, fault):
    resources, data = tmp_path / "resources", tmp_path / "data"
    resource_tree(resources)
    with child(tmp_path, resources, data) as owned:
        owned.send()
        ready = owned.record()
        if fault == "wrong-nonce":
            owned.send("shutdown", nonce="c" * 32)
        else:
            owned.process.stdin.write(b"x" * 4097 if fault == "oversize" else b"[]\n")
            owned.process.stdin.flush()
        owned.finish(code=1)
        with client_for(ready) as api, pytest.raises(httpx.ConnectError):
            api.get("/api/v1/health")


def test_parent_eof_before_start_does_not_create_workspace(tmp_path):
    with child(tmp_path, tmp_path / "absent-resources", tmp_path / "data") as owned:
        owned.process.stdin.close()
        owned.finish()
        assert owned.process.stdout.read() == b""
    assert not (tmp_path / "data").exists()


def test_fixed_intake_mode_uses_same_worker_and_preserves_originals(tmp_path):
    source = metadata(tmp_path / "datasets")
    raw = source.read_bytes()
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(
        json.dumps(
            {
                "intake": {"source": "local", "path": "fixture"},
                "local_root": str(tmp_path / "datasets"),
            }
        )
    )
    command = [
        sys.executable,
        str(SIDE / "entrypoint.py"),
        "intake-worker",
        str(request),
        str(result),
    ]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "packages/core/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(command, env=env, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    saved = result.read_bytes()
    assert json.loads(saved)["result"]["total_frames"] == 20
    assert source.read_bytes() == raw
    repeated = subprocess.run(command, env=env, capture_output=True, timeout=30)
    assert repeated.returncode == 1
    assert result.read_bytes() == saved


def test_parent_eof_after_handshake_before_start_does_not_open_workspace(tmp_path):
    resources = tmp_path / "resources"
    resource_tree(resources)
    with child(tmp_path, resources, tmp_path / "data") as owned:
        owned.send()
        owned.process.stdin.close()
        message = owned.record()
        assert message["event"] == "stopped" and message["reason"] == "parent_eof"
        owned.finish()
    assert not (tmp_path / "data").exists()
