"""Forced owner termination with real API jobs and workers; no external networking."""

import json
import os
import socket
import socketserver
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]


def eventually(check, label, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {label}")


class ConnectGate(socketserver.ThreadingTCPServer):
    """Hold real HTTPX worker connections without opening an upstream socket."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self):
        self.release = threading.Event()
        self.requests = []
        super().__init__(("127.0.0.1", 0), ConnectHandler)


class ConnectHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        request = self.rfile.readline(4096).decode("ascii").strip()
        # Do not forward the CONNECT request or perform any remote DNS lookup.
        self.server.requests.append(request)
        while self.rfile.readline(4096) not in {b"\r\n", b"\n", b""}:
            pass
        self.server.release.wait(timeout=25)
        try:
            self.wfile.write(
                b"HTTP/1.1 502 Recovery test gate released\r\nContent-Length: 0\r\n\r\n"
            )
        except BrokenPipeError, ConnectionResetError:
            pass


@contextmanager
def application(tmp_path, environment, iteration):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with (tmp_path / f"owner-{iteration}.log").open("w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "vla_platform.cli", "serve", "--port", str(port)],
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:

                def healthy():
                    if process.poll() is not None:
                        log.seek(0)
                        raise AssertionError(f"Application startup failed: {log.read()}")
                    try:
                        return client.get("/api/v1/health", timeout=0.5).status_code == 200
                    except httpx.HTTPError:
                        return False

                eventually(healthy, f"owner {iteration} readiness")
                yield process, client
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def test_forced_stop_reconciles_real_jobs_in_new_processes(tmp_path):
    metadata = tmp_path / "datasets" / "fixture" / "meta"
    metadata.mkdir(parents=True)
    (metadata / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 1,
                "total_frames": 3,
                "fps": 10,
                "features": {
                    "action": {"dtype": "float32", "shape": [2]},
                    "observation.state": {"dtype": "float32", "shape": [2]},
                },
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    gate = ConnectGate()
    thread = threading.Thread(target=gate.serve_forever, daemon=True)
    thread.start()
    proxy = f"http://127.0.0.1:{gate.server_address[1]}"
    environment = {
        **{
            key: value
            for key, value in os.environ.items()
            if key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
        },
        "FIREBIRD_DATA_DIR": str(workspace),
        "FIREBIRD_LOCAL_DATA_ROOT": str(tmp_path / "datasets"),
        "FIREBIRD_WEB_DIR": str(tmp_path / "no-web"),
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "ALL_PROXY": proxy,
        "NO_PROXY": "127.0.0.1,localhost,::1",
    }
    running = []
    proof = {"fixture": "loopback CONNECT gate; actual CLI/API/metadata workers"}
    try:
        with application(tmp_path, environment, 1) as (first, client):
            project = client.post("/api/v1/projects", json={"name": "Forced-stop fixture"})
            assert project.status_code == 201
            endpoint = f"/api/v1/projects/{project.json()['id']}"

            def submit(payload):
                response = client.post(endpoint + "/intakes", json=payload)
                assert response.status_code == 202, response.text
                return response.json()["id"]

            def finished(job_id, status):
                record = client.get(f"/api/v1/jobs/{job_id}").json()
                return record if record["status"] == status else None

            succeeded = submit({"source": "local", "path": "fixture"})
            eventually(lambda: finished(succeeded, "succeeded"), "completed local fixture")
            failed = submit({"source": "local", "path": "missing"})
            eventually(lambda: finished(failed, "failed"), "failed local fixture")
            for _ in range(2):
                running.append(submit({"repo_id": "recovery-fixture/blocked"}))
            eventually(lambda: len(gate.requests) == 2, "two held real worker connections")
            cancelled = submit({"repo_id": "recovery-fixture/cancelled"})
            response = client.post(f"/api/v1/jobs/{cancelled}/cancel")
            assert response.status_code == 200 and response.json()["status"] == "cancelled"
            queued = submit({"repo_id": "recovery-fixture/queued"})
            before = {job["id"]: job for job in client.get(endpoint + "/jobs").json()}
            assert [before[job]["status"] for job in running] == ["running", "running"]
            assert before[queued]["status"] == "queued"
            terminals = {job: before[job] for job in [succeeded, failed, cancelled]}
            proof["first_owner_pid"] = first.pid
            proof["before_kill"] = {job: record["status"] for job, record in before.items()}
            # kill() is SIGKILL on POSIX and TerminateProcess on Windows: no lifespan cleanup.
            first.kill()
            first.wait(timeout=10)
            with sqlite3.connect(workspace / "workspace.sqlite3") as database:
                persisted = dict(database.execute("select id, status from jobs"))
            assert persisted == proof["before_kill"]
            proof["hard_stop_exit_code"] = first.returncode

        with application(tmp_path, environment, 2) as (second, client):
            assert second.pid != first.pid
            recovered = {job["id"]: job for job in client.get(endpoint + "/jobs").json()}
            for job in [*running, queued]:
                assert recovered[job]["status"] == "interrupted"
                assert recovered[job]["result"] is None
                assert "Submit a new inspection" in recovered[job]["error"]
            assert {job: recovered[job] for job in terminals} == terminals
            assert not (workspace / "jobs" / queued).exists()
            assert not (workspace / "jobs" / cancelled).exists()
            for job in running:
                assert not (workspace / "jobs" / job / "result.json").exists()
            gate.release.set()
            for job in running:
                result_path = workspace / "jobs" / job / "result.json"
                eventually(result_path.is_file, f"orphan worker {job} late result")
                late = json.loads(result_path.read_text(encoding="utf-8"))
                assert late["result"] is None and "ProxyError" in late["error"]
            assert {job["id"]: job for job in client.get(endpoint + "/jobs").json()} == recovered
            assert gate.requests == ["CONNECT huggingface.co:443 HTTP/1.1"] * 2
            proof["second_owner_pid"] = second.pid
            proof["after_restart"] = {job: record["status"] for job, record in recovered.items()}
            proof["late_worker_files_ignored"] = len(running)
            proof["terminal_records_unchanged"] = [
                record["status"] for record in terminals.values()
            ]

        with application(tmp_path, environment, 3) as (third, client):
            assert third.pid not in {first.pid, second.pid}
            assert {job["id"]: job for job in client.get(endpoint + "/jobs").json()} == recovered
            assert len(gate.requests) == 2
            assert not (workspace / "jobs" / queued).exists()
            proof["third_owner_pid"] = third.pid
            proof["second_restart_idempotent"] = True
            proof["worker_connect_attempts"] = len(gate.requests)
    finally:
        # Release every held worker even on assertion failure; no descendant PID guessing.
        gate.release.set()
        gate.shutdown()
        gate.server_close()
        thread.join(timeout=5)
        for job in running:
            if (workspace / "jobs" / job / "request.json").is_file():
                eventually(
                    (workspace / "jobs" / job / "result.json").is_file,
                    f"controlled worker {job} cleanup",
                    timeout=25,
                )
    proof["cleanup"] = "all owners stopped; held workers released and result files observed"
    print("RECOVERY_PROOF " + json.dumps(proof, sort_keys=True))
