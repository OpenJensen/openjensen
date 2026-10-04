"""CPU-only crash evidence using unmodified app/worker entry points and real HTTP.

The POSIX process test deliberately leaves its small workspace/logs under /tmp for
review. A test-only startup gate holds metadata workers before their real entrypoint,
without patching the scheduler or seeding SQLite. SIGKILL targets only the owned server.
Orphans are released with valid metadata, proving late results stay unpublished;
failure cleanup signals only verified worker commands inside this unique workspace.
Native Windows process-lifecycle evidence remains a separate validation gate.
"""

import asyncio
import errno
import hashlib
import json
import os
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.dml import Update
from vla_platform.contracts import IntakeRequest, WorkerResult, now
from vla_platform.datasets.inspect import profile
from vla_platform.execution import Execution
from vla_platform.settings import Settings
from vla_platform.storage import Storage

GATE_NAME = "startup-gate"

METADATA = json.dumps(
    {
        "codebase_version": "v3.0",
        "total_episodes": 1,
        "total_frames": 2,
        "fps": 10,
        "features": {
            "action": {"dtype": "float32", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [1]},
        },
    }
).encode()


def eventually(check, description, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.025)
    raise AssertionError(f"Timed out waiting for {description}")


class ProcessHarness:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="firebird-job002-", dir="/tmp"))
        self.workspace = self.root / "workspace"
        self.datasets = self.root / "fixtures"
        self.datasets.mkdir()
        # Delay only owned metadata workers; keep metadata itself a regular file.
        (self.root / "sitecustomize.py").write_text(
            textwrap.dedent(f"""\
            import json
            import sys
            from pathlib import Path

            args = sys.orig_argv
            if args[-4:-2] == ["-m", "vla_platform.datasets.worker"]:
                request = json.loads(Path(args[-2]).read_text())
                gate = (
                    Path(request["local_root"]) / request["intake"]["path"] / "meta" / {GATE_NAME!r}
                )
                if gate.exists():
                    with gate.open("rb") as stream:
                        stream.read()
            """)
        )
        self.servers = []
        self.logs = []
        self.writers = {}
        self.job_ids = set()
        self.child_commands = {}
        # Pass an already-bound socket, avoiding a free-port discovery/bind race.
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.socket.getsockname()[1]}/api/v1",
            timeout=1,
            trust_env=False,
        )
        self.event("workspace", path=str(self.workspace))

    def event(self, event, **details):
        with (self.root / "evidence.jsonl").open("a") as handle:
            handle.write(json.dumps({"at": now(), "event": event, **details}) + "\n")

    def start(self):
        log = (self.root / f"server-{len(self.servers) + 1}.log").open("w")
        self.logs.append(log)
        env = {
            **os.environ,
            "FIREBIRD_DATA_DIR": str(self.workspace),
            "FIREBIRD_LOCAL_DATA_ROOT": str(self.datasets),
            "FIREBIRD_WEB_DIR": str(self.root / "no-static-assets"),
            "PYTHONPATH": os.pathsep.join((str(self.root), os.environ.get("PYTHONPATH", ""))),
        }
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "vla_platform.api:create_app",
            "--factory",
            "--fd",
            str(self.socket.fileno()),
        ]
        server = subprocess.Popen(
            command,
            cwd=self.root,
            env=env,
            pass_fds=(self.socket.fileno(),),
            start_new_session=True,
            stdout=log,
            stderr=log,
        )
        self.servers.append(server)
        self.event("server_started", pid=server.pid, command=command, cwd=str(self.root))

        def ready():
            assert server.poll() is None, Path(log.name).read_text()
            try:
                return self.client.get("/health").status_code == 200
            except httpx.HTTPError:
                return False

        eventually(ready, f"server {server.pid} readiness")
        self.event("server_ready", pid=server.pid)
        return server

    def api(self, method, path, payload=None, status=200):
        response = self.client.request(method, path, json=payload)
        assert response.status_code == status, response.text
        return response.json()

    def submit(self, project_id, path):
        job = self.api(
            "POST",
            f"/projects/{project_id}/intakes",
            {"source": "local", "path": path},
            status=202,
        )
        assert job["status"] == "queued"
        self.job_ids.add(job["id"])
        self.event("submitted", job=job)
        return job["id"]

    def job(self, job_id):
        return self.api("GET", f"/jobs/{job_id}")

    def wait_job(self, job_id, status):
        def matching():
            job = self.job(job_id)
            return job if job["status"] == status else None

        job = eventually(matching, f"job {job_id}: {status}")
        self.event("observed", job=job)
        return job

    def fixture(self, name, blocked=False):
        path = self.datasets / name / "meta" / "info.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(METADATA)
        if blocked:
            path = path.with_name(GATE_NAME)
            os.mkfifo(path)
        return path

    def hold_reader(self, path):
        def open_writer():
            try:
                return os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                return None

        # Opening the writer proves the real worker reached this exact FIFO.
        self.writers[path] = eventually(open_writer, f"worker reading {path}")

    def release_reader(self, path):
        writer = self.writers.pop(path)
        try:
            assert os.write(writer, METADATA) == len(METADATA)
        finally:
            os.close(writer)
        self.event("orphan_input_released", path=str(path))

    def workers(self):
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,stat=,command="],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        # Settings resolves macOS /tmp to /private/tmp in worker arguments.
        workspace = self.workspace.resolve()
        expected = {
            (
                "-m",
                "vla_platform.datasets.worker",
                str(workspace / "jobs" / job_id / "request.json"),
                str(workspace / "jobs" / job_id / "result.json"),
            )
            for job_id in self.job_ids
        }
        found = {}
        for line in output.splitlines():
            fields = line.split(None, 3)
            if len(fields) != 4 or str(workspace) not in fields[3]:
                continue
            pid, parent, state, command = fields
            if tuple(shlex.split(command)[-4:]) not in expected or state.startswith("Z"):
                continue
            pid, parent = int(pid), int(parent)
            # Discover only our servers' children; reparented children must be known.
            if parent in {server.pid for server in self.servers}:
                self.child_commands[pid] = command
            if self.child_commands.get(pid) == command:
                found[pid] = {"ppid": parent, "command": command}
        return found

    def records(self):
        # Read-only DB verification; all records originate in real API submissions.
        with sqlite3.connect(
            f"file:{self.workspace / 'workspace.sqlite3'}?mode=ro", uri=True
        ) as connection:
            rows = connection.execute("SELECT id, status, record FROM jobs").fetchall()
        records = {}
        for job_id, status, raw in rows:
            record = json.loads(raw)
            assert record["status"] == status
            records[job_id] = record
        return records

    def stop(self, server):
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        self.event("server_stopped", pid=server.pid, returncode=server.returncode)

    def close(self):
        try:
            for server in self.servers:
                self.stop(server)
            for writer in self.writers.values():
                os.close(writer)
            self.writers.clear()
            # Error-path cleanup only: re-check the exact unique workspace command
            # before every signal. Never signal unrelated processes or process groups.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for pid in self.workers():
                    try:
                        os.kill(pid, sig)
                        self.event("owned_orphan_cleanup", pid=pid, signal=sig.name)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + 2
                while self.workers() and time.monotonic() < deadline:
                    time.sleep(0.025)
                if not self.workers():
                    break
            assert not self.workers(), "Harness left a verified worker alive"
            self.event("cleanup_complete", workers=[])
        finally:
            self.client.close()
            self.socket.close()
            for log in self.logs:
                log.close()


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="POSIX SIGKILL/FIFO proof")
def test_sigkill_restart_fences_real_queued_running_and_late_results():
    harness = ProcessHarness()
    try:
        first = harness.start()
        project = harness.api("POST", "/projects", {"name": "JOB-002 CPU crash proof"}, 201)
        project_id = project["id"]
        harness.fixture("ready")
        success_id = harness.submit(project_id, "ready")
        succeeded = harness.wait_job(success_id, "succeeded")
        failed_id = harness.submit(project_id, "missing")
        failed = harness.wait_job(failed_id, "failed")

        running = {}
        pipes = []
        for name in ("blocked-a", "blocked-b"):
            pipe = harness.fixture(name, blocked=True)
            pipes.append(pipe)
            job_id = harness.submit(project_id, name)
            harness.hold_reader(pipe)
            running[job_id] = harness.wait_job(job_id, "running")
        children = harness.workers()
        assert len(children) == 2
        assert all(child["ppid"] == first.pid for child in children.values())
        harness.event("workers_blocked", server_pid=first.pid, workers=children)

        harness.fixture("queued")
        queued_id = harness.submit(project_id, "queued")
        queued = harness.wait_job(queued_id, "queued")
        harness.fixture("cancelled")
        cancelled_id = harness.submit(project_id, "cancelled")
        cancelled = harness.api("POST", f"/jobs/{cancelled_id}/cancel")
        assert cancelled["status"] == "cancelled"
        terminal = {success_id: succeeded, failed_id: failed, cancelled_id: cancelled}
        interrupted = {**running, queued_id: queued}
        before = {**terminal, **interrupted}
        assert harness.records() == before
        request_hashes = {
            job_id: hashlib.sha256(
                (harness.workspace / "jobs" / job_id / "request.json").read_bytes()
            ).hexdigest()
            for job_id in running
        }
        for job_id in interrupted:
            assert not (harness.workspace / "jobs" / job_id / "result.json").exists()
        assert not (harness.workspace / "jobs" / queued_id).exists()
        harness.event("before_sigkill", pid=first.pid, jobs=before)
        # Popen.kill is SIGKILL on POSIX, equivalent to kill -9 <this server PID>.
        first.kill()
        assert first.wait(timeout=5) == -signal.SIGKILL
        harness.event("sigkill_reaped", pid=first.pid, returncode=first.returncode)
        assert harness.records() == before, "SIGKILL must bypass graceful reconciliation"
        orphans = harness.workers()
        assert set(orphans) == set(children)
        assert all(child["ppid"] != first.pid for child in orphans.values())
        harness.event("orphans_survived", workers=orphans)

        second = harness.start()
        assert second.pid != first.pid
        assert harness.api("GET", "/projects") == [project]
        recovered = {}
        for job_id, previous in interrupted.items():
            job = harness.wait_job(job_id, "interrupted")
            assert job["result"] is None
            assert f"recovered {previous['status']} job" in job["error"]
            assert previous["updated_at"] in job["error"]
            assert job["updated_at"] > previous["updated_at"]
            assert job["created_at"] == previous["created_at"]
            assert job["request"] == previous["request"]
            if job_id in running:
                assert "orphan metadata worker" in job["error"]
            recovered[job_id] = job
        expected = {**terminal, **recovered}
        assert harness.records() == expected
        for job_id, record in expected.items():
            assert harness.job(job_id) == record
            assert harness.api("POST", f"/jobs/{job_id}/cancel") == record
        assert all(child["ppid"] != second.pid for child in harness.workers().values())

        # Let the OLD real children finish successfully AFTER recovery. Their valid
        # result files are retained as evidence, but cannot publish a success badge.
        for pipe in pipes:
            harness.release_reader(pipe)
        for job_id in running:
            result_path = harness.workspace / "jobs" / job_id / "result.json"
            eventually(result_path.is_file, f"late result for {job_id}")
            result = WorkerResult.model_validate_json(result_path.read_bytes())
            assert result.result is not None and result.error is None
            assert result.result.total_frames == 2
            harness.event("late_worker_success", job_id=job_id, result=result.model_dump())
        eventually(lambda: not harness.workers(), "old children to exit after FIFO release")
        harness.event("orphans_exited", pids=sorted(children))
        for _ in range(5):
            assert {
                job["id"]: job for job in harness.api("GET", f"/projects/{project_id}/jobs")
            } == expected
            assert not harness.workers()
            assert not (harness.workspace / "jobs" / queued_id).exists()
            time.sleep(0.05)
        for job_id, digest in request_hashes.items():
            assert (
                hashlib.sha256(
                    (harness.workspace / "jobs" / job_id / "request.json").read_bytes()
                ).hexdigest()
                == digest
            )
        assert harness.records() == expected
        harness.event("late_results_ignored_no_rerun", jobs=expected)

        # Re-open yet again with stale success files already present: errors/timestamps
        # must be durable and reconciliation must not change any terminal record.
        harness.stop(second)
        # Uvicorn can re-raise the captured SIGTERM after graceful lifespan shutdown.
        assert second.returncode in {0, -signal.SIGTERM}
        assert "Application shutdown complete." in Path(harness.logs[1].name).read_text()
        third = harness.start()
        assert third.pid not in {first.pid, second.pid}
        assert harness.records() == expected
        assert {
            job["id"]: job for job in harness.api("GET", f"/projects/{project_id}/jobs")
        } == expected
        assert not (harness.workspace / "jobs" / queued_id).exists()
        assert not harness.workers()
        harness.event("third_process_durable", pid=third.pid, jobs=expected)
        harness.stop(third)
        assert third.returncode in {0, -signal.SIGTERM}
        assert "Application shutdown complete." in Path(harness.logs[2].name).read_text()
        harness.event("PASS", server_pids=[server.pid for server in harness.servers])
    except BaseException as exc:
        harness.event("FAIL", error=repr(exc))
        raise
    finally:
        harness.close()


def test_recovery_batch_rolls_back_and_terminal_guards_preserve_evidence(tmp_path, monkeypatch):
    async def exercise():
        storage = Storage(tmp_path)
        await storage.initialize()
        execution = Execution(storage, Settings(data_dir=tmp_path))
        execution.slots = asyncio.Semaphore(0)
        try:
            request = IntakeRequest(source="local", path="fixture")
            first = await execution.submit("p", request)
            second = await execution.submit("p", request)
            cancelled = await execution.submit("p", request)
            cancelled = await execution.cancel(cancelled.id)
            first.error = "Earlier diagnostic evidence"
            await execution.save(first)
            before = await execution.list("p")
            execute = AsyncConnection.execute
            writes = 0

            async def fail_second_update(connection, statement, *args, **kwargs):
                nonlocal writes
                if isinstance(statement, Update) and statement.table.name == "jobs":
                    assert execution.lock.locked(), "Recovery must serialize with finish/cancel"
                    writes += 1
                    if writes == 2:
                        raise RuntimeError("injected recovery transaction failure")
                return await execute(connection, statement, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(AsyncConnection, "execute", fail_second_update)
                with pytest.raises(RuntimeError, match="injected recovery"):
                    await execution.reconcile()
            assert writes == 2
            assert await execution.list("p") == before, "Partial recovery must roll back"
            await execution.reconcile()
            recovered = await execution.list("p")
            assert (await execution.get(first.id)).error.startswith(first.error + "\n")
            assert (await execution.get(first.id)).status == "interrupted"
            assert (await execution.get(second.id)).status == "interrupted"
            assert await execution.get(cancelled.id) == cancelled
            revision = "metadata-sha256:" + hashlib.sha256(METADATA).hexdigest()
            late_success = WorkerResult(result=profile(METADATA, request, revision))
            for job in recovered:
                await execution.finish(job.id, late_success)
                await execution.finish(job.id, WorkerResult(error="late worker failure"))
                assert await execution.cancel(job.id) == job
            await execution.reconcile()
            assert await execution.list("p") == recovered
        finally:
            await execution.close()
            await storage.close()

    asyncio.run(exercise())
