"""Opt-in macOS frozen-sidecar experiment in a new, disposable workspace.

This exercises only fixed serve/intake modes. It neither activates the desktop payload
nor claims installed-app upgrade, clean-machine, signing or model-runtime acceptance.
"""

import argparse
import errno
import hashlib
import json
import os
import platform
import select
import shutil
import signal
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from native_dependencies import inspect as inspect_native
from payload_inventory import inventory, verify
from sidecar_resources import load_resources

NONCE = "4d" * 16
ARM64 = 0x100000C


class Child:
    """Own exactly one frozen child and its private control pipes."""

    def __init__(self, executable: Path, root: Path, resources: Path, data: Path, local: Path):
        self.log_path = root / f"child-{time.monotonic_ns()}.log"
        self.log = self.log_path.open("xb")
        try:
            self.process = spawn_owned(
                [
                    str(executable),
                    "serve",
                    "--resources",
                    str(resources),
                    "--data-dir",
                    str(data),
                    "--local-root",
                    str(local),
                ],
                cwd=root,
                env=clean_environment(root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                start_new_session=True,
            )
        except BaseException:
            self.log.close()
            raise
        self.pending = bytearray()

    def send(self, command: str = "start", nonce: str = NONCE) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps({"schema_version": 1, "command": command, "nonce": nonce}).encode() + b"\n"
        )
        self.process.stdin.flush()

    def record(self, timeout: float = 90) -> dict[str, Any]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if b"\n" in self.pending:
                line, _, self.pending = self.pending.partition(b"\n")
                if len(line) > 4096:
                    raise AssertionError("Oversized private record")
                return json.loads(line)
            if select.select(
                [self.process.stdout], [], [], min(0.1, max(0, deadline - time.monotonic()))
            )[0]:
                block = os.read(self.process.stdout.fileno(), 4097)
                if not block:
                    raise AssertionError(
                        f"Child exited without a private record; inspect {self.log_path}"
                    )
                self.pending.extend(block)
                if len(self.pending) > 8192:
                    raise AssertionError("Oversized private pipe")
        raise TimeoutError("Frozen child did not produce its private record within budget")

    def finish(self, expected: int = 0) -> None:
        assert self.process.wait(timeout=30) == expected, self.log_path

    def close(self) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        if not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            # The leader can exit while its descendants remain. Reconcile the owned
            # group even after a normal leader exit; never report that case as reaped.
            cleanup_group(self.process)
        finally:
            self.process.stdout.close()
            self.log.close()

    def __enter__(self) -> Child:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def spawn_owned(command: list[str], **kwargs: Any) -> subprocess.Popen:
    """Register the returned child before delivering an interrupt during Popen."""
    pending: list[int] = []
    previous = {
        value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    }
    process = None
    try:
        for value in previous:
            signal.signal(value, lambda number, _frame: pending.append(number))
        try:
            process = subprocess.Popen(command, **kwargs)
        finally:
            for value, handler in previous.items():
                signal.signal(value, handler)
        if pending:
            raise InterruptedError("Interrupted while registering owned frozen child")
        return process
    except BaseException:
        if process is not None:
            cleanup_group(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        raise


def group_exists(process: subprocess.Popen) -> bool:
    """Only kernel-confirmed disappearance resolves Darwin's transient exit EPERM."""
    process.poll()
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as original:
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            time.sleep(0.01)
            process.poll()
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                continue
        raise original


def cleanup_group(process: subprocess.Popen) -> None:
    """Bounded POSIX cleanup of this harness-created session, including descendants."""

    def signal_group(value: int) -> None:
        try:
            os.killpg(process.pid, value)
        except ProcessLookupError:
            pass
        except PermissionError:
            if group_exists(process):
                raise

    if group_exists(process):
        signal_group(signal.SIGTERM)
        deadline = time.monotonic() + 5
        while group_exists(process) and time.monotonic() < deadline:
            time.sleep(0.02)
        if group_exists(process):
            signal_group(signal.SIGKILL)
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while group_exists(process) and time.monotonic() < deadline:
        time.sleep(0.02)
    if group_exists(process):
        raise RuntimeError("Owned process group cleanup remains unknown")


def clean_environment(root: Path) -> dict[str, str]:
    """No repository path, interpreter lookup, inherited configuration or credentials."""
    home, temporary = root / "home", root / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    return {"HOME": str(home), "TMPDIR": str(temporary), "PATH": str(root / "no-executables")}


def request(port: int, path: str, body: dict | None = None) -> tuple[int, bytes]:
    if not path.startswith("/") or not 0 < port < 65536:
        raise ValueError("Invalid loopback request")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with opener.open(req, timeout=5) as response:
        raw = response.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2:
            raise AssertionError("HTTP response exceeded bound")
        return response.status, raw


def assert_listener_closed(port: int) -> None:
    try:
        request(port, "/api/v1/health")
    except urllib.error.HTTPError as exc:
        raise AssertionError("Owned listener is still serving HTTP") from exc
    except urllib.error.URLError as exc:
        if not isinstance(exc.reason, OSError) or exc.reason.errno != errno.ECONNREFUSED:
            raise AssertionError("Owned listener closure is unverified") from exc
    else:
        raise AssertionError("Owned listener is still serving HTTP")


def expect_json(port: int, path: str, body: dict | None = None, status: int = 200) -> Any:
    code, raw = request(port, path, body)
    assert code == status, (path, code, raw[:2000])
    return json.loads(raw)


def ready(child: Child, expected_resources: Any, data: Path) -> dict[str, Any]:
    child.send()
    value = child.record()
    assert value["schema_version"] == 1 and value["event"] == "ready"
    assert value["nonce"] == NONCE and value["host"] == "127.0.0.1"
    assert type(value["port"]) is int and 0 < value["port"] < 65536
    assert value["build_id"] == expected_resources.build_id
    assert value["app_version"] == expected_resources.app_version
    assert value["resources_sha256"] == expected_resources.manifest_sha256
    assert value["workspace_id"] == hashlib.sha256(os.fsencode(data.resolve())).hexdigest()
    return value


def run(payload: Path, output: Path, build_id: str, resource_sha256: str) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This experiment targets macOS ARM64 only")
    if not payload.is_absolute() or not output.is_absolute():
        raise ValueError("Absolute paths are required")
    payload = payload.resolve(strict=True)
    if output.resolve().is_relative_to(payload) or payload.is_relative_to(output.resolve()):
        raise ValueError("Experiment output and original payload must be disjoint")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    original = inventory(payload)
    (output / "original-inventory.json").write_text(json.dumps(original, indent=2) + "\n")
    native = {
        name: entry["macho_cpu_types"]
        for name, entry in original["entries"].items()
        if entry["kind"] == "file" and entry["macho_cpu_types"] is not None
    }
    assert native and all(arches == [ARM64] for arches in native.values()), native
    relocated = output / "relocated/payload"
    shutil.copytree(payload, relocated, symlinks=True)
    verify(relocated, original)
    native_links = inspect_native(relocated, original)
    (output / "native-dependencies.json").write_text(json.dumps(native_links, indent=2) + "\n")
    executable = relocated / "firebird-sidecar"
    assert executable.is_file() and os.access(executable, os.X_OK)
    resources_path = relocated / "_internal/sidecar-resources"
    resources = load_resources(resources_path)
    assert resources.build_id == build_id and resources.manifest_sha256 == resource_sha256
    metadata = output / "datasets/fixture/meta/info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "generated_metadata_fixture",
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
    metadata_before = hashlib.sha256(metadata.read_bytes()).hexdigest()
    local, data = output / "datasets", output / "workspace"
    with Child(executable, output, resources_path, data, local) as child:
        first_ready = ready(child, resources, data)
        port = first_ready["port"]
        assert expect_json(port, "/api/v1/health")["status"] == "ok"
        assert expect_json(port, "/api/v1/decision/status")["configured"] is False
        assert expect_json(port, "/api/v1/augmentation-options")["auth_mode"] == "unconfigured"
        static = json.loads((resources_path / "resources.json").read_text())["files"]
        for name, entry in static.items():
            code, raw = request(port, "/" if name == "index.html" else "/" + name)
            assert code == 200 and len(raw) == entry["bytes"]
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name
        project = expect_json(
            port, "/api/v1/projects", {"name": "Frozen generated metadata verification"}, 201
        )
        job = expect_json(
            port,
            f"/api/v1/projects/{project['id']}/intakes",
            {"source": "local", "path": "fixture"},
            202,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            job = expect_json(port, f"/api/v1/jobs/{job['id']}")
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded" and job["result"]["total_frames"] == 20, job
        with Child(executable, output, resources_path, data, local) as competing:
            competing.send()
            competing.finish(expected=1)
            assert competing.process.stdout is not None and competing.process.stdout.read() == b""
        assert expect_json(port, "/api/v1/health")["status"] == "ok"
        child.send("shutdown")
        stopped = child.record(timeout=30)
        assert stopped["event"] == "stopped" and stopped["reason"] == "shutdown"
        assert stopped["nonce"] == NONCE
        child.finish()
    with Child(executable, output, resources_path, data, local) as restarted:
        restart_ready = ready(restarted, resources, data)
        assert expect_json(restart_ready["port"], "/api/v1/projects") == [project]
        assert expect_json(restart_ready["port"], f"/api/v1/jobs/{job['id']}") == job
        assert restarted.process.stdin is not None
        restarted.process.stdin.close()
        eof = restarted.record(timeout=30)
        assert eof["event"] == "stopped" and eof["reason"] == "parent_eof" and eof["nonce"] == NONCE
        restarted.finish()
    with Child(executable, output, resources_path, output / "never-created", local) as early:
        assert early.process.stdin is not None
        early.process.stdin.close()
        early.finish()
        assert early.process.stdout is not None and early.process.stdout.read() == b""
        assert not (output / "never-created").exists()
    tampered = output / "tampered-resources"
    shutil.copytree(resources_path, tampered)
    (tampered / "web/index.html").write_bytes(b"tampered")
    with Child(executable, output, tampered, output / "tampered-workspace", local) as invalid:
        invalid.send()
        invalid.finish(expected=1)
        assert invalid.process.stdout is not None and invalid.process.stdout.read() == b""
        assert not (output / "tampered-workspace").exists()
    with Child(executable, output, resources_path, data, local) as invalid_control:
        invalid_ready = ready(invalid_control, resources, data)
        invalid_control.send("shutdown", nonce="ff" * 16)
        invalid_control.finish(expected=1)
        assert_listener_closed(invalid_ready["port"])
    # Reject the generic Python bridge without executing supplied code.
    for forbidden in (["-c", "raise RuntimeError('must not run')"], ["-m", "vla_platform.cli"]):
        result = subprocess.run(
            [str(executable), *forbidden],
            cwd=output,
            env=clean_environment(output),
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 2 and b"invalid choice" in result.stderr
    dbs = list(data.glob("workspace.sqlite3"))
    assert len(dbs) == 1, dbs
    with sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True) as database:
        migration = database.execute("select version_num from alembic_version").fetchall()
        assert migration
        assert database.execute("pragma integrity_check").fetchall() == [("ok",)]
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == metadata_before
    verify(payload, original)
    verify(relocated, original)
    report = {
        "schema_version": 1,
        "source_build_id": build_id,
        "resources_sha256": resource_sha256,
        "payload_identity_sha256": original["identity_sha256"],
        "native_files": native,
        "elapsed_seconds": time.monotonic() - started,
        "ready": first_ready,
        "restart_ready": restart_ready,
        "static_files_verified_over_http": len(static),
        "project": project,
        "intake_job": job,
        "sqlite_revision": migration,
        "fixture_kind": "generated_metadata_only_no_recorded_media",
        "source_and_relocated_payloads_unchanged": True,
        "workspace_owner_exclusion": True,
        "shutdown_and_parent_eof_exit_zero": True,
        "restart_preserved_project_and_job": True,
        "original_metadata_unchanged": True,
        "parent_eof_before_start_created_no_workspace": True,
        "tampered_static_rejected_before_workspace": True,
        "wrong_control_nonce_shut_down_owned_listener": True,
        "environment_keys": sorted(clean_environment(output)),
        "production_activation": False,
        "native_tauri_owner_executed": False,
        "upgrade_or_clean_machine_verified": False,
        "model_runtime_verified": False,
    }
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--resource-sha256", required=True)
    args = parser.parse_args()

    def interrupted(_number: int, _frame: Any) -> None:
        raise InterruptedError("Frozen verification interrupted")

    signal.signal(signal.SIGTERM, interrupted)

    def expired(_number: int, _frame: Any) -> None:
        raise TimeoutError("Frozen experiment exceeded its 600-second outer deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    try:
        report = run(args.payload, args.output, args.build_id, args.resource_sha256)
    finally:
        signal.alarm(0)
    print(
        json.dumps(
            {
                "receipt": str(args.output / "receipt.json"),
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
