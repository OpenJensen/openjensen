"""Opt-in, network-capable wheel install and local API/CLI persistence verification.

Run from the repository with Python 3.14 and uvx available. Dependencies are
installed into a disposable environment using the repository lock. Application
operations use only local synthetic fixtures; no model, dataset or cloud downloads.
The socket handoff is POSIX-only. This is not a Windows installation certificate.
"""

import argparse
import hashlib
import io
import json
import os
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV = ["uvx", "--from", "uv==0.12.19", "uv"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(evidence: Path):
    if os.name != "posix":
        raise SystemExit("This clean-install harness requires POSIX socket handoff")
    evidence.mkdir(parents=True, exist_ok=True)
    # Never leave a stale passing receipt next to a failed rerun log.
    (evidence / "receipt.json").unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="firebird-clean-install-") as temporary:
        root = Path(temporary)
        log = (evidence / "install.log").open("w", encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        server = None
        channel = socket.socket()
        channel.bind(("127.0.0.1", 0))
        origin = f"http://127.0.0.1:{channel.getsockname()[1]}"

        def run(command, cwd=root):
            log.write(json.dumps({"command": list(map(str, command)), "cwd": str(cwd)}) + "\n")
            log.flush()
            result = subprocess.run(
                list(map(str, command)),
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=180,
            )
            log.write(result.stdout + result.stderr)
            log.flush()
            if result.returncode:
                raise RuntimeError(f"Command failed ({result.returncode}); see {log.name}")
            return result.stdout

        def api(method, path, value=None):
            data = json.dumps(value).encode() if value is not None else None
            request = urllib.request.Request(
                origin + "/api/v1" + path,
                data=data,
                method=method,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.load(response)

        def wait(check, message):
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                value = check()
                if value:
                    return value
                time.sleep(0.05)
            raise AssertionError(message)

        def start():
            nonlocal server
            server = subprocess.Popen(
                [
                    str(python),
                    "-I",
                    "-m",
                    "uvicorn",
                    "vla_platform.api:create_app",
                    "--factory",
                    "--fd",
                    str(channel.fileno()),
                    "--log-level",
                    "warning",
                ],
                cwd=root,
                env=env,
                pass_fds=(channel.fileno(),),
                stdout=log,
                stderr=log,
            )

            def ready():
                assert server.poll() is None, f"Server exited; see {log.name}"
                try:
                    return api("GET", "/health")["status"] == "ok"
                except urllib.error.URLError:
                    return False

            wait(ready, "Installed application did not become ready")

        def stop():
            if server and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)

        try:
            run(
                [
                    *UV,
                    "export",
                    "--frozen",
                    "--no-dev",
                    "--no-emit-workspace",
                    "--output-file",
                    root / "requirements.txt",
                ],
                ROOT,
            )
            run(
                [*UV, "build", "--package", "vla-platform", "--wheel", "--out-dir", root / "wheel"],
                ROOT,
            )
            (wheel,) = (root / "wheel").glob("*.whl")
            run([*UV, "venv", "--python", "3.14.7", root / "environment"])
            python = root / "environment/bin/python"
            entrypoint = root / "environment/bin/firebird"
            run([*UV, "pip", "sync", "--python", python, root / "requirements.txt"])
            run([*UV, "pip", "install", "--python", python, "--no-deps", wheel])
            imports = json.loads(
                run(
                    [
                        python,
                        "-I",
                        "-c",
                        """
import importlib.util,json,pathlib,platform,sys
import vla_platform.api
for name in ('torch','pyarrow'):
 assert importlib.util.find_spec(name) is None, name
 assert name not in sys.modules, name
print(json.dumps({'python':platform.python_version(),'api_file':vla_platform.api.__file__,
 'heavy_dependencies_absent':['torch','pyarrow']}))
""",
                    ]
                )
            )
            assert Path(imports["api_file"]).is_relative_to(root / "environment")
            run([entrypoint, "--help"])
            source = root / "synthetic-source"
            source.write_bytes(b"Synthetic fixture: not policy weights")
            config = root / "runtimes.json"
            config.write_text(
                json.dumps(
                    {
                        "runtimes": [
                            {
                                "id": "browser-success",
                                "label": "Synthetic clean-install fixture",
                                "python": str(python),
                                "worker_root": str(ROOT / "tests/fixtures/browser_worker"),
                                "vendor": str(root),
                                "build": str(root),
                            }
                        ],
                        "sources": [
                            {
                                "id": "synthetic-source",
                                "label": "Synthetic source",
                                "path": str(source),
                                "sha256": sha(source),
                            }
                        ],
                    }
                )
            )
            env.update(
                FIREBIRD_DATA_DIR=str(root / "workspace"),
                FIREBIRD_RUNTIME_CONFIG=str(config),
                FIREBIRD_API_URL=origin,
                FIREBIRD_LOCAL_DATA_ROOT=str(ROOT / "tests/fixtures"),
                FIREBIRD_WEB_DIR=str(root / "no-bundled-web-assets"),
            )

            def cli(*args):
                return json.loads(run([entrypoint, *args]))

            def completed(job):
                record = api("GET", "/jobs/" + job["id"])
                return record if record["status"] not in {"queued", "running"} else None

            start()
            project = cli("projects", "create", "Synthetic clean wheel verification")
            intake = cli("inspect", project["id"], "--path", "lerobot_v3_preview")
            inspected = wait(lambda: completed(intake), "Metadata intake did not complete")
            assert inspected["status"] == "succeeded", inspected
            assert inspected["result"]["inspection_scope"] == "metadata_only"
            recipe = root / "recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "operation": "policy.workflow",
                        "runtime_id": "browser-success",
                        "source_id": "synthetic-source",
                        "evaluation": {"mode": "engine"},
                    }
                )
            )
            job = cli("policy", "submit", project["id"], str(recipe))
            finished = wait(lambda: completed(job), "Policy fixture did not complete")
            assert finished["status"] == "succeeded", finished
            assert finished["result"]["decision"] == "diagnostics_only"
            assert cli("jobs", "show", job["id"]) == finished
            artifacts = cli("policy", "artifacts", project["id"])
            assert artifacts == finished["result"]["artifacts"]
            assert artifacts and all(item["metadata"]["fixture_only"] for item in artifacts)
            events = cli("jobs", "events", job["id"])
            assert any(event["message"] == "Optimizer step 1" for event in events)
            artifact = next(item for item in artifacts if item["parent_ids"])
            url = origin + f"/api/v1/projects/{project['id']}/artifacts/{artifact['id']}/download"
            with urllib.request.urlopen(url, timeout=5) as response:
                downloaded = response.read()
            with tarfile.open(fileobj=io.BytesIO(downloaded)) as archive:
                manifest_bytes = archive.extractfile("policy/manifest.json").read()
                assert hashlib.sha256(manifest_bytes).hexdigest() == artifact["manifest_sha256"]
                for name, expected in json.loads(manifest_bytes)["files"].items():
                    assert (
                        hashlib.sha256(archive.extractfile("policy/" + name).read()).hexdigest()
                        == expected
                    )
            stop()
            start()
            assert cli("jobs", "show", job["id"]) == finished
            assert cli("policy", "artifacts", project["id"]) == artifacts
            assert cli("jobs", "events", job["id"]) == events
            with urllib.request.urlopen(url, timeout=5) as response:
                assert response.read() == downloaded
            receipt = {
                "scope": "Clean wheel, local metadata and synthetic subprocess protocol only",
                "application_revision": run(["git", "rev-parse", "HEAD"], ROOT).strip(),
                "wheel_sha256": sha(wheel),
                "locked_requirements_sha256": sha(root / "requirements.txt"),
                **imports,
                "job": finished,
                "intake": inspected,
                "archive_sha256": hashlib.sha256(downloaded).hexdigest(),
                "restart_preserved_job_artifacts_events_and_archive": True,
                "network": (
                    "Dependency/build setup may use network; "
                    "application operations use local fixtures only"
                ),
            }
            (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"passed": True, "receipt": str(evidence / "receipt.json")}))
        finally:
            stop()
            channel.close()
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    verify(parser.parse_args().evidence_dir.resolve())
