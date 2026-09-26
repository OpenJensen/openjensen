"""Opt-in public metadata journey; never collected by pytest or run by normal CI.

Run with --allow-public-metadata --output-dir PATH after building the static web app.
The unchanged web transport runs in Node: this is not interactive browser evidence.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
SOURCE = "codywang/so101_pickup_test"
REVISION = "ecef85bc07005f771ad86deeff1427f9d72953ed"
METADATA_SHA256 = "2257f8360a272ff10bff3d5aeeb4d365c9716141dee5a74ecfe7fe5e8cfca6be"


def command(arguments, environment, timeout=25):
    result = subprocess.run(
        arguments, cwd=REPO, env=environment, capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


@contextmanager
def owner(environment, port, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "vla_platform.cli", "serve", "--port", str(port)],
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 20
            with httpx.Client(base_url=environment["FIREBIRD_API_URL"], trust_env=False) as client:
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text(encoding="utf-8")
                    try:
                        if client.get("/api/v1/health", timeout=0.5).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.05)
                else:
                    raise AssertionError("Application startup timed out")
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def journey(output):
    node = shutil.which("node")
    assert node, "Node is required for the unchanged web transport"
    static = REPO / "apps/web/out"
    assert (static / "index.html").is_file(), "Run pnpm build:web first"
    # Each invocation owns a new directory; never reuse or overwrite an existing workspace.
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    environment = {
        **os.environ,
        "FIREBIRD_DATA_DIR": str(output / "workspace"),
        "FIREBIRD_WEB_DIR": str(static),
        "FIREBIRD_API_URL": origin,
        "NEXT_PUBLIC_API_URL": origin,
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
    }
    environment.pop("FIREBIRD_LOCAL_DATA_ROOT", None)

    def cli(*arguments):
        return json.loads(
            command([sys.executable, "-m", "vla_platform.cli", *arguments], environment)
        )

    def web(expression, timeout=25):
        source = (
            "import { api } from './apps/web/src/lib/api.ts';\n"
            f"console.log(JSON.stringify(await ({expression})));"
        )
        return json.loads(
            command([node, "--input-type=module", "-e", source], environment, timeout)
        )

    proof = {
        "evidence_kind": "live_public_metadata",
        "source": SOURCE,
        "source_revision": REVISION,
        "application_revision": command(["git", "rev-parse", "HEAD"], environment),
        "started_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "node": command([node, "--version"], environment),
        "web_transport_sha256": hashlib.sha256(
            (REPO / "apps/web/src/lib/api.ts").read_bytes()
        ).hexdigest(),
        "interactive_browser_verified": False,
    }
    with owner(environment, port, output / "owner-before.log") as first:
        proof["first_owner_pid"] = first.pid
        with httpx.Client(base_url=origin, trust_env=False) as client:
            page = client.get("/")
            assert page.status_code == 200
            assert page.content == (static / "index.html").read_bytes()
            assets = sorted(set(re.findall(r'(?:src|href)="(/_next/[^"?]+)', page.text)))
            assert assets, "Expected static JavaScript/CSS references"
            for asset in assets:
                assert client.get(asset).status_code == 200, asset
            proof["static_http"] = {
                "status": page.status_code,
                "index_sha256": hashlib.sha256(page.content).hexdigest(),
                "asset_count": len(assets),
                "all_assets_status": 200,
                "rendering_verified": False,
            }
        project = web("api.createProject('INT-001 pinned public metadata journey')")
        assert cli("projects", "list") == [project]
        payload = {"source": "huggingface", "repo_id": SOURCE, "revision": REVISION}
        submitted = web(f"api.inspect('{project['id']}', {json.dumps(payload)})")
        assert submitted["status"] == "queued"
        states = web(
            "(async () => { const states = []; const deadline = Date.now() + 100000; "
            f"while (Date.now() < deadline) {{ const jobs = await api.jobs('{project['id']}'); "
            "const job = jobs[0]; states.push(job); "
            "if (!['queued', 'running'].includes(job.status)) return states; "
            "await new Promise(resolve => setTimeout(resolve, 250)); } "
            "throw new Error('Public metadata polling timed out'); })()",
            timeout=110,
        )
        completed = states[-1]
        # Persist observed failure too; an unavailable source must not become fabricated success.
        (output / "observed-job.json").write_text(
            json.dumps(completed, indent=2) + "\n", encoding="utf-8"
        )
        assert completed["status"] == "succeeded", completed
        profile = completed["result"]
        assert profile["revision"] == REVISION
        assert profile["repo_id"] == SOURCE and profile["source"] == "huggingface"
        assert profile["metadata_sha256"] == METADATA_SHA256
        assert profile["inspection_scope"] == "metadata_only"
        assert profile["total_episodes"] == 30 and profile["total_frames"] == 4500
        assert profile["warnings"] and len(profile["features"]) == 8
        assert cli("jobs", "show", completed["id"]) == completed
        assert cli("jobs", "list", project["id"]) == [completed]
        assert web(f"api.cancel('{completed['id']}')") == completed
        assert cli("jobs", "cancel", completed["id"]) == completed
        # Local negative paths do not make additional requests to the public source.
        assert web("api.jobs('missing').catch(error => error.message)") == "Project not found"
        assert "at least 1 character" in web("api.createProject(' ').catch(e => e.message)")
        disabled = web(
            f"api.inspect('{project['id']}', {{source: 'local', path: 'missing'}})"
            ".catch(error => error.message)"
        )
        assert disabled == "Local intake is disabled; configure FIREBIRD_LOCAL_DATA_ROOT"
        assert cli("projects", "list") == [project]
        assert cli("jobs", "list", project["id"]) == [completed]
        proof.update(
            project=project,
            completed_job=completed,
            observed_poll_statuses=list(dict.fromkeys(job["status"] for job in states)),
            negative_paths=["missing project", "empty project name", "disabled local intake"],
            terminal_cancel_preserved=True,
            rejected_requests_created_no_jobs=True,
        )
    assert first.poll() is not None
    proof["first_owner_exit_code"] = first.returncode
    with owner(environment, port, output / "owner-after.log") as second:
        assert second.pid != first.pid
        proof["second_owner_pid"] = second.pid
        assert web("api.projects()") == cli("projects", "list") == [project]
        assert web(f"api.jobs('{project['id']}')") == [completed]
        assert cli("jobs", "show", completed["id"]) == completed
        assert cli("jobs", "list", project["id"]) == [completed]
        proof["fresh_process_records_identical"] = True
    proof["second_owner_exit_code"] = second.returncode
    proof["all_owners_stopped"] = first.poll() is not None and second.poll() is not None
    proof["finished_at"] = datetime.now(UTC).isoformat()
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print("LIVE_INTAKE_PROOF " + json.dumps(proof, sort_keys=True))
    print("PASS: pinned live metadata persists identically through web transport and CLI restart.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-public-metadata", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.allow_public_metadata:
        parser.error("Explicit --allow-public-metadata opt-in is required; this contacts the Hub")
    journey(args.output_dir.resolve())
