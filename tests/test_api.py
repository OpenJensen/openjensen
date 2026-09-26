"""Versioned API, local access boundary, and real web/CLI transport integration."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.settings import Settings

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as value:
        yield value


def test_validation_and_missing_resources_do_not_schedule_work(client):
    for payload in [{"name": " "}, {"name": "x" * 101}, {"name": "Valid", "extra": 1}]:
        response = client.post("/api/v1/projects", json=payload)
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)
    project = client.post("/api/v1/projects", json={"name": " API project "}).json()
    assert project["name"] == "API project"
    endpoint = f"/api/v1/projects/{project['id']}/intakes"
    for payload in [
        {"source": "unknown"},
        {"repo_id": "not-a-repo"},
        {"repo_id": "x/y", "path": "."},
        {"source": "local", "path": "."},
        {"repo_id": "x/y", "operation": "train"},
    ]:
        response = client.post(endpoint, json=payload)
        assert response.status_code == 422, response.text
        assert response.json()["detail"]
    malformed = client.post(endpoint, content="{", headers={"Content-Type": "application/json"})
    assert malformed.status_code == 422
    assert client.get(f"/api/v1/projects/{project['id']}/jobs").json() == []
    for method, path, payload, detail in [
        ("GET", "/projects/missing/jobs", None, "Project not found"),
        ("POST", "/projects/missing/intakes", {"repo_id": "x/y"}, "Project not found"),
        ("GET", "/jobs/missing", None, "Job not found"),
        ("POST", "/jobs/missing/cancel", None, "Job not found"),
    ]:
        response = client.request(method, "/api/v1" + path, json=payload)
        assert response.status_code == 404
        assert response.json() == {"detail": detail}
        assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/v2/projects").status_code == 404
    assert client.get("/api/v1/events").status_code == 404  # SSE is not implemented.


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Origin": "https://attacker.invalid"}, 403),
        ({"Origin": "null"}, 403),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Host": "attacker.invalid", "Origin": "http://attacker.invalid"}, 400),
        ({"Origin": "http://localhost:3000", "Sec-Fetch-Site": "cross-site"}, 201),
        ({"Origin": "http://127.0.0.1:8000"}, 201),
        ({"Host": "localhost:8123", "Origin": "http://localhost:8123"}, 201),
        ({"Sec-Fetch-Site": "same-origin"}, 201),
        ({}, 201),  # CLI requests have no browser Origin header.
    ],
)
def test_local_access_boundary_checks_before_mutation(client, headers, status):
    response = client.post("/api/v1/projects", json={"name": "Boundary"}, headers=headers)
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    assert len(client.get("/api/v1/projects").json()) == (1 if status == 201 else 0)


def test_development_preflight_and_uncached_polling(client):
    response = client.options(
        "/api/v1/projects",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    for path in ["health", "projects", "capabilities"]:
        response = client.get(f"/api/v1/{path}", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_queued_cancellation_is_visible_and_idempotent(client):
    # Hold dispatch only: use the real API, persistence and cancellation implementation.
    client.app.state.execution.slots = asyncio.Semaphore(0)
    project = client.post("/api/v1/projects", json={"name": "Cancel"}).json()
    endpoint = f"/api/v1/projects/{project['id']}"
    submitted = client.post(endpoint + "/intakes", json={"repo_id": "fixture/data"})
    assert submitted.status_code == 202
    queued = submitted.json()
    assert queued["status"] == "queued" and queued["result"] is None
    cancelled = client.post(f"/api/v1/jobs/{queued['id']}/cancel").json()
    assert cancelled["status"] == "cancelled" and cancelled["result"] is None
    assert cancelled["error"]
    assert client.get(f"/api/v1/jobs/{queued['id']}").json() == cancelled
    assert client.get(endpoint + "/jobs").json() == [cancelled]
    assert client.post(f"/api/v1/jobs/{queued['id']}/cancel").json() == cancelled


@pytest.fixture
def live_api(tmp_path):
    root = tmp_path / "datasets"
    metadata = root / "fixture" / "meta"
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
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    environment = {
        **os.environ,
        "FIREBIRD_DATA_DIR": str(tmp_path / "workspace"),
        "FIREBIRD_LOCAL_DATA_ROOT": str(root),
        "FIREBIRD_WEB_DIR": str(tmp_path / "no-static-build"),
        "FIREBIRD_API_URL": origin,
        "NEXT_PUBLIC_API_URL": origin,
    }
    with (tmp_path / "server.log").open("w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "vla_platform.cli", "serve", "--port", str(port)],
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail(f"Application exited during startup: {log.read()}")
                try:
                    if httpx.get(origin + "/api/v1/health", timeout=0.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                pytest.fail("Application did not become ready")
            yield environment
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for actual web-client test")
def test_real_web_and_cli_share_versioned_operations_and_poll_results(live_api):
    def cli(*arguments, success=True):
        result = subprocess.run(
            [sys.executable, "-m", "vla_platform.cli", *arguments],
            cwd=REPO,
            env=live_api,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == (0 if success else 1), result.stderr
        return json.loads(result.stdout) if success else result.stderr

    def web(expression):
        # Node executes the unchanged web transport with real fetch against the live owner.
        source = (
            "import { api } from './apps/web/src/lib/api.ts';\n"
            f"console.log(JSON.stringify(await ({expression})));"
        )
        result = subprocess.run(
            [shutil.which("node"), "--input-type=module", "-e", source],
            cwd=REPO,
            env=live_api,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    project = web("api.createProject('Created by actual web client')")
    assert cli("projects", "list") == [project]
    other = cli("projects", "create", "Created by actual CLI")
    assert {item["id"] for item in web("api.projects()")} == {project["id"], other["id"]}
    assert web("api.health()")["status"] == "ok"
    assert web("api.capabilities()") == cli("capabilities")
    submitted = cli("inspect", project["id"], "--path", "fixture")
    assert submitted["status"] == "queued"
    # Poll using the actual browser transport; completion comes from a real metadata subprocess.
    records = web(
        "(async () => { const states = []; const deadline = Date.now() + 15000; "
        f"while (Date.now() < deadline) {{ const jobs = await api.jobs('{project['id']}'); "
        "const job = jobs[0]; states.push(job); "
        "if (!['queued', 'running'].includes(job.status)) return states; "
        "await new Promise(resolve => setTimeout(resolve, 50)); } "
        "throw new Error('Polling timed out'); })()"
    )
    completed = records[-1]
    assert completed["status"] == "succeeded", completed
    assert completed["result"]["total_frames"] == 3
    assert completed["result"]["inspection_scope"] == "metadata_only"
    assert cli("jobs", "show", submitted["id"]) == completed
    assert cli("jobs", "list", project["id"]) == [completed]
    assert web(f"api.cancel('{submitted['id']}')") == completed
    assert cli("jobs", "cancel", submitted["id"]) == completed
    failed_request = web(f"api.inspect('{other['id']}', {{ source: 'local', path: 'missing' }})")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        failed = cli("jobs", "show", failed_request["id"])
        if failed["status"] not in {"queued", "running"}:
            break
        time.sleep(0.05)
    assert failed["status"] == "failed" and failed["result"] is None and failed["error"]
    assert "API error (404)" in cli("jobs", "show", "missing", success=False)
    assert web("api.jobs('missing').catch(error => error.message)") == "Project not found"
    assert "API error (422)" in cli("projects", "create", " ", success=False)
    assert "at least 1 character" in web("api.createProject(' ').catch(error => error.message)")
