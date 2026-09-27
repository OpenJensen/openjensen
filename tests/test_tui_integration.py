"""Real local API owner, persisted workspace and supervised fixture jobs; no ML claims."""

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Input, OptionList, Select, TextArea
from vla_platform.tui import FirebirdApp


@pytest.fixture
def api_server(tmp_path):
    dataset = tmp_path / "datasets" / "fixture" / "meta"
    dataset.mkdir(parents=True)
    (dataset / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "generated_tui_fixture",
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
    source = tmp_path / "source"
    source.write_bytes(b"Synthetic protocol fixture, not policy weights")
    runtime = tmp_path / "runtimes.json"
    runtime.write_text(
        json.dumps(
            {
                "runtimes": [
                    {
                        "id": "slow",
                        "label": "slow fixture",
                        "python": sys.executable,
                        "worker_root": str(Path(__file__).parent / "fixtures/native_worker"),
                        "vendor": str(tmp_path),
                        "build": str(tmp_path),
                    }
                ],
                "sources": [
                    {
                        "id": "source",
                        "label": "Fixture",
                        "path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("FIREBIRD_", "GOOGLE_", "AWS_", "LIVEKIT_", "OPENAI_", "OPENROUTER_"))
    }
    environment.update(
        {
            "FIREBIRD_DATA_DIR": str(tmp_path / "workspace"),
            "FIREBIRD_LOCAL_DATA_ROOT": str(dataset.parents[1]),
            "FIREBIRD_WEB_DIR": str(tmp_path / "no-web"),
            "FIREBIRD_RUNTIME_CONFIG": str(runtime),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    process = None
    log = (tmp_path / "api.log").open("w")

    def stop():
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def start():
        nonlocal process
        process = subprocess.Popen(
            [sys.executable, "-m", "vla_platform.cli", "serve", "--port", str(port)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError((tmp_path / "api.log").read_text())
            try:
                if (
                    httpx.get(base + "/api/v1/projects", timeout=0.2, trust_env=False).status_code
                    == 200
                ):
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise AssertionError("Disposable API did not start")

    try:
        start()
        yield base, environment, stop, start
    finally:
        stop()
        log.close()


async def wait_for(predicate):
    async with asyncio.timeout(20):
        while not predicate():
            await asyncio.sleep(0.03)


def test_keyboard_to_real_intake_cancel_restart_and_cli_share_records(api_server):
    base, environment, stop, start = api_server
    ids = {}

    async def journey():
        app = FirebirdApp(base, poll_seconds=0.2)
        async with app.run_test(size=(100, 34)) as pilot:
            await wait_for(lambda: app.connected)
            await pilot.press("ctrl+n")
            app.screen.query_one("#name", Input).value = "Real terminal fixture"
            await pilot.press("enter")
            await wait_for(lambda: len(app.projects) == 1)
            await pilot.press("f1", "enter")
            await wait_for(lambda: app.project_id is not None)
            ids["project"] = app.project_id
            await pilot.press("f4")
            app.screen.query_one("#source", Select).value = "local"
            await pilot.pause()
            app.screen.query_one("#target", Input).value = "fixture"
            app.screen.query_one("#submit", Button).press()
            await wait_for(lambda: app.jobs and app.jobs[0]["status"] == "succeeded")
            ids["intake"] = app.jobs[0]["id"]
            await pilot.press("f2", "enter")
            await wait_for(lambda: '"total_frames": 20' in app.query_one("#detail", TextArea).text)
            assert '"metadata_only"' in app.query_one("#detail", TextArea).text
            async with httpx.AsyncClient(base_url=base, trust_env=False) as client:
                assert (
                    await client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
                ).status_code == 200
                response = await client.post(
                    f"/api/v1/projects/{app.project_id}/policy-jobs",
                    json={
                        "operation": "policy.import",
                        "runtime_id": "slow",
                        "source_id": "source",
                    },
                )
                assert response.status_code == 202, response.text
                ids["slow"] = response.json()["id"]
            await wait_for(
                lambda: any(j["id"] == ids["slow"] and j["status"] == "running" for j in app.jobs)
            )
            await pilot.press("f2")
            app.query_one("#jobs", OptionList).highlighted = next(
                i for i, j in enumerate(app.jobs) if j["id"] == ids["slow"]
            )
            await pilot.pause()
            await pilot.press("f8")
            app.screen.query_one("#confirm", Button).press()
            await wait_for(
                lambda: any(j["id"] == ids["slow"] and j["status"] == "cancelled" for j in app.jobs)
            )
            await pilot.resize_terminal(48, 18)
            await pilot.press("ctrl+q")

    asyncio.run(journey())
    output = subprocess.run(
        [sys.executable, "-m", "vla_platform.cli", "projects", "list"],
        env=environment | {"FIREBIRD_API_URL": base},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert output.returncode == 0, output.stderr
    assert json.loads(output.stdout)[0]["id"] == ids["project"]
    stop()
    start()
    with httpx.Client(base_url=base, trust_env=False) as client:
        assert client.get("/api/v1/jobs/" + ids["intake"]).json()["result"]["total_frames"] == 20
        assert client.get("/api/v1/jobs/" + ids["slow"]).json()["status"] == "cancelled"
