"""Real application/API and supervised CPU protocol worker, with no model-quality claim."""

import asyncio
import json

import httpx
import pytest

pytest.importorskip("textual")
from test_native_quantization import application  # noqa: F401
from test_tui import choose_project, until
from textual.widgets import Button, Checkbox, Select, TextArea
from vla_platform.tui import FirebirdApp
from vla_platform.tui_client import ApiClient
from vla_platform.tui_lifecycle_forms import LifecycleForm


def test_tui_real_application_submits_and_observes_supervised_protocol_job(application, tmp_path):  # noqa: F811
    async def scenario():
        app_api, fixture_api, pid, source, _ = application
        posts = []

        async def bridge(request):
            if request.method == "POST":
                posts.append(request.url.path)
            response = await asyncio.to_thread(
                fixture_api.request,
                request.method,
                request.url.path,
                content=request.content,
                headers={"Content-Type": "application/json"},
            )
            return httpx.Response(response.status_code, content=response.content)

        client = ApiClient("http://testserver", transport=httpx.MockTransport(bridge))
        app = FirebirdApp(client=client, poll_seconds=0.05, journal_dir=tmp_path / "journal")
        async with app.run_test(size=(100, 40)) as pilot:
            await choose_project(app, pilot)
            assert app.project_id == pid
            app.action_lifecycle()
            await until(lambda: isinstance(app.screen, LifecycleForm) and app.screen.is_mounted)
            await pilot.pause()
            form = app.screen
            form.query_one("#lifecycle-mode", Select).value = "quantize"
            await pilot.pause()
            form.query_one("#recipe-editor", TextArea).load_text(
                json.dumps(
                    {
                        "operation": "policy.quantize",
                        "runtime_id": "native-cpu",
                        "artifact_id": source.id,
                        "native_quantization": {
                            "format": "firebird_quant",
                            "bits": 8,
                            "group_size": 64,
                        },
                        "timeout_seconds": 60,
                    }
                )
            )
            await pilot.pause()
            form.query_one("#recipe-review", Button).press()
            await until(lambda: form.reviewed is not None and not form.busy)
            form.query_one("#recipe-consent", Checkbox).value = True
            await pilot.pause()
            assert not posts
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: app.screen is app.default_screen and app.job_id is not None)
            await until(
                lambda: any(j["id"] == app.job_id and j["status"] == "succeeded" for j in app.jobs),
                timeout=15,
            )
            receipt = fixture_api.get(f"/api/v1/jobs/{app.job_id}").json()
            assert receipt["result"]["artifacts"][0]["parent_ids"] == [source.id]
            assert receipt["result"]["artifacts"][0]["metadata"]["quality_verified"] is False
            assert len(posts) == 1
            assert form.journal.read() is None

    asyncio.run(scenario())
