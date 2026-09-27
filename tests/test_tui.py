"""Keyboard/client behavior; HTTP fixtures never represent robot quality evidence."""

import asyncio
import json

import httpx
import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static, TextArea
from vla_platform.tui import CancelForm, FirebirdApp, IntakeForm
from vla_platform.tui_client import ApiClient, ApiError, endpoint, plain

STAMP = "2026-09-27T00:00:00Z"


def project(ident="p", name="Project"):
    return {"id": ident, "name": name, "created_at": STAMP}


def job(ident="j", pid="p", status="running", *, policy=False):
    return {
        "id": ident,
        "project_id": pid,
        "kind": "policy.finetune" if policy else "dataset.inspect",
        "status": status,
        "created_at": STAMP,
        "updated_at": STAMP,
        "request": (
            {"operation": "policy.finetune", "runtime_id": "fixture", "dataset_job_id": "intake"}
            if policy
            else {"source": "huggingface", "repo_id": "fixture/data"}
        ),
    }


class Server:
    def __init__(self):
        self.projects = [project(), project("q", "Second")]
        self.jobs = [job(), job("trained", status="failed", policy=True)]
        self.calls = []
        self.offline = False
        self.post_timeout = False
        self.delay_jobs = None

    async def handle(self, request):
        self.calls.append((request.method, request.url.path, request.content))
        if self.offline:
            raise httpx.ConnectError("offline")
        path = request.url.path.removeprefix("/api/v1")
        if request.method == "GET":
            if path == "/projects":
                return httpx.Response(200, json=self.projects)
            if path.startswith("/projects/"):
                if self.delay_jobs:
                    await self.delay_jobs.wait()
                pid = path.split("/")[2]
                return httpx.Response(200, json=[j for j in self.jobs if j["project_id"] == pid])
            if path.endswith("/events"):
                return httpx.Response(200, json=[{"message": "[red]literal\u001b[31m"}])
            return httpx.Response(
                200, json=next(j for j in self.jobs if j["id"] == path.split("/")[2])
            )
        if self.post_timeout:
            raise httpx.ReadTimeout("lost response")
        if path == "/projects":
            item = project("new", json.loads(request.content)["name"])
            self.projects.append(item)
            return httpx.Response(201, json=item)
        if path.endswith("/cancel"):
            item = next(j for j in self.jobs if j["id"] == path.split("/")[2])
            item["status"] = "cancelled"
            return httpx.Response(200, json=item)
        item = job("intake", pid=path.split("/")[2], status="queued")
        self.jobs.append(item)
        return httpx.Response(202, json=item)

    def client(self):
        return ApiClient("http://127.0.0.1:9999", transport=httpx.MockTransport(self.handle))


async def until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def choose_project(app, pilot, index=0):
    await until(lambda: app.connected)
    app.query_one("#projects", OptionList).highlighted = index
    app.query_one("#projects", OptionList).focus()
    await pilot.press("enter")
    await until(lambda: app.project_id is not None and app.read_worker.is_finished)


def test_navigation_literal_details_resize_and_offline_recovery():
    async def scenario():
        server = Server()
        server.projects[0]["name"] = "[red]literal\u001b[31m"
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test(size=(100, 32)) as pilot:
            await choose_project(app, pilot)
            assert len(app.jobs) == 2
            await pilot.press("enter")
            await until(lambda: '"events"' in app.query_one("#detail", TextArea).text)
            assert "[red]literal" in app.query_one("#detail", TextArea).text
            assert "\x1b" not in app.query_one("#detail", TextArea).text
            await pilot.resize_terminal(48, 18)
            await pilot.press("f1")
            assert app.focused.id == "projects"
            server.offline = True
            app.action_refresh()
            await until(lambda: not app.connected)
            assert app.query_one("#intake", Button).disabled
            assert len(app.jobs) == 2  # Last known data is retained with stale status.
            server.offline = False
            app.action_refresh()
            await until(lambda: app.connected)
            await pilot.press("ctrl+q")
        assert not any(method == "POST" for method, _, _ in server.calls)

    asyncio.run(scenario())


def test_forms_create_project_and_local_snapshot_exactly_once():
    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=0.05)
        async with app.run_test() as pilot:
            await until(lambda: app.connected)
            await pilot.press("ctrl+n")
            app.screen.query_one("#name", Input).value = "New project"
            await pilot.press("enter")
            await until(lambda: any(p["id"] == "new" for p in app.projects))
            await choose_project(app, pilot)
            app.action_intake()
            await pilot.pause()
            assert isinstance(app.screen, IntakeForm)
            app.screen.query_one("#source", Select).value = "local"
            await pilot.pause()
            app.screen.query_one("#target", Input).value = "/operator/dataset"
            app.screen.query_one("#snapshot", Checkbox).value = True
            app.screen.query_one("#submit", Button).press()
            await until(lambda: any(j["id"] == "intake" for j in app.jobs))
            posts = [
                (path, json.loads(raw)) for method, path, raw in server.calls if method == "POST"
            ]
            assert len(posts) == 2
            assert posts[1][1] == {
                "source": "local",
                "path": "/operator/dataset",
                "revision": "main",
                "snapshot_for_training": True,
            }

    asyncio.run(scenario())


def test_cancel_needs_explicit_confirmation_and_actual_status():
    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            await choose_project(app, pilot)
            app.action_cancel_job()
            await pilot.pause()
            assert isinstance(app.screen, CancelForm)
            await pilot.press("enter")  # The safe default keeps the job.
            assert not any(method == "POST" for method, _, _ in server.calls)
            app.action_cancel_job()
            await pilot.pause()
            app.screen.query_one("#confirm", Button).press()
            await until(lambda: server.jobs[0]["status"] == "cancelled")
            await until(lambda: not app.writing)
            assert len([c for c in server.calls if c[0] == "POST"]) == 1
            assert app.query_one("#cancel-job", Button).disabled

    asyncio.run(scenario())


def test_cancel_does_not_target_changed_selection_or_terminal_job():
    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            await choose_project(app, pilot)
            app.action_cancel_job()
            await pilot.pause()
            app.query_one("#jobs", OptionList).highlighted = 1
            app.screen.query_one("#confirm", Button).press()
            await pilot.pause()
            assert not any(c[0] == "POST" for c in server.calls)
            app.query_one("#jobs", OptionList).highlighted = 0
            app.controls()
            app.action_cancel_job()
            await pilot.pause()
            server.jobs[0]["status"] = "succeeded"
            app.screen.query_one("#confirm", Button).press()
            await until(lambda: not app.writing)
            assert not any(c[0] == "POST" for c in server.calls)

    asyncio.run(scenario())


def test_timed_out_submission_is_never_retried_by_poll():
    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=0.05)
        async with app.run_test() as pilot:
            await choose_project(app, pilot)
            server.post_timeout = True
            app.mutate("intake", {"source": "huggingface", "repo_id": "fixture/data"}, "p")
            await until(lambda: not app.writing)
            await pilot.pause(0.15)
            assert len([c for c in server.calls if c[0] == "POST"]) == 1
            assert "Outcome unknown" in str(app.query_one("#notice", Static).content)

    asyncio.run(scenario())


def test_project_switch_ignores_slow_previous_jobs_and_removal_blocks_actions():
    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            await until(lambda: app.connected)
            server.delay_jobs = asyncio.Event()
            app.query_one("#projects", OptionList).focus()
            await pilot.press("enter")
            await pilot.press("f1", "down", "enter")
            server.delay_jobs.set()
            await until(lambda: app.read_worker.is_finished)
            assert app.project_id == "q" and app.jobs == []
            server.projects = [project()]
            app.action_refresh()
            await until(lambda: app.read_worker.is_finished)
            assert app.project_id is None
            assert app.query_one("#intake", Button).disabled

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad",
    [
        "file:///tmp/x",
        "http://user:secret@host",
        "http://x/?secret=x",
        "http://x/#frag",
        "http://x:bad",
    ],
)
def test_endpoint_rejects_ambiguous_or_secret_urls(bad):
    with pytest.raises(ValueError):
        endpoint(bad)


def test_transport_invalid_oversized_redirect_and_deadline_are_bounded(monkeypatch):
    from vla_platform import tui_client

    async def scenario():
        for reply, match in [
            (httpx.Response(200, text="not json"), "invalid JSON"),
            (httpx.Response(302, headers={"Location": "http://elsewhere"}), "302"),
            (httpx.Response(200, text="x" * 20), "limit"),
        ]:
            calls = []

            def handler(request):
                calls.append(request)
                return reply

            monkeypatch.setattr(tui_client, "MAX_RESPONSE", 16)
            client = ApiClient("http://127.0.0.1", transport=httpx.MockTransport(handler))
            with pytest.raises(ApiError, match=match):
                await client.request("POST", "/projects", {"name": "one"})
            assert len(calls) == 1
            await client.close()

        async def slow(_):
            await asyncio.sleep(10)

        client = ApiClient("http://127.0.0.1", transport=httpx.MockTransport(slow), deadline=0.03)
        with pytest.raises(ApiError, match="Outcome unknown"):
            await client.request("POST", "/projects", {})
        await client.close()

    asyncio.run(scenario())


def test_plain_does_not_allow_terminal_control_characters():
    assert "\x1b" not in plain("\x1b]8;;http://bad\x07[red]literal")
    assert "[red]literal" in plain("[red]literal")


def test_invalid_or_cross_project_records_are_rejected():
    async def scenario():
        for records, method, match in [
            ([project(), project()], "projects", "duplicate"),
            ([project() | {"created_at": "bad"}], "projects", "invalid record"),
            ([job(pid="other")], "jobs", "another project"),
            (job("different"), "job", "another job"),
        ]:
            client = ApiClient(
                "http://127.0.0.1",
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=records)),
            )
            with pytest.raises(ApiError, match=match):
                if method == "projects":
                    await client.projects()
                elif method == "jobs":
                    await client.jobs("p")
                else:
                    await client.job("j")
            await client.close()

    asyncio.run(scenario())


def test_removed_project_and_stale_keyboard_events_cannot_submit():
    from types import SimpleNamespace

    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            await until(lambda: app.connected)
            assert app.query_one("#intake", Button).disabled
            await choose_project(app, pilot)
            await pilot.press("f4")
            server.projects = []
            app.action_refresh()
            await until(lambda: app.project_id is None)
            app.screen.query_one("#target", Input).value = "fixture/data"
            app.screen.query_one("#submit", Button).press()
            await pilot.pause()
            app.project_selected(SimpleNamespace(option=SimpleNamespace(id="removed")))
            app.job_selected(SimpleNamespace(option=SimpleNamespace(id="removed")))
            assert app.project_id is None and app.job_id is None
            assert not any(c[0] == "POST" for c in server.calls)

    asyncio.run(scenario())


def test_invalid_url_does_not_reflect_secret_port_or_credentials():
    for value in ("http://operator:SECRET@localhost", "http://localhost:SECRET"):
        with pytest.raises(ValueError) as error:
            endpoint(value)
        assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("mismatch", ["project", "operation"])
def test_intake_does_not_acknowledge_another_project_or_operation(mismatch):
    async def scenario():
        server = Server()
        original = server.handle

        async def handle(request):
            if request.method == "POST" and request.url.path.endswith("/intakes"):
                server.calls.append((request.method, request.url.path, request.content))
                wrong = job(pid="other") if mismatch == "project" else job(policy=True)
                return httpx.Response(202, json=wrong)
            return await original(request)

        server.handle = handle
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            await choose_project(app, pilot)
            app.mutate("intake", {"source": "huggingface", "repo_id": "fixture/data"}, "p")
            await until(lambda: not app.writing)
            message = str(app.query_one("#notice", Static).content)
            assert "identity differs" in message and "Intake accepted" not in message
            assert len([call for call in server.calls if call[0] == "POST"]) == 1

    asyncio.run(scenario())
