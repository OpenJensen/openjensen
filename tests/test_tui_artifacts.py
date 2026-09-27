"""Terminal artifact transfer through the real bounded client; generated HTTP bytes only."""

import asyncio
import hashlib
import tarfile

import httpx
import pytest

pytest.importorskip("textual")
from test_cli_client import Bytes, artifact
from test_native_quantization import application  # noqa: F401
from test_tui import Server as BaseServer
from test_tui import choose_project, until
from textual.widgets import Button, Input, Select, TextArea
from vla_platform.tui import FirebirdApp
from vla_platform.tui_artifacts import ArtifactDownloadForm, DownloadDraft
from vla_platform.tui_client import ApiClient, ApiError


class Server(BaseServer):
    def __init__(self, stream=None):
        super().__init__()
        self.artifact = artifact(project_id="p")
        self.stream = stream or Bytes()

    async def handle(self, request):
        if request.url.path.endswith(("/artifacts", "/download")):
            self.calls.append((request.method, request.url.path, request.content))
            assert request.method == "GET"
            if request.url.path.endswith("/artifacts"):
                return httpx.Response(200, json=[self.artifact])
            return httpx.Response(
                200, headers={"content-type": "application/x-tar"}, stream=self.stream
            )
        return await super().handle(request)


class DrainingBytes(Bytes):
    def __init__(self, hold):
        super().__init__(hold=hold)
        self.closing = asyncio.Event()
        self.release_close = asyncio.Event()
        self.close_interrupted = False

    async def aclose(self):
        self.closing.set()
        try:
            await self.release_close.wait()
        except asyncio.CancelledError:
            self.close_interrupted = True
            raise
        self.closed = True


async def form_for(app, pilot, output):
    await choose_project(app, pilot)
    await pilot.press("f6")
    await until(lambda: isinstance(app.screen, ArtifactDownloadForm))
    form = app.screen
    form.query_one("#download-artifact", Select).value = "job:policy"
    form.query_one("#download-output", Input).value = str(output)
    await pilot.pause()
    return form


async def review(form):
    form.query_one("#download-review", Button).press()
    await until(lambda: form.reviewed is not None)


def test_keyboard_download_preserves_connection_and_explicit_destination(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREBIRD_API_URL", "http://wrong.invalid")

    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        output = tmp_path / "policy.tar"
        async with app.run_test(size=(48, 18)) as pilot:
            form = await form_for(app, pilot, output)
            form.query_one("#download-sha", Input).value = hashlib.sha256(
                server.stream.data
            ).hexdigest()
            await pilot.pause()
            await review(form)
            assert not output.exists() and not any(
                p.endswith("/download") for _, p, _ in server.calls
            )
            form.query_one("#download-submit", Button).focus()
            await pilot.press("enter")
            await until(lambda: form.receipt is not None and not form.busy)
            assert output.read_bytes() == server.stream.data
            assert form.receipt["expected_sha256_matched"] is True
            assert '"sha256"' in form.query_one("#download-receipt", TextArea).text
            assert not app.client.http.is_closed
            await pilot.press("escape")
            await until(lambda: app.screen is app.default_screen)
            assert await app.client.projects()
            assert all(m == "GET" for m, _, _ in server.calls)
            assert len([p for _, p, _ in server.calls if p.endswith("/download")]) == 1
        assert app.client.http.is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["registry", "project", "destination"])
def test_review_changes_prevent_transfer(tmp_path, change):
    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        output = tmp_path / "policy.tar"
        async with app.run_test() as pilot:
            form = await form_for(app, pilot, output)
            await review(form)
            if change == "registry":
                server.artifact["manifest_sha256"] = "b" * 64
            elif change == "project":
                app.project_id = "q"
            else:
                output.write_bytes(b"existing")
            form.submit()
            await until(lambda: not form.busy)
            assert form.reviewed is None
            assert not any(p.endswith("/download") for _, p, _ in server.calls)
            assert not list(tmp_path.glob(".firebird-download-*.partial"))
            assert (
                output.read_bytes() == b"existing"
                if change == "destination"
                else not output.exists()
            )
            assert not app.client.http.is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize("quit_app", [False, True])
def test_transfer_cancellation_cleans_before_dismiss_or_connection_close(tmp_path, quit_app):
    async def scenario():
        hold = asyncio.Event()
        server = Server(Bytes(hold=hold))
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            form = await form_for(app, pilot, tmp_path / "cancelled.tar")
            await review(form)
            form.submit()
            await asyncio.wait_for(hold.wait(), 5)
            if quit_app:
                app.exit()
            else:
                await pilot.press("escape")
                await until(lambda: not form.busy)
                assert app.screen is form
                assert not app.client.http.is_closed and await app.client.projects()
                await pilot.press("escape")
                await until(lambda: app.screen is app.default_screen)
        assert form.transfer_task.done() and server.stream.closed and app.client.http.is_closed
        assert list(tmp_path.iterdir()) == []
        assert all(method == "GET" for method, _, _ in server.calls)

    asyncio.run(scenario())


def test_cancel_before_transfer_task_starts_unlocks_form(tmp_path):
    async def scenario():
        server = Server()
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            form = await form_for(app, pilot, tmp_path / "out")
            await review(form)
            form.submit()
            form.stop()  # Before the transfer coroutine gets its first event-loop turn.
            await until(lambda: form.transfer_task.done() and not form.busy)
            assert not any(p.endswith("/download") for _, p, _ in server.calls)
            await pilot.press("escape")
            await until(lambda: app.screen is app.default_screen)
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_repeated_stop_and_cancelled_parent_drain_wait_for_http_close(tmp_path):
    async def scenario():
        hold = asyncio.Event()
        stream = DrainingBytes(hold)
        server = Server(stream)
        app = FirebirdApp(client=server.client(), poll_seconds=100)
        async with app.run_test() as pilot:
            form = await form_for(app, pilot, tmp_path / "out")
            await review(form)
            form.submit()
            await asyncio.wait_for(hold.wait(), 5)
            form.stop()
            await asyncio.wait_for(stream.closing.wait(), 5)
            try:
                form.stop()
                drain = asyncio.create_task(form.stop_transfer())
                await asyncio.sleep(0)
                drain.cancel()
                await asyncio.sleep(0)
                drain.cancel()
                await asyncio.sleep(0)
                assert not form.transfer_task.done() and form.busy
                assert not stream.closed and not stream.close_interrupted
                assert not app.client.http.is_closed
                form.action_back()
                assert app.screen is form
            finally:
                stream.release_close.set()
            await asyncio.wait_for(drain, 5)
            await until(lambda: not form.busy)
            assert stream.closed and not stream.close_interrupted
            assert list(tmp_path.iterdir()) == []
            assert await app.client.projects()
            await pilot.press("escape")
            await until(lambda: app.screen is app.default_screen)

    asyncio.run(scenario())


def test_terminal_downloads_a_real_api_archive_without_changing_source(application, tmp_path):  # noqa: F811
    async def scenario():
        app_api, fixture_api, pid, source, source_root = application
        calls = []
        before = {
            p.relative_to(source_root): p.read_bytes()
            for p in source_root.rglob("*")
            if p.is_file()
        }

        async def bridge(request):
            calls.append(request.method)
            response = await asyncio.to_thread(
                fixture_api.request, request.method, request.url.path
            )
            return httpx.Response(
                response.status_code, headers=response.headers, stream=Bytes(response.content)
            )

        client = ApiClient("http://testserver", transport=httpx.MockTransport(bridge))
        app = FirebirdApp(client=client, poll_seconds=100)
        output = tmp_path / "saved-policy.tar"
        async with app.run_test() as pilot:
            await choose_project(app, pilot)
            await pilot.press("f6")
            await until(lambda: isinstance(app.screen, ArtifactDownloadForm))
            form = app.screen
            form.query_one("#download-artifact", Select).value = source.id
            form.query_one("#download-output", Input).value = str(output)
            await pilot.pause()
            await review(form)
            form.submit()
            await until(lambda: form.receipt is not None and not form.busy)
            assert form.receipt["project_id"] == pid
            assert form.receipt["registered_manifest_sha256"] == source.manifest_sha256
            with tarfile.open(output) as archive:
                for name, content in before.items():
                    assert archive.extractfile("policy/" + name.as_posix()).read() == content
            assert all(method == "GET" for method in calls)
        assert before == {
            p.relative_to(source_root): p.read_bytes()
            for p in source_root.rglob("*")
            if p.is_file()
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "limit,timeout", [("true", "60"), ("100", "0"), ("1.5", "60"), ("100", "3601")]
)
def test_download_draft_rejects_unbounded_or_noninteger_values(tmp_path, limit, timeout):
    with pytest.raises(ApiError):
        DownloadDraft.parse("policy", str(tmp_path / "out"), limit, timeout, "")
