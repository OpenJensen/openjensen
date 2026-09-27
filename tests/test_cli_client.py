"""Read-only wait and atomic byte-download contracts; no provider/model calls."""

import asyncio
import hashlib
import json
import os
import stat

import httpx
import pytest
from typer.testing import CliRunner
from vla_platform import cli, cli_client
from vla_platform.tui_client import ApiClient, ApiError

TIME = "2026-09-27T10:00:00Z"


def job(status="succeeded", ident="job"):
    return {
        "id": ident,
        "project_id": "project",
        "status": status,
        "request": {"source": "huggingface", "repo_id": "generated/dataset", "revision": "fixture"},
        "created_at": TIME,
        "updated_at": TIME,
    }


def artifact(**changes):
    return {
        "id": "job:policy",
        "project_id": "project",
        "job_id": "job",
        "label": "Generated archive",
        "format": "native_quantized",
        "path": "server-owned",
        "manifest_sha256": "a" * 64,
        "file_bytes": 4,
        "metadata": {},
        **changes,
    }


class Bytes(httpx.AsyncByteStream):
    def __init__(self, data=b"generated archive bytes", fail=False, hold=None):
        self.data, self.fail, self.hold, self.closed = data, fail, hold, False

    async def __aiter__(self):
        yield self.data
        if self.fail:
            raise httpx.ReadError("PRIVATE_URL_OR_TOKEN")
        if self.hold is not None:
            self.hold.set()
            await asyncio.sleep(20)

    async def aclose(self):
        self.closed = True


def setup(monkeypatch, handler):
    calls = []

    async def handle(request):
        calls.append(request)
        value = handler(request)
        return await value if asyncio.iscoroutine(value) else value

    api = ApiClient("http://127.0.0.1:9999", transport=httpx.MockTransport(handle))
    monkeypatch.setattr(cli_client, "client", lambda: api)
    return api, calls


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled", "interrupted"])
def test_wait_returns_each_real_terminal_status_without_mutation(monkeypatch, status):
    states = iter(["queued", "running", status])
    api, calls = setup(monkeypatch, lambda _: httpx.Response(200, json=job(next(states))))
    result = asyncio.run(cli_client.wait_job("job", 3, 0.1))
    assert result["status"] == status and len(calls) == 3
    assert {r.method for r in calls} == {"GET"} and api.http.is_closed


@pytest.mark.parametrize("payload", [job(ident="other"), job(status="unknown"), {"error": "bad"}])
def test_wait_rejects_mismatched_or_invalid_records(monkeypatch, payload):
    api, calls = setup(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(ApiError):
        asyncio.run(cli_client.wait_job("job", 1, 0.1))
    assert len(calls) == 1 and api.http.is_closed


def test_wait_deadline_never_cancels_job(monkeypatch):
    api, calls = setup(monkeypatch, lambda _: httpx.Response(200, json=job("running")))
    with pytest.raises(cli_client.WaitDeadline, match="not cancelled"):
        asyncio.run(cli_client.wait_job("job", 1, 0.1))
    assert calls and all(r.method == "GET" for r in calls) and api.http.is_closed


def test_wait_bounds_even_a_stalled_request(monkeypatch):
    async def stalled(_):
        await asyncio.sleep(20)

    api, calls = setup(monkeypatch, stalled)
    with pytest.raises(cli_client.WaitDeadline):
        asyncio.run(cli_client.wait_job("job", 1, 0.1))
    assert len(calls) == 1 and api.http.is_closed


def test_wait_cancellation_only_closes_observation(monkeypatch):
    async def scenario():
        waiting = asyncio.Event()

        async def stalled(_):
            waiting.set()
            await asyncio.sleep(20)

        api, calls = setup(monkeypatch, stalled)
        task = asyncio.create_task(cli_client.wait_job("job", 10, 0.1))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1 and calls[0].method == "GET" and api.http.is_closed

    asyncio.run(scenario())


def download_server(monkeypatch, stream=None, headers=None, records=None, status=200, after=None):
    stream = stream or Bytes()
    headers = {"content-type": "application/x-tar", **(headers or {})}
    count = 0

    def handle(request):
        nonlocal count
        if request.url.path.endswith("/artifacts"):
            count += 1
            if count == 2 and after:
                after()
            return httpx.Response(200, json=records if records is not None else [artifact()])
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(status, headers=headers, stream=stream)

    return (*setup(monkeypatch, handle), stream)


def test_download_publishes_new_exact_bytes_with_honest_receipt(tmp_path, monkeypatch):
    api, calls, stream = download_server(monkeypatch)
    output = tmp_path / "policy.tar"
    result = asyncio.run(
        cli_client.download_artifact(
            "project", "job:policy", output, expected_sha256=hashlib.sha256(stream.data).hexdigest()
        )
    )
    assert output.read_bytes() == stream.data and result["bytes"] == len(stream.data)
    assert result["sha256"] == hashlib.sha256(stream.data).hexdigest()
    assert result["expected_sha256_matched"] is True
    assert result["registered_manifest_sha256"] == "a" * 64
    assert "not independently verified" in result["verification_scope"]
    assert len(calls) == 3 and all(r.method == "GET" for r in calls)
    assert "job%3Apolicy" in calls[1].url.raw_path.decode()
    assert stream.closed and api.http.is_closed and not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize(
    "headers,status,match",
    [
        ({"content-length": "999"}, 200, "byte bound"),
        ({"content-length": "abc"}, 200, "Content-Length"),
        ({"content-length": "99"}, 200, "did not match"),
        ({"content-length": "1"}, 200, "did not match"),
        ({"content-type": "text/html"}, 200, "TAR"),
        ({"content-encoding": "gzip"}, 200, "Compressed"),
        ({"location": "https://example.invalid"}, 302, "HTTP 302"),
        ({}, 500, "HTTP 500"),
    ],
)
def test_download_rejects_bad_http_without_output(tmp_path, monkeypatch, headers, status, match):
    api, calls, stream = download_server(monkeypatch, headers=headers, status=status)
    with pytest.raises(ApiError, match=match):
        asyncio.run(
            cli_client.download_artifact(
                "project", "job:policy", tmp_path / "out.tar", max_bytes=100
            )
        )
    assert len(calls) == 2 and api.http.is_closed and stream.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("data,limit", [(b"", 100), (b"x" * 101, 100)])
def test_download_checks_actual_bytes_even_without_content_length(
    tmp_path, monkeypatch, data, limit
):
    _, _, stream = download_server(monkeypatch, stream=Bytes(data))
    with pytest.raises(ApiError):
        asyncio.run(
            cli_client.download_artifact("project", "job:policy", tmp_path / "out", max_bytes=limit)
        )
    assert stream.closed and list(tmp_path.iterdir()) == []


def test_download_stream_error_is_bounded_and_does_not_reflect_private_details(
    tmp_path, monkeypatch
):
    _, _, stream = download_server(monkeypatch, stream=Bytes(fail=True))
    with pytest.raises(ApiError, match="interrupted") as error:
        asyncio.run(cli_client.download_artifact("project", "job:policy", tmp_path / "out"))
    assert "PRIVATE" not in str(error.value) and stream.closed and list(tmp_path.iterdir()) == []


def test_download_checksum_mismatch_keeps_no_output(tmp_path, monkeypatch):
    download_server(monkeypatch)
    with pytest.raises(ApiError, match="SHA256"):
        asyncio.run(
            cli_client.download_artifact(
                "project", "job:policy", tmp_path / "out", expected_sha256="0" * 64
            )
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "records",
    [
        [],
        [artifact(project_id="foreign")],
        [artifact(), artifact()],
        [artifact(manifest_sha256="bad")],
    ],
)
def test_download_requires_unique_project_owned_manifest_before_bytes(
    tmp_path, monkeypatch, records
):
    _, calls, _ = download_server(monkeypatch, records=records)
    with pytest.raises(ApiError):
        asyncio.run(cli_client.download_artifact("project", "job:policy", tmp_path / "out"))
    assert len(calls) == 1 and list(tmp_path.iterdir()) == []


def test_download_rechecks_manifest_before_publication(tmp_path, monkeypatch):
    records = [artifact()]
    download_server(
        monkeypatch, records=records, after=lambda: records[0].update(manifest_sha256="b" * 64)
    )
    with pytest.raises(ApiError, match="changed"):
        asyncio.run(cli_client.download_artifact("project", "job:policy", tmp_path / "out"))
    assert list(tmp_path.iterdir()) == []


def test_destination_race_never_replaces_an_existing_file(tmp_path, monkeypatch):
    output = tmp_path / "out"
    download_server(monkeypatch, after=lambda: output.write_bytes(b"other operator"))
    with pytest.raises(FileExistsError):
        asyncio.run(cli_client.download_artifact("project", "job:policy", output))
    assert output.read_bytes() == b"other operator" and list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_destination_is_rejected_before_network(tmp_path, monkeypatch, symlink):
    output = tmp_path / "out"
    if symlink:
        try:
            output.symlink_to(tmp_path / "missing")
        except OSError:
            pytest.skip("Creating symlinks requires local permission")
    else:
        output.write_bytes(b"existing")
    monkeypatch.setattr(
        cli_client, "client", lambda: pytest.fail("Network must not be initialized")
    )
    with pytest.raises(ApiError, match="already exists"):
        asyncio.run(cli_client.download_artifact("project", "job:policy", output))
    assert output.is_symlink() if symlink else output.read_bytes() == b"existing"


def test_cancellation_removes_partial_and_closes_stream(tmp_path, monkeypatch):
    async def scenario():
        waiting = asyncio.Event()
        api, _, stream = download_server(monkeypatch, stream=Bytes(hold=waiting))
        task = asyncio.create_task(
            cli_client.download_artifact("project", "job:policy", tmp_path / "out")
        )
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed and api.http.is_closed and list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_download_total_deadline_cleans_partial(tmp_path, monkeypatch):
    _, _, stream = download_server(monkeypatch, stream=Bytes(hold=asyncio.Event()))
    with pytest.raises(ApiError, match="timed out"):
        asyncio.run(
            cli_client.download_artifact("project", "job:policy", tmp_path / "out", timeout=1)
        )
    assert stream.closed and list(tmp_path.iterdir()) == []


def test_file_sync_failure_never_publishes_output(tmp_path, monkeypatch):
    download_server(monkeypatch)
    monkeypatch.setattr(
        os, "fsync", lambda _: (_ for _ in ()).throw(OSError("generated sync failure"))
    )
    with pytest.raises(OSError):
        asyncio.run(cli_client.download_artifact("project", "job:policy", tmp_path / "out"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is POSIX-specific")
@pytest.mark.parametrize("error", [OSError, TimeoutError])
def test_failed_directory_sync_removes_only_our_published_inode(tmp_path, monkeypatch, error):
    output = tmp_path / "out"
    download_server(monkeypatch)
    original = os.fsync

    def failed(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise error("generated directory sync failure")
        original(fd)

    monkeypatch.setattr(os, "fsync", failed)
    with pytest.raises((OSError, ApiError)):
        asyncio.run(cli_client.download_artifact("project", "job:policy", output))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("replacement", [False, True])
def test_interruption_at_real_link_cleans_only_owned_output(tmp_path, monkeypatch, replacement):
    output = tmp_path / "out"
    download_server(monkeypatch)
    original = os.link

    def interrupted(source, destination):
        original(source, destination)
        if replacement:
            output.unlink()
            output.write_bytes(b"other operator")
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(cli_client.download_artifact("project", "job:policy", output))
    if replacement:
        assert output.read_bytes() == b"other operator"
        assert list(tmp_path.iterdir()) == [output]
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "status,exit_code", [("succeeded", 0), ("failed", 1), ("cancelled", 1), ("interrupted", 1)]
)
def test_wait_cli_emits_final_json_with_meaningful_exit(monkeypatch, status, exit_code):
    setup(monkeypatch, lambda _: httpx.Response(200, json=job(status)))
    result = CliRunner().invoke(cli.app, ["jobs", "wait", "job", "--timeout", "1"])
    assert result.exit_code == exit_code and json.loads(result.stdout)["status"] == status
    assert not result.stderr
