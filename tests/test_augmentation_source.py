"""Unpaid media download regressions, independent of FFmpeg and preview budgets."""

import asyncio
import hashlib

import httpx
import pytest
from vla_platform.augmentation import source
from vla_platform.datasets.explore import Budget, ExplorationError, HubReader

URL = "https://huggingface.co/datasets/fixture/robot/resolve/" + "a" * 40 + "/video.mp4"


class VideoStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def test_video_larger_than_preview_limit_streams_to_disk(tmp_path):
    chunk = b"v" * 65536
    count = 49 * 16  # 49 MiB, larger than the old 48 MiB preview budget.
    streams = []

    def respond(request):
        stream = VideoStream(chunk for _ in range(count))
        streams.append(stream)
        return httpx.Response(
            200, headers={"content-length": str(len(chunk) * count)}, stream=stream
        )

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            reader = HubReader(client)
            with pytest.raises(ExplorationError, match="preview size limit"):
                await reader.read(URL, 48 * 1024 * 1024, Budget())
            target = tmp_path / "source.mp4"
            await source.download_video(client, URL, target)
            assert target.stat().st_size == len(chunk) * count
            digest = hashlib.sha256()
            for _ in range(count):
                digest.update(chunk)
            with target.open("rb") as saved:
                assert hashlib.file_digest(saved, "sha256").hexdigest() == digest.hexdigest()
            assert not reader.cache
        assert all(stream.closed for stream in streams)

    asyncio.run(check())


@pytest.mark.parametrize("declared_size", [False, True])
def test_oversized_video_is_rejected_and_partial_file_removed(tmp_path, monkeypatch, declared_size):
    monkeypatch.setattr(source, "MAX_SOURCE_BYTES", 65536)
    stream = VideoStream([b"v" * 65536, b"extra"])

    async def check():
        headers = {"content-length": "65541"} if declared_size else {}
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers=headers, stream=stream)
            )
        ) as client:
            target = tmp_path / "source.mp4"
            with pytest.raises(ValueError, match="augmentation download limit"):
                await source.download_video(client, URL, target)
            assert not target.exists()
        assert stream.closed

    asyncio.run(check())


def test_allowed_hub_redirect_is_followed(tmp_path):
    calls = []

    def respond(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(302, headers={"location": "https://cdn.hf.co/pinned.mp4"})
        return httpx.Response(200, content=b"video")

    async def check():
        # Even a caller configured to follow redirects cannot bypass the allowlist.
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True
        ) as client:
            target = tmp_path / "source.mp4"
            await source.download_video(client, URL, target)
            assert target.read_bytes() == b"video"
        assert calls == [URL, "https://cdn.hf.co/pinned.mp4"]

    asyncio.run(check())


@pytest.mark.parametrize(
    "destination", ["https://example.com/video.mp4", "http://127.0.0.1/private"]
)
def test_external_redirect_is_rejected_before_request(tmp_path, destination):
    calls = []

    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": destination})

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True
        ) as client:
            with pytest.raises(ValueError, match="outside Hugging Face"):
                await source.download_video(client, URL, tmp_path / "source.mp4")
        assert calls == [URL]
        assert not list(tmp_path.iterdir())

    asyncio.run(check())


def test_redirect_loop_is_bounded(tmp_path):
    calls = []

    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": URL})

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ValueError, match="redirect limit"):
                await source.download_video(client, URL, tmp_path / "source.mp4")
        assert len(calls) == 6

    asyncio.run(check())


@pytest.mark.parametrize(
    "status,message",
    [(401, "gated or private"), (403, "gated or private"), (404, "inspected revision")],
)
def test_missing_or_private_media_is_actionable(tmp_path, status, message):
    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(status))
        ) as client:
            with pytest.raises(ValueError, match=message):
                await source.download_video(client, URL, tmp_path / "source.mp4")
        assert not list(tmp_path.iterdir())

    asyncio.run(check())


def test_empty_video_is_removed(tmp_path):
    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ) as client:
            with pytest.raises(ValueError, match="empty"):
                await source.download_video(client, URL, tmp_path / "source.mp4")
        assert not list(tmp_path.iterdir())

    asyncio.run(check())


def test_cancelled_download_closes_stream_and_removes_partial_file(tmp_path):
    async def check():
        blocked = asyncio.Event()

        class BlockedStream(VideoStream):
            async def __aiter__(self):
                yield b"v" * 65536
                blocked.set()
                await asyncio.Future()

        stream = BlockedStream([])
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        ) as client:
            target = tmp_path / "source.mp4"
            task = asyncio.create_task(source.download_video(client, URL, target))
            await asyncio.wait_for(blocked.wait(), 2)
            assert target.exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not target.exists()
            assert stream.closed

    asyncio.run(check())


def test_existing_file_is_not_overwritten_or_removed(tmp_path):
    async def check():
        target = tmp_path / "source.mp4"
        target.write_bytes(b"existing")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"video"))
        ) as client:
            with pytest.raises(FileExistsError):
                await source.download_video(client, URL, target)
        assert target.read_bytes() == b"existing"

    asyncio.run(check())
