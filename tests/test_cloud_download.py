"""The HTTP download process streams with backpressure and dies on disconnect."""

import asyncio
import sys

import pytest
from vla_platform.lifecycle import cloud_download


@pytest.fixture
def command(monkeypatch, tmp_path):
    real_spawn = asyncio.create_subprocess_exec
    script = tmp_path / "stream.py"
    processes = []

    async def spawn(*args, **kwargs):
        assert args[2] == "stream"
        process = await real_spawn(sys.executable, str(script), **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(cloud_download.sky_runner, "executable", lambda: "/fixture/sky")
    monkeypatch.setattr(
        cloud_download.cloud_compute_catalog, "sky_python", lambda _: sys.executable
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return script, processes


def test_explicit_download_yields_artifact_bytes_in_bounded_chunks(command, tmp_path):
    script, processes = command
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'FIREBIRD_ARCHIVE_READY\\n')\n"
        "sys.stdout.buffer.write(b'x' * (3 * 1024**2 + 17))\n"
        "sys.stdout.buffer.flush()\n"
    )

    async def exercise():
        download = await cloud_download.open_download(tmp_path, "checkpoint.tar")
        chunks = [chunk async for chunk in download]
        assert sum(map(len, chunks)) == 3 * 1024**2 + 17
        assert max(map(len, chunks)) <= 1024**2
        assert processes[0].returncode == 0
        assert download.filename == "checkpoint.tar"

    asyncio.run(exercise())


def test_failed_preflight_never_returns_a_successful_download(command, tmp_path):
    script, processes = command
    script.write_text("import sys\nsys.stderr.write('checksum mismatch')\nsys.exit(1)\n")

    async def exercise():
        with pytest.raises(ValueError, match="integrity check"):
            await cloud_download.open_download(tmp_path, "checkpoint.tar")
        assert processes[0].returncode is not None

    asyncio.run(exercise())


def test_disconnect_terminates_cloud_reader(command, tmp_path):
    script, processes = command
    script.write_text(
        "import sys, time\n"
        "sys.stdout.buffer.write(b'FIREBIRD_ARCHIVE_READY\\n')\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(600)\n"
    )

    async def exercise():
        download = await cloud_download.open_download(tmp_path, "checkpoint.tar")
        await asyncio.wait_for(download.aclose(), 3)
        assert processes[0].returncode is not None

    asyncio.run(exercise())


def test_midstream_integrity_failure_is_not_a_completed_response(command, tmp_path):
    script, _ = command
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'FIREBIRD_ARCHIVE_READY\\npartial tar')\n"
        "sys.stdout.buffer.flush()\n"
        "sys.exit(1)\n"
    )

    async def exercise():
        download = await cloud_download.open_download(tmp_path, "checkpoint.tar")
        with pytest.raises(RuntimeError, match="integrity"):
            _ = [chunk async for chunk in download]

    asyncio.run(exercise())
