"""Stream GCS downloads through the credentialed SkyPilot environment."""

import asyncio
from pathlib import Path

from vla_platform import cloud_compute_catalog
from vla_platform.lifecycle import sky_runner


class CloudDownload:
    def __init__(self, process, filename):
        self.process = process
        self.filename = filename
        self.errors = asyncio.create_task(self._read_errors())

    async def _read_errors(self):
        captured = bytearray()
        while block := await self.process.stderr.read(8192):
            if len(captured) < 8192:
                captured.extend(block[: 8192 - len(captured)])
        return bytes(captured)

    async def aclose(self):
        if self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        await self.errors

    def __aiter__(self):
        return self.chunks()

    async def chunks(self):
        try:
            while block := await self.process.stdout.read(1024**2):
                yield block
            if await self.process.wait():
                raise RuntimeError("Cloud artifact download failed its integrity check")
        finally:
            await self.aclose()


async def open_download(directory: Path, filename: str):
    python = cloud_compute_catalog.sky_python(sky_runner.executable())
    if not python:
        raise ValueError("The cloud download environment is unavailable")
    process = await asyncio.create_subprocess_exec(
        python,
        str(Path(__file__).with_name("cloud_storage.py")),
        "stream",
        str(directory.resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    download = CloudDownload(process, filename)
    try:
        ready = await asyncio.wait_for(process.stdout.readline(), 60)
        if ready != b"FIREBIRD_ARCHIVE_READY\n":
            raise ValueError("Cloud artifact is unavailable or its stored integrity check failed")
    except TimeoutError as exc:
        await download.aclose()
        raise ValueError("Cloud artifact download preparation timed out") from exc
    except BaseException:
        await download.aclose()
        raise
    return download
