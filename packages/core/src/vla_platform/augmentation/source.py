"""Stream immutable Hub camera videos to disk without using the preview byte budget."""

import asyncio
from pathlib import Path
from urllib.parse import urljoin

import httpx

from vla_platform.datasets.explore import allowed_url

MAX_SOURCE_BYTES = 512 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 300


async def download_video(client: httpx.AsyncClient, url: str, target: Path) -> None:
    """Download one bounded source shard; never cache its bytes in preview memory."""
    created = False
    try:
        async with asyncio.timeout(DOWNLOAD_TIMEOUT_SECONDS):
            for _ in range(6):
                if not allowed_url(url):
                    raise ValueError("Source camera video redirects outside Hugging Face storage")
                async with client.stream("GET", url, follow_redirects=False) as response:
                    if response.is_redirect:
                        url = urljoin(str(response.url), response.headers.get("location", ""))
                        continue
                    if response.status_code in {401, 403}:
                        raise ValueError(
                            "Source camera video is gated or private; use a public dataset"
                        )
                    if response.status_code == 404:
                        raise ValueError(
                            "Source camera video was not found at the inspected revision"
                        )
                    response.raise_for_status()
                    size = response.headers.get("content-length", "")
                    message = "Source camera video exceeds the 512 MiB augmentation download limit"
                    if size.isdigit() and int(size) > MAX_SOURCE_BYTES:
                        raise ValueError(message)
                    with target.open("xb") as output:
                        created = True
                        downloaded = 0
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            downloaded += len(chunk)
                            if downloaded > MAX_SOURCE_BYTES:
                                raise ValueError(message)
                            output.write(chunk)
                    if not downloaded:
                        raise ValueError("Source camera video is empty")
                    return
            raise ValueError("Source camera video exceeded the redirect limit")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
