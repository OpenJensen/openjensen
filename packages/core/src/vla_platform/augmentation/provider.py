"""Bounded Gemini Omni video editing through the official Interactions API.

Verified against https://ai.google.dev/gemini-api/docs/omni and
https://ai.google.dev/api/interactions-api on 2026-09-26. Cloud login uses
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/edit-videos
and its global Interactions endpoint with short-lived gcloud bearer tokens.
Uploaded editing clips must be at most 10 seconds (the caller validates duration).
Developer API uploaded-video editing is unavailable in the EEA, Switzerland, and UK.
Appearance prompts do not guarantee
preserved motion or labels; callers must review generated datasets before training.
System instructions and sampling controls are unsupported. Generated videos carry
SynthID; generated Files expire after 48 hours. We request no stored interaction,
use inline source media, and download generated Files immediately on the Developer
API route. Cloud uses inline output and does not require a Cloud Storage bucket.
"""

import asyncio
import base64
import binascii
import json
import os
import re
import shutil
import signal
from urllib.parse import urljoin, urlsplit

import httpx

MODEL = "gemini-omni-1.1-flash"
CLOUD_MODEL = "gemini-omni-1.1-flash-preview"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
MAX_INPUT_VIDEO_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 20 * 1024 * 1024
MAX_OUTPUT_VIDEO_BYTES = 40 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * ((MAX_OUTPUT_VIDEO_BYTES + 2) // 3) + 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
FILE_POLL_INTERVAL_SECONDS = 5
FILE_POLL_TIMEOUT_SECONDS = 180


class AugmentationError(ValueError):
    """An actionable augmentation failure safe to include in a job result."""


async def google_cloud_access_token() -> str:
    """Read a short-lived token from the operator's Google Cloud CLI login.

    Never expose CLI output in errors or save the token to application storage.
    The CLI owns credential refresh; this command never starts a login prompt.
    """
    executable = shutil.which("gcloud")
    if executable is None:
        raise AugmentationError("Install Google Cloud CLI and connect Google Cloud in Settings.")
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "auth",
            "print-access-token",
            "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={
                **os.environ,
                "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
                "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK": "1",
            },
            start_new_session=os.name == "posix",
        )
    except OSError:
        raise AugmentationError("Could not start Google Cloud CLI. Check Settings.") from None
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=20)
    except (TimeoutError, asyncio.CancelledError) as error:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.communicate(), timeout=3)
        except TimeoutError:
            pass
        if isinstance(error, asyncio.CancelledError):
            raise
        raise AugmentationError("Google Cloud sign-in check timed out. Try again.") from None
    if process.returncode or len(stdout) > 16384:
        raise AugmentationError("Google Cloud sign-in expired or failed. Reconnect in Settings.")
    try:
        token = stdout.decode("ascii").strip()
    except UnicodeDecodeError:
        token = ""
    if not re.fullmatch(r"[A-Za-z0-9._~+/=-]{1,16384}", token):
        raise AugmentationError(
            "Google Cloud returned an invalid access token. Reconnect in Settings."
        )
    return token


class GeminiOmniProvider:
    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        google_cloud_project: str | None = None,
    ):
        if google_cloud_project is not None and not re.fullmatch(
            r"[a-z][a-z0-9-]{4,28}[a-z0-9]", google_cloud_project
        ):
            raise AugmentationError("Choose a valid Google Cloud project in Settings.")
        if google_cloud_project is None and (not isinstance(api_key, str) or not api_key.strip()):
            raise AugmentationError("Set GEMINI_API_KEY to enable Gemini augmentation.")
        self._api_key = api_key.strip() if api_key else None
        self._cloud_project = google_cloud_project
        self._access_token = None
        self.model = CLOUD_MODEL if google_cloud_project else MODEL
        self._api_root = (
            f"https://aiplatform.googleapis.com/v1beta1/projects/{google_cloud_project}/locations/global"
            if google_cloud_project
            else API_ROOT
        )
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        limit: int,
        body: dict | None = None,
        authenticated: bool = True,
        redirects: bool = False,
    ) -> bytes:
        headers = {"Accept": "application/json" if body is not None else "*/*"}
        if authenticated:
            if self._cloud_project:
                self._access_token = await google_cloud_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                headers["x-goog-user-project"] = self._cloud_project
            else:
                headers["x-goog-api-key"] = self._api_key
                headers["Api-Revision"] = "2026-05-20"
        try:
            async with self._client.stream(
                method,
                url,
                headers=headers,
                json=body,
                timeout=httpx.Timeout(300, connect=20),
                follow_redirects=False,
            ) as response:
                if redirects and response.status_code in {301, 302, 303, 307, 308}:
                    target = urljoin(url, response.headers.get("location", ""))
                    if not self._safe_download_redirect(target):
                        raise AugmentationError("Gemini returned an invalid video download URL.")
                    # Follow at most one redirect and never forward the API key to storage.
                    return await self._request(
                        "GET",
                        target,
                        limit=limit,
                        authenticated=False,
                    )
                if not 200 <= response.status_code < 300:
                    raise AugmentationError(
                        f"Gemini request failed (HTTP {response.status_code}); "
                        "check model access, API quota, and account region."
                    )
                content_length = response.headers.get("content-length")
                if content_length and content_length.isdigit() and int(content_length) > limit:
                    raise AugmentationError("Gemini response exceeds the configured size limit.")
                result = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(result) + len(chunk) > limit:
                        raise AugmentationError(
                            "Gemini response exceeds the configured size limit."
                        )
                    result.extend(chunk)
                return bytes(result)
        except httpx.TimeoutException:
            raise TimeoutError("Gemini video request timed out.") from None
        except httpx.HTTPError:
            raise AugmentationError("Unable to reach the Gemini API.") from None

    @staticmethod
    def _safe_download_redirect(uri: str) -> bool:
        try:
            parsed = urlsplit(uri)
            host = parsed.hostname or ""
            return (
                parsed.scheme == "https"
                and parsed.port in {None, 443}
                and parsed.username is None
                and parsed.password is None
                and (
                    host == "storage.googleapis.com"
                    or host.endswith(".storage.googleapis.com")
                    or host.endswith(".googleusercontent.com")
                )
            )
        except ValueError:
            return False

    @staticmethod
    def _json(data: bytes) -> dict:
        try:
            document = json.loads(data)
        except ValueError, UnicodeDecodeError:
            raise AugmentationError("Gemini returned an invalid JSON response.") from None
        if not isinstance(document, dict):
            raise AugmentationError("Gemini returned an invalid response object.")
        return document

    @staticmethod
    def _file_id(uri: str) -> str:
        # Rebuild URLs from a validated resource name rather than fetching model output URLs.
        match = re.fullmatch(
            r"(?:https://generativelanguage\.googleapis\.com/v1beta/)?"
            r"files/([A-Za-z0-9_-]+)(?::download\?alt=media)?",
            uri,
        )
        if not match:
            raise AugmentationError("Gemini returned an invalid video file reference.")
        return match.group(1)

    async def _download(self, uri: str) -> bytes:
        if self._cloud_project:
            # Cloud mode requests inline output; do not send its token to Files
            # or follow arbitrary model-returned URLs or Cloud Storage objects.
            raise AugmentationError("Google Cloud did not return inline video data.")
        file_id = self._file_id(uri)
        async with asyncio.timeout(FILE_POLL_TIMEOUT_SECONDS):
            while True:
                metadata = self._json(
                    await self._request(
                        "GET",
                        f"{API_ROOT}/files/{file_id}",
                        limit=MAX_METADATA_BYTES,
                    )
                )
                state = metadata.get("state")
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise AugmentationError("Gemini video processing failed.")
                if state != "PROCESSING":
                    raise AugmentationError("Gemini returned an unsupported file state.")
                await asyncio.sleep(FILE_POLL_INTERVAL_SECONDS)
            return await self._request(
                "GET",
                f"{API_ROOT}/files/{file_id}:download?alt=media",
                limit=MAX_OUTPUT_VIDEO_BYTES,
                redirects=True,
            )

    async def edit(self, video: bytes, prompt: str) -> tuple[bytes, str | None]:
        if not video or len(video) > MAX_INPUT_VIDEO_BYTES:
            raise AugmentationError("Source video must be nonempty and at most 20 MiB.")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
            raise AugmentationError("Augmentation prompt must contain 1 to 8000 characters.")
        body = {
            "model": self.model,
            "input": [
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "video",
                            "mime_type": "video/mp4",
                            "data": base64.b64encode(video).decode("ascii"),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "generation_config": {"video_config": {"task": "edit"}},
            "response_format": {"type": "video", "delivery": "uri", "resolution": "720p"},
            "background": False,
            "store": False,
            "stream": False,
        }
        if self._cloud_project:
            body["response_format"] = [
                {"type": "video", "delivery": "inline", "resolution": "720p"}
            ]
        # Include base64 expansion and prompt/JSON overhead in the inline-request cap.
        encoded_size = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
        if encoded_size > MAX_REQUEST_BYTES:
            raise AugmentationError("Encoded Gemini request exceeds 20 MiB; use a smaller clip.")
        # One paid POST only: retrying uncertain failures can create duplicate charges.
        response = self._json(
            await self._request(
                "POST",
                f"{self._api_root}/interactions",
                body=body,
                limit=MAX_RESPONSE_BYTES,
            )
        )
        if response.get("status") != "completed":
            raise AugmentationError("Gemini did not complete the video edit.")
        videos = []
        steps = response.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                content = step.get("content")
                if isinstance(content, list):
                    videos.extend(
                        item
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "video"
                    )
        # MIME is optional in VideoContent; the caller validates the actual MP4 container.
        if len(videos) != 1 or videos[0].get("mime_type") not in {None, "video/mp4"}:
            raise AugmentationError("Gemini did not return exactly one MP4 video.")
        output = videos[0]
        if isinstance(output.get("data"), str) and output["data"]:
            if len(output["data"]) > 4 * ((MAX_OUTPUT_VIDEO_BYTES + 2) // 3):
                raise AugmentationError("Gemini video exceeds the 40 MiB output limit.")
            try:
                result = base64.b64decode(output["data"], validate=True)
            except ValueError, binascii.Error:
                raise AugmentationError("Gemini returned invalid video data.") from None
        elif isinstance(output.get("uri"), str):
            result = await self._download(output["uri"])
        else:
            raise AugmentationError("Gemini returned no downloadable video data.")
        if not result or len(result) > MAX_OUTPUT_VIDEO_BYTES:
            raise AugmentationError("Gemini video is empty or exceeds the 40 MiB output limit.")
        interaction_id = response.get("id")
        if (
            not isinstance(interaction_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,512}", interaction_id)
            or (self._api_key and self._api_key in interaction_id)
            or (self._access_token and self._access_token in interaction_id)
        ):
            interaction_id = None
        return result, interaction_id
