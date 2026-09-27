import asyncio
import base64
import json

import httpx
import pytest
from vla_platform.augmentation import provider as module
from vla_platform.augmentation.provider import AugmentationError, GeminiOmniProvider


def completed(content=None, **extra):
    return {
        "id": "v1_test",
        "status": "completed",
        "steps": [
            {"type": "user_input", "content": [{"type": "video", "data": "ignored"}]},
            {
                "type": "model_output",
                "content": content
                or [
                    {
                        "type": "video",
                        "mime_type": "video/mp4",
                        "data": base64.b64encode(b"edited mp4").decode(),
                    }
                ],
            },
        ],
        **extra,
    }


def run_provider(handler, video=b"source mp4", prompt="Use warm lighting."):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = GeminiOmniProvider("secret-key", client)
            try:
                return await provider.edit(video, prompt)
            finally:
                await provider.close()
                assert not client.is_closed

    return asyncio.run(run())


def test_omni_actual_rest_request_and_output_shape():
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == f"{module.API_ROOT}/interactions"
        assert request.headers["x-goog-api-key"] == "secret-key"
        body = json.loads(request.content)
        assert body["model"] == "gemini-omni-1.1-flash"
        assert body["generation_config"] == {"video_config": {"task": "edit"}}
        assert body["response_format"] == {
            "type": "video",
            "delivery": "uri",
            "resolution": "720p",
        }
        assert body["background"] is body["stream"] is body["store"] is False
        source, prompt = body["input"][0]["content"]
        assert source["type"] == "video" and source["mime_type"] == "video/mp4"
        assert base64.b64decode(source["data"]) == b"source mp4"
        assert prompt == {"type": "text", "text": "Use warm lighting."}
        return httpx.Response(200, json=completed())

    assert run_provider(handler) == (b"edited mp4", "v1_test")
    assert len(requests) == 1


def test_cloud_omni_uses_login_token_and_cloud_model_without_api_key(monkeypatch):
    token_calls = []

    async def access_token():
        token_calls.append(True)
        return "short-lived-secret"

    monkeypatch.setattr(module, "google_cloud_access_token", access_token)

    def handler(request):
        assert str(request.url) == (
            "https://aiplatform.googleapis.com/v1beta1/projects/robotics-demo/locations/global/interactions"
        )
        assert request.headers["authorization"] == "Bearer short-lived-secret"
        assert request.headers["x-goog-user-project"] == "robotics-demo"
        assert "x-goog-api-key" not in request.headers
        assert "api-revision" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == module.CLOUD_MODEL
        assert body["response_format"] == [
            {"type": "video", "delivery": "inline", "resolution": "720p"}
        ]
        assert body["store"] is False
        return httpx.Response(200, json=completed(id="short-lived-secret"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = GeminiOmniProvider(client=client, google_cloud_project="robotics-demo")
            assert await provider.edit(b"source mp4", "Warm lighting") == (b"edited mp4", None)
            assert await provider.edit(b"source mp4", "Cool lighting") == (b"edited mp4", None)

    asyncio.run(run())
    assert len(token_calls) == 2  # CLI refreshes credentials before each paid request.


def test_cloud_omni_does_not_fetch_model_returned_uris(monkeypatch):
    async def access_token():
        return "short-lived-secret"

    monkeypatch.setattr(module, "google_cloud_access_token", access_token)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=completed(
                [{"type": "video", "uri": "https://storage.googleapis.com/private/video.mp4"}]
            ),
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = GeminiOmniProvider(client=client, google_cloud_project="robotics-demo")
            with pytest.raises(AugmentationError, match="inline video"):
                await provider.edit(b"source mp4", "Warm lighting")

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize("project", ["", "../../other", "name?query=bad", "short"])
def test_cloud_project_validation(project):
    with pytest.raises(AugmentationError, match="valid Google Cloud project"):
        GeminiOmniProvider(google_cloud_project=project)


@pytest.mark.parametrize(
    "returncode,stdout",
    [
        (1, b"secret-token"),
        (0, b"bad\nheader"),
        (0, b"\xff"),
        (0, b"x" * 16385),
    ],
)
def test_cloud_token_failures_are_sanitized(monkeypatch, returncode, stdout):
    class Process:
        async def communicate(self):
            return stdout, None

    process = Process()
    process.returncode = returncode

    async def create(*args, **kwargs):
        assert args == ("/fixture/gcloud", "auth", "print-access-token", "--quiet")
        assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
        assert kwargs["env"]["CLOUDSDK_CORE_DISABLE_PROMPTS"] == "1"
        return process

    monkeypatch.setattr(module.shutil, "which", lambda _: "/fixture/gcloud")
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", create)
    with pytest.raises(AugmentationError) as error:
        asyncio.run(module.google_cloud_access_token())
    assert "secret-token" not in str(error.value)
    assert "Reconnect in Settings" in str(error.value)


def test_cloud_token_reads_cli_output_without_persisting_it(monkeypatch):
    class Process:
        returncode = 0

        async def communicate(self):
            return b"short-lived-token\n", None

    async def create(*args, **kwargs):
        return Process()

    monkeypatch.setattr(module.shutil, "which", lambda _: "/fixture/gcloud")
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", create)
    assert asyncio.run(module.google_cloud_access_token()) == "short-lived-token"


def test_file_delivery_polls_then_downloads(monkeypatch):
    monkeypatch.setattr(module, "FILE_POLL_INTERVAL_SECONDS", 0)
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(
                200,
                json=completed(
                    [
                        {
                            "type": "video",
                            "mime_type": "video/mp4",
                            "uri": f"{module.API_ROOT}/files/generated-1:download?alt=media",
                        }
                    ]
                ),
            )
        if request.url.path.endswith(":download"):
            return httpx.Response(200, content=b"downloaded mp4")
        return httpx.Response(
            200,
            json={
                "state": "PROCESSING" if len(calls) == 2 else "ACTIVE",
            },
        )

    assert run_provider(handler) == (b"downloaded mp4", "v1_test")
    assert len(calls) == 4


def test_download_redirect_never_forwards_key():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(
                200,
                json=completed(
                    [
                        {
                            "type": "video",
                            "mime_type": "video/mp4",
                            "uri": "files/generated",
                        }
                    ]
                ),
            )
        if request.url.host == "storage.googleapis.com":
            assert "x-goog-api-key" not in request.headers
            return httpx.Response(200, content=b"mp4")
        if request.url.path.endswith(":download"):
            return httpx.Response(
                302,
                headers={
                    "location": "https://storage.googleapis.com/generated/video?signature=abc",
                },
            )
        return httpx.Response(200, json={"state": "ACTIVE"})

    assert run_provider(handler)[0] == b"mp4"


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
def test_api_failures_redacted_and_never_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "secret-key sensitive prompt"})

    with pytest.raises(AugmentationError) as error:
        run_provider(handler)
    assert str(status) in str(error.value)
    assert "secret-key" not in str(error.value)
    assert "sensitive" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        completed(status="in_progress"),
        completed(status="requires_action"),
        completed(status="failed"),
        completed(status="new_unknown_state"),
        {"status": "completed", "output_video": {"data": "SDK-only"}},
        completed([{"type": "text", "text": "secret-key content filtered"}]),
        completed([{"type": "video", "mime_type": "video/webm", "data": "aA=="}]),
        completed([{"type": "video", "mime_type": "video/mp4", "data": "invalid"}]),
        completed([{"type": "video", "mime_type": "video/mp4", "data": ""}]),
        completed([{"type": "video", "mime_type": "video/mp4", "uri": "https://evil.test"}]),
        completed([{"type": "video", "mime_type": "video/mp4", "uri": "files/../secret"}]),
        [],
    ],
)
def test_bad_results_fail_closed(response):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response)

    with pytest.raises(AugmentationError) as error:
        run_provider(handler)
    assert "secret-key" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("state", ["FAILED", "UNKNOWN", None])
def test_unusable_file_state_fails_without_download(state):
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json=completed(
                    [
                        {
                            "type": "video",
                            "mime_type": "video/mp4",
                            "uri": "files/generated",
                        }
                    ]
                ),
            )
        return httpx.Response(200, json={"state": state})

    with pytest.raises(AugmentationError):
        run_provider(handler)
    assert len(calls) == 2


def test_size_limits_before_post_and_after_decode(monkeypatch):
    monkeypatch.setattr(module, "MAX_INPUT_VIDEO_BYTES", 3)
    with pytest.raises(AugmentationError, match="20 MiB"):
        run_provider(lambda _: pytest.fail("must not send oversized input"))
    monkeypatch.setattr(module, "MAX_INPUT_VIDEO_BYTES", 100)
    monkeypatch.setattr(module, "MAX_OUTPUT_VIDEO_BYTES", 2)
    with pytest.raises(AugmentationError, match="40 MiB"):
        run_provider(lambda _: httpx.Response(200, json=completed()))


def test_response_stream_size_is_bounded(monkeypatch):
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 5)

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"123"
            yield b"456"
            pytest.fail("reader must stop at cap")

    with pytest.raises(AugmentationError, match="size limit"):
        run_provider(lambda _: httpx.Response(200, stream=Chunks()))


def test_content_length_rejected_before_read(monkeypatch):
    monkeypatch.setattr(module, "MAX_RESPONSE_BYTES", 5)
    with pytest.raises(AugmentationError, match="size limit"):
        run_provider(lambda _: httpx.Response(200, content=b"123456"))


def test_encoded_request_cap_before_post(monkeypatch):
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 600)
    with pytest.raises(AugmentationError, match="Encoded Gemini request"):
        run_provider(lambda _: pytest.fail("must not send oversized request"), video=b"x" * 300)


def test_optional_mime_and_sensitive_interaction_id():
    response = completed([{"type": "video", "data": "bXA0"}], id="secret-key")
    assert run_provider(lambda _: httpx.Response(200, json=response)) == (b"mp4", None)


def test_download_size_limit_and_bad_redirect(monkeypatch):
    monkeypatch.setattr(module, "MAX_OUTPUT_VIDEO_BYTES", 2)

    def handler(request):
        if request.method == "POST":
            return httpx.Response(
                200,
                json=completed(
                    [
                        {
                            "type": "video",
                            "mime_type": "video/mp4",
                            "uri": "files/generated",
                        }
                    ]
                ),
            )
        if request.url.path.endswith(":download"):
            return httpx.Response(200, content=b"mp4")
        return httpx.Response(200, json={"state": "ACTIVE"})

    with pytest.raises(AugmentationError, match="size limit"):
        run_provider(handler)

    def bad_redirect(request):
        if request.url.path.endswith(":download"):
            return httpx.Response(302, headers={"location": "https://evil.test/secret-key"})
        return handler(request)

    with pytest.raises(AugmentationError, match="invalid video download URL"):
        run_provider(bad_redirect)


def test_transport_failure_redacted_and_timeout_and_cancel_propagate():
    def network_failure(request):
        raise httpx.ConnectError("secret-key", request=request)

    with pytest.raises(AugmentationError, match="Unable to reach") as error:
        run_provider(network_failure)
    assert "secret-key" not in str(error.value)

    def timeout(request):
        raise httpx.ReadTimeout("secret-key", request=request)

    with pytest.raises(TimeoutError, match="timed out") as error:
        run_provider(timeout)
    assert "secret-key" not in str(error.value)

    def cancel(_):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_provider(cancel)
