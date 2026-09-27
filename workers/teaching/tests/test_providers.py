"""Offline provider-contract fixtures; no live inference or robot-control acceptance."""

import asyncio
import base64
import copy
import io
import json
from dataclasses import asdict, replace

import httpx
import pytest
from PIL import Image

from firebird_teaching import providers as p

KEY = "fixture-openrouter-key-not-real"
CONTEXT = p.SourceContext("generated-session", "generated-episode", 2)
CHOICES = (p.Choice("eligible-a", "Already admitted option A"), p.Choice("eligible-b", "Option B"))
NOW = 8_000_000_000


class Stream(httpx.AsyncByteStream):
    def __init__(self, body, *, delay=0, chunks=None):
        self.body, self.delay, self.chunks = body, delay, chunks

    async def __aiter__(self):
        for part in self.chunks or [self.body]:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield part


def response(body=None, *, status=200, raw=None, headers=None, delay=0, chunks=None):
    raw = json.dumps(body).encode() if raw is None else raw
    return httpx.Response(
        status,
        headers={"Content-Type": "application/json", **(headers or {})},
        stream=Stream(raw, delay=delay, chunks=chunks),
    )


def decision():
    return {
        "id": "generated-request",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "usage": {"cost": 0.0001},
        "answers": {
            "selection": {
                "type": "choice",
                "choice": "eligible-a",
                "confidence": 0.6,
                "probabilities": {"eligible-a": 0.8, "eligible-b": 0.2},
            }
        },
    }


def perception(content=None):
    return {
        "model": p.PERCEPTION_MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        content
                        or {
                            "summary": "Generated colored squares.",
                            "uncertain": True,
                            "observations": [{"label": "square", "box": [0.1, 0.2, 0.5, 0.8]}],
                        },
                        indent=2,
                    )
                },
            }
        ],
    }


def frame():
    return p.FrameInput(CONTEXT, 3, 2, 2, bytes([15, 40, 120]) * 4, NOW)


def client(reply=None, **settings):
    requests = []

    async def handler(request):
        requests.append(request)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return await reply(request)
        return reply or response(decision())

    return (
        p.Providers(
            p.ProviderSettings(KEY, **settings),
            transport=httpx.MockTransport(handler),
            clock_ns=lambda: NOW,
        ),
        requests,
    )


def decide(api, **kwargs):
    return asyncio.run(
        api.decide(
            CONTEXT,
            "Generated state, no measurements.",
            "Choose an eligible ID.",
            CHOICES,
            current=lambda: CONTEXT,
            **kwargs,
        )
    )


def perceive(api, source=None, current=None):
    source = source or frame()
    return asyncio.run(
        api.perceive(
            source, "Describe the visible fixture.", current=current or (lambda: source.identity)
        )
    )


def test_documented_jev_request_and_receipt_are_bounded_eligible_only():
    api, requests = client()
    result = decide(api)
    assert result.choice == "eligible-a"
    assert result.receipt.context == CONTEXT
    assert result.receipt.reported_cost_usd == 0.0001
    assert result.receipt.generated_proposal is True
    assert result.receipt.elapsed_seconds == 0
    sent = requests[0]
    assert str(sent.url) == p.JEV_URL
    body = json.loads(sent.content)
    assert body["provider"] == {"allow_fallbacks": False}
    assert body["model"] == p.JEV_MODEL
    assert body["questions"]["selection"]["type"] == "choice"
    assert set(body["questions"]["selection"]["criteria"]) == {"eligible-a", "eligible-b"}
    assert len(result.receipt.request_sha256) == 64
    assert KEY not in repr(result) and KEY not in repr(api.settings)
    assert sent.headers["authorization"] == "Bearer " + KEY
    assert sent.headers["accept-encoding"] == "identity"
    assert len(requests) == 1


def test_perception_documented_png_json_schema_and_source_identity():
    api, requests = client(response(perception()))
    source = frame()
    result = perceive(api, source)
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == p.PERCEPTION_URL
    assert body["model"] == p.PERCEPTION_MODEL
    assert body["provider"] == {"allow_fallbacks": False, "require_parameters": True}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["max_tokens"] == 1024 and body["stream"] is False
    assert "tools" not in body and "models" not in body
    encoded = body["messages"][1]["content"][1]["image_url"]["url"]
    assert encoded.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(encoded.split(",")[1]))) as image:
        assert image.mode == "RGB" and image.size == (2, 2)
        assert image.tobytes() == source.rgb
    assert result.receipt.frame == source.identity
    assert result.receipt.reported_cost_usd is None
    assert result.receipt.response_id is None
    assert result.observations[0].box == (0.1, 0.2, 0.5, 0.8)
    assert result.uncertain is True
    assert KEY not in json.dumps(asdict(result))


@pytest.mark.parametrize("key", ["", "tiny", "x" * 4097, "x" * 20 + "\n", "é" * 20])
def test_missing_or_malformed_credentials_never_echo(key):
    with pytest.raises(p.ProviderError) as caught:
        p.ProviderSettings.from_env({"OPENROUTER_API_KEY": key})
    assert caught.value.code == "configuration_missing"
    assert str(caught.value) == "Set a valid OPENROUTER_API_KEY."


@pytest.mark.parametrize(
    "changes",
    [{"timeout_seconds": True}, {"timeout_seconds": 31}, {"max_frame_age_seconds": float("nan")}],
)
def test_settings_limits(changes):
    with pytest.raises(p.ProviderError):
        p.ProviderSettings(KEY, **changes)


def test_credential_configuration_reads_only_explicit_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    assert p.ProviderSettings.from_env() == p.ProviderSettings(KEY)
    with pytest.raises(p.ProviderError):
        p.ProviderSettings.from_env({})


@pytest.mark.parametrize(
    "status,code",
    [
        (301, "provider_unavailable"),
        (401, "authentication_failed"),
        (402, "credit_required"),
        (429, "rate_limited"),
        (503, "provider_unavailable"),
    ],
)
def test_no_retry_redirect_fallback_or_error_body_disclosure(status, code):
    api, requests = client(
        response(status=status, raw=KEY.encode(), headers={"Location": "https://untrusted.invalid"})
    )
    with pytest.raises(p.ProviderError) as caught:
        decide(api)
    assert caught.value.code == code and KEY not in str(caught.value)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "raw", [b'{"model":1,"model":2}', b'{"x":NaN}', b"[]", b"not-json", b"\xff", b"[" * 2000]
)
def test_malformed_duplicate_nonfinite_and_nested_json(raw):
    api, _ = client(response(raw=raw))
    with pytest.raises(p.ProviderError, match="invalid JSON"):
        decide(api)


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "text/html"},
        {"Content-Encoding": "gzip"},
        {"Content-Length": "wat"},
        {"Content-Length": str(p.MAX_RESPONSE + 1)},
    ],
)
def test_bad_headers_and_declared_sizes(headers):
    api, _ = client(response(decision(), headers=headers))
    with pytest.raises(p.ProviderError):
        decide(api)


def test_response_stream_and_total_deadline_are_bounded_without_retry():
    api, requests = client(response(raw=b"", chunks=[b" " * 65536] * 3))
    with pytest.raises(p.ProviderError) as caught:
        decide(api)
    assert caught.value.code == "response_too_large"
    assert len(requests) == 1
    api, requests = client(response(decision(), delay=0.1), timeout_seconds=0.01)
    with pytest.raises(p.ProviderError) as caught:
        decide(api)
    assert caught.value.code == "timeout" and len(requests) == 1


@pytest.mark.parametrize("error", [httpx.ReadTimeout(KEY), httpx.ConnectError(KEY)])
def test_transport_exceptions_are_redacted(error):
    api, requests = client(error)
    with pytest.raises(p.ProviderError) as caught:
        decide(api)
    assert KEY not in str(caught.value) and caught.value.__suppress_context__
    assert len(requests) == 1


def test_request_and_success_response_cannot_echo_credentials():
    api, requests = client()
    with pytest.raises(p.ProviderError):
        asyncio.run(api.decide(CONTEXT, KEY, "Q", CHOICES, current=lambda: CONTEXT))
    assert not requests
    api, _ = client(response(raw=json.dumps({"error": KEY}).encode()))
    with pytest.raises(p.ProviderError, match="contains credentials"):
        decide(api)
    api, _ = client()
    with pytest.raises(p.ProviderError):
        asyncio.run(api._post(p.JEV_URL, {"large": "a" * p.MAX_REQUEST}))


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(model="typesafe/jev-2"),
        lambda d: d.update(error="upstream error"),
        lambda d: d.update(usage={"cost": -1}),
        lambda d: d.update(usage={"cost": 10**1000}),
        lambda d: d.update(answers={}),
        lambda d: d["answers"]["selection"].update(choice="not-eligible"),
        lambda d: d["answers"]["selection"].update(choice=[]),
        lambda d: d["answers"]["selection"].update(type="noul"),
        lambda d: d["answers"]["selection"].update(confidence=True),
        lambda d: d["answers"]["selection"].update(
            probabilities={"eligible-a": 0.2, "eligible-b": 0.2}
        ),
        lambda d: d["answers"]["selection"].update(
            probabilities={"eligible-a": 1.2, "eligible-b": -0.2}
        ),
        lambda d: d["answers"]["selection"].update(extra="motion"),
    ],
)
def test_decision_response_validation(edit):
    body = decision()
    edit(body)
    api, _ = client(response(body))
    with pytest.raises(p.ProviderError):
        decide(api)


def test_decision_context_change_rejected_before_and_after_network():
    for before in (True, False):
        state = [replace(CONTEXT, revision=3) if before else CONTEXT]

        async def handler(request):
            state[0] = replace(CONTEXT, session_id="new-session")
            return response(decision())

        api, requests = client(handler)
        with pytest.raises(p.ProviderError) as caught:
            asyncio.run(api.decide(CONTEXT, "S", "Q", CHOICES, current=lambda: state[0]))
        assert caught.value.code == "stale_context"
        assert len(requests) == (0 if before else 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"session_id": "bad id"},
        {"episode_id": ""},
        {"revision": True},
        {"revision": -1},
    ],
)
def test_source_context_validation(changes):
    with pytest.raises(p.ProviderError):
        replace(CONTEXT, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 0},
        {"height": True},
        {"width": 1921},
        {"rgb": b"wrong"},
        {"rgb": bytearray(12)},
        {"received_monotonic_ns": -1},
        {"step": -1},
        {"context": "wrong"},
        {"width": 1920, "height": 1920},
    ],
)
def test_frame_input_validation(changes):
    with pytest.raises(p.ProviderError):
        replace(frame(), **changes)


def test_frame_identity_and_choice_input_guards():
    with pytest.raises(p.ProviderError):
        p.FrameIdentity(CONTEXT, 0, "not-a-hash", 2, 2)
    with pytest.raises(p.ProviderError):
        p.FrameIdentity("wrong", 0, "a" * 64, 2, 2)
    with pytest.raises(p.ProviderError):
        p.Choice("id", "")
    api, requests = client()
    for choices in [(CHOICES[0],), (CHOICES[0], CHOICES[0]), ("wrong", "object")]:
        with pytest.raises(p.ProviderError):
            asyncio.run(api.decide(CONTEXT, "S", "Q", choices, current=lambda: CONTEXT))
    assert not requests
    with pytest.raises(p.ProviderError):
        asyncio.run(api.decide("wrong", "S", "Q", CHOICES, current=lambda: CONTEXT))


@pytest.mark.parametrize("changed", ["age", "future", "session", "step", "rgb"])
def test_stale_frame_rejected_before_request(changed):
    source = frame()
    current = source.identity
    if changed == "age":
        source = replace(source, received_monotonic_ns=0)
    if changed == "future":
        source = replace(source, received_monotonic_ns=NOW + 1)
    if changed == "session":
        current = replace(current, context=replace(CONTEXT, session_id="changed"))
    if changed == "step":
        current = replace(current, step=4)
    if changed == "rgb":
        current = replace(current, rgb_sha256="a" * 64)
    api, requests = client(response(perception()))
    with pytest.raises(p.ProviderError) as caught:
        perceive(api, source, lambda: current)
    assert caught.value.code == "stale_frame" and not requests


def test_frame_change_and_age_during_response_rejected():
    for reason in ("identity", "age"):
        source = frame()
        current = [source.identity]
        now = [NOW]

        async def handler(request):
            if reason == "identity":
                current[0] = replace(source.identity, step=4)
            else:
                now[0] += 6_000_000_000
            return response(perception())

        api, requests = client(handler)
        api._clock_ns = lambda: now[0]
        with pytest.raises(p.ProviderError) as caught:
            perceive(api, source, lambda: current[0])
        assert caught.value.code == "stale_frame" and len(requests) == 1


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(model="perceptron/perceptron-mk1"),
        lambda d: d.update(choices=[]),
        lambda d: d["choices"][0].update(finish_reason="length"),
        lambda d: d["choices"][0]["message"].update(tool_calls=[{"motion": True}]),
        lambda d: d["choices"][0]["message"].update(refusal="no"),
        lambda d: d["choices"][0]["message"].update(content=None),
        lambda d: d["choices"][0]["message"].update(content="{}"),
        lambda d: d["choices"][0]["message"].update(content=""),
    ],
)
def test_bad_perception_envelope(edit):
    body = perception()
    edit(body)
    api, _ = client(response(body))
    with pytest.raises(p.ProviderError):
        perceive(api)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(uncertain="yes"),
        lambda d: d.update(summary=""),
        lambda d: d.update(observations=[{"label": "a", "box": [0, 0, 1, 1]}] * 17),
        lambda d: d.update(observations="objects"),
        lambda d: d.update(motion="command"),
        lambda d: d["observations"][0].update(box=[0, 0, 0, 1]),
        lambda d: d["observations"][0].update(box=[0, 0, 2, 1]),
        lambda d: d["observations"][0].update(box=[0, 0, True, 1]),
        lambda d: d["observations"][0].update(box=[0, 0, 1]),
        lambda d: d["observations"][0].update(label="a" * 65),
    ],
)
def test_bad_generated_geometry_and_schema_are_not_accepted(edit):
    body = json.loads(perception()["choices"][0]["message"]["content"])
    edit(body)
    api, _ = client(response(perception(body)))
    with pytest.raises(p.ProviderError):
        perceive(api)


def test_empty_observation_list_is_honest_and_allows_abstention():
    api, _ = client(
        response(
            perception({"summary": "No reliable objects.", "uncertain": True, "observations": []})
        )
    )
    assert perceive(api).observations == ()


def test_future_model_version_not_silently_substituted():
    body = copy.deepcopy(decision())
    body["model"] = p.JEV_MODEL
    api, _ = client(response(body))
    assert decide(api).receipt.returned_model == p.JEV_MODEL


def test_frame_dimensions_and_oversized_header_are_bound():
    source = frame()
    api, calls = client(response(perception()))
    with pytest.raises(p.ProviderError, match="stale"):
        perceive(api, source, lambda: replace(source.identity, width=3))
    assert not calls
    with pytest.raises(p.ProviderError):
        replace(source.identity, width=0)
    api, _ = client(response(decision(), headers={"Content-Length": "9" * 5000}))
    with pytest.raises(p.ProviderError) as caught:
        decide(api)
    assert caught.value.code == "response_too_large"


def test_cancellation_propagates_without_retry():
    calls = []

    async def scenario():
        entered = asyncio.Event()

        async def handler(request):
            calls.append(request)
            entered.set()
            await asyncio.Event().wait()

        api = p.Providers(p.ProviderSettings(KEY), transport=httpx.MockTransport(handler))
        task = asyncio.create_task(api.decide(CONTEXT, "S", "Q", CHOICES, current=lambda: CONTEXT))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(calls) == 1


def test_json_escaped_key_and_invalid_unicode_are_rejected():
    escaped = "".join("\\u%04x" % ord(char) for char in KEY)
    api, _ = client(response(raw=('{"secret":"' + escaped + '"}').encode()))
    with pytest.raises(p.ProviderError, match="contains credentials"):
        decide(api)
    api, _ = client(response(perception()))
    with pytest.raises(p.ProviderError):
        asyncio.run(api.decide(CONTEXT, "\ud800", "Q", CHOICES, current=lambda: CONTEXT))
    body = perception()
    body["choices"][0]["message"]["content"] = "\ud800"
    api, _ = client(response(body))
    with pytest.raises(p.ProviderError):
        perceive(api)


@pytest.mark.parametrize("field", ["summary", "label"])
def test_nested_escaped_perception_secret_never_returns(field):
    content = json.loads(perception()["choices"][0]["message"]["content"])
    if field == "summary":
        content["summary"] = KEY
    else:
        content["observations"][0]["label"] = KEY
    escaped = "".join("\\u%04x" % ord(char) for char in KEY)
    nested = json.dumps(content).replace(KEY, escaped)
    body = perception()
    body["choices"][0]["message"]["content"] = nested
    api, calls = client(response(body))
    with pytest.raises(p.ProviderError) as caught:
        perceive(api)
    assert caught.value.code == "invalid_response"
    assert KEY not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("variant", ["overflow", "utf16"])
def test_nonfinite_unused_envelope_or_non_utf8_json_rejected(variant):
    raw = json.dumps(decision())
    if variant == "overflow":
        body = raw[:-1].encode() + b', "unused": 1e309}'
    else:
        body = raw.encode("utf-16")
    api, _ = client(response(raw=body))
    with pytest.raises(p.ProviderError) as caught:
        decide(api)
    assert caught.value.code == "invalid_response"
