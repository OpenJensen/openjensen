# Optional decision and frame-perception proposals

`firebird_teaching.providers` contains real outbound OpenRouter adapters, separately callable by trusted server code. The optional Teaching intelligence panel now reaches them through the authenticated loopback broker; they remain separate from voice-agent tools and the optimizer. See the [application integration](../../docs/teaching-intelligence.md). Importing the module does not make a request. Nothing here starts recording, changes controller targets, ranks unadmitted recipes or marks task success. The rules-only workflow continues without these optional providers.

Install `requirements-providers.txt` in an isolated Python3.12 environment. It pins HTTPX0.28.1, the existing application HTTP version. The voice environment already supplies that dependency; do not install it into NVIDIA's Isaac runtime. Set server-side `OPENROUTER_API_KEY` explicitly. `ProviderSettings.from_env()` reads that variable only; it does not discover dotenv files or copy credentials into results. Missing configuration raises `ProviderError` with `configuration_missing`. Use the operator's existing secret-loading mechanism and preserve the current room/cloud configuration.

## Supported contract

| Adapter | Fixed request | Typed output |
|---|---|---|
| `Providers.decide` | `POST https://openrouter.ai/api/alpha/decisions`, `typesafe/jev-1.13`, one Choice question | A caller-admitted choice ID, probabilities for exactly the supplied IDs and provider confidence |
| `Providers.perceive` | `POST https://openrouter.ai/api/v1/chat/completions`, `perceptron/perceptron-mk1.5`, one inline PNG and strict JSON schema | Description, uncertainty flag and at most16 labelled normalized image boxes |

Jev uses its own Decisions API, not chat completions. Its documented dated response model suffix is retained in the receipt; another version/family is rejected. Mk1.5 must return the exact requested model. Neither adapter substitutes a provider/model or automatically retries. Both requests disable provider fallback. Perception requires parameter support and rejects truncation, refusal, tool calls, malformed/duplicate/nonfinite JSON and invalid box geometry. A generated image box has no calibrated depth, robot coordinate transform or grasp target. Confidence/probabilities are provider outputs, not calibrated safety guarantees.

Each receipt records the original session/episode/revision, exact request fingerprint, requested/returned model, optional provider response ID/reported cost, and local elapsed time. Frame receipts also bind step, dimensions and raw RGB hash. No raw prompt, pixels, key or upstream error body is written to logs by this module. Cost is reported only when provided; absent cost is unknown, not zero. A timeout/cancellation can occur after an upstream request was billed; the client never retries it automatically.

A minimal trusted-server call, with caller-owned eligibility and current context:

```python
from firebird_teaching.providers import Choice, Providers, ProviderSettings, SourceContext

provider = Providers(ProviderSettings.from_env())
context = SourceContext(session_id="actual-session-id", episode_id=None, revision=2)
# Populate from admitted candidates and an actual immutable session snapshot.
proposal = await provider.decide(
    context,
    state="The user's target and measured evidence, with no credentials.",
    question="Which already eligible configuration best matches the stated preference?",
    choices=(Choice("candidate-a", "Measured eligible configuration A"),
             Choice("candidate-b", "Measured eligible configuration B")),
    current=lambda: current_session_snapshot,
)
```

`current_session_snapshot` must be maintained by the caller; it is deliberately not defined as a constant example that bypasses restart checks. The selected ID must still pass the optimizer's authoritative evidence/hardware gates at use time. This adapter contribution does not complete DEC-001's integrated recipe-ranking acceptance.

## Frame provenance and integration gate

Create `FrameInput` only from a single atomically session-bound observation. For the teaching transport, call `frame_snapshot.admit_frame(...)` with the expected session and local request-start/receipt timestamps, then `.provider_input()`. This preserves validated context, exact RGB fingerprint, step, dimensions, source age plus full request elapsed time, and receipt time from the consuming process's clock. Pass `current=` returning the current `FrameIdentity`. The adapter checks identity and age before sending and again before returning. Callbacks must be nonblocking and backed by current authoritative snapshots. Any later consumer must recheck identity at use time; an inference result is never authorization to act.

The [worker v1 `/frame` envelope](FRAME.md) commits image, measured joints and acquisition context together. Its strict admission helper rejects a frozen/current context mismatch, invalid hash or stale source; it must not be reconstructed from separate `/state` and `/frame` reads. The adapters remain separately callable. The explicit UI broker re-reads authoritative context and the atomic frame after inference before exposing a result; it never uses the advice to actuate. A caller must maintain its `current=` identity from newly admitted atomic observations and invalidate it on unavailable/stale/faulted sources. Do not compare a remote simulator's monotonic timestamp to this process's clock. Repeated polling cannot refresh the age of unchanged stale pixels. The timestamp records executor receipt after observation validation, not hardware exposure time. Provider freshness includes preserved source/request age plus subsequent local residence time.

Perception accepts at most1920pixels per dimension and1,048,576pixels total. Raw RGB is encoded losslessly to PNG without public storage or external image URLs. Invoking `perceive` explicitly sends those pixels to OpenRouter and its configured provider; this is remote inference, not private local model execution. The calling product must make that transmission intentional. No images are sent during offline tests.

## Bounds and failure behavior

- One request; total network deadline8s by default, configurable only within0.01–30s. Frame conservative-age allowance5s by default, bounded to0.01–30s.
- Request at most5MiB; response at most128KiB, checked both from declared size and streamed bytes. Identity encoding is required; compressed responses are rejected before decoding.
- Two to16 distinct pre-admitted choices, state at most16384characters and question at most512characters. Input is single-line text; invalid control characters are rejected explicitly.
- HTTPS endpoint fixed in source; no redirect following or environment proxies. TLS certificate verification remains enabled. Exceptions disclose stable errors, not upstream details or configured secret values.
- `ProviderError.code`: `configuration_missing`, `invalid_input`, `invalid_response`, `stale_context`, `stale_frame`, `response_too_large`, `timeout`, `authentication_failed`, `credit_required`, `rate_limited`, `provider_unavailable`. Cancellation propagates to the caller.

## Verification and remaining acceptance

Run offline tests using the existing teaching test environment (pytest, Pillow for independent PNG decoding, pytest-cov, mypy and Ruff):

```sh
PYTHONPATH=workers/teaching:workers/isaac_sim python -m pytest workers/teaching/tests/test_providers.py -q --cov=firebird_teaching.providers --cov-report=term-missing
python -m mypy --strict --follow-imports=skip workers/teaching/firebird_teaching/providers.py
python -m ruff check workers/teaching/firebird_teaching/providers.py workers/teaching/tests/test_providers.py
python -m ruff format --check workers/teaching/firebird_teaching/providers.py workers/teaching/tests/test_providers.py
```

All responses in those tests are generated contract fixtures. They check the documented request shape, actual PNG decoding, context/frame changes, cancellation/deadlines, body bounds, provider errors and secret redaction. They do not establish provider availability, inference quality, billing access, perception latency, camera calibration or robot success. Live provider verification remains pending configured credentials and an explicitly selected sample. Neither task should be marked done solely from these tests.

## Current primary references

API contracts checked2026-09-27:

- [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev)
- [Decisions request/response schema](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
- [Mk1.5 official model listing](https://openrouter.ai/perceptron/perceptron-mk1.5)
- [Inline image inputs](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)
- [Structured outputs and endpoint support](https://openrouter.ai/docs/guides/features/structured-outputs)
- [Provider routing/fallback control](https://openrouter.ai/docs/guides/routing/provider-selection)
