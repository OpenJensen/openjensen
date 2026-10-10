# Invoke optional Teaching providers

## Install and configure

In an isolated Python **3.12** environment install `requirements-providers.txt`
(HTTPX **0.28.1**), or reuse the voice environment that supplies it. Set
server-side `OPENROUTER_API_KEY` through the operator's existing secret loader.
For the web flow follow [Teaching intelligence setup](../../docs/teaching-intelligence.md).

## Call a provider

| Adapter | Fixed request | Required input |
| --- | --- | --- |
| `Providers.decide` | OpenRouter Decisions API, `typesafe/jev-1.13` | One question and 2–16 unique caller-admitted choices. |
| `Providers.perceive` | OpenRouter chat completions, `perceptron/perceptron-mk1.5` | One admitted RGB frame, encoded as inline PNG, and the typed description schema. |

Supply a current immutable session context and callback from trusted server code:

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

Maintain `current_session_snapshot` from the active session; recheck the chosen
ID against the caller's eligible choices when consuming the returned proposal.
Use single-line state/question strings: up to 16,384 and 512 characters.

## Supply a frame

Call `frame_snapshot.admit_frame(...)` using
[the frame schema and local timestamps](FRAME.md), then `.provider_input()`.
Pass `current=` returning the current `FrameIdentity`; update it from newly
admitted atomic observations and invalidate it on stale/unavailable sources.
Recheck it before displaying the provider response.

Use at most 1920 pixels per dimension and 1,048,576 total pixels. Make the
OpenRouter image transmission explicit to the operator. Keep the request within
5 MiB and response within 128 KiB. Use the 8-second default network deadline
(allowed 0.01–30 seconds) and 5-second frame-age allowance (0.01–30 seconds).

Catch `ProviderError` and handle its stable code:
`configuration_missing`, `invalid_input`, `invalid_response`, `stale_context`,
`stale_frame`, `response_too_large`, `timeout`, `authentication_failed`,
`credit_required`, `rate_limited` or `provider_unavailable`. Propagate cancellation;
review an ambiguous/billed request before explicitly making a new call. Store
returned session/request/model identities and any reported cost with the request.

## Local checks

Use the existing teaching test environment with pytest, Pillow, pytest-cov,
mypy and Ruff:

```sh
PYTHONPATH=workers/teaching:workers/isaac_sim python -m pytest workers/teaching/tests/test_providers.py -q --cov=firebird_teaching.providers --cov-report=term-missing
python -m mypy --strict --follow-imports=skip workers/teaching/firebird_teaching/providers.py
python -m ruff check workers/teaching/firebird_teaching/providers.py workers/teaching/tests/test_providers.py
python -m ruff format --check workers/teaching/firebird_teaching/providers.py workers/teaching/tests/test_providers.py
```

## API references

- [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev)
- [Decisions request/response schema](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
- [Mk1.5 model listing](https://openrouter.ai/perceptron/perceptron-mk1.5)
- [Inline image inputs](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)
- [Structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
