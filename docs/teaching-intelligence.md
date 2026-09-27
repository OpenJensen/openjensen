# Optional Teaching intelligence

Teaching has three separate optional connections: its authenticated simulator executor, an OpenRouter advisory broker, and a LiveKit voice room. Manual controls and the atomic camera preview do not require voice or OpenRouter. Nothing on this page automatically launches a simulator, cloud job, microphone, room or inference request.

## Design and ownership

The browser talks only to `/api/v1/teaching`. Its optional intelligence panel saves settings and sends explicitly consented, session-bound requests. The application relays to the fixed operator-configured loopback broker, authenticated with the existing private control token. The broker reads the executor's fixed `/state` and `/frame` endpoints and invokes the existing isolated HTTPX provider adapters. It never calls the executor's command endpoint for an advisory request. Provider keys stay server-side; no lifecycle scheduler or cloud configuration participates.

Jev (`typesafe/jev-1.13`) selects among server-owned **review instruction**, **inspect camera**, and, when a recording exists, **review recording** suggestions. None authorizes motion, recipe execution or job submission. This is teaching assistance, not optimizer recipe ranking. Mk1.5 (`perceptron/perceptron-mk1.5`) receives one validated image and returns a generated description. Neither output proves calibration or task success.

## Explicit private configuration

Use an isolated Python3.12 voice environment with the pinned [voice requirements](../workers/teaching/requirements-voice.txt). The existing [Teaching setup](../workers/teaching/README.md) establishes the executor, private token and separate LiveKit teaching broker. No dependencies are added to Isaac.

The page's **Configure optional services** form saves only `openrouter_api_key` and an explicit `provider/model` voice chat identifier in `<application data directory>/teaching-intelligence.json`. It creates a private file, replaces it atomically, and never returns its secret. Removing settings deletes only that managed file. The application does not edit existing `.env`, LiveKit, Google Cloud or operator files. File encryption and a multi-user secret store are not provided; retain the local application's existing host/origin boundary and protect its data directory.

Opt the separate broker and voice agent into this exact file using `FIREBIRD_TEACHING_INTELLIGENCE_FILE` with a real absolute path. Existing `OPENROUTER_API_KEY` and `FIREBIRD_VOICE_MODEL` environment values take precedence. Missing environment values may come from that explicitly configured file; nothing searches for other secret files. Restart the voice agent after changing its model. The broker reads the selected configuration at each advisory request. Saving a file does not prove that a different process is using it; the UI compares its loaded revision when available.

Continue supplying LiveKit URL/key/secret and executor settings using the existing operator setup. The managed file has no LiveKit or Google Cloud fields. Jev is not a chat model and is rejected as the voice model. Default voice speech still requires LiveKit Cloud; the existing explicit custom-speech mode remains available.

## Readiness and deliberate requests

- Saved settings mean **saved locally**, never authenticated or funded.
- Broker readiness means the configured loopback process can load its key; no provider request is made by polling.
- Voice readiness checks local configuration and installed pinned dependencies. It does not create a room or imply an agent is running.
- **Connect voice** explicitly obtains the separate teaching room token and enables the microphone. The page distinguishes microphone connection from an actual `ParticipantKind.AGENT` present in that room. Presence is not successful inference or execution.
- Advice needs the current executor session/revision and explicit consent to transmit text/context; camera requests also transmit an image. Each click creates one request ID. There is no automatic inference retry, model fallback or automatic follow-up.

## Freshness, cancellation and limits

Both pre- and post-inference executor context must match the requested session, episode and revision. Perception additionally requires the exact same atomically acquired frame before and after inference. Its conservative age includes executor-reported source age, local request elapsed time and subsequent local residence; the five-second provider frame allowance is unchanged. A moving scene can invalidate a result. Pausing does not acquire a fresh frame. Such results are withheld instead of being relabelled current. Displayed accepted results are captured suggestions; later context changes mark them historical. They are never live camera observations or control authorization.

One advisory request can run in the broker at a time. Its whole deadline is12seconds, with the provider adapter's existing8-second network deadline. The application relay has a16-second deadline; the browser stops waiting at20seconds. Payloads, response bodies and prompts are bounded. Paths, URLs, files, pixels and eligible choice sets are not accepted from the browser. The private configuration reader rejects symlinks, nonregular files, excessive size and unsafe POSIX permissions. The broker uses fixed authenticated routes, refuses browser Origin headers and bounds socket/body reads.

Explicit cancellation, page departure, context change or disconnection aborts the owned request and requests cancellation from the broker. Repeated cancellation is shielded while owned cleanup completes. Cancellation and timeouts can occur after a provider has billed a request; billing and remote completion remain unverified. The worker's recent256request IDs prevent duplicate attempts only within that broker process. This is not durable cross-restart idempotency. Review an ambiguous outcome before explicitly making another request.

## Validation scope

Local tests use generated responses with the real provider request/response adapters and authenticated loopback transport. They exercise stale context/frame rejection, consent, request identity, cancellation, bounds, missing configuration, private settings and secret redaction. Browser tests cover the production page on desktop/mobile and never enable a real microphone or call a provider. Existing voice/executor tests remain regression gates. These tests do not establish live Jev/Mk1.5 availability, billing access, voice-agent speech, simulator calibration or policy quality. Those remain explicit operator acceptance checks with configured credentials and a chosen sample.

Official contracts checked2026-09-27: [Jev](https://openrouter.ai/docs/guides/community/jev), [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request), [Mk1.5](https://openrouter.ai/perceptron/perceptron-mk1.5), [OpenRouter authentication](https://openrouter.ai/docs/api/reference/authentication), [LiveKit OpenAI-compatible models](https://docs.livekit.io/agents/models/llm/openai/).

## Source-integration gate

This Teaching source slice intentionally leaves the shared generated API files to the integration owner, because native policy/distillation contracts are being developed separately. Before accepting the combined application tree, regenerate `packages/core/openapi.json` with `scripts/export_openapi.py`, regenerate `apps/web/src/lib/api.generated.ts` with the pinned `generate:client` command, then run `test_checked_in_openapi_matches_runtime_contract`, web typecheck and the relevant browser suites. The standalone authored tree's checked-in-schema equality test remains stale until that step; its passing Teaching API/worker/browser tests do not replace this integration check.
