# Configure Teaching voice and advice

## Explicit private configuration

Install the pinned [voice requirements](../workers/teaching/requirements-voice.txt) in an isolated Python 3.12 environment. Follow [Teaching worker setup](../workers/teaching/README.md) to start the executor, private token and broker.

Open **Teaching → Configure optional services** and save `openrouter_api_key` plus an explicit `provider/model` voice identifier. The application writes `<application data directory>/teaching-intelligence.json`.

Set `FIREBIRD_TEACHING_INTELLIGENCE_FILE` to that file's absolute path for the broker and voice agent. Set `OPENROUTER_API_KEY` and `FIREBIRD_VOICE_MODEL` directly in those processes if using environment configuration instead. Restart the voice agent after changing its model.

Supply LiveKit URL/key/secret and speech services through the worker setup. Use LiveKit Cloud speech or configure the worker's custom speech endpoints.

## Connect and request

1. Open Teaching after starting the configured executor and wait for its current session.
2. Select **Connect voice** and allow microphone access.
3. Confirm the voice agent appears in the room.
4. Enter a question, review transmission consent, and select **Ask Jev for a review suggestion**.
5. For a camera description, select **Ask Mk1.5 about the camera** with the current frame.

Use Jev `typesafe/jev-1.13` for advisory requests and Mk1.5 `perceptron/perceptron-mk1.5` for image descriptions. Choose a chat model for the voice agent.

## Stop or troubleshoot

Disconnect voice to leave the room. Cancel the selected advisory request before changing executor context. For missing-key/model errors, check the broker's selected configuration file and restart the affected process. For stale-frame errors, wait for a new executor frame and make a new explicit request.

Provider references: [Jev](https://openrouter.ai/docs/guides/community/jev), [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request), [Mk1.5](https://openrouter.ai/perceptron/perceptron-mk1.5), [OpenRouter authentication](https://openrouter.ai/docs/api/reference/authentication), [LiveKit models](https://docs.livekit.io/agents/models/llm/openai/).
