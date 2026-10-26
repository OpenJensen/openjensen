# Set up AI teleoperation voice and advice

These optional services add two things to the **AI teleoperation** page: a voice agent you talk to through LiveKit, and advice from OpenRouter models about the session or the camera image. Both send data to paid providers. They work with an [existing executor](teaching-connections.md), not with managed sessions.

## What you need

- A running executor and its private token, set up as in [Connect the simulation executor](teaching-connections.md).
- An isolated Python 3.12 environment with the pinned [voice requirements](../workers/teaching/requirements-voice.txt).
- The broker and voice agent, started as described in [Teaching worker setup](../workers/teaching/README.md#start-voice-services). Point `JENSEN_TEACHING_VOICE_URL` on the application host at the broker. Advice also goes through the broker.
- An OpenRouter API key, plus a LiveKit URL, key and secret.
- Speech services: LiveKit Cloud speech, or the worker's custom speech endpoints.

## 1. Save the OpenRouter settings

1. On **AI teleoperation**, open **Configure optional services** under **Advice**.
2. Enter the **OpenRouter API key** (16–4096 printable ASCII characters).
3. Enter an explicit **Voice chat model** as `provider/model`. Jev models (`typesafe/...`) are rejected here.
4. Select **Save private settings**. The app writes `openrouter_api_key` and `voice_model` to `<workspace>/teaching-intelligence.json`, with owner-only permissions. **Remove saved settings** deletes it.

Saving does not check provider access.

## 2. Load the settings in the broker and voice agent

- Set `JENSEN_TEACHING_INTELLIGENCE_FILE` to the file's absolute path in the environment of both the broker and the voice agent.
- Run both as the same user as the server, and use a real path without symlinks. They refuse the file if other users can access it.
- To use environment configuration instead, set `OPENROUTER_API_KEY` and `JENSEN_VOICE_MODEL` directly in those processes. Environment values take precedence over the file.
- Supply the LiveKit URL, key and secret and the speech services through the [worker setup](../workers/teaching/README.md#start-voice-services).

The broker rereads the settings on each request. Restart the voice agent after changing its model. Until the broker has read the latest save, the page shows "The broker has not loaded these settings yet."

## 3. Talk to the voice agent

1. Start the configured executor, open **AI teleoperation** and wait for its current session.
2. Select **Connect voice** and allow microphone access. Your microphone streams through LiveKit and OpenRouter for up to 5 minutes, and provider charges apply.
3. Check that the status reads "Voice agent present in room".
4. Select **Disconnect microphone** to leave the room.

## 4. Ask for advice

1. Enter a **Question** of up to 512 characters.
2. Tick the consent box. It covers sending the question and session context to OpenRouter, plus one frame for camera questions. Provider charges apply.
3. Select one:
   - **Suggest what to review** uses Jev `typesafe/jev-1.13` through the OpenRouter Decisions API.
   - **Ask about the camera image** sends the current frame to Mk1.5 `perceptron/perceptron-mk1.5`.
4. To stop a pending request, select **Cancel intelligence request**. Billing for a cancelled request is not verified.

The app fixes these two advice models; you only choose the voice chat model. Only one advice request runs at a time, and failed requests are never retried automatically. If the session, episode or revision changes, the app cancels the pending request and withholds its answer.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| A missing-key or missing-model error | Check which configuration file the broker reads, then restart the affected process. |
| The camera frame changed or expired | Wait for a new executor frame and make a new request. |
| OpenRouter rejected its key, requires credit, or rate limited the request | Fix the key or account, then ask again. Nothing is retried for you. |
| **Connect voice** stays disabled | The broker is unreachable or its configuration or dependencies are missing. Check **Connection details**. |
| Voice disconnects after 5 minutes, or the executor restarted | The room lease ended. Restart the voice services as the [worker setup](../workers/teaching/README.md#start-voice-services) describes, then select **Connect voice** again. |
| "Voice connection closed. Reconnect to continue." | The connection dropped. Select **Connect voice** again. |

## Provider references

[Jev](https://openrouter.ai/docs/guides/community/jev), [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request), [Mk1.5](https://openrouter.ai/perceptron/perceptron-mk1.5), [OpenRouter authentication](https://openrouter.ai/docs/api/reference/authentication), [LiveKit models](https://docs.livekit.io/agents/models/llm/openai/). For direct provider calls, see [Teaching providers](../workers/teaching/PROVIDERS.md).
