# Connect the Teaching executor

## Configure the API host

Start the [Teaching worker](../workers/teaching/README.md) and its authenticated loopback executor. Set `FIREBIRD_TEACHING_URL` and `FIREBIRD_TEACHING_CONTROL_TOKEN_FILE` on the API host, then restart the application. Use a private token file shared with the executor.

For optional voice, also set `FIREBIRD_TEACHING_VOICE_URL` and configure its LiveKit/OpenRouter services through [voice setup](teaching-intelligence.md).

## Record

1. Open **Teaching** under **Data**.
2. Wait for the configured executor's current session and camera frame.
3. Enter an instruction and select **Start recording**.
4. Pause, correct named joints or mark a failure as needed.
5. Select **Finish episode**, then [prepare the capture](recording-preparation.md).

## Atomic preview

Direct clients read `GET /api/v1/teaching/frame?session_id=...` using the selected session ID. Supply the worker's [schema 1 frame envelope](../workers/teaching/FRAME.md), including its image, joint values, timestamp and hash.

If the frame is stale or the executor disconnects, inspect the worker and reconnect to its current session. For a command timeout, read the executor's current state/command acknowledgement before sending it again.
