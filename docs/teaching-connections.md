# Connect the Teaching executor

Connect OpenJensen to a Teaching executor you run yourself (Isaac Sim with the Teaching worker), then record demonstration episodes from the browser. To let OpenJensen start the simulator for you, see [managed Teaching sessions](managed-teaching.md).

## What you need

- The [Teaching worker](../workers/teaching/README.md) running with its authenticated executor on loopback. For a remote executor, a loopback tunnel to the application host.
- A private token file shared with the executor. It must be a regular file, not a symlink, with owner-only permissions (for example `0600`) and at most 4096 bytes. It holds a token of at least 32 characters: letters, digits, `.`, `_` or `-`.

## 1. Configure the application host

Set these on the application host, then restart the server:

| Variable | Value |
| --- | --- |
| `JENSEN_TEACHING_URL` | The executor address, exactly `http://127.0.0.1:<port>`, for example `http://127.0.0.1:8768`. |
| `JENSEN_TEACHING_CONTROL_TOKEN_FILE` | Absolute path of the private token file. |
| `JENSEN_TEACHING_VOICE_URL` | Optional. The voice and advice broker, also `http://127.0.0.1:<port>`, for example `http://127.0.0.1:8769`. Set up its LiveKit and OpenRouter services with [voice and advice setup](teaching-intelligence.md). |

Setting `JENSEN_TEACHING_URL` adds **Teaching** to the sidebar under **Robot**.

## 2. Record episodes

1. Open **Teaching** and keep **Existing executor** selected under **Connection**.
2. Wait for **Executor reachable**, the executor's current session and a camera frame.
3. Enter a **Task instruction**, select **Set instruction**, then **Start recording**.
4. As needed: **Pause**, nudge a joint with **Joint correction** and **+ 0.05 rad** or **− 0.05 rad**, **Mark failure**, or **Reset scene**.
5. Select **Finish episode**, then [prepare the capture](recording-preparation.md).

## Troubleshooting

| What you see | What to do |
| --- | --- |
| The frame is stale or the executor disconnects | Inspect the worker and reconnect to its current session. |
| No acknowledgement received for a command | Read the executor's current state and the command acknowledgement before sending it again. |
| The executor stopped after a fault | Inspect the worker and its local log before continuing. |
| The connection configuration is invalid | Check that the URL is exactly `http://127.0.0.1:<port>` and that the token file meets the rules above. |

Open **Connection details** on the page to see the executor and voice service status.

## API

The application relays these requests to the executor. Browser clients use them instead of calling the executor directly.

| Request | Use |
| --- | --- |
| `GET /api/v1/teaching/state` | Current mode, session, episode and revision. |
| `GET /api/v1/teaching/frame?session_id=...` | The atomic camera preview for the selected session ID. |
| `POST /api/v1/teaching/commands` | Send `task`, `start`, `pause`, `reset`, `correct`, `mark_failure` or `finish` with the current session, episode and revision. |
| `GET /api/v1/teaching/commands/{command_id}` | Poll a command's acknowledgement instead of resending it. |

- The worker supplies each frame as its [schema 1 frame envelope](../workers/teaching/FRAME.md), with the image, joint values, timestamp and hash. Frames older than 5 seconds are rejected.
- A `correct` command moves one named joint by a nonzero amount of at most 0.15 radians.
- Instructions and failure reasons are limited to 256 characters.
