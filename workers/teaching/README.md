# Configure and run simulator teaching

## Prepare runtimes

Use separate runtimes for the following processes:

| Process | Required runtime |
| --- | --- |
| `firebird_teaching.isaac` | Existing Isaac **6.1** Python and simulator dependencies. |
| `firebird_teaching.voice_join`, `firebird_teaching.livekit_agent` | Python **3.12** with `requirements-voice.txt`. |
| `firebird_teaching.dataset` | Python **3.12** with `requirements-dataset.txt`, CPU Torch/torchvision and FFmpeg. |

Use the pinned LiveKit Agents **1.8.3** / RTC **1.1.18** pair. For Linux dataset
installation apply the pinned CPU wheel URLs/hashes from the ACT worker lock;
check the resolved packages before installing. The writer requires pinned
LeRobot **0.6.2** at `e595b7902714ba51f91e47523f66f89c5181b649`.
For prepared reader environments follow [local CPU setup](../local_cpu/README.md).

From the repository root set
`PYTHONPATH=workers/teaching:workers/isaac_sim` for each process. Use Linux/macOS
for dataset conversion's directory-descriptor checks.

## Configure and run

Copy [settings.example.json](settings.example.json) to an operator-owned
configuration directory. Resolve `scene` relative to that copied file and keep
all referenced USD assets together. Set joint order, native radians, camera,
FPS, `max_steps` and `max_episode_bytes` for the recording.

Set `FIREBIRD_TEACHING_CONTROL_TOKEN_FILE` to a private owner-only file containing
at least 32 nonspace ASCII characters. Alternatively set
`FIREBIRD_TEACHING_CONTROL_TOKEN`; choose exactly one token source. Keep the
executor/broker on loopback; use a configured tunnel for a remote executor.
Create the capture output parent, then select a new session directory.

Validate settings in the worker environment:

```sh
python -m firebird_teaching.isaac --settings workers/teaching/settings.example.json --output /operator/capture/new-session --validate-only
```

Run on the GPU host in its Isaac runtime after accepting NVIDIA's license:

```sh
/isaac-sim/python.sh --no-ros-env -m firebird_teaching.isaac --settings /operator/teaching.json --output /operator/capture/new-session --port 8768 --max-seconds 300
```

Keep an outer container timeout and kill grace around the session process.
Send SIGTERM for graceful closure and preserve unfinished raw episodes after an
interruption. Use one Isaac process for the selected scene.

## Issue manual commands

```sh
export FIREBIRD_TEACHING_CONTROL_URL=http://127.0.0.1:8768
python -m firebird_teaching.command task --arguments '{"instruction":"Move shoulder_pan slightly"}'
python -m firebird_teaching.command start
python -m firebird_teaching.command correct --arguments '{"joint":"shoulder_pan","delta_rad":0.03}'
python -m firebird_teaching.command pause
python -m firebird_teaching.command finish
```

Poll each command receipt rather than resubmitting a pending command. Use
`task`, `start`, `pause`, `reset`, `correct`, `mark_failure` and `finish` with the
current session/episode/revision. For `correct`, choose a named joint and finite
nonzero `delta_rad` of at most 0.15 radians.

## Start voice services

Supply these values through the operator's existing `--env-file`:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.
- `OPENROUTER_API_KEY` and explicit `FIREBIRD_VOICE_MODEL`.
- `FIREBIRD_TEACHING_CONTROL_URL` and the private control token source.
- Default `FIREBIRD_VOICE_MODE=openrouter` uses LiveKit Cloud STT/TTS:
  `deepgram/flux-general`, `fishaudio/s2.1-pro`, voice
  `fa4c9eb3dccc4806b382b40d61c6b10a`. Override `FIREBIRD_STT_MODEL`,
  `FIREBIRD_TTS_MODEL`, `FIREBIRD_TTS_VOICE` as needed.
- For custom speech set `FIREBIRD_VOICE_MODE=openrouter-custom-speech` with
  `FIREBIRD_STT_BASE_URL`, `FIREBIRD_STT_API_KEY`, `FIREBIRD_TTS_BASE_URL`,
  `FIREBIRD_TTS_API_KEY` and explicit speech model/voice settings.

```sh
python -m firebird_teaching.livekit_agent --env-file /operator/livekit.env --check-config
python -m firebird_teaching.livekit_agent --env-file /operator/livekit.env dev
python -m firebird_teaching.voice_join --env-file /operator/livekit.env --port 8769
```

Start the agent and broker in separate shells with the same executor session.
Use the Teaching page's explicit voice connection control. Restart voice services
after room-lease expiry or executor restart; use manual pause/finish as needed.

## HTTP requests

Authenticate each request with `Authorization: Bearer <control secret>` from
trusted server code. Use the application relay for browser clients.

| Request | Input / next action |
| --- | --- |
| Executor `GET /state` | Read current mode, session, episode and revision before issuing a command. |
| Executor `POST /commands` | Send exact `{command_id, session_id, episode_id, expected_revision, operation, arguments}`. Poll `GET /commands/<command_id>` after the queued receipt. |
| Executor `GET /frame` | Read and admit [the atomic frame envelope](FRAME.md). |
| Broker `POST /voice/join` | Send exact `{session_id}`; use the returned room URL/token before expiry. Keep tokens out of logs. |

Command arguments are `task:{instruction}`, `start:{}`, `pause:{}`, `reset:{}`,
`correct:{joint,delta_rad}`, `mark_failure:{reason}` or `finish:{}`.

## Finalize dataset

Finish selected episodes, then run in the isolated dataset environment:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m firebird_teaching.dataset /operator/capture/new-session /operator/datasets/new-snapshot --episode EPISODE_ID --repo-id local/teaching
```

Use explicit episode IDs and a new output directory. Fit the input selection
within 100 episodes, 50,000 inventory entries, 8 GiB total and 3,600 frames per
episode. Preserve raw captures while conversion validates/readbacks the output.

To combine compatible captures, add
`--additional-captures /operator/captures.json`. Supply this manifest (up to
64 KiB); resolve relative capture paths from it:

```json
{"schema_version":1,"captures":[{"path":"../capture/second-session","episodes":["SECOND_EPISODE_ID"]}]}
```

Choose matching joint order, camera key/prim/dimensions, rates, controller, units
and timebase. Keep roots nonoverlapping and session/episode IDs unique. For a
held-out training split supply distinct declared scene-lineage groups.
Inspect output Parquet/video plus `meta/firebird-lineage.json` and
`meta/firebird-demonstrations.json` before selecting the dataset for training.

## Optional services and application preparation

Configure [provider calls](PROVIDERS.md) or follow
[Teaching intelligence setup](../../docs/teaching-intelligence.md) for the broker
and private settings file. Use [recording preparation](../../docs/recording-preparation.md)
to register finalized host-local captures as a project dataset.

## Local checks

Run in the isolated dataset/voice environment with pytest installed:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest workers/teaching/tests -q
```

References: [LiveKit voice tools](https://docs.livekit.io/agents/logic/tools/),
[OpenRouter plugin](https://docs.livekit.io/agents/models/llm/openrouter/),
[LiveKit Inference](https://docs.livekit.io/agents/models/inference/),
[LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).
