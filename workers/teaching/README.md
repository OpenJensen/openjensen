# Simulator teaching and LeRobot recording

This worker connects one LiveKit voice agent and explicit operator controls to the existing Isaac `reset/observe/apply/step` interface. It records camera pixels and **applied** joint targets locally, then invokes the pinned upstream LeRobot v3 writer. A spoken task supplies a label; it does not supply a pickup controller. No task success, physical calibration, training quality, or distillation is implied.

## Processes and ownership

| Process | Runtime | Responsibility |
|---|---|---|
| `firebird_teaching.isaac` | Existing NVIDIA Isaac6.1 Python | Main-thread simulator executor, motion guard, durable capture, authenticated loopback control8768 |
| `firebird_teaching.voice_join` | Isolated Python3.12 voice requirements | Loopback credential broker8769; creates a distinct bounded room and microphone-only token |
| `firebird_teaching.livekit_agent` | Same voice environment | One LiveKit Agents tool loop, OpenRouter LLM, LiveKit Inference speech, camera preview |
| `firebird_teaching.dataset` | Isolated Python3.12 dataset requirements | Validates finalized capture, creates/reads genuine LeRobot v3 dataset, records lineage |

Reuse LeRobot, LiveKit and the existing Isaac adapter. Do not install voice/LeRobot dependencies into Isaac or the ACT0.6.1 exporter. The two requirements files intentionally serve separate environments. On Linux explicitly constrain CPU Torch/torchvision wheel URLs and hashes from the approved ACT worker lock; a resolver backend flag alone did not select CPU-only wheels in the tested uv0.12.19 setup. Check the resolved package set before installing; CUDA packages are unnecessary for dataset conversion. Install FFmpeg for video assembly. RTC1.1.18 is the version required by Agents1.8.3; independently upgrading RTC to1.1.20 conflicts with that SDK.

Set `PYTHONPATH=workers/teaching:workers/isaac_sim` from the repository root in each process. The simulator adapter is imported without starting NVIDIA until the Isaac command explicitly runs. This module is an internal worker, not a public network service. Secure source admission currently requires POSIX directory descriptors (Linux/macOS); Windows writer acceptance is pending.

## Configure and run

Copy [settings.example.json](settings.example.json) into an operator-owned configuration directory and resolve `scene` relative to that file. The example uses the existing experimental SO101 USD and native radians. Preserve all referenced USD assets. The stored scene fingerprint covers the root USD bytes only; it does not claim an inventory of referenced assets. Repeated resets retain the same lineage group.

All control clients use a shared private file configured by `FIREBIRD_TEACHING_CONTROL_TOKEN_FILE` (at least32 nonspace ASCII characters; owner-only permissions). For environments that already supply secret variables, `FIREBIRD_TEACHING_CONTROL_TOKEN` is also accepted; do not set both. Never give this secret to the browser. Control listeners bind only127.0.0.1 and reject browser Origin headers. Remote simulator use requires an explicitly configured loopback tunnel or colocated processes; this worker does not open firewall ports.

Validate settings without starting Isaac:

```sh
python -m firebird_teaching.isaac --settings workers/teaching/settings.example.json --output /operator/capture/new-session --validate-only
```

Run in the existing Isaac runtime on the GPU host with its accepted NVIDIA license environment:

```sh
/isaac-sim/python.sh --no-ros-env -m firebird_teaching.isaac --settings /operator/teaching.json --output /operator/capture/new-session --port 8768 --max-seconds 300
```

Create the operator-owned output parent first, then use a unique output directory. `max_steps` bounds each episode, `max_episode_bytes` bounds raw capture, and `max-seconds` bounds the session loop. A blocking NVIDIA SDK call can outlast the loop deadline, so deployment also needs the existing outer container timeout and kill grace. SIGTERM requests graceful closure; a hard kill preserves raw files but leaves the episode unfinalized. Do not start a second Isaac process inside an active rollout container.

Manual commands use the same executor as voice:

```sh
export FIREBIRD_TEACHING_CONTROL_URL=http://127.0.0.1:8768
python -m firebird_teaching.command task --arguments '{"instruction":"Move shoulder_pan slightly"}'
python -m firebird_teaching.command start
python -m firebird_teaching.command correct --arguments '{"joint":"shoulder_pan","delta_rad":0.03}'
python -m firebird_teaching.command pause
python -m firebird_teaching.command finish
```

Voice runtime variables:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`: reuse the operator's existing file with `--env-file`; values are not copied into this repository.
- `OPENROUTER_API_KEY`, `FIREBIRD_VOICE_MODEL`: required explicit OpenRouter key/model; no automatic LLM fallback.
- `FIREBIRD_TEACHING_CONTROL_URL` and the private control token file: server-side executor connection.
- Default `FIREBIRD_VOICE_MODE=openrouter` uses LiveKit Cloud Inference for STT/TTS with the existing LiveKit credentials. Entitlement/connectivity are runtime checks. Defaults are `deepgram/flux-general` and `fishaudio/s2.1-pro` with documented voice `fa4c9eb3dccc4806b382b40d61c6b10a`; override `FIREBIRD_STT_MODEL`, `FIREBIRD_TTS_MODEL`, `FIREBIRD_TTS_VOICE` explicitly.
- Self-hosted speech uses `FIREBIRD_VOICE_MODE=openrouter-custom-speech`, plus `FIREBIRD_STT_BASE_URL`, `FIREBIRD_STT_API_KEY`, `FIREBIRD_TTS_BASE_URL`, `FIREBIRD_TTS_API_KEY` and explicit speech model/voice settings. Compatible local endpoints may use loopback HTTP. Hosted inference is optional, not an offline-local claim.

```sh
python -m firebird_teaching.livekit_agent --env-file /operator/livekit.env --check-config
python -m firebird_teaching.livekit_agent --env-file /operator/livekit.env dev
python -m firebird_teaching.voice_join --env-file /operator/livekit.env --port 8769
```

The broker creates `firebird-teaching-<uuid>` only, at most two participants, an empty-room timeout60s, and a five-minute microphone-only operator token. Dispatch binds the voice agent to that operator identity and executor session. Joining does not claim microphone, inference, or simulator connection success. One broker leases one room; restart voice services after lease expiry or executor restart. Participant disconnect triggers best-effort pause; manual pause and the executor's bounds remain independent. Voice controls are task/start/pause/reset/named-joint correction/failure annotation/finish, never generated code.

Existing LivekitTeleOP/Unity rooms are not used. The inspected Unity prototype consumes arbitrary data packets as joint floats without checking topic/sender. This worker publishes camera/audio through standard LiveKit tracks and does not send robot data into those rooms. Jev is a separate typed decision API, not a chat model; Muose is not an execution gate. Neither is silently substituted for the configured LLM.

## HTTP contract

Every request requires `Authorization: Bearer <control secret>`. No browser receives this secret; the application proxies to fixed operator-configured loopback origins.

- Executor `GET /state`: mode (`starting/idle/running/paused/faulted/closed`), `episode_id`, `revision`, `instruction`, `outcome`, `steps`, `sim_time`, native `state_rad`, `joints`, `fault`, `session_id`, `last_applied_command_id`, `last_applied_step`. Starting state has only mode/episode/revision until the simulator is ready.
- Executor `POST /commands`: exact `{command_id,session_id,episode_id,expected_revision,operation,arguments}`. Returns202 queued receipt; poll `GET /commands/<command_id>` for `queued/executing/acknowledged/rejected`. A correction stays executing until a physics step, following observation and durable trajectory row exist. Rejected/stale/expired commands do not become success. Client timeout may return pending; poll the receipt instead of blindly retrying.
- Commands: `task:{instruction}`, `start:{}`, `pause:{}`, `reset:{}`, `correct:{joint,delta_rad}`, `mark_failure:{reason}`, `finish:{}`. Delta is finite, nonzero and at most0.15 radians. The existing articulation limit/2rad/s guard constrains the applied target. Pause/reset/finish receive queue priority; version checks invalidate older commands. Pause stops simulation ticks and does not claim a certified physical stop.
- `GET /frame`: RGB base64 plus width/height, episode/step/sim_time and `published_monotonic_ns`, or `{available:false}`. This is preview transport; dataset pixels come from the local pre-action observation. Published time is not capture/speech latency.
- Voice broker `POST /voice/join`: exact `{session_id}`. Returns200 `{status:"room_created",url,token,room,operator_identity,session_id,expires_in_seconds,agent_name,microphone_or_executor_connection_verified:false}`. Operator identity is server-generated.401 denotes unauthorized access;400 denotes unavailable/mismatched session/configuration. Tokens must not be logged.

## Finalize dataset

Explicitly select finalized capture episode IDs:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m firebird_teaching.dataset /operator/capture/new-session /operator/datasets/new-snapshot --episode EPISODE_ID --repo-id local/teaching
```

The writer checks native LeRobot0.6.2 and critical source-file hashes from `e595b7902714ba51f91e47523f66f89c5181b649`, validates source frame/event hashes and dimensions, calls upstream create/add_frame/save_episode/finalize, decodes all frames with the upstream reader and verifies numeric action/state equality. Missing local files cannot trigger a Hub fallback. Output publishes only after verification; the raw source remains unchanged. Incomplete, torn, altered, symlinked or unsupported-success captures are rejected. Admission is bounded across all sources to100 selected episodes,50000 inventory entries,8GiB input inventory and3600frames/episode; conversion uses an operator-owned output parent without concurrent writers.

To combine compatible captures, add `--additional-captures /operator/captures.json` to the command. The primary capture still requires explicit `--episode` selections. The additional manifest (maximum64KiB) has this exact shape; relative paths are relative to the manifest:

```json
{"schema_version":1,"captures":[{"path":"../capture/second-session","episodes":["SECOND_EPISODE_ID"]}]}
```

The Python equivalent is `convert_captures([CaptureSelection(first, (episode_a,)), CaptureSelection(second, (episode_b,))], output)`. All captures must share joint order, camera key/prim/dimensions, rates, controller, native units and timebase. Capture roots cannot overlap; session and episode identities must be unique. Sources are inventoried independently and rechecked before publication, while aggregate bounds apply to the whole selection. Per-source scene/session identity, origin and hashes are retained in `firebird-demonstrations.json.sources`, and each output episode references its source session.

Capture combination never invents independent groups. Repeated sessions from the same scene lineage retain the same group, so combining them alone still cannot satisfy the platform's two-group training/validation admission. Use recordings with genuinely distinct declared scene lineage; synthetic test scenes stay labeled synthetic, and group declarations alone do not prove physical independence or task success.

`action` contains applied native joint targets, while `teaching.requested_action`, `teaching.next_state`, simulation timestamps and intervention flags retain control provenance. H.264 output is lossy; original raw RGB remains in capture storage. `meta/firebird-lineage.json` covers every episode and conservatively groups scene resets. `meta/firebird-demonstrations.json` records units, controller, joint order, events, unknown/operator-reported-failure outcomes, source hashes and the writer pin. Recordings alone are behavioral-cloning data, not teacher-policy distillation. These native simulator coordinates require the same controller convention in training/replay; they are not automatically compatible with arbitrary existing SO101 normalized datasets.

## Verification and current limits

Run the tests in the isolated dataset/voice environment with `pytest` installed:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest workers/teaching/tests -q
```

Tests use a clearly marked generated CPU simulator fixture, genuine upstream LeRobot MP4/Parquet writer/readback, real loopback HTTP, and actual LiveKit tool/token classes with network service calls substituted. They test clipping, applied acknowledgement, pause/reset, stale/expired/duplicate commands, interruption evidence, malformed captures, session-bound room dispatch and token scope. They do not prove Isaac motion, speech recognition, OpenRouter tool selection, pickup success, or training quality. Actual bounded GPU/room/microphone execution and measured speech-to-applied latency remain separate acceptance gates.

Primary references: [LiveKit voice tools](https://docs.livekit.io/agents/logic/tools/), [OpenRouter plugin](https://docs.livekit.io/agents/models/llm/openrouter/), [LiveKit Inference](https://docs.livekit.io/agents/models/inference/), [robotics](https://docs.livekit.io/robotics/), [transport](https://docs.livekit.io/transport/), [frontends](https://docs.livekit.io/frontends/), [self hosting](https://docs.livekit.io/transport/self-hosting/), [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), [Jev decision API](https://openrouter.ai/docs/guides/community/jev), [Muose model card](https://huggingface.co/muose/Muose-50M-Decision).
