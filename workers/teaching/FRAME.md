# Atomic teaching observation envelope

`GET /frame` is authenticated by the existing operator-owned loopback bearer token. The simulator owner thread freezes an immutable observation after its existing observe/validation path; the mailbox publishes that snapshot and executor state under one lock. Network handlers never read the simulator or issue a reset, step, apply or observation call.

The available schema1 response has exactly these fields:

| Fields | Meaning |
|---|---|
| `schema_version`, `available` | Exact integer1 and booleantrue. An unavailable response contains only these fields with `available:false`. |
| `session_id`, `revision`, `active_episode_id`, `mode` | Frozen command context of the observation; not replaced when a later command is accepted. |
| `current_context` | Those same four keys from the current owner-thread publication, atomically paired with the frozen observation. Admission requires exact equality with the frozen context. |
| `episode_id`, `step`, `sim_time` | Observation identity, nonnegative frame step and simulator seconds. Idle previews have a `preview-...` episode and `active_episode_id:null`; active recording uses the same episode in both fields. |
| `camera_key`, `camera_prim` | `observation.images.front` and the configured absolute USD camera prim. |
| `joints`, `state_rad`, `units` | Ordered native joint names and finite measured radians from this same observation; `units` is exactly `rad`. No coordinate conversion, physical calibration or clipping of measured state is implied. The existing controller still guards applied targets separately. |
| `width`, `height`, `rgb_base64`, `rgb_sha256` | Complete immutable RGB24 pixels and SHA256 of the decoded raw bytes. Each dimension is1..1920, at most1920² pixels. Current recording settings further require even dimensions of at least2. |
| `observation_received_monotonic_ns` | Executor-local time after the observation returned and validation completed. This is not the camera exposure or device acquisition timestamp. |
| `published_monotonic_ns` | Latest owner-thread mailbox publication time on that executor clock. Re-publishing never changes the observation receipt timestamp. |
| `source_age_ns` | Executor elapsed duration since observation receipt, recomputed while serializing each locked response. |

The original seven preview fields remain present. Consumers that require atomic identity must explicitly require schema1 instead of silently treating an older preview response as trustworthy. The RGB/joints and frozen acquisition context never change when task, pause, correction or annotation commands are accepted. Their new current context invalidates the old observation until an existing actual observation occurs. A reset creates a new preview; restarting the executor creates a new session.

A pause remains successful even when rendering is unavailable. There is deliberately no additional read-only observation on pause/task: the current Isaac observation path performs rendering without a bounded per-call deadline. No extra motion, rendering or recorded row is introduced to refresh a display. Faults and close also invalidate an old snapshot through `current_context`; finish/step-limit publishes unavailable once its observation is cleared.

## Consumer admission

```python
import time
from firebird_teaching.frame_snapshot import admit_frame

started = time.monotonic_ns()
payload = trusted_loopback_client.request("/frame")
received = time.monotonic_ns()
frame = admit_frame(
    payload,
    expected_session_id=operator_selected_session,
    request_started_monotonic_ns=started,
    received_monotonic_ns=received,
)
provider_frame = frame.provider_input()  # Optional dependency imported only here.
```

The caller owns the expected session selection. A mismatch, unavailable response, unsupported/excess fields, invalid dimensions, base64 length, RGB fingerprint, joint metadata, nonfinite number or inconsistent timing raises `ValueError`. Source/request age is conservatively `source_age_ns + (received - started)`, with a default5s maximum, configurable only up to30s. Absolute clock values from different machines are never subtracted. Both consumer timestamps must come from the same local process; the source acquisition/publication pair belongs only to the executor clock.

The optional provider bridge keeps that source/request age and adds subsequent local elapsed time before and after inference. A current-identity callback must come from fresh admitted frames, invalidate when unavailable/stale, and be checked again when presenting a proposal. A frame or proposal never authorizes robot motion. The provider's existing1024² total-pixel cap is intentionally narrower than preview transport; a larger preview cannot be submitted through the provider bridge.

Transport reads must remain bounded; the existing client allows16MiB total and a3s socket timeout. A consumer may impose tighter dimensions/body/deadline limits. An in-flight command can invalidate an already received frame immediately after admission; this is why later consumers must recheck current identity. The snapshot is atomic at owner-thread publication, not a promise that the robot stops changing during inference.

## Evidence boundary

The regression suite uses deterministic generated CPU simulator observations and real authenticated loopback HTTP. It covers stale commands, pause/resume/reset/restart/fault/close, failed recording, immutable publication during racing reads, malicious envelope fields/hash/length, distinct host clock epochs and cumulative age. It neither launches Isaac nor calls a remote provider. Hardware camera timing, live inference quality and integrated application acceptance remain separate gates.
