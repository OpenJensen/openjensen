# Read and admit a teaching frame

Call `GET /frame` through the configured loopback client with the executor's
operator-owned bearer token. Select the expected executor session from a fresh
`GET /state` read. Require schema version 1 and these response fields:

| Fields | Meaning |
|---|---|
| `schema_version`, `available` | Exact integer 1 and boolean true. An unavailable response contains only these fields with `available:false`. |
| `session_id`, `revision`, `active_episode_id`, `mode` | Frozen command context of the observation; not replaced when a later command is accepted. |
| `current_context` | Those same four keys from the current owner-thread publication, atomically paired with the frozen observation. Admission requires exact equality with the frozen context. |
| `episode_id`, `step`, `sim_time` | Observation identity, nonnegative frame step and simulator seconds. Idle previews have a `preview-...` episode and `active_episode_id:null`; active recording uses the same episode in both fields. |
| `camera_key`, `camera_prim` | `observation.images.front` and the configured absolute USD camera prim. |
| `joints`, `state_rad`, `units` | Ordered native joint names and finite measured radians from this same observation; require `units: rad`. |
| `width`, `height`, `rgb_base64`, `rgb_sha256` | Complete immutable RGB24 pixels and SHA256 of the decoded raw bytes. Each dimension is 1..1920, at most 1920² pixels. Current recording settings further require even dimensions of at least 2. |
| `observation_received_monotonic_ns` | Executor-local time after the observation returned and validation completed. |
| `published_monotonic_ns` | Latest owner-thread mailbox publication time on that executor clock. Re-publishing never changes the observation receipt timestamp. |
| `source_age_ns` | Executor elapsed duration since observation receipt, recomputed while serializing each locked response. |

## Consumer admission

Capture request-start and receipt times from the same local monotonic clock,
then pass the response and expected session to the admission helper:

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

Use the returned immutable frame only within its age allowance: 5 seconds by
default, configurable up to 30 seconds. Add the full local request elapsed time
to `source_age_ns`; keep remote absolute clock values separate from local ones.

Keep transport reads bounded to 16 MiB and a 3-second socket timeout or tighter
client limits. When state/frame context changes, replace the selected identity
with a newly admitted observation. Recheck identity before presenting a provider
response. For `.provider_input()` fit the frame within 1,048,576 total pixels.
See [provider invocation](PROVIDERS.md) for that call's current-identity callback.
