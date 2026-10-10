# Managed local teaching captures

This opt-in backend starts the existing `firebird_teaching.isaac` entrypoint in an explicitly configured local Linux Isaac installation. It does not provision a GPU, connect to remote compute, start LiveKit, or change the existing manual Teaching relay. An unavailable local Isaac runtime remains unavailable; there is no generated backend fallback.

A start request creates an ordinary durable application job with operation `teaching.capture`. Use `Idempotency-Key` and the existing project-scoped submission lookup to recover the same job after a lost response. An accepted key never launches a replacement worker, including after application restart. The job stores only a configured profile identity and bounded duration; clients cannot submit a command, executable, destination, credential or URL.

## Explicit operator configuration

Set `FIREBIRD_TEACHING_SESSION_CONFIG` to a new operator-owned JSON file only when ready to enable this feature. `FIREBIRD_RECORDING_CONFIG` must already map the project to a distinct publication root using the existing recording preparation configuration. No setup command or endpoint changes these files.

```json
{
  "schema_version": 1,
  "profiles": [{
    "id": "local-isaac",
    "label": "Local Isaac teaching",
    "project_id": "existing-project-id",
    "isaac_python": "/operator/isaac/python.sh",
    "worker_root": "/operator/openjensen/workers/teaching",
    "settings_path": "/operator/teaching/settings.json",
    "lease_path": "/operator/teaching/isaac-resource.lock",
    "control_port": 8878,
    "accept_eula": true,
    "max_seconds": 300,
    "max_capture_bytes": 536870912
  }]
}
```

The settings file is the existing strict teaching settings format. Native joint order, radians, requested and applied targets, source origin, camera, simulation clock and lineage are preserved. The scene identity covers root USD bytes only; it does not establish referenced-asset closure or physical calibration. Repeated sessions of one lineage remain one group. Training requires independent source groups; this runner does not invent them.

Profiles expose configured/available state and `runtime_verified: false`. Linux support and structural configuration do not prove Isaac readiness. Start checks the exact profile, source, settings, executable target and recording registry identities again. Only a matching authenticated worker session and measured executor state establish readiness. No executable is probed while merely listing profiles.

## API and ownership

All routes below are under `/api/v1/projects/{project_id}/teaching` and use the application's existing local-access boundary.

- `GET /profiles` lists fixed profiles and their identity hashes.
- `POST /sessions` accepts `operation`, `profile_id`, `profile_sha256` and `timeout_seconds` and returns the ordinary job. Duration is 1–3600 seconds and cannot exceed the operator cap.
- `GET /sessions/{job_id}` returns `{job, ready, session_id, stop_requested}`.
- `POST /sessions/{job_id}/stop` requests graceful finish. Repeated calls are safe; it never launches work.
- The session's `/state`, `/frame`, `/commands` and `/commands/{command_id}` reuse the strict typed Teaching protocol, bound to this project, job and live executor identity.
- Existing job cancellation aborts publication. There is no automatic capture conversion, training, provider request or voice-room creation.

A private kernel lease excludes another configured owner. The lease descriptor is inherited by the fixed supervisor and Isaac worker. A parent-liveness pipe, deadline, output caps and bounded shutdown are independent of the browser. Parent loss, cancellation, deadlines and malformed ownership receipts cannot publish a capture. Application restart marks unfinished jobs interrupted and never signals a stored PID or adopts late output.

The supervisor and Isaac child occupy one newly created POSIX process group. On exit, the supervisor writes a private child-exit receipt and kills that entire group, including itself. Its expected signal exit is an ownership mechanism, not execution success. The API independently reaps the supervisor and verifies group absence. An uncertain cleanup keeps the failure explicit; closing the API's lease descriptor cannot release a lease still inherited by a live child.

## Capture publication

Only explicit graceful stop, Isaac child exit zero, matching live session identity and verified process cleanup proceed to capture validation. The existing raw-capture inspector checks every episode's finalized receipt, intervention journal, contiguous timestamps, measured and applied coordinates and all RGB bytes. Any unfinished episode rejects publication. Empty sessions also produce no catalog entry.

Validation is bounded to 100 episodes, 50,000 entries and the configured byte cap (at most 8 GiB). Live disk monitoring is periodic, so the capture may exceed its cap by data written before the next monitor check; that session is rejected and retained privately. The profile must be sized to the operator's actual disk headroom. Logs are capped separately.

Verified bytes are copied to a private staging directory outside the catalog, rehashed against the unchanged raw source, then published with an atomic no-replace directory operation. Existing captures are never replaced. Failed or interrupted staging and raw inputs remain private evidence. A filesystem publication followed by an unavailable job database can retain a valid catalog capture with an explicit job-receipt failure; it must not cause automatic re-execution.

The existing recording catalog and durable `dataset.inspect` preparation path consume the published session and exact episode hashes. LeRobot writing, full dataset readback, immutable snapshot creation and explicit Training selection remain separate existing steps. Capture validation alone claims neither trainability of one lineage group nor task quality.

## Verification scope

The tests distinguish generated Session/Journal raw-capture fixtures and real disposable CPU process ownership from real Isaac execution. An actual configured Isaac host, rendered episode, finalized capture publication, genuine writer/readback and Training handoff are required before claiming live simulator acceptance. These checks do not establish live runtime acceptance. LiveKit credentials, provider entitlement, Unity transport and remote capture transfer remain separate operational work.
