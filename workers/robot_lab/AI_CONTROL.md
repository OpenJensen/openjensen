# Governed SO101 execution

The companion can execute one bounded joint-position segment for an explicitly armed AI teleoperation session. The language model remains outside the motor loop. The backend's fixed private adapter forwards typed requests; it does not expose the companion credential or a serial device to the model.

## Ownership and arming

1. The operator selects the connected follower, saved calibration and camera, then explicitly starts full-arm joint control. Existing control startup verifies calibration and holds the measured pose before enabling torque.
2. `agent/acquire` transfers that powered manual controller to one agent owner. It requires `ARM AI CONTROL`, the current `session.generation`, and exact `operator_present:true` and `workspace_clear:true` acknowledgements. It never opens a serial port or enables torque on its own. Selected-motor sessions, a moving rest/middle operation and an active human recording episode are rejected.
3. Core renews the owner lease only while its explicit operator/browser lease remains fresh. Each companion heartbeat identifies the owner, authority epoch and a strictly increasing nonce. Model inference does not supply the heartbeat. A duplicated heartbeat returns its receipt without extending the lease.
4. The local watchdog disables torque when no current heartbeat arrives for two seconds. Stop invalidates controller generation and agent authority before waiting for camera or serial completion. Releasing AI ownership uses the same torque-off Stop; it is not a gravity-compensated hold. Manual control requires a new explicit connection.

`generation` starts from a fresh random integer on each companion process and increments on controller transitions. This fences delayed preflight requests after restart. Authority epochs and command identities are private durable records. Manual heartbeat names `expected_generation`. Automatic manual cleanup names both `expected_generation` and `expected_mode:"manual"`; the mode fence also rejects cleanup after a same-generation transfer to AI ownership. An empty Stop request remains the unconditional emergency path. A manual heartbeat cannot renew an agent lease.

Pause and Cancel inference revoke pending planning work while retaining the locally supervised pose hold and current ownership. A segment already in progress finishes its admitted bounded motion. Model/network failures also pause planning; they do not supply motor heartbeat. The operator must continue the browser presence lease. Explicit Stop, lease expiry or a local motion fault disables torque.

The watchdog is software running in the companion. It does not establish a motor-firmware watchdog or guarantee torque-off after process death, USB failure or loss of motor power. The arm can fall when torque is disabled; support it and keep physical power removal available.

## Observation contract

`agent/observe` acquires encoder samples before and after a fresh camera read. It never uses the recording preview's cached JPEG. An admitted snapshot retains its image hash, named joint vector, camera and encoder host times, calibration hash, controller generation/epoch, embodiment, units, limits and a unique observation identity/hash.

- Arm joint positions use **degrees**; gripper position uses **percent of calibrated travel**.
- The combined read interval must be at most 150 ms, with at most 1 degree or 2 gripper-percent of measured drift across it.
- The execution snapshot expires after one second on the companion's monotonic clock. Core accounts conservatively for transport delay without comparing unsynchronized clocks or changing the hashed source.
- OpenCV does not expose portable sensor/exposure timestamps for these cameras. The contract explicitly says `image_sensor_timestamp_verified:false` and `sensor_time_policy:host_read_interval`. Host read bounds do not prove sensor-clock synchronization.
- Open the selected camera preview before arming so permission prompts and camera startup are outside the short admission interval.

A cloud plan can outlive this execution window. Supervised re-admission must therefore retain the original model observation and obtain a distinct fresh execution observation, with explicit operator review of the current scene and exact proposal. Refreshing an observation must never relabel old pixels or silently replace the model's original evidence.

Observations are private evidence, bounded to 128 MiB per session, 512 MiB in the store, and 2 MiB per image envelope. Archival or free space is required once a bound is reached.

## Segment contract

`agent/execute` accepts the fresh observation ID/hash, named absolute `targets`, exact per-target `units`, and `duration_seconds`. Optional host provenance records the proposal, provider/model and original model observation ID/hash. The executor rechecks ownership, calibration, source age, current measured state and the admitted joint manifest before creating a segment.

Every target lies inside the calibrated margin. A segment may change an arm joint by at most 5 degrees or the gripper by at most 10 percent. Duration is 0.5–5 seconds. A quintic position interpolation is admitted only if its commanded speed is at most 10 units/s and commanded acceleration at most 20 units/s². Delayed ticks advance by at most 75 ms, preventing a large catch-up goal.

These limits describe the **commanded position profile**, not measured physical acceleration or collision safety. Local measured tracking errors above 3 degrees or 6 gripper-percent stop the session, including while holding without an active segment. Encoder reads and repeated hold commands continue in the local loop during model calls. A successful local read/write must remain younger than 500 ms; browser/network heartbeat cannot renew a stalled controller. Completion requires measured state within 1 degree or 2 gripper-percent of the target, with a bounded tracking deadline. The gripper is position controlled; its percentage is not a contact-force command.

Only one segment can be active. Observation acquisition and another segment are rejected until its receipt is available. Requested targets, transmitted targets, measured next state, units, source/calibration bindings, timing and termination are retained. A completed segment never independently asserts task success. Physical AI episodes are not passed through the existing human recording path.

This capability does not provide Cartesian motion, inverse kinematics, object localization, collision avoidance or a grasp-force controller. Object pickup requires appropriate task grounding and controlled acceptance; a natural-language success statement does not verify a lift.

## Wire protocol and evidence

Requests use this outer envelope:

```json
{"command_id":"unique_id","agent_session_id":"owner_id","authority_epoch":2,"arguments":{}}
```

| Route | Arguments |
| --- | --- |
| `POST /v1/agent/acquire` | `confirm`, `expected_generation`, `camera_index`, `operator_present`, `workspace_clear`; outer epoch is 0 |
| `POST /v1/agent/observe` | Empty object |
| `POST /v1/agent/execute` | `observation_id`, `observation_sha256`, `targets`, `units`, `duration_seconds`; optional host provenance |
| `POST /v1/agent/heartbeat` | Increasing `nonce`, `operator_present:true` |
| `POST /v1/agent/release` | Empty object; disables torque |
| `GET /v1/agent/executions/{command_id}` | Reconcile the original command identity |

Canonical request hashes bind the operation and entire payload. Identical IDs return the existing receipt; changed payloads are rejected. The core adapter verifies receipt hashes and never reposts a segment after a lost acknowledgement. The initial identity is durable before motor execution. Process restart marks accepted/in-progress work unknown and never resumes it.

Serial Stop is not delayed by evidence writes: revocation occurs in memory first, motor torque is disabled, and revocation records are flushed afterwards. Segment completion is persisted outside the serial lock. Recording, calibration, model provenance and credentials remain separate domains.
