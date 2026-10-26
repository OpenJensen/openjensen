# Robot Lab companion API, v1

- The app backend forwards `/api/v1/robot-lab/*` to this loopback-only companion's `/v1/*` with a private Bearer token. Browser JavaScript never receives the token.
- Requests and responses are JSON, except camera frames and dataset files.
- Errors use HTTP 400, 409, 422 or 503 with `{"error": "Human-readable reason", "code": "invalid_request"}`.
- Status is cached: polling and device refreshes never start a hardware operation or move a motor.
- Nothing uploads to Hugging Face, and there is no dataset deletion endpoint.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/status` | Companion, session, calibration and recording state |
| `GET /v1/devices` | Serial ports and cameras, without opening cameras |
| `GET /v1/camera.jpg` | One JPEG frame from a camera |
| `POST /v1/probe` | Read-only motor scan |
| `POST /v1/setup/motor` | Assign the ID of one isolated motor |
| `POST /v1/setup/assembled/{start,observe,wiggle,accept,finish,cancel}` | Experimental assembled-arm ID setup |
| `POST /v1/mapping/wiggle`, `POST /v1/mapping/confirm` | Check a saved mapping joint by joint |
| `POST /v1/calibration/{start,center,finish,cancel}` | Staged calibration |
| `POST /v1/control/{start,jog,heartbeat,stop}`, `POST /v1/stop` | Powered control, watchdog and Stop |
| `POST /v1/agent/{acquire,observe,execute,heartbeat,release}`, `GET /v1/agent/executions/{command_id}` | Governed AI ownership, observation and segment receipts; see [AI control](AI_CONTROL.md) |
| `POST /v1/recording/{start,finish}`, `POST /v1/recording/episode/{start,save,discard}` | Recording |
| `GET /v1/datasets`, `GET /v1/datasets/{id}/manifest`, `GET /v1/datasets/{id}/files/{path}` | Finalized datasets for import |

Joint names and canonical IDs are fixed: `shoulder_pan` 1, `shoulder_lift` 2, `elbow_flex` 3, `wrist_flex` 4, `wrist_roll` 5, `gripper` 6. Every `positions`, `mins` and `maxes` object is keyed by these names. Motion positions are degrees, except the gripper (percent of calibrated travel); calibration positions and ranges are raw encoder ticks.

## Status and devices

`GET /v1/status`:

```json
{"ok":true,"version":"1","platform":"Darwin","dependencies":{"lerobot":{"installed":true,"version":"0.4.4"},"serial":{"installed":true,"version":"3.5"},"cv2":{"installed":true,"version":"4.12.0"},"scservo_sdk":{"installed":true,"version":"1.0.0"}},"capabilities":{"probe":true,"camera":true,"motor_setup":true,"calibration":true,"control":true,"recording":true},"session":{"mode":"idle","robot_id":null,"port":null,"role":null,"positions":{},"torque_enabled":false,"watchdog_seconds":2.0,"error":null},"calibration":{"phase":"idle","mins":{},"maxes":{},"offsets":{},"path":null},"recording":{"active":false,"dataset_id":null,"episode_active":false,"frames":0,"episodes":0,"fps":10,"error":null},"calibrations":[],"datasets":[],"joints":[{"name":"shoulder_pan","id":1,"label":"Shoulder pan","unit":"degrees"}],"setup_order":["gripper","wrist_roll","wrist_flex","elbow_flex","shoulder_lift","shoulder_pan"]}
```

- `session.mode`: idle, setup, verification, calibration, manual, leader, handguide or agent. `calibration.phase`: idle, center, ranges, saved or interrupted.
- Also reported: `backend:"feetech-sdk"`, `dataset_format:"lerobot-v3"`, `controller` (the active control profile, see [Control](#control)), dependency entries for `numpy`, `PIL` and `pyarrow`, and `recording.actual_fps` (number or null).
- `dependencies.lerobot.installed:false` is expected on the lightweight hardware runtime and does not disable any capability.
- `session.generation` fences controller handoffs. `agent_control` reports the owner, epoch, units, limits and two-second lease separately from hardware readiness; `capabilities.agent_control` advertises protocol support. See [AI control](AI_CONTROL.md) for explicit arming and physical acceptance boundaries.
- `capabilities` also include `assembled_setup`, `mapping_verification`, `arm_readiness`, `calibration_persistence`, `lerobot_position_control`, `move_to_middle`, `middle_tolerance`, `partial_control`, `joint_sliders` and `large_joint_steps`. The app hides controls an older companion does not advertise.
- `calibrations` entries are `{robot_id, role, path}` plus `valid`, `sha256`, `setup_port`, `setup_fresh` and `error`. All six joints are validated and backup files are excluded. These describe the files, not whether the connected hardware matches.

`GET /v1/devices`:

```json
{"ports":[{"device":"/dev/cu.usbmodem...","description":"USB Serial","manufacturer":"...","serial_number":"..."}],"cameras":[{"index":0,"name":"Camera 0","width":640,"height":480}],"camera_error":null}
```

Camera inventory does not open devices, so width and height are requested defaults until a preview confirms them. Linux lists V4L2 capture nodes (indices 0–63) and omits metadata-only nodes. Preview and recording need macOS camera permission or Linux device read/write permission. Serial ports must be devices enumerated on this computer.

`GET /v1/camera.jpg?index=0` returns a JPEG. Optional `width` and `height` apply on first open (default 640×480). An idle stream is released after 30 seconds.

## Probe

`POST /v1/probe {port, baudrate?:1000000}` is read-only: it never writes registers or enables torque.

```json
{"port":"...","baudrate":1000000,"motors":[{"id":1,"model":777,"stable":true,"responses":3}],"ready":false,"expected_ids":[1,2,3,4,5,6],"warning":"A response at one ID cannot prove that only one physical motor is connected."}
```

With `robot_id` and `role`, the response adds `profile.state`: `missing`, `invalid`, `stale`, `unavailable`, `mismatch` or `matched`. Only a stable six-motor scan with matching offsets and limits for every joint reports `matched`.

## Motor IDs

### One isolated motor

`POST /v1/setup/motor {port, joint, baudrate?:1000000, isolated:true, confirm:"ASSIGN MOTOR ID"}` returns `{"assigned":{"joint":"gripper","id":6,"baudrate":1000000},"warning":"..."}`.

Shared factory IDs cannot be told apart by a scan, so this requires a physical-isolation acknowledgement for every motor, three stable model checks, and a single responding ID across 1–253. Each motor must be connected on its own, as in the official LeRobot setup. Torque is never enabled.

### Assembled arm (experimental)

The joint illustration is browser-only: picking or replaying it sends no motor traffic. After setup starts, selecting a joint can begin a read-only encoder observation; it never saves an ID or commands a movement.

`POST /v1/setup/assembled/start {port, supported:true, confirm:"START ASSEMBLED SETUP"}` starts a background session and returns status immediately. `supported:true` confirms a supported, physically held chain of six STS3215 servos and permission to reboot servos and change persistent IDs. Keep every motor cable connected. The reboot-timing method depends on firmware and adapter timing and has no fallback if the wake edge cannot be measured. It never sends Goal_Position or Torque_Enable=1, but reboots and torque release can let the arm fall: support it throughout.

`status.assembled` reports `phase`, `port`, `ids`, `positions` (raw ticks by current address), `selected_joint`, `detected_id`, `candidate_delta`, `mapping` (joint to current address), `message`, `error`, `log`, `progress`, `experimental:true` and `calibration_required`. Phases: idle, scanning, splitting, ready, observing, review, applying, complete, error, stopped. An active session uses `session.mode:"setup"` and needs `/v1/control/heartbeat` every 500 ms (10-second timeout).

Identify each joint, once six independent temporary addresses are verified, in one of two ways:

- **By hand.** `POST /v1/setup/assembled/observe {joint}` re-baselines the encoders and watches that joint without writing registers. Small, ambiguous or multi-joint movement, already-assigned motors and stale evidence are rejected. `POST /v1/setup/assembled/accept {joint, motor_id}` confirms the detected address.
- **By wiggle.** `POST /v1/setup/assembled/wiggle {motor_id, supported:true, confirm:"WIGGLE MOTOR"}` re-verifies the inventory, requires position mode with torque released, and makes one fixed ±24-tick motion on that address with a 10% SRAM torque limit. The client cannot set position, duration or torque. No EEPROM or calibration register changes; cancelling disables torque, and SRAM settings are restored only after torque-off read-back. With no measurable movement it falls back to identification by hand. Then `POST /v1/setup/assembled/accept {joint, motor_id, method:"wiggle", confirm:"CONFIRM OBSERVED JOINT"}` records the joint the operator saw move, within 60 seconds of a completed wiggle. Support the arm and keep every joint clear of its stops.

Each address can belong to only one joint. After all six, review the mapping and `POST /v1/setup/assembled/finish {confirm:"SAVE MOTOR IDS"}`. It renumbers through unused temporary IDs, re-locks EEPROM, and after a final reboot verifies the six addresses, model, lock and torque-off state before reporting complete. Keep the heartbeat running while the phase is `applying`. Previous calibrations stay on disk but cannot authorize movement for the new setup; calibrate again.

`POST /v1/setup/assembled/cancel {}` or Stop cancels the session; an in-flight ID transaction finishes its EEPROM re-lock first, and no new one starts. Failures keep a private journal of discovered addresses, the mapping and any uncertain rename. A retry scans the whole address space instead of assuming factory IDs. Setup never erases calibrations or datasets. Simulated tests cover the protocol and states, not every physical arm.

### Check a saved mapping

`POST /v1/mapping/wiggle {port, joint, supported:true, confirm:"WIGGLE SAVED MOTOR"}` needs a completed mapping journal and an idle arm. It verifies the six canonical addresses directly (stable encoder samples, STS3215 model, stored ID, torque off), then performs the same bounded SRAM-only wiggle. It never reboots, rewrites IDs, changes the journal or needs a calibration.

The status shows `session.mode:"verification"` and a `verification` object with `phase` (idle, checking, wiggling, done, error, stopped), `port`, `joint`, `motor_id`, `ticks`, `motion_active`, `error` and a unique `run_id`. Keep the 500 ms heartbeat running (10-second timeout); Stop and hiding the page cancel it, and serial closes after torque-off cleanup.

`POST /v1/mapping/confirm {port, joint, run_id, correct:boolean, confirm:"CONFIRM PHYSICAL JOINT"}` records whether the operator saw the expected joint move. It accepts only the latest completed, cleaned-up wiggle on an unchanged mapping, never commands hardware, and repeats of the same answer are idempotent. Status `mapping_checks` lists `port`, the journal `revision` and per-joint `run_id`, `correct`, `checked_at` and `error`. Confirmations are kept on this computer, tied to the mapping, and cleared by a new setup. Completed mappings are restored from the journal on restart without opening serial ports.

## Calibration

1. `POST /v1/calibration/start {port, robot_id, role:"follower"|"leader", confirm:"START CALIBRATION"}` checks all motors, disables torque and enters the `center` phase.
2. `POST /v1/calibration/center {}` computes homing offsets from the pose held by hand and enters `ranges`, sampling live minimums and maximums as each joint is moved by hand.
3. `POST /v1/calibration/finish {confirm:"SAVE CALIBRATION"}` validates the five measured ranges (wrist roll uses the full 0–4095), writes a LeRobot-compatible calibration, disables torque and closes serial. The phase becomes `saved`.

`POST /v1/calibration/cancel {}` disables torque, closes serial and keeps the previous calibration file, restoring staged motor changes when possible. Calibration writes EEPROM, and each stage is an explicit action.

Calibration runs on the companion with torque off and continues if the browser disconnects (`session.watchdog_seconds:null`). Private checkpoints keep the captured middle and ranges, so a reloaded page can rejoin. `POST /v1/calibration/start {resume:true, confirm:"START CALIBRATION"}` resumes a checkpoint after verifying motor IDs, the saved-profile checksum and motor settings. Cancel restores the previous calibration and archives the checkpoint; Save archives it once the new calibration is verified.

**Interrupted calibration.** A communication or safety failure closes the session, restores the previous calibration if the bus allows, and reports `phase:"interrupted"`, `interrupted_stage:"center"|"ranges"` and `error`. Starting again is explicit. Missing, incomplete or checksum-invalid read replies get at most three attempts, discarding stale input first; replies must match the requested motor ID, register width, frame length, header and checksum. Motor-reported faults are not retried. A missing configuration-write acknowledgement is resolved by an exact register read-back, never by replaying the write. Motion, torque, ID and baud-rate writes are never retried this way. Encoder wraps or implausible jumps abort the session.

## Control

`POST /v1/control/start {port, robot_id, mode:"manual"|"leader"|"handguide", confirm:"ENABLE MOTION"}` needs six IDs and a matching calibration. The arm holds its current position before torque is enabled, so it does not jump.

- `manual`: on-screen joint control. Optional `joints` (unique canonical names) limits probing, read-back, torque, encoder reads, jogs and Stop to those motors; omitted means all six, and `session.active_joints` lists them. Selected motors must keep responding.
- `manual` with `move_to_middle:true`: sends one full goal to the recorded middle, 2047 ticks per controlled joint, rejecting a middle outside the calibrated range. The app requests this only when all six joints are selected. Jogging and recording wait until every joint is within `middle_tolerance` (0.5–5, default 2; degrees, or percent for the gripper). `session.centering` reports `active`, `progress` and `tolerance`.
- `leader` with `leader_port` and `leader_id`: a second calibrated arm drives the follower.
- `handguide`: torque stays off, for moving the arm by hand and recording passive demonstrations.

Hand guiding, leader mode and recording need all six joints.

**Profile.** Control follows the pinned LeRobot SO101 position profile. `controller` reports `upstream`, `target_mode:"direct_position"`, `max_relative_target:null`, acceleration 254, maximum acceleration 254 and goal velocity 0, and `capabilities.lerobot_position_control` is true. Goals are full calibrated positions, resent as batched position packets every control cycle. Leader tracking sends the leader's full calibrated positions. A joint lagging its goal, for example while lifting, is expected and does not stop control.

`POST /v1/control/jog` (manual only) takes exactly one of:

- `{joint, delta}`: a step of up to ±10 degrees (gripper ±10 percent), at least 100 ms after the previous jog. It advances the last commanded target, not the measured position.
- `{joint, target}`: an absolute slider position within `session.joint_limits`, the calibrated range with an 8-tick margin.

Targets are clipped to that range and sent directly as one goal. `session.targets` holds the commanded positions, which recordings use as actions, next to measured `session.positions`. Unselected motors receive no commands. Jogs are refused while moving to the middle.

`POST /v1/control/heartbeat {}` refreshes the 2-second watchdog; send it every 500 ms. `POST /v1/control/stop {}` or `POST /v1/stop {}` stops motion, disables torque, closes ports and ends an active episode while keeping its frames for save or discard. Heartbeat expiry, serial or hardware errors, invalid encoder values and range violations do the same, and `session.error` explains why. Torque must be re-enabled explicitly. Every operation returns the current status. The companion never runs a policy.

## Recording

`POST /v1/recording/start {dataset_id:"pick-block", task:"Pick up the block", fps:10, camera_index:0}` creates a new LeRobot v3.0 dataset in private local storage. It needs an active calibrated control session (any mode), a new safe slug as `dataset_id`, a task, and `fps` 1–20. `camera_index:null` records state and actions only. Features are `observation.state`, `action` and `observation.images.front`, in degrees and gripper percent.

1. `POST /v1/recording/episode/start {}` starts an episode.
2. `POST /v1/recording/episode/save {}` saves it, or `POST /v1/recording/episode/discard {}` drops it.
3. `POST /v1/recording/finish {}` finalizes the metadata once no episode is pending.

Hand-guided actions are the observed positions; manual and leader actions are the targets actually sent. Failed or late frames raise an error and stop collection; frames are never invented. Capture runs on fixed deadlines and pauses when the actual rate stays below 90% for 2 seconds; `openjensen.json` keeps each episode's measured FPS. Starting needs 1 GiB free, capture keeps a 512 MiB reserve, and each episode is limited to 5 minutes.

`POST /v1/recording/finish {abandon_empty:true, confirm:"ABANDON EMPTY RECORDING"}` closes a recording that has no saved episodes or frames, so a new dataset name or lower FPS can be chosen. It pauses collection, keeps every file and pending frame on this computer, and sends no motor command. A missing or wrong confirmation, a non-boolean flag or saved episodes are rejected, and a metadata-write failure keeps the recorder attached for a retry. The dataset stays `finalized:false`, cannot be imported, and is listed with `abandoned:true` and `pending_frames:N` (existing records default to `false` and `0`).

## Datasets

`GET /v1/datasets`:

```json
{"datasets":[{"id":"pick-block","path":"...","episodes":2,"frames":300,"fps":10,"task":"Pick up the block","format":"lerobot","finalized":true}]}
```

`GET /v1/datasets/pick-block/manifest` (or `/files`) lists a finalized dataset:

```json
{"id":"pick-block","name":"pick-block","dataset_id":"pick-block","files":[{"path":"meta/info.json","size":1234,"sha256":"64 hex characters"}],"total_bytes":1234}
```

`GET /v1/datasets/pick-block/files/meta/info.json` (or `/files?path=meta/info.json`) returns the file's bytes. Dot paths, symlinks, hidden files and traversal are rejected. The app's importer copies the complete manifest into a project through the app backend.

### Rest position

`POST /v1/control/rest/save` with `{"confirm":"SAVE REST POSITION"}` saves the actual
encoder positions of all six joints for the active follower and calibration, including
hand guiding. `POST /v1/control/rest/return` with `{"confirm":"RETURN TO REST"}` requires
powered full-arm control and a saved pose. It pauses leader tracking, ramps at at most
10 degrees/percent per second, verifies arrival within 2 degrees/percent, and holds
torque in manual mode. A timeout holds the observed position and reports an error.
Recording episodes, partial control, expired heartbeats and concurrent positioning
block both operations. Calibration changes invalidate the saved pose. Stop, browser
visibility/disconnect safety, and shutdown retain their immediate torque-off behavior.
`session.rest_position` reports saved/phase/progress/targets; `capabilities.rest_position`
allows older companions to be detected without issuing unsupported commands.

## Governed AI segments

The fixed private agent protocol, units, observation bounds, commanded segment profile and reconciliation semantics are specified in [AI_CONTROL.md](AI_CONTROL.md). `control/heartbeat` accepts optional `expected_generation`; both Stop routes accept the same optional fence for automatic cleanup. Omit it only for an unconditional operator Stop.
