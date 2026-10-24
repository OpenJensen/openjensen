# Robot Lab companion API, v1

The app backend forwards `/api/v1/robot-lab/*` to this loopback-only companion's `/v1/*` with a private Bearer token. Browser JavaScript never receives that token. All responses JSON except camera/file bytes. Errors use HTTP 400/409/422/503 and `{ "error": "Human-readable reason", "code": "invalid_request" }`. Inputs below are JSON. No automatic Hugging Face upload.

`GET /v1/status`:
```json
{"ok":true,"version":"1","platform":"Darwin","dependencies":{"lerobot":{"installed":true,"version":"0.4.4"},"serial":{"installed":true,"version":"3.5"},"cv2":{"installed":true,"version":"4.12.0"},"scservo_sdk":{"installed":true,"version":"1.0.0"}},"capabilities":{"probe":true,"camera":true,"motor_setup":true,"calibration":true,"control":true,"recording":true},"session":{"mode":"idle","robot_id":null,"port":null,"role":null,"positions":{},"torque_enabled":false,"watchdog_seconds":2.0,"error":null},"calibration":{"phase":"idle","mins":{},"maxes":{},"offsets":{},"path":null},"recording":{"active":false,"dataset_id":null,"episode_active":false,"frames":0,"episodes":0,"fps":10,"error":null},"calibrations":[],"datasets":[],"joints":[{"name":"shoulder_pan","id":1,"label":"Shoulder pan","unit":"degrees"}],"setup_order":["gripper","wrist_roll","wrist_flex","elbow_flex","shoulder_lift","shoulder_pan"]}
```
`session.mode`: idle, setup, verification, calibration, manual, leader, handguide. `calibration.phase`: idle, center, ranges, saved. Joint names/IDs fixed shoulder_pan/1, shoulder_lift/2, elbow_flex/3, wrist_flex/4, wrist_roll/5, gripper/6. All .positions/mins/maxes objects keyed by these names. Motion positions are degrees except gripper (percent); calibration positions/ranges are raw encoder ticks. `calibrations` entries `{robot_id,role,path}`; `datasets` entries below.

`GET /v1/devices`: `{"ports":[{"device":"/dev/cu.usbmodem...","description":"USB Serial","manufacturer":"...","serial_number":"..."}],"cameras":[{"index":0,"name":"Camera 0","width":640,"height":480}],"camera_error":null}`. Camera inventory does not open devices; width/height are requested defaults until preview verifies them. macOS camera permission or Linux device read/write permission is needed for preview/record. Linux lists V4L2 capture nodes and omits metadata-only nodes; indices 0–63 are supported. `GET /v1/camera.jpg?index=0` returns JPEG; optional width/height set on first open (defaults640x480). No stream held when idle for more than30s. `POST /v1/probe {port,baudrate?:1000000}` returns `{"port":"...","baudrate":1000000,"motors":[{"id":1,"model":777,"stable":true,"responses":3}],"ready":false,"expected_ids":[1,2,3,4,5,6],"warning":"A response at one ID cannot prove that only one physical motor is connected."}`. Read only. All serial ports must be actual locally enumerated devices.

`POST /v1/setup/motor {port,joint,baudrate?:1000000,isolated:true,confirm:"ASSIGN MOTOR ID"}` returns `{"assigned":{"joint":"gripper","id":6,"baudrate":1000000},"warning":"..."}`. Requires physical isolation acknowledgement every motor; three stable model checks and sole responding ID across1–253. Shared factory IDs cannot be identified from an ordinary scan, so this legacy isolated-motor endpoint requires physical isolation. The assembled workflow below is a separate experimental procedure. No torque enabled. Failed assignment returns error. Each motor must be separately connected as official LeRobot setup requires.

`POST /v1/calibration/start {port,robot_id,role:"follower"|"leader",confirm:"START CALIBRATION"}` checks all motors then disables torque and enters center phase. Returns status. `POST /v1/calibration/center {}` computes homing offsets, enters ranges phase with live min/max sampler. `POST /v1/calibration/finish {confirm:"SAVE CALIBRATION"}` validates all five measured joint ranges, writes official LeRobot-compatible calibration, disables torque and closes serial. Returns status (phase saved). Wrist roll is full0–4095 per official process. `/v1/calibration/cancel {}` disables torque/closes serial, previous file retained; staged motor changes restored when possible. User moves each joint by hand during ranges. Calibration changes EEPROM; each stage is an explicit UI action. Calibration runs locally with torque off and continues across browser disconnection. Periodic private checkpoints preserve captured middle/ranges; resume verifies the saved profile and known motor configuration before restoring unpowered capture.

`POST /v1/control/start {port,robot_id,mode:"manual"|"leader"|"handguide",leader_port?:string,leader_id?:string,confirm:"ENABLE MOTION"}` requires six IDs and existing matching calibration. It holds current position before torque enable to avoid a jump. `manual`: only small explicit jog actions; `leader`: streams second calibrated arm at20Hz with a bounded step; `handguide`: torque remains disabled, useful for passive observation/recording. `POST /v1/control/jog {joint,delta}` manual only; delta max2degrees or2percent; bounded to measured joint limits. `POST /v1/control/heartbeat {}` refreshes2-second watchdog (heartbeat every500ms). `POST /v1/control/stop {}` and `/v1/stop {}` stop motion, disable torque, close ports, end active episode collection, leave captured frames pending for save/discard. Watchdog expiry or serial failure does the same; .session.error explains why. Torque must be enabled again explicitly. Return current status for every operation. No policy execution.

`POST /v1/recording/start {dataset_id:"pick-block",task:"Pick up the block",fps:10,camera_index:0}` creates a new trainable LeRobot v3.0 dataset under private local storage; requires active calibrated control session; fps1–20. Dataset ID must be new safe slug, task required. Optional camera_index:null records state/actions only. Uses normalized joint positions/degrees and gripper percentage; features `observation.state`, `action`, `observation.images.front`. `POST /v1/recording/episode/start {}` starts collection; `/v1/recording/episode/save {}` ends and saves buffered episode; `/v1/recording/episode/discard {}` clears current pending episode; `/v1/recording/finish {}` requires episode saved or discarded, finalizes metadata. Return status. Failed/too-slow frames surface error and stop collection, never silently invent camera frames. Handguide actions equal observed positions; manual/leader actions are actual bounded targets.

`POST /v1/recording/finish {abandon_empty:true,confirm:"ABANDON EMPTY RECORDING"}` explicitly closes a recording only when it has no saved episodes or saved frames. It pauses active collection, retains every dataset file and pending camera frame/JSON row on the robot computer, and clears the active recorder so a new dataset name and lower FPS can be chosen. No motor command is sent. Missing/wrong confirmation, a non-boolean flag, or saved episodes are rejected without detaching the recorder. Metadata-write failure also leaves the paused recorder attached for retry. The retained dataset stays `finalized:false` and cannot be imported; its list entry reports `abandoned:true` and `pending_frames:N`. Existing records default to `abandoned:false,pending_frames:0`. Normal `{}` finish behavior is unchanged.

`GET /v1/datasets`: `{"datasets":[{"id":"pick-block","path":"...","episodes":2,"frames":300,"fps":10,"task":"Pick up the block","format":"lerobot","finalized":true}]}`.
`GET /v1/datasets/pick-block/manifest` or `/files`: `{"id":"pick-block","name":"pick-block","dataset_id":"pick-block","files":[{"path":"meta/info.json","size":1234,"sha256":"64 hex characters"}],"total_bytes":1234}`; finalized datasets only. `GET /v1/datasets/pick-block/files/meta/info.json` or `/files?path=meta/info.json` returns file bytes. Dot paths, symlinks, hidden files and traversal rejected. Importer can copy complete manifest into Xbox project through app backend. No dataset deletion endpoint.

Additional status fields: `backend:"feetech-sdk"`, `dataset_format:"lerobot-v3"`, dependency entries `numpy`, `PIL`, `pyarrow`, and `recording.actual_fps` (number or null). `dependencies.lerobot.installed:false` is expected for the lightweight runtime and does not disable hardware/recording capabilities. Capture uses fixed deadlines and pauses when sustained actual FPS falls below 90% for 2 seconds. Dataset `openjensen.json` retains per-episode measured FPS. Recording-start requires 1 GiB available disk; capture preserves a 512 MiB reserve and limits each episode to 5 minutes.

## Experimental assembled-arm ID setup

The joint illustration is browser-only. Picking or replaying an illustration before preparation sends no motor traffic. After explicit preparation, joint selection can start a read-only encoder observation; it never saves an ID or commands a movement target.

`POST /v1/setup/assembled/start {port,supported:true,confirm:"START ASSEMBLED SETUP"}` starts a background setup session and immediately returns status. `supported:true` acknowledges a supported, physically held six-STS3215 SO-101 chain and permission to reboot servos and update persistent IDs. Keep the motor cables connected. The reboot-timing method is experimental and depends on firmware/adapter timing; it does not send Goal_Position or Torque_Enable=1 packets. Reboots and torque release can let the arm fall, so support it throughout. There is no default-timing fallback if the wake edge cannot be measured.

`status.assembled` includes `phase`, `port`, `ids`, `positions` (raw ticks keyed by current address), `selected_joint`, `detected_id`, `candidate_delta`, `mapping` (joint name to current address), `message`, `error`, `log`, `progress`, `experimental:true` and `calibration_required`. Phases are idle, scanning, splitting, ready, observing, review, applying, complete, error or stopped. Status is cached: polling never initiates a hardware operation. An active session uses `session.mode:"setup"` and requires the existing `/v1/control/heartbeat` every 500 ms, with a 10-second timeout.

`POST /v1/setup/assembled/observe {joint}` re-baselines encoder readings and watches that joint without register writes. It requires six independent supported responders. Detection rejects small movement, ambiguous multi-joint movement, already assigned motors and stale evidence. `POST /v1/setup/assembled/accept {joint,motor_id}` confirms the currently detected address; each address can belong to only one joint. Repeat for all six and review the mapping.

`POST /v1/setup/assembled/finish {confirm:"SAVE MOTOR IDS"}` requires a complete unique mapping, then starts asynchronous canonical renumbering through unused temporary IDs. It re-locks EEPROM and verifies the six addresses, model, lock and torque-off state after a final reboot before reporting complete. Preserve the heartbeat while phase is applying. A new calibration is required after ID setup; previous calibration files remain on disk but cannot authorize movement for the new setup revision.

`POST /v1/setup/assembled/cancel {}` and the normal Stop endpoint cancel the session. An in-flight ID transaction can finish its essential EEPROM re-lock before Stop returns; no subsequent transaction can begin. Bounded failures retain a private journal of discovered addresses, mapping and any uncertain rename. Retrying scans the complete address space instead of assuming factory IDs or flattening the bus. Setup does not erase calibration files or datasets. Simulator tests prove protocol/state behavior, not acceptance on a particular physical arm.

### Joint identification modes

After six independent temporary addresses are verified, `setup/assembled/observe`
plus `accept` provides passive hand identification. The separate POST
`setup/assembled/wiggle` accepts `{motor_id, supported: true, confirm: "WIGGLE MOTOR"}`.
It re-verifies the inventory, requires position mode and released torque, and
executes one fixed ±24-tick motion on that address with a 10% maximum SRAM torque
limit. Its position, duration and torque limits cannot be supplied by the client.
No EEPROM or calibration registers are modified. Cancellation disables torque;
SRAM settings are restored only after torque-off read-back. Missing movement
returns to passive identification without manufacturing an address mapping.

The operator must support the arm and keep every joint clear of its stops.
POST `setup/assembled/accept` with `{joint, motor_id, method: "wiggle",
confirm: "CONFIRM OBSERVED JOINT"}` records the observed joint only after a
successful, completed wiggle, with a 60-second confirmation window. It does not
change persistent IDs until the existing six-joint review/finish action.

## Check a saved mapping

`POST /v1/mapping/wiggle {port,joint,supported:true,confirm:"WIGGLE SAVED MOTOR"}`
requires a completed canonical mapping journal and an idle arm. It reads and
verifies the current six-address inventory before the existing bounded SRAM-only
wiggle. It does not reboot, rewrite IDs, modify the setup journal, or require
calibration. Only this explicit action commands motion; CAD rotation, zoom and
joint selection are visual controls.

The immediate status contains `session.mode:"verification"` and `verification`
fields `phase` (idle/checking/wiggling/done/error/stopped), `port`, `joint`,
`motor_id`, `ticks`, `motion_active`, `error`, and a unique `run_id`. Maintain the
500 ms heartbeat while active (10-second timeout); Stop and hidden-page cleanup
cancel verification. The operation closes serial after torque-off cleanup.
The UI only marks a motion check correct after the operator confirms the joint
that actually moved. Operator confirmations are retained on the robot computer and tied to the exact completed mapping journal; a new setup invalidates them.
Completed ID mappings are restored from the private journal on companion restart
without opening serial hardware.

### Interrupted calibration

A communication or safety failure during center/range capture closes the session,
restores the previous calibration when the bus permits, and reports calibration
`phase: "interrupted"`, `interrupted_stage: "center" | "ranges"`, and `error`.
Starting calibration again is explicit; cancelling normally returns to `idle`.
Missing, incomplete or checksum-invalid read replies receive at most three attempts,
with stale input discarded before each fresh request. Read replies must match the
requested motor ID and register width, frame length, header and checksum before
decoding; a short write acknowledgement cannot become a register value. Unexpected
center-capture exceptions also enter the stop/restore path. Motor-reported faults are
not retried; persistent errors stop the session. Missing configuration-write
acknowledgements require an exact register read-back, without replaying the write.
Motion, torque, ID and baudrate writes do not use this recovery.

Saved-mapping wiggles verify the six canonical addresses directly (stable encoder
samples, STS3215 model, stored ID and torque-off). They do not repeat address
separation or scan unused addresses; the completed journal is required first.

### Moving to the recorded middle

`POST /v1/control/start` accepts optional `move_to_middle: true` for `mode: manual`.
The browser's **Enable & move to recorded middle** button requests this behavior.
The captured half-turn center is 2047 encoder ticks per joint; it is not the average
of the subsequently captured minimum and maximum. Startup rejects a center outside
the calibrated range, holds a freshly sampled pose with verified goal and torque
read-back, then approaches the center at no more than 4 degrees/second (4 percent
of calibrated travel/second for the gripper). A delayed control tick cannot cause
a catch-up jump. Session status reports `centering.active` and `centering.progress`.
Jogging and recording are blocked until measured positions reach the center.
Stop, hidden-page cancellation, heartbeat expiry, motor faults, range checks and
trajectory timeout remain active. Manual control also stops on tracking deviation
greater than 5 degrees (5 percent for the gripper). The app never expands calibration
limits or raises torque limits to complete a move.

Manual jogs advance the previous requested target, not the measured encoder position. Pending targets cannot exceed 5° (5% for the gripper) ahead of the measured position. A measured tracking error over that threshold stops motion. `session.targets` exposes requested normalized joint values alongside measured `session.positions`.

Manual `/control/start` accepts an optional nonempty `joints` array of unique canonical names. Omitted means all six. In a selected-motor session, startup probing, calibration read-back, torque commands, encoder reads, jogs and Stop address only those selected IDs. Excluded joints are absent from `session.positions` and listed selection is in `session.active_joints`. Selected IDs must respond reliably; their disappearance remains fatal. Hand guiding, leader mode and dataset recording require all six. The UI holds the current pose for partial selection and retains automatic middle movement when all six are selected.

`capabilities.calibration_persistence` means calibration is owned by the local companion, with `session.watchdog_seconds:null` only in unpowered calibration. Powered control retains its two-second heartbeat watchdog. A hidden or reloaded browser may rejoin active calibration without repeating middle capture. `POST /calibration/start {resume:true,confirm:"START CALIBRATION"}` resumes a recoverable checkpoint after verifying motor IDs, saved-profile checksum and motor settings. Explicit Cancel restores the prior calibration and archives the checkpoint; Save archives it after persisted calibration verification.


Manual `/control/jog` also accepts `{joint,target}` instead of `{joint,delta}` for an absolute slider position in degrees (gripper percent). Exactly one of target/delta is required. Targets must lie within `session.joint_limits`, the active calibration with an eight-tick end margin. Releasing a slider or a keyboard adjustment commits one target. The companion interpolates at 4°/s or 4%/s, caps elapsed-time advances to 100 ms, verifies tracking before each target write, and retains the existing heartbeat/Stop cancellation. A new slider target replaces that joint's trajectory from the current commanded position; a delta jog cancels that joint's slider trajectory. `session.slider_targets` shows final desired positions, while `session.targets` remains the actual intermediate command used in recordings. Excluded motors receive no commands. `capabilities.joint_sliders` gates the UI for older companions.


The recorded-middle trajectory has no wall-clock completion deadline. Slow serial cycles continue with the existing 100 ms per-tick cap until all encoders enter the configured middle arrival band. Stop, tab-hidden Stop, the two-second control heartbeat watchdog, calibrated limits and tracking-error stops remain active. Slider trajectories retain their separate deadline.


Manual `/control/start` accepts `middle_tolerance` (finite number 0.5–5, default 2) for `move_to_middle`. Units are degrees for angular joints and percent of calibrated gripper travel. The UI offers 2°/2% and 5°/5%, persists the choice, and locks it while a session is active. `capabilities.middle_tolerance` gates this setting; `session.centering.tolerance` reports its value. Once the scheduled trajectory finishes and every measured joint is inside its band, the companion unlocks controls without changing the stable hold target. Encoder feedback need not reach exact 2047-tick alignment. Existing five-degree/five-percent tracking protection and calibrated limits remain unchanged; the arrival band does not bypass tracking checks.


### Arm readiness

Status advertises `capabilities.arm_readiness` and passive `mapping_checks` metadata
(`port`, mapping journal `revision`, `checks` keyed by joint with `run_id`, `correct`,
`checked_at`, plus `error`). `POST /mapping/confirm` accepts
`{port,joint,run_id,correct:boolean,confirm:"CONFIRM PHYSICAL JOINT"}` only for the
most recent completed, cleaned-up wiggle and unchanged mapping. It stores operator
observations separately from setup journals, never commands hardware, and supports
idempotent same-answer retries. A new wiggle is required to change an observation.

Saved calibration inventory validates all six entries, excludes backup filenames,
and reports `valid`, `sha256`, `setup_port`, `setup_fresh`, `error`. These are file /
setup-revision facts, not a claim that current hardware matches. An explicit
read-only `/probe` may include `robot_id` and `role`; its `profile.state` is
`missing`, `invalid`, `stale`, `unavailable`, `mismatch` or `matched`. Only a stable
six-motor scan plus matching offsets and limits for every joint reports `matched`.
A scan never writes registers or enables torque. Status/device refresh never scans
motors. The UI retains this as a scan snapshot and clears it on USB or identity change.


### Larger joint steps

`capabilities.large_joint_steps` enables 5° and 10° choices in the Step selector
(gripper 5% and 10%). Manual `/control/jog {joint,delta}` accepts finite nonzero
steps up to +/-10. Steps above 2 schedule the existing bounded trajectory at
4°/s (gripper 4%/s) instead of sending a large immediate position jump. Repeated
large steps advance the pending final target and clip to calibrated limits with
the eight-tick margin. Each trajectory starts at the current commanded position;
there is no replay of motor writes. Smaller steps retain the existing direct jog
and five-degree tracking guard. The 100 ms request spacing, Stop cancellation,
heartbeat watchdog, tracking checks, partial-joint selection and intermediate
action values used for recording remain unchanged. This selector sets distance
per click; it does not raise motor speed, torque or tracking tolerances.
