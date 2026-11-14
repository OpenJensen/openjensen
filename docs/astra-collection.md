# Supervised AI collection through Pi

Open **Robot Lab → AI collection** inside a project. Start with **Practice run**:
it runs the pinned Pi Agent SDK with a deterministic model, a synthetic SO101
bus and two moving synthetic camera views. It never opens USB or calls an API.
The resulting private dataset appears in the project dataset library for normal
review/replay and ZIP export. Practice-run artifacts are explicitly synthetic.

For hardware, update the USB computer's Robot Lab companion from this release,
select the connected calibrated SO101 in Teleoperation and enable full-arm joint
control. Configure the OpenAI API key in existing Settings. Choose **Real arm**,
pick the AI model and the wrist and side cameras, then tick operator presence and
workspace clearance. The page offers the current image-capable models and accepts
any other model ID; the model is called through the Responses API with low
reasoning effort. Subscriptions are not API credentials. Account and model access
remain a live-run check.

Every proposed physical segment is displayed for approval. Check the scene and
the joint targets before approving. The executor reobserves locally, checks the
model observation identity and pose drift, and validates calibrated bounds,
speed, acceleration, tracking, authority and freshness. Model observations older
than 60 seconds cannot be approved; local execution observations expire after
one second. This is bounded joint-position interpolation, not Cartesian planning,
collision avoidance or force control. Stop if the model cannot safely identify
the next small movement. A successful pick–place requires supervised acceptance.

Keep the page visible. Its 400 ms heartbeat expires after 1.5 seconds in Core;
the companion's independent two-second lease and local-health watchdog remain in
force. Stop collection cancels the runtime and releases the agent controller.
The Robot Lab stop and physical stop remain available. No motion resumes after
a process restart. Each run has immutable command IDs: duplicate tool calls reuse
their result, and the existing companion deduplicates executions durably.
Ambiguous POSTs are reconciled by GET receipt; physical POSTs are never replayed.
Provider retries are disabled. Run steps, elapsed time, output tokens per request
and cumulative reported tokens are bounded. The cumulative token threshold is
checked between requests; a single request's input usage can cross it.

## Runtime and deployment

Pi packages are pinned at 1.5.1 in workers/pi_collection/package-lock.json.
Use Node 24 and npm ci --ignore-scripts in that directory. Set JENSEN_PI_NODE to
the Node executable; JENSEN_COLLECTION_PYTHON (or FIREBIRD_CPU_READER_PYTHON)
selects an existing Python environment with numpy, Pillow and pyarrow for dry runs.
No Pi fork, coding-agent shell, filesystem tool, OAuth subscription or arbitrary
tool is exposed. The replaceable worker uses NDJSON over private process pipes.
Keys are passed only over stdin and are excluded from persisted configuration.

The collection API is project-scoped under /robot-lab/collection-runs so the
existing public gateway denies it. Runs and exposed model/tool traces live in
the private workspace's agent-collection directory. The companion stores datasets
under its existing datasets directory. Import uses the existing hash-verified
private Robot Lab importer. The original recordings are retained on the companion.

## Dataset and audit contract

Each run currently produces one episode. LeRobot v3 exports contain
observation.images.wrist and observation.images.side as lossless RGB image
features, observation.state and action with six values in the documented
SO101 joint order. Arm joints use degrees; gripper uses percent, never radians.

- evidence/recording.json: calibration, camera-role mapping, clock origin,
  configuration, completion/failure state and synchronization semantics.
- evidence/controller.jsonl: every successfully returned local control tick,
  measured state, actual sent setpoints/raw ticks, command ID, encoder sample
  time and transport-write-return time. A returned write is not claimed to be
  a per-motor acknowledgment; measured completion is a separate command receipt.
- evidence/frames.jsonl: each paired sample's raw monotonic camera timestamps,
  assembly timestamp, controller sample ID and active command ID.
- evidence/trace.jsonl: run/episode/tool/command IDs, exposed model outputs,
  requests, validation observations, terminal receipts, interventions and usage.
  It contains no hidden chain of thought.
- evidence/observations: exact model input images and validation observation
  images with hashes and state metadata. The additional side model image is
  explicitly timestamped on Core; it is not claimed to be simultaneous with the
  wrist model snapshot. Training pairs use companion camera timestamps.
- raw/: original PNG frames and rows, preserved before Parquet finalization.

Camera timestamps are host read completions, not verified sensor exposure times.
Training pairs reject camera skew above 150 ms, camera staleness above 500 ms,
and measured-state staleness above 250 ms. The regular LeRobot timestamp is
frame_index/fps; raw irregular timestamps remain available for later compilation.
Frames map to the latest successful controller sample before assembly. Holding,
contact, settling and thinking intervals are retained. Requested/rejected actions
are trace events, never training action labels.

A missing/stale camera, writer failure or evidence queue overflow stops the arm.
Partial episodes with at least two frames are finalized and remain clearly marked
failed/stopped. Shorter or crashed captures remain raw. Interrupted runs are marked
on reload and never resumed automatically; recover pending data on the companion
before creating a new recording. There is no automatic scene reset or training.

## Verification boundary

Automated acceptance exercises Pi itself, fake transport, both cameras, recording,
trace linkage, persistence, importer and export. Existing agent safety regressions
cover stale observations, command retries, authority and stop races. This establishes
software behavior only. Real camera load/timing, calibration, mechanical motion,
tracking tolerances, API account access and one physical pick–place remain
supervised hardware acceptance checks.
