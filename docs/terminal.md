# Terminal workbench

The optional Textual client uses the same application API and persisted projects/jobs as the web UI and JSON CLI. It does not start a server, schedule work independently, configure cloud accounts, or launch training automatically.

From a source checkout with the pinned uv version:

```sh
uv sync --frozen --all-packages --extra tui
uv run --frozen --extra tui firebird serve
```

In another interactive terminal:

```sh
uv run --frozen --extra tui firebird tui
```

The default application URL is `http://127.0.0.1:8000`. Override it with `FIREBIRD_API_URL` or `firebird tui --api-url http://127.0.0.1:8001`. The endpoint accepts HTTP(S), without embedded credentials, query parameters or fragments. This client does not add remote authentication or change the server's host/origin boundaries. A packaged installation can use the `vla-platform[tui]` extra; the lock pins Textual 8.2.8. The ordinary CLI and application do not require Textual.

| Key | Action |
| --- | --- |
| F1 | Projects |
| F2 | Jobs in the selected project |
| F3 | Details and events for the selected job |
| F4 | Dataset intake |
| F5 | New lifecycle recipe |
| F6 | Review and save a policy archive |
| Ctrl+N | Create a project |
| Ctrl+R | Refresh |
| F8 | Review cancellation of the highlighted job |
| Ctrl+Q | Quit the client |
| Up/Down, Enter | Highlight and open a project/job |
| Tab / Shift+Tab | Move between controls |
| Escape | Close a form without submitting |

Choose a project before intake. Hugging Face intake accepts a repository and revision. Local paths refer to the **application host's allowed dataset folder**, not necessarily the computer running the terminal. “Prepare immutable local training copy” requests the existing complete snapshot validator and copy; it does not start training or promise that a dataset satisfies split/admission requirements.

The Jobs view includes intake, training and other native policy jobs submitted through any client. The details view shows the API record, its actual status, results, errors and events. It does not reinterpret job completion as robot task success. Use F5 to prepare a lifecycle recipe in the terminal, or `firebird policy submit PROJECT RECIPE.json` for an existing recipe file.

Reads poll every three seconds. Each request has a 15-second total deadline and an 8 MiB response limit; displayed job details are capped at 120,000 characters with an explicit truncation notice. Offline/error states retain the last data with its last refresh time. No write is retried automatically. If a write times out or returns an uncertain response, inspect the jobs/projects before submitting again. Closing the terminal stops observation, not application jobs.

Cancellation requires an explicit confirmation whose safe default is “Keep job.” The client rereads the selected job before sending cancellation and refuses if its identity, project, selection or active state changed. The resulting status is the server's response; cancellation does not establish that cloud resources have been deleted.

The noninteractive CLI continues to emit JSON to stdout and errors to stderr. `firebird --version` and command help work without the extra. `firebird inspect PROJECT --path DATASET --snapshot-for-training` exposes the same local snapshot request. Recipe files must be regular JSON objects of at most 1 MiB; reads are bounded and named pipes are rejected.

## Review and submit lifecycle recipes

F5 opens an editable composer for **Fine-tune / train**, **ACT teacher-to-student
Distill**, **ACT INT8/INT4 Quantize**, **CPU observation Replay**, **Resume saved
training**, **Export inference policy**, **SmolVLA GGUF quantization**, **Evaluate
engine / LIBERO**, and **Run SmolVLA engine / LIBERO**. It reads the
selected project's saved datasets/artifacts and the application's advertised
runtime/model catalog. An empty selector means there is no currently compatible
choice; the terminal never installs a runtime or invents connectivity.

Choose the workflow and inputs, then use **Build / replace draft**. Fixed adapter
identifiers come from the API; required budgets, training method/model, episodes,
frames, coordinate units and attestations remain null/empty until explicitly
chosen. Use **Edit with labelled fields** for the supported settings. It offers
registered model/method choices, dataset cameras, explicit budgets, episode/frame
selections, coordinate units and evaluation thresholds. The **Advanced: complete
JSON recipe** section remains available for adapter-specific settings without a
separate recipe file. Applying fields validates the draft locally; it sends no
request, invalidates any previous consent and still requires fresh review. Escape
keeps the original draft and review unchanged. Saved settings are preserved;
changing a model/method with advanced training settings requires an explicit new
draft so incompatible settings are not carried silently between adapters.
The model catalog lists registered repository IDs and methods. The application
still enforces model-specific camera, dataset and hardware requirements. **Copy
selected job recipe** copies the entire saved request for editing; it neither
retries nor resumes that job. Use the separate Resume workflow for checkpoint recovery.

Resume selects either an exact registered training checkpoint or an interrupted
training job. **Build / replace draft** retains the original dataset and method;
choose a new execution timeout, not a changed training budget. The review binds
the visible saved lineage and rejects recipe overrides. The server independently
verifies the checkpoint files and restores its saved training recipe, optimizer
and RNG state. With an interrupted job it chooses the latest complete checkpoint;
a failed job may have no usable checkpoint. The terminal does not claim to have
read server-side checkpoint files. Cloud checkpoints require the cloud training
runtime; selecting them does not implicitly materialize a local resumable copy.

Export consumes a training checkpoint. ACT requires its separately configured
local CPU exporter, full-training metadata and pinned Hugging Face dataset lineage;
local-snapshot ACT export is unsupported. A completed, reload-verified cloud ACT
checkpoint can be materialized by the existing backend (up to 4 GiB) before export.
SmolVLA export requires its local training adapter and a local checkpoint. Export
does not establish task success or simulator compatibility.

Engine Evaluate/Run require a compatible SmolVLA GGUF or deployment package.
Choose explicit `evaluation.mode`, `suite`, `warmups` and `repetitions`. Engine
mode measures inference behavior/timing and cannot score a robot task. Choose the
protocol selector before **Build / replace draft** to expose its required fields.
For LIBERO,
also supply `task_id` (Object) or `task_ids` (Spatial), disjoint `initial_states`
and `final_states`, `seed` and `steps`. Spatial requires a CUDA simulator, its
280-step horizon and explicit `parity_limits`; the artifact must declare the same
suite. Google Cloud currently supports engine checks, not LIBERO scoring. These
forms do not add ACT/Isaac scoring support. GGUF quantization chooses an explicit
`precision.language`; native packed ACT uses its distinct workflow.

**Review exact recipe** rereads current context, validates the existing API
schema, and shows the complete normalized recipe plus source manifest and compute
identity. Review and consent are invalidated by an edit. Submission rereads the
source/dataset/runtime again and refuses changed context. Authorize the exact
source/recipe/compute explicitly; Google Cloud also requires a separate paid-job
checkbox. Timeouts are not spending caps. Navigating, copying or reviewing never
submits work. Each submission sends one POST and validates its acknowledgment.

The client stores a small private attempt record per API endpoint/project under
`$XDG_STATE_HOME/openjensen/tui` (default `~/.local/state/openjensen/tui`) or
`%LOCALAPPDATA%/OpenJensen/state/tui` on Windows. Records contain attempt/recipe/context
hashes and operation/runtime identities, not full recipes or credentials. On POSIX
the directory is mode0700 and records mode0600. An OS lock prevents another TUI
from clearing an active submission. A pending or uncertain record survives client
closure/restart and blocks another submission. Storage errors also block writes.
This is process-restart recovery, not a hardware power-loss guarantee: file and
journal-directory changes are flushed, but first-use ancestor-directory creation
is not independently certified durable through a system crash.

To resolve uncertainty, reopen F5 for the same endpoint/project, load fresh saved
job history, inspect it for the displayed exact attempt/recipe, then explicitly
acknowledge that attempt. Clearing the block does not submit anything; every new
request still needs review and consent. No client-generated idempotency key is
promised by the server. If a server acknowledgment arrives but local cleanup fails,
its accepted job ID remains visible and **Back to jobs** follows it; do not submit
it again. Closing the terminal does not cancel an accepted application job.

These recipe paths expose existing backend adapters. CPU replay is not an
Isaac rollout or task-success evaluation. Quantization reports drift/reload evidence,
not calibration, hardware performance or robot quality. The labelled editors cover
the nine workflows above; adapter-specific options remain in the advanced recipe.
Resume and Export expose only the execution timeout in the labelled editor,
preserving their source identities and saved training method.

## Save an artifact from the terminal

F6 opens the selected project's registered policy artifacts. Choose a policy, an
explicit **new local output file**, a byte limit and a deadline. Optionally enter an
independently known archive SHA256; the registered manifest hash is a different
identity. The destination directory must already exist. The default bounds are
4 GiB and 600 seconds, and the archive includes packaging overhead.

**Review download** reloads the project-owned registry and shows the exact artifact,
manifest, destination and bounds. **Save archive** explicitly starts one transfer
over the same connection as the terminal. It refuses a registry changed since review,
then uses the CLI's existing streaming checks, post-transfer registry comparison and
atomic no-overwrite publication. It neither extracts the TAR nor changes a model.

**Stop transfer** or Escape during a transfer cancels only that download, removes its
own partial output and leaves application jobs running. Wait for cleanup, then use
Escape again to close the form. Quitting drains the transfer before closing the
terminal connection. No transfer is retried automatically. A successful receipt
retains the local path, received bytes, archive checksum and registered manifest;
it does not certify model quality or archive contents. Network deadlines do not
bound synchronous disk writes/flushes. The noninteractive command below remains
available without Textual.

## Follow lifecycle jobs from the CLI

Existing commands remain noninteractive and emit JSON. The application remains the
only scheduler: `policy submit` can submit the current fine-tuning, native ACT
distillation/quantization/replay, or supported engine recipe; the server validates
the complete recipe and configured capability. A successful submission is a job
receipt, not completed training or robot-task acceptance.

```sh
firebird policy options
firebird policy submit PROJECT recipe.json
firebird jobs wait JOB --timeout 600 --interval 2
firebird jobs events JOB
firebird policy artifacts PROJECT
firebird policy download PROJECT ARTIFACT --output ./new-policy.tar
```

Replace `PROJECT`, `JOB` and `ARTIFACT` with the IDs in the actual API records.
Waiting sends only GET requests. It prints the final job record and exits `0` for
`succeeded`, `1` for `failed`, `cancelled` or `interrupted`, and `124` when its
observation deadline expires. The timeout range is 1–86400 seconds; polling is
0.1–30 seconds. An unavailable/invalid API response stops the wait with an error.
Ctrl+C exits the client (`130`) without cancelling the job. Cancellation remains a
separate explicit `firebird jobs cancel JOB` command.
Interrupting a write, including a cancellation request, has an unknown outcome:
the server may already have accepted it. Inspect the saved job/project before
submitting that write again. A redirect after a write is also treated as uncertain
and is never followed or retried automatically.

Downloads use an explicit **new file on the CLI computer**, never a server path.
The parent directory must exist. The command resolves the project-owned registry
record, streams a TAR response to a private temporary file, and rechecks that the
registered artifact is unchanged before publishing without replacement. Existing
files and symlinks are rejected, including a destination created during transfer.
Failures/cancellation remove only the command's own partial file. No archive is
extracted and no registered source is modified.

The default byte limit is 4 GiB and the deadline is 600 seconds. Adjust them explicitly
with `--max-bytes BYTES` (1 byte through 100 GiB) and `--timeout SECONDS` (1–3600).
Both declared and actual response lengths are checked; redirects, compressed HTTP
responses and unexpected media types are refused. An independently known archive
checksum can be required with `--expected-sha256 HEX`. The deadline bounds async
network observation; synchronous local writes and filesystem flushes can take
longer before control returns. It is not a hard filesystem-operation time limit.

The receipt reports received bytes, their SHA256 and the unchanged registered
manifest identity. The archive checksum and registered manifest checksum are
**different identities**. Without `--expected-sha256`, the receipt's checksum is a
measurement of received bytes, not comparison against an independent source. The
CLI does not independently certify archive contents, runtime compatibility,
calibration or task quality.

All CLI JSON requests share the terminal transport's endpoint validation,
15-second total request deadline and 8 MiB response cap, and ignore proxy environment
variables. HTTP redirects are not successful responses. A lost/invalid response or
server error after a POST reports an unknown outcome and never automatically
repeats that POST; inspect the recorded jobs/projects before making another request.
Successful write acknowledgments must match the project/job, operation and stable
input identities. The server may resolve dataset branches/local paths or restore
checkpoint-owned training settings; the CLI does not invent those identities.

## Verification

```sh
uv run --frozen --extra tui pytest tests/test_tui.py tests/test_tui_cli.py tests/test_tui_integration.py tests/test_cli_client.py tests/test_cli_workflow.py tests/test_cli_ack_types.py tests/test_tui_lifecycle.py tests/test_tui_lifecycle_integration.py tests/test_tui_policy_modes.py tests/test_tui_policy_api.py tests/test_tui_artifacts.py -q
```

Pilot tests exercise keyboard navigation, forms, 48×18 terminal resizing, stale/invalid responses, selection changes, offline recovery, explicit cancellation and non-retried ambiguous writes. The integration test starts a disposable loopback API on an ephemeral port, creates a real project, runs actual local metadata intake and a supervised slow protocol fixture, cancels that fixture, reads the same records with the CLI, and restarts the API to verify persistence. It never contacts a model provider or cloud service. Fixture evidence is not training, robot-quality or hardware evidence.

Lifecycle Pilot cases additionally exercise complete recipes for all four workflows,
changed ownership/capabilities, separate cloud consent, persisted uncertainty and
failed attempt storage. A second integration test uses the actual application API
and its supervised CPU protocol fixture through an in-process HTTP adapter; it
verifies one explicit quantization submission and persisted results without real
model inference. This is application protocol evidence, not an ML benchmark.

macOS testing is recorded with this implementation. Native Windows terminal behavior and a desktop installer require their own verification; terminal checks do not establish those claims. [Textual testing](https://textual.textualize.io/guide/testing/) and [workers](https://textual.textualize.io/guide/workers/) describe the upstream UI/test APIs used here.
