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
| Ctrl+N | Create a project |
| Ctrl+R | Refresh |
| F8 | Review cancellation of the highlighted job |
| Ctrl+Q | Quit the client |
| Up/Down, Enter | Highlight and open a project/job |
| Tab / Shift+Tab | Move between controls |
| Escape | Close a form without submitting |

Choose a project before intake. Hugging Face intake accepts a repository and revision. Local paths refer to the **application host's allowed dataset folder**, not necessarily the computer running the terminal. “Prepare immutable local training copy” requests the existing complete snapshot validator and copy; it does not start training or promise that a dataset satisfies split/admission requirements.

The Jobs view includes intake, training and other native policy jobs submitted through any client. The details view shows the API record, its actual status, results, errors and events. It does not reinterpret job completion as robot task success. Training recipes and other policy operations can still be submitted through `firebird policy submit PROJECT RECIPE.json`; this first terminal slice does not include a training recipe editor.

Reads poll every three seconds. Each request has a 15-second total deadline and an 8 MiB response limit; displayed job details are capped at 120,000 characters with an explicit truncation notice. Offline/error states retain the last data with its last refresh time. No write is retried automatically. If a write times out or returns an uncertain response, inspect the jobs/projects before submitting again. Closing the terminal stops observation, not application jobs.

Cancellation requires an explicit confirmation whose safe default is “Keep job.” The client rereads the selected job before sending cancellation and refuses if its identity, project, selection or active state changed. The resulting status is the server's response; cancellation does not establish that cloud resources have been deleted.

The noninteractive CLI continues to emit JSON to stdout and errors to stderr. `firebird --version` and command help work without the extra. `firebird inspect PROJECT --path DATASET --snapshot-for-training` exposes the same local snapshot request. Recipe files must be regular JSON objects of at most 1 MiB; reads are bounded and named pipes are rejected.

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

All CLI JSON requests now share the terminal transport's endpoint validation,
15-second total request deadline and 8 MiB response cap, and ignore proxy environment
variables. HTTP redirects are not successful responses. A lost/invalid response or
server error after a POST reports an unknown outcome and never automatically
repeats that POST; inspect the recorded jobs/projects before making another request.
Successful write acknowledgments must match the project/job, operation and stable
input identities. The server may resolve dataset branches/local paths or restore
checkpoint-owned training settings; the CLI does not invent those identities.

## Verification

```sh
uv run --frozen --extra tui pytest tests/test_tui.py tests/test_tui_cli.py tests/test_tui_integration.py tests/test_cli_client.py tests/test_cli_workflow.py -q
```

Pilot tests exercise keyboard navigation, forms, 48×18 terminal resizing, stale/invalid responses, selection changes, offline recovery, explicit cancellation and non-retried ambiguous writes. The integration test starts a disposable loopback API on an ephemeral port, creates a real project, runs actual local metadata intake and a supervised slow protocol fixture, cancels that fixture, reads the same records with the CLI, and restarts the API to verify persistence. It never contacts a model provider or cloud service. Fixture evidence is not training, robot-quality or hardware evidence.

macOS testing is recorded with this implementation. Native Windows terminal behavior and a desktop installer require their own verification; this terminal slice does not establish those claims. [Textual testing](https://textual.textualize.io/guide/testing/) and [workers](https://textual.textualize.io/guide/workers/) describe the upstream UI/test APIs used here.
