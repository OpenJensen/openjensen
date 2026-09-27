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

## Verification

```sh
uv run --frozen --extra tui pytest tests/test_tui.py tests/test_tui_cli.py tests/test_tui_integration.py -q
```

Pilot tests exercise keyboard navigation, forms, 48×18 terminal resizing, stale/invalid responses, selection changes, offline recovery, explicit cancellation and non-retried ambiguous writes. The integration test starts a disposable loopback API on an ephemeral port, creates a real project, runs actual local metadata intake and a supervised slow protocol fixture, cancels that fixture, reads the same records with the CLI, and restarts the API to verify persistence. It never contacts a model provider or cloud service. Fixture evidence is not training, robot-quality or hardware evidence.

macOS testing is recorded with this implementation. Native Windows terminal behavior and a desktop installer require their own verification; this terminal slice does not establish those claims. [Textual testing](https://textual.textualize.io/guide/testing/) and [workers](https://textual.textualize.io/guide/workers/) describe the upstream UI/test APIs used here.
