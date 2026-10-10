# Run the terminal workbench

## Install and start

From the source checkout, install the optional TUI environment and start the API:

```sh
uv sync --frozen --all-packages --extra tui
uv run --frozen --extra tui firebird serve
```

In a second interactive terminal:

```sh
uv run --frozen --extra tui firebird tui
```

Set `FIREBIRD_API_URL` or use `firebird tui --api-url http://127.0.0.1:8001` for a different API address.

| Key | Action |
| --- | --- |
| F1 / F2 / F3 | Projects / Jobs / Job details and events. |
| F4 | Dataset intake. |
| F5 | Review a lifecycle recipe. |
| F6 | Save a policy archive. |
| Ctrl+N / Ctrl+R | Create project / Refresh. |
| F8 | Review cancellation of the selected job. |
| Ctrl+Q | Quit. |
| Tab / Shift+Tab | Move between controls. |
| Up/Down, Enter | Highlight and open a project/job. |
| Escape | Close the current form. |

## Review and submit lifecycle recipes

1. Select a project and open F5.
2. Choose the workflow and registered inputs.
3. Select **Build / replace draft**, then edit the labelled fields or full JSON recipe.
4. Supply budgets, model/method, observations and thresholds required by that workflow.
5. Select **Review exact recipe**, review source/compute and consent, then submit.
6. Open F2/F3 to follow the accepted job.

For Resume, select a saved checkpoint/interrupted training job and set the new execution timeout. For Export, select the checkpoint and configured CPU exporter. Install the required workers using [policy setup](policy-workflow.md) before submitting.

For an uncertain submission, reopen F5 for the same endpoint/project, read fresh job history and reconcile the displayed attempt before another request. Attempt records are stored under `$XDG_STATE_HOME/openjensen/tui` (default `~/.local/state/openjensen/tui`) or `%LOCALAPPDATA%/OpenJensen/state/tui`.

## Save an artifact from the terminal

Open F6, select the registered artifact and a new local destination file in an existing directory. Set the byte limit/deadline, optionally supply an independent archive SHA256, then select **Review download → Save archive**. Use **Stop transfer** or Escape to stop the download. Defaults are 4 GiB and 600 seconds.

## Follow lifecycle jobs from the CLI

```sh
firebird policy options
firebird policy submit PROJECT recipe.json
firebird jobs wait JOB --timeout 600 --interval 2
firebird jobs events JOB
firebird policy artifacts PROJECT
firebird policy download PROJECT ARTIFACT --output ./new-policy.tar
```

Replace `PROJECT`, `JOB` and `ARTIFACT` with returned IDs. Use `firebird jobs cancel JOB` to cancel. For `jobs wait`, set `--timeout` from 1–86400 seconds and `--interval` from 0.1–30 seconds.

For downloads, use `--max-bytes` from 1 byte to 100 GiB, `--timeout` from 1–3600 seconds, and `--expected-sha256 HEX` when you have an independent checksum. Choose a new destination file and inspect the returned path/byte/checksum receipt after the transfer.

If a write is interrupted or its response is lost, read the saved job/project before submitting again. Refresh F2/F3 after reconnecting to the API.
