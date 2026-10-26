# Use the terminal UI and CLI

The terminal UI is a keyboard-driven view of projects, jobs, dataset intake and recipes. The `jensen` CLI covers the same jobs for scripts. Both talk to a running OpenJensen server.

## Start the terminal UI

Install the optional terminal dependencies and start the server:

```sh
uv sync --frozen --all-packages --extra tui
uv run --frozen --extra tui jensen serve
```

In a second terminal:

```sh
uv run --frozen --extra tui jensen tui
```

For a server at another address, set `JENSEN_API_URL` or run `jensen tui --api-url http://127.0.0.1:8001`.

| Key | Action |
| --- | --- |
| F1 / F2 / F3 | Projects / Jobs / Job details and events |
| F4 | Dataset intake |
| F5 | Review a recipe |
| F6 | Save a policy archive |
| F8 | Review cancelling the selected job |
| Ctrl+N / Ctrl+R | Create a project / Refresh |
| Up/Down, Enter | Highlight and open a project or job |
| Tab / Shift+Tab | Move between controls |
| Escape | Close the current form |
| Ctrl+Q | Quit |

## Review and submit a recipe

Install the workers the workflow needs first; see [policy setup](policy-workflow.md#configure-the-execution-host).

1. Select a project and press F5.
2. Choose the workflow and its registered inputs.
3. Select **Build / replace draft**, then edit the labelled fields or the full JSON recipe.
4. Fill in what the workflow needs: budgets, model and method, observations and thresholds.
5. Select **Review exact recipe**, check the source, compute and consent, and submit.
6. Follow the accepted job in F2 and F3.

- **Resume:** select a saved checkpoint or interrupted training job and set a new execution timeout.
- **Export:** select the checkpoint and the configured CPU exporter.
- **Uncertain submission:** if a response was lost, reopen F5 for the same server and project, read the fresh job history and reconcile the shown attempt before submitting again. Attempt records live in `$XDG_STATE_HOME/openjensen/tui` (default `~/.local/state/openjensen/tui`) or `%LOCALAPPDATA%/OpenJensen/state/tui` on Windows.

## Save an artifact

Press F6, select the registered artifact and a new file in an existing directory. Set the size limit and deadline (defaults 4 GiB and 600 seconds), optionally add an independent archive SHA-256, then select **Review download → Save archive**. **Stop transfer** or Escape stops the download.

## Follow lifecycle jobs from the CLI

Replace `PROJECT`, `JOB` and `ARTIFACT` with the IDs the commands return:

```sh
jensen policy options
jensen policy submit PROJECT recipe.json
jensen jobs wait JOB --timeout 600 --interval 2
jensen jobs events JOB
jensen jobs cancel JOB
jensen policy artifacts PROJECT
jensen policy download PROJECT ARTIFACT --output ./new-policy.tar
```

| Option | Range |
| --- | --- |
| `jobs wait --timeout` | 1–86400 seconds |
| `jobs wait --interval` | 0.1–30 seconds |
| `policy download --max-bytes` | 1 byte to 100 GiB |
| `policy download --timeout` | 1–3600 seconds |
| `policy download --expected-sha256 HEX` | An independent checksum, when you have one |

Download to a new file, then check its path, size and checksum. If a write is interrupted or its response is lost, read the saved job or project before submitting again, and refresh F2 and F3 after reconnecting.
