# Local restart recovery protocol

The application owns its workspace with an OS-backed file lock before opening
SQLite and reconciling jobs. The API starts accepting work only after startup
reconciliation finishes. A hard process termination releases that ownership;
the next owner does not rely on deleting the `owner.lock` file.

Every persisted `queued` or `running` job becomes terminal `interrupted`, with
its result cleared and an error instructing the user to submit a new inspection.
There is no automatic retry or resume. Reconciliation never starts a worker or
reads an existing `result.json`. Jobs already `succeeded`, `failed`, `cancelled`,
or `interrupted` remain byte-for-byte unchanged as API records, including their
timestamps. A second restart therefore leaves the recovery result unchanged.
If recovery itself is interrupted, another startup finishes any remaining
nonterminal rows before exposing the API.

Only the supervisor publishes a result to SQLite. The metadata worker writes a
bounded per-job JSON file, not application records. `finish()` reloads the
persisted job under the same lock as reconciliation/cancellation and refuses to
modify any terminal job. This also rejects a valid but late successful result.
Job IDs are unique for each explicit submission, so a user retry does not reuse
the old worker's request/result paths.

Graceful shutdown cancels and joins all supervised tasks, terminates remaining
child processes, and then marks unfinished work interrupted. A hard kill can
leave an already spawned metadata child running, particularly on Windows. A
network metadata worker has 20-second HTTP inactivity timeouts and bounded
metadata reads, but the dead supervisor's 90-second wall-clock limit no longer
applies; slow streams or unusual local filesystem stalls can outlive that bound.
The next owner does not attempt unsafe PID-based termination of unknown or
reused processes. Old children cannot publish database results; late result
files remain inert. Old request/result files are retained for inspection, with
no automatic deletion or general orphan-process reclamation implemented.

Reproduce the forced-stop check from the repo root with:

```powershell
.venv/Scripts/python.exe -m pytest -q -s tests/test_restart_recovery.py
```

On Linux use `.venv/bin/python` for the same command. The test starts the actual
CLI application and actual metadata workers, submits work through HTTP, kills
the owner without shutdown, and starts two fresh owners against the same
workspace. Two real Hub workers are held by a loopback HTTP CONNECT gate so a
third job stays queued. The gate never contacts a remote host. It later rejects
CONNECT, allowing orphan workers to write real failure files that recovery must
ignore. A separate local metadata fixture supplies the completed success case;
a missing local path supplies the failed case. These are controlled recovery
fixtures, not live Hub, real robotics corpus, simulator, or GPU evidence.

The companion execution tests check clearing inconsistent nonterminal results,
late success/failure rejection, idempotence, and graceful queued-task cleanup.
