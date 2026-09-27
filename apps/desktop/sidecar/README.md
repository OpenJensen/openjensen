# Python sidecar foundation

This is the first DESK-002 Python boundary. Normal-interpreter tests exercise the actual
application and its intake worker in disposable workspaces. The Tauri shell does not
start this sidecar yet. No frozen build, relocated application, signing, distribution,
Windows support, or packaged model runtime has been verified by this slice.

## Fixed modes and ownership

`entrypoint.py` accepts only:

- `serve --resources ABSOLUTE_DIRECTORY --data-dir ABSOLUTE_DIRECTORY
  [--local-root ABSOLUTE_DIRECTORY]`
- `intake-worker ABSOLUTE_JOB_REQUEST_JSON ABSOLUTE_JOB_RESULT_JSON`

The second mode requires the same job directory and the exact names `request.json` and
`result.json`. It calls the existing metadata intake worker. Source installations keep
exactly their existing `python -m vla_platform.datasets.worker` command; a frozen
installation uses the same executable with the fixed `intake-worker` mode. There is no
user-selectable module, script, shell, `-m`, or `-c` execution mode.

The future native owner supplies absolute trusted paths and private standard pipes.
It must also construct a minimal environment instead of forwarding parent credentials.
`serve` removes child-local `FIREBIRD_*` and `GEMINI_API_KEY` environment entries
before API imports, including optional Decision/Teaching/augmentation activation settings.
The parent environment and persistent configuration files are never modified. It ignores
the working directory for application paths.
Application data and bundle resources must be disjoint. Local dataset access is disabled
unless that owner supplies an explicit local root. This slice does not configure or
activate model, cloud, provider, simulation, or training runtimes.

The existing API lifespan retains its workspace `owner.lock`, database migrations,
recovery and job shutdown behavior. Opening an existing workspace can therefore run
its existing recovery logic; it is not a read-only attach operation. An upgrade backup
and schema compatibility policy remains required before a packaged release adopts
existing data. External-backend attachment must continue using the current Tauri
connection path and must never send this private shutdown protocol to that server.

## Private control protocol

The parent must send one UTF-8 JSON line within ten seconds:

```json
{"schema_version":1,"command":"start","nonce":"0123456789abcdef0123456789abcdef"}
```

The nonce is 32 lowercase hexadecimal characters. It correlates messages on the owned
private pipe; it is not an HTTP credential. The child validates bundled file hashes,
binds its own ephemeral `127.0.0.1` socket, and starts the real API. It emits `ready` only
after successful lifespan startup, including the nonce, loopback port, app version,
source build ID, resource-manifest SHA256 and an opaque workspace-path digest. It does
not emit private filesystem paths on stdout. Logs go to stderr.

To stop, send `{"schema_version":1,"command":"shutdown","nonce":"..."}` with the
same nonce, or close the pipe. Malformed, oversized, or mismatched control messages
also initiate shutdown and cause failure. Each control record is at most 4 KiB; only
start and shutdown are consumed. A parent that disappears before readiness receives
no late ready announcement. Startup polling has a 60-second limit, but synchronous
imports and filesystem validation are not preempted by that poll timer.

Shutdown awaits the existing application cleanup and releases its workspace lock.
A `stopped` record is emitted only afterward. Uvicorn limits HTTP task draining to five
seconds; this does **not** impose a hard limit on the application's lifespan cleanup.
The future native supervisor must implement a separate final owned-child deadline,
reaping, crash reporting, and restart policy. This Python foundation does not claim
arbitrary nested detached process cleanup after a forced kill.

## Static resources and future build

`prepare_resources.py --web ABSOLUTE_STATIC_EXPORT --output NEW_ABSOLUTE_DIRECTORY
--build-id EXACT_40_CHARACTER_COMMIT` copies an already-built static export and records
every file's SHA256 and byte count. It rejects links/special files, missing index,
more than 10,000 files, and more than 512 MiB total payload. Resource verification uses
64 KiB chunks. Existing outputs are never replaced; interrupted build directories
remain for inspection. The manifest is limited to 4 MiB. Runtime validation checks an
exact file inventory before serving. Bundle immutability after that check relies on
the installed application's filesystem boundary; this is not a hostile-writer sandbox.

`firebird-sidecar.spec` is an unexecuted one-directory PyInstaller build specification.
It includes the bounded resources plus real core Python files needed by Alembic and
fixed helper subprocesses. A future reviewed build should use an isolated build-only
environment with the existing core lock and a pinned freezer, then test from a new
location with no repository or system Python/Node access. No freezer was installed for
this slice. The complete one-directory payload, including `_internal`, must be shipped
as a Tauri resource; copying just the executable is insufficient. Model runtimes stay
separate. Native start/stop UI and capability changes are deliberately a later review.

## Opt-in macOS ARM64 frozen experiment

The packaging tools in this directory are an experimental verification path. The
production Rust payload pin remains absent, and the desktop Start control remains
disabled. A prepared spec or a passed harness unit test is not a frozen-package result.

`requirements-build-macos-arm64.txt` pins six prebuilt build-tool wheels by exact public
URL and SHA256. Use a new isolated CPython 3.14 build environment, `--require-hashes`,
`--no-deps`, and the exact macOS ARM64 production core dependency closure selected from
`uv.lock`. Do not use the development/TUI/ML extras or modify an existing environment.
The spec makes core source discoverable before dynamic hook collection and explicitly
targets ARM64 on macOS; no other platform build is established by this experiment.

Prepare a resource directory from the independently verified static-export inputs.
When an existing export retains older hashed assets, copy only the files in its verified
build manifest first. Record the source input hashes, static output hashes, tool wheel
hashes, installed versions, spec hash and build log separately. The manifest build ID
identifies the core/static source; packaging-source changes must have their own binding.

Run PyInstaller's fixed `firebird-sidecar.spec` with `FIREBIRD_DESKTOP_RESOURCES` pointing
to that absolute directory and fresh scratch work/dist directories. Keep the complete
one-directory output. `payload_inventory.py PAYLOAD OUTPUT_JSON` records every regular
file's bytes, SHA256, permissions and Mach-O CPU types, all directories, and literal
safe relative symlink targets. It rejects escaping, absolute, dangling or cyclic links
and special entries. Limits are 20,000 entries and 2 GiB. This is evidence for a trusted
same-user build tree, not a hostile-writer sandbox or a production payload pin.

The opt-in acceptance command is:

```text
python apps/desktop/sidecar/verify_frozen.py \
  --payload /absolute/frozen/firebird-sidecar \
  --output /absolute/new/experiment \
  --build-id EXACT_SOURCE_COMMIT \
  --resource-sha256 EXACT_RESOURCES_JSON_SHA256
```

It copies the complete payload to a new location preserving links, checks every native
file with the fixed system `otool`, and rejects unresolved/private absolute dependencies.
It runs the executable from outside the repository with only disposable HOME/TMPDIR and
a PATH containing no executables; no PYTHONPATH, provider configuration or credentials
are inherited. Actual HTTP checks cover every recorded static file, health, migrations,
SQLite integrity, a generated metadata-only intake, duplicate workspace-owner refusal,
normal stop, parent EOF and same-build restart preserving the project/job. Negative checks
cover static tampering, a wrong control nonce and generic Python invocation. This does
not decode robotics video or execute a model.

The harness has a 600-second outer deadline, 90-second readiness/intake observer bounds,
30-second orderly child-exit bounds and bounded TERM/KILL cleanup. It reconciles its owned
process group even when the direct child already exited; only kernel-confirmed absence
is cleanup success. An unresolved cleanup error prevents a successful receipt. These
are bounded experiment limits, not startup or latency guarantees.

A receipt is written only after all checks and final original/relocated inventory checks
pass. Retain initial failures and the complete logs. Even a passing local experiment does
not establish Tauri-owned packaged lifecycle, upgrades, existing workspace adoption,
clean-machine installation, notarization, other platforms or separate model runtimes.
Those gates remain open before production payload activation.
