# OPEN JENSEN desktop workbench

This Tauri 2 client opens the existing Python-served static OPEN JENSEN web app. This build still requires an existing application: **Start desktop workspace is unavailable because no frozen Python payload is pinned or bundled.** The native ownership implementation and private protocol are a separate, testable foundation for DESK-002; they do not establish clean-machine installation. Attached applications and cloud jobs remain running when the desktop closes.

## Run and build

Use Node 24.21.x, pnpm 12.6.0 and Rust 1.98.1 with rustfmt/clippy and your platform's native Tauri prerequisites. On macOS, Xcode command-line tools and the macOS SDK are required. The desktop dependency set has its own Cargo and pnpm lockfiles; it does not change the root web workspace. The desktop workspace keeps a 24-hour minimum dependency release age, with reviewed exceptions for the 12 exact Tauri CLI/platform 2.12.0 packages listed in `pnpm-workspace.yaml`; later versions are not exempt. Their npm registry integrity values were checked against the lockfile.

```sh
cd apps/desktop
pnpm install --frozen-lockfile --ignore-scripts
pnpm exec tauri dev
```

Start the existing application separately with `firebird serve`, after building the shared web interface using the repository's normal setup. The connection screen checks both `/api/v1/health` and the root HTML page before opening it. API-only servers, wrong application versions, redirects, connection failures and oversized responses are rejected with a visible retry state. The packaged frontend here is only the connection screen; the full static Next interface continues to be served by the existing application.

The default endpoint is `http://127.0.0.1:8000/`. To use another port, set `FIREBIRD_DESKTOP_PORT` to a decimal integer from 1 to 65535 before launching the desktop executable. Hosts, URLs, credentials and path prefixes are not accepted as desktop configuration. The native **OPEN JENSEN → Connection** menu (Cmd/Ctrl+Shift+R) returns to the connection screen after a backend restart. No backend restart is triggered by that menu.

```sh
# macOS application bundle; Node is a build dependency, not a runtime requirement.
pnpm exec tauri build --bundles app
# The executable also supports a bounded, read-only readiness check.
src-tauri/target/release/firebird-desktop --self-check
```

The local bundle is `src-tauri/target/release/bundle/macos/OPEN JENSEN.app`. A locally built application is not a signed/notarized public distribution or proof of clean-machine installation. Windows/Linux packages and runtime acceptance remain unverified until those operating systems are tested. No updater, installer service or autostart is installed.

## Ownership and permissions

- Five fixed native commands provide status, the existing loopback probe, and owned start/stop/restart. The only ownership input is explicit workspace confirmation; no command accepts an executable, path, shell, address or arbitrary Python mode. The external probe never follows redirects or environment proxy settings. Each HTTP read has a two-second timeout and byte bound.
- Only the bundled connection page receives these command permissions. The HTTP-served workbench has no Tauri remote capability grant; the command also verifies the invoking window's current bundled origin.
- Top-level navigation is restricted to the configured loopback origin and the bundled connection page. New windows and off-origin links are blocked. Use the ordinary browser for external documentation. No shell, filesystem, process, HTTP plugin or arbitrary execution bridge is exposed to web content.
- macOS microphone purpose text explains the explicitly chosen voice teaching flow. This metadata does not grant permission or prove native microphone, speech-provider or robot-control operation.
- Existing cloud/Hugging Face/voice settings remain owned by the Python application. No credentials or configuration files are copied into the bundle.
- A healthy loopback service is not cryptographic server authentication. This slice relies on the same trusted local-machine boundary as the existing application.

## Verification

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml
node --test tests/connection.test.cjs
```

Rust tests exercise endpoint policy, successful API/static-page probing, bad and oversized responses, redirects, refused connections, the permission manifest, and private owned-child protocol/workspace/cleanup behavior. Test-only Unix shell fixtures cannot be selected by production commands. Native window checks must additionally cover connection, a stopped-server retry, blocked navigation, the native Connection menu and backend health after the desktop closes. Unit tests alone do not prove native runtime behavior or an installable Python sidecar.

## Verified upstream basis

Verified stable releases on 2026-09-27: Rust 1.98.1, Tauri crate/CLI 2.12.0, tauri-build 2.7.0. Registry metadata and exact resolved dependencies are separate from application acceptance.

- [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)
- [Next static export integration](https://v2.tauri.app/start/frontend/nextjs/)
- [Tauri capabilities and application command permissions](https://v2.tauri.app/security/capabilities/)
- [Tauri application configuration](https://v2.tauri.app/reference/config/)
- [Python sidecar mechanism and platform-specific binaries](https://v2.tauri.app/develop/sidecar/)
- [macOS distribution](https://v2.tauri.app/distribute/macos-application-bundle/)

## Owned backend foundation (not a packaged release)

Production startup is fail-closed at `PRODUCTION_PAYLOAD = None`. Merely placing an executable in a resource directory cannot enable it. A future separately reviewed payload must pin the complete frozen build and supply the exact executable/resource identities; the present resolver checks the fixed executable and static manifest, while the child verifies every static file before ready. This is not independent verification of arbitrary `_internal` dependencies. A [local frozen-sidecar experiment](sidecar/README.md#recorded-local-experiment) now verifies the Python payload separately; no Tauri bundle resource mapping, installer or production runtime activation is included yet.

The fixed intended resource layout is `sidecar/firebird-sidecar/firebird-sidecar` (with `.exe` on Windows) plus the complete `_internal` tree. Native code resolves it from Tauri's resource directory, never PATH, current working directory or browser input. The child receives a cleared environment with only OS runtime temporary-directory/SystemRoot fields retained; application/provider/cloud and Python override variables are excluded.

Start requires explicit confirmation. It may create only `app_local_data_dir()/workspaces/desktop-v1` and a bounded private ownership marker. An existing unmarked directory, symlink, foreign owner or different build/resource identity is refused and preserved. There is no implicit adoption of `.firebird`, no copying of credentials, and no automatic cross-build migration. A recognized same-build workspace retains the existing core database and owner-lock behavior.

Each owned start uses a fresh random 128-bit nonce and an increasing in-process generation. Private pipe messages are strict UTF-8 JSON, at most 4 KiB each. Ready must match the nonce, build, application version, manifest SHA256, canonical workspace digest, loopback host and integer port. A healthy port alone never proves ownership. Stderr is continuously drained into a bounded 64 KiB in-memory tail, never returned to remote pages. These controls are a trusted-local-bundle boundary, not an OS sandbox against a malicious same-user filesystem writer.

The supervisor permits at most 90 seconds from spawn to ready, 30 seconds for graceful shutdown, then five seconds to observe reaping after killing the exact owned `Child` handle. These are finite supervisor budgets, not performance guarantees or a cap on preceding synchronous local resource hashing. Closing/Quit waits for cleanup off the GUI thread. Parent pipe EOF asks the Python sidecar to run its normal lifespan shutdown. There is no process-name search, saved-PID signaling or process-group kill.

A valid stopped receipt plus successful process exit establishes ordinary child shutdown. A hard stop, missing receipt or inspection/reaping failure remains `cleanup_unknown`, revokes navigation and blocks another owned start in that desktop process. It does not prove detached worker or remote resource cleanup. Data and application recovery evidence are preserved. Unexpected exits are observed on the next native status/navigation check; no automatic retry or relaunch follows. Connecting to an external backend neither owns nor stops it, and cannot erase a prior cleanup warning.

The native Connection menu only returns to the bundled screen. Only its controls can request ownership changes; HTTP-loaded workbench pages have no native IPC grant. Stop/restart invalidate the allowed backend origin; a new ready generation supplies a new allowed loopback origin. Window close and Quit are intercepted before process exit. Native GUI lifecycle behavior, real frozen-sidecar relocation, active-worker hard-stop behavior, upgrade policy, signing/notarization and Windows/Linux acceptance remain separate gates.
