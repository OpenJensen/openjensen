# OPEN JENSEN desktop workbench

This Tauri 2 client opens the existing Python-served static OPEN JENSEN web app. It does not start, package, configure or stop Python, GPU workers or cloud jobs. Closing the window exits only the desktop process. DESK-002 remains the separate Python-sidecar and clean-machine installation milestone.

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

- Exactly one custom native command performs a fixed loopback readiness probe. It accepts no browser-supplied address and never follows redirects or environment proxy settings. Each of the two HTTP reads has a two-second timeout and a byte bound.
- Only the bundled connection page receives that command permission. The HTTP-served workbench has no Tauri remote capability grant; the command also verifies the invoking window's current bundled origin.
- Top-level navigation is restricted to the configured loopback origin and the bundled connection page. New windows and off-origin links are blocked. Use the ordinary browser for external documentation. No shell, filesystem, process, HTTP plugin or arbitrary execution bridge is exposed to web content.
- macOS microphone purpose text explains the explicitly chosen voice teaching flow. This metadata does not grant permission or prove native microphone, speech-provider or robot-control operation.
- Existing cloud/Hugging Face/voice settings remain owned by the Python application. No credentials or configuration files are copied into the bundle.
- A healthy loopback service is not cryptographic server authentication. This slice relies on the same trusted local-machine boundary as the existing application.

## Verification

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml
```

Rust tests exercise endpoint policy, successful API/static-page probing, bad and oversized responses, redirects, refused connections, and the permission manifest. Native window checks must additionally cover connection, a stopped-server retry, blocked navigation, the native Connection menu and backend health after the desktop closes. Unit tests alone do not prove native runtime behavior or an installable Python sidecar.

## Verified upstream basis

Verified stable releases on 2026-09-27: Rust 1.98.1, Tauri crate/CLI 2.12.0, tauri-build 2.7.0. Registry metadata and exact resolved dependencies are separate from application acceptance.

- [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)
- [Next static export integration](https://v2.tauri.app/start/frontend/nextjs/)
- [Tauri capabilities and application command permissions](https://v2.tauri.app/security/capabilities/)
- [Tauri application configuration](https://v2.tauri.app/reference/config/)
- [Python sidecar mechanism and platform-specific binaries](https://v2.tauri.app/develop/sidecar/)
- [macOS distribution](https://v2.tauri.app/distribute/macos-application-bundle/)
