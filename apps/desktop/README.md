# Run the OPEN JENSEN desktop client

Use Node 24.21.x, pnpm 12.6.0, Rust 1.98.1 with rustfmt/clippy, and the [Tauri platform prerequisites](https://v2.tauri.app/start/prerequisites/). Install the Xcode command-line tools and macOS SDK for a macOS build.

## Run and build

Start the application from the repository root:

```sh
pnpm build:web
uv run --frozen firebird serve
```

In another terminal:

```sh
cd apps/desktop
pnpm install --frozen-lockfile --ignore-scripts
pnpm exec tauri dev
```

Connect to `http://127.0.0.1:8000/`. To use another port, set `FIREBIRD_DESKTOP_PORT` to the backend's port before launching the desktop executable. After restarting the backend, choose **OPEN JENSEN → Connection** (Cmd/Ctrl+Shift+R) and reconnect.

Build a macOS application and check the executable:

```sh
pnpm exec tauri build --bundles app
src-tauri/target/release/firebird-desktop --self-check
```

The application is written to `src-tauri/target/release/bundle/macos/OPEN JENSEN.app`.

## Verification

From `apps/desktop`:

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --features custom-protocol -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml
node --test tests/connection.test.cjs
```

## Prepare a local payload build

Follow the [sidecar build instructions](sidecar/README.md#opt-in-macos-arm64-frozen-experiment), then the [native candidate preparation steps](sidecar/README.md#local-native-candidate-preparation) to generate the pin and Tauri resource overlay. Use the generated overlay when building with `local-payload-experiment`.

For signing and distribution, follow [Tauri's macOS application guide](https://v2.tauri.app/distribute/macos-application-bundle/).
