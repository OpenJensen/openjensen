# Build and run the Python sidecar

Use absolute paths for the static resources, application workspace and job files. Keep application data in a separate directory from the packaged resources.

## Prepare static resources

Build the frontend from the repository root, then copy the export into a new resource directory:

```sh
pnpm build:web
python apps/desktop/sidecar/prepare_resources.py \
  --web /absolute/openjensen/apps/web/out \
  --output /absolute/new/resources \
  --build-id EXACT_40_CHARACTER_COMMIT
```

## Start the sidecar

Start the sidecar from an environment with the core application installed:

```sh
python apps/desktop/sidecar/entrypoint.py serve \
  --resources /absolute/new/resources \
  --data-dir /absolute/workspace
```

Add `--local-root /absolute/datasets` when using local datasets. To invoke metadata intake, place `request.json` and the new `result.json` path in the same job directory:

```sh
python apps/desktop/sidecar/entrypoint.py intake-worker \
  /absolute/job/request.json /absolute/job/result.json
```

## Start and stop through stdin

After starting `serve`, send this UTF-8 JSON line through its stdin pipe within ten seconds:

```json
{"schema_version":1,"command":"start","nonce":"0123456789abcdef0123456789abcdef"}
```

Choose a fresh 32-character lowercase hexadecimal nonce. Read the `ready` message on stdout for the loopback port. Read process logs from stderr. To stop the process, send a `shutdown` message with the same nonce, or close the pipe:

```json
{"schema_version":1,"command":"shutdown","nonce":"0123456789abcdef0123456789abcdef"}
```

## Opt-in macOS ARM64 frozen experiment

Use an isolated CPython 3.14 macOS ARM64 build environment. From the repository root, install the production core dependencies and pinned build tools:

```sh
UV_PROJECT_ENVIRONMENT=/absolute/build-env uv sync --frozen --no-dev --no-editable
uv pip install --python /absolute/build-env/bin/python --require-hashes --no-deps \
  -r apps/desktop/sidecar/requirements-build-macos-arm64.txt
```

Build with the prepared resources and new work and dist directories:

```sh
FIREBIRD_DESKTOP_RESOURCES=/absolute/new/resources \
  /absolute/build-env/bin/python -I -B -m PyInstaller \
  --workpath /absolute/new/work --distpath /absolute/new/dist \
  apps/desktop/sidecar/firebird-sidecar.spec
```

Keep the complete `/absolute/new/dist/firebird-sidecar` directory, including `_internal`.

Inventory the payload and run the frozen-package checks from the repository root:

```sh
python apps/desktop/sidecar/payload_inventory.py \
  /absolute/new/dist/firebird-sidecar /absolute/new/payload-inventory.json
python apps/desktop/sidecar/verify_frozen.py \
  --payload /absolute/new/dist/firebird-sidecar \
  --output /absolute/new/experiment \
  --build-id EXACT_SOURCE_COMMIT \
  --resource-sha256 EXACT_RESOURCES_JSON_SHA256
```

Use the resource manifest's SHA256 and its exact source commit. Keep the generated receipt and logs with the payload.

## Local native candidate preparation

Supply the payload, inventory and frozen-check receipt to the scratch preparer:

```sh
python -B -s apps/desktop/sidecar/prepare_local_tauri.py \
  --payload /absolute/accepted/firebird-sidecar \
  --manifest /absolute/payload-inventory.json \
  --acceptance /absolute/acceptance/receipt.json \
  --output /absolute/new-native-candidate
```

The output contains `local-payload-pin.rs`, `tauri.local.json`, resources and a preparation receipt. From `apps/desktop`, build with that pin and overlay:

```sh
FIREBIRD_DESKTOP_EXPERIMENT_PIN=/absolute/new-native-candidate/local-payload-pin.rs \
  CARGO_NET_OFFLINE=true pnpm exec tauri build --features local-payload-experiment \
  --config /absolute/new-native-candidate/tauri.local.json --bundles app --no-sign
```

Compare the finished application's resource inventory with the pin before launching it. Use a clean build environment with the cached toolchain and lockfiles.
