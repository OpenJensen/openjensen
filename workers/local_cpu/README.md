# Install local ACT CPU runtimes

Use Apple Silicon macOS or Linux x86_64, an existing Python 3.12 interpreter,
uv 0.12.19 and a complete repository checkout. Install FFmpeg/ffprobe separately
on the application host. Keep the checkout and installation in persistent paths.

Run the following commands from the repository root, replacing every example
path. Create the installation's parent directory first and choose a new root.
The installer creates `act-model/` and `dataset-reader/` beneath it.

## Plan

```sh
python3 workers/local_cpu/manage.py plan \
  --root /absolute/persistent/cpu-20260927 \
  --python /absolute/existing/python3.12 \
  --uv /absolute/existing/uv
```

## Install

```sh
python3 workers/local_cpu/manage.py install --execute \
  --root /absolute/persistent/cpu-20260927 \
  --python /absolute/existing/python3.12 \
  --uv /absolute/existing/uv
```

Verify the installed environments:

```sh
python3 workers/local_cpu/manage.py verify --root /absolute/persistent/cpu-20260927
```

To reuse a uv download cache, add the same
`--cache /absolute/existing/uv-cache` argument to `plan` and `install --execute`.
Use an existing cache with `CACHEDIR.TAG`, separate from the new installation.

## Complete an installed partial attempt

Supply the original paths from the saved plan:

```sh
python3 workers/local_cpu/manage.py complete \
  --root /absolute/existing/partial-installation \
  --python /absolute/original/python3.12 \
  --uv /absolute/original/uv \
  --cache /absolute/original/uv-cache
```

Review the printed commands, then add `--execute` to run verification and write
`installation.json`.

## Generate application configuration

Choose a new output file; `--base` supplies an existing configuration to extend:

```sh
python3 workers/local_cpu/manage.py config \
  --root /absolute/persistent/cpu-20260927 \
  --base /absolute/existing/runtimes.json \
  --output /absolute/new/runtimes-with-local-cpu.json
```

Set `FIREBIRD_RUNTIME_CONFIG` to the generated file and restart the application
when no owned job is active. The runtime IDs are `local-act-distillation-cpu`,
`local-act-quantization-cpu` and `local-act-replay-cpu`.

Prepare the required policy and data using the
[distillation](../policy_distillation/README.md),
[packing](../firebird_quant/NATIVE_ACT.md) and
[replay](../isaac_sim/NATIVE_REPLAY.md) recipes.

## Regenerate dependency locks

```sh
uv pip compile workers/local_cpu/reader-macos-arm64.in \
  --python 3.12 --python-platform aarch64-apple-darwin \
  --generate-hashes --no-annotate --no-header \
  --output-file /absolute/new/reader-macos-arm64.lock
uv pip compile workers/local_cpu/reader-linux-x86_64.in \
  --python 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-annotate --no-header \
  --output-file /absolute/new/reader-linux-x86_64.lock
```

## Shell completion

Set `OPENJENSEN_REPO` to the absolute checkout path and source
`workers/local_cpu/completions.bash`. The `openjensen-cpu-setup` wrapper uses the
same commands.
