# CPU quantization run recipe

## Install and test

From `workers/vla_cpp`, use Python 3.11 or 3.12:

```sh
uv sync --locked --extra quantize --extra test
uv run python -m pytest -q -rs
uv run policykit --help
```

## Prepare and quantize

Review `configs/benchmark.docker.yaml`, create the `artifacts/docker/` parent
directories and make the output paths writable by the container user. The
preparation command downloads and builds the models/runtime in that manifest:

```sh
docker compose run --rm policykit --config configs/benchmark.docker.yaml prepare
docker compose run --rm policykit --config configs/benchmark.docker.yaml quantize --model smolvla --preset bf16
docker compose run --rm policykit --config configs/benchmark.docker.yaml quantize --model smolvla --preset q4_0_vision
```

Keep generated files under `artifacts/`. Use the [packed-loader recipe](../patches/README.md)
for rebuilding native targets and the [engine screen](quantization-benchmark.md)
for running inference calls.
