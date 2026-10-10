# SmolVLA GGUF worker setup

## Install

From `workers/vla_cpp`, use Python 3.11 or 3.12:

```sh
uv sync --locked --extra quantize --extra test
uv run policykit-worker --help
uv run python -m pytest -q
```

## Prepare the native runtime

Check out vla.cpp at `52439f7c6c362d7bee218b400b9080cc32d75cc3`. Follow the
[build recipe](patches/README.md) to capture the floating control, apply
`policykit/patches/vla-cpp-smolvla-packed.patch` and rebuild. Then inspect the runtime:

```sh
uv run policykit-worker --describe-runtime /absolute/path/to/vla.cpp
```

Use the returned identity in the quantization recipe. Repeat this step after
changing worker source or dependencies.

## Submit a conversion

Prepare an F32/BF16 SmolVLA GGUF and a new writable output directory. Write the
job envelope with its source SHA256 and the resolved runtime object:

```json
{
  "schema_version": 1,
  "run_id": "run-example",
  "attempt_id": "attempt-1",
  "output_dir": "/absolute/path/to/run",
  "recipe_sha256": "SHA256_OF_CANONICAL_RECIPE_JSON",
  "recipe": {
    "schema_version": 1,
    "operation": "policy.quantize",
    "backend": "vla_cpp_smolvla",
    "policy": {"architecture": "smolvla"},
    "source": {"path": "/absolute/path/to/float.gguf", "sha256": "SOURCE_SHA256"},
    "runtime": "REPLACE_WITH_DESCRIBE_RUNTIME_OBJECT",
    "target": "local-cpu-conversion",
    "language": "Q4_0",
    "vision": "Q8_0"
  }
}
```

Replace every placeholder. Select `language: Q8_0` or `Q4_0`; set `vision: Q8_0`
or `null` to keep source vision precision. Compute `recipe_sha256` from UTF-8
canonical JSON with sorted keys, separators `(',', ':')` and `allow_nan=False`.

```sh
uv run policykit-worker --job /absolute/path/to/run/job.json
```

Read `result.json`, `events.jsonl`, the precision audit and `bundle/` beneath the
output directory. The direct quantizer invocation is
`policykit-quantize --in ... --out ... --vendor-script ... --type Q4_0 --vision-type Q8_0`.

## Additional recipes

- [CPU preparation and commands](docs/cpu-benchmark.md)
- [CUDA preparation and commands](docs/gpu-setup.md)
- [Native worker tests](docs/native-acceptance.md)
- [Spatial asset setup](docs/spatial-application.md)
- [SO101 preflight](docs/so101-evaluation.md)
- [Target configuration](docs/deployment-targets.md)
