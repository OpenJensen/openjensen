# CPU quantization setup and measurement

This layer adds runtime/model preparation, CPU engine screens, repeated and
balanced-order measurements, output fidelity, cost reports, comparison-table
rendering, and optional dataset/LIBERO evaluation scaffolding. It reuses the
standalone module's quantizer and packed-runtime patch. The shared application core
and QLoRA training use separate packages and environments.

## Run

From `workers/vla_cpp`:

```sh
uv sync --locked --extra quantize --extra test
uv run python -m pytest -q -rs
uv run policykit --help
```

For the native experiment, use the existing Docker runner and manifests:

```sh
docker compose run --rm policykit --config configs/benchmark.docker.yaml prepare
docker compose run --rm policykit --config configs/benchmark.docker.yaml quantize --model smolvla --preset bf16
docker compose run --rm policykit --config configs/benchmark.docker.yaml quantize --model smolvla --preset q4_0_vision
```

Preparation downloads/builds dependencies and checkpoints, including all models
listed in the chosen manifest. Do not run preparation just to execute the unit
tests. Native runtime preparation, the unpatched float-control capture and packed
screen reproduction are detailed in [the patch guide](../patches/README.md) and
[benchmark protocol](quantization-benchmark.md). The measurement scripts
expect their documented `artifacts/docker/...` layout; this worker does not make old
absolute paths portable or introduce a new application scheduler.

Generated weights, vendor checkouts, binaries, logs, videos and reports remain
ignored under `artifacts/`. Keep model and dataset bytes outside Git. Use a new
run directory and preserve source identities for each comparison.

## Product boundaries

`benchmark.py` and `selection.py` preserve the original experiment behavior.
The legacy selection field has incomplete admission gates and must not certify
a deployment. The locked product objective is smallest tested deployed-policy
weights subject to explicit quality, memory and latency constraints, with an
independent final evaluation. Candidate ordering alone cannot satisfy those constraints.

The generic evaluator's execution smoke, per-phase logging, denominator handling,
target measurement and final-validation protocol still need correction before
integration into the shared application. This worker provides CPU measurement tools; it does
not claim a finished optimizer, closed-loop benchmark or complete deployment export.
