# CPU quantization experiments

PR: `feat/quantization_benchmark_cpu` → codebase `main`.
Depends on `feat/quantization_module`; merge the module PR first.
Related task: [EVAL-002 (#21)](https://github.com/sobhanb-eth/firebird-hackathon-codebase/issues/21).

This layer adds runtime/model preparation, CPU engine screens, repeated and
balanced-order measurements, output fidelity, cost reports, historical leaderboard
rendering, and optional dataset/LIBERO evaluation scaffolding. It reuses the
standalone module's quantizer and packed-runtime patch. The shared application core
and QLoRA training are separate branches.

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
[benchmark protocol](quantization-benchmark.md). Recorded-run scripts currently
expect their documented `artifacts/docker/...` layout; this PR does not make old
absolute paths portable or introduce a new application scheduler.

Generated weights, vendor checkouts, binaries, logs, videos and reports remain
ignored under `artifacts/`. The existing experiment artifacts are in the original
Documents checkout; they were not copied into this branch. The committed
[prep evidence manifest](https://github.com/sobhanb-eth/firebird-hackathon-prep/blob/c002464716cc73e9b7d19782b40ae37f7bc56269/docs/quantization/imported-evidence.json) records original
report/result hashes and local locations. Preserve source evidence when relocating
it, and verify hashes before reusing it.

## Saved evidence

The saved Docker CPU experiment covers the pinned SmolVLA checkpoint and four
packed candidates. The smallest GGUF is 566.99 MiB versus 1074.18 MiB for the
floating reference. The saved patched-runtime run verified packed resident weights
and finite synthetic action output. Original failed loading attempts remain part
of the evidence. The floating timing control drifted by 25.6%; these results do
not establish a reliable speedup or slowdown due to quantization.

These are previously recorded results, not a fresh full-model benchmark performed
while splitting the branches. No task-success rate, CUDA performance or pi0
quantization support is established. Synthetic action agreement is a numerical
diagnostic, not a robot-task score.

## Product boundaries

`benchmark.py` and `selection.py` preserve the original experiment behavior.
The legacy selection field has incomplete admission gates and must not certify
a deployment. The locked product objective is smallest tested deployed-policy
weights subject to explicit quality, memory and latency constraints, with an
independent final evaluation. Earlier experiment documents specifying fastest-first
or preset-order selection are historical proposals, superseded by that lock-in.

The generic evaluator's execution smoke, per-phase logging, denominator handling,
target measurement and final-validation protocol still need correction before
integration into the shared application. This PR exposes inspectable CPU experiment evidence; it does
not claim a finished optimizer, closed-loop benchmark or complete deployment export.
