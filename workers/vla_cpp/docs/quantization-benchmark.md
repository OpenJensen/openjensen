# Run the CPU engine screen

From `workers/vla_cpp`, prepare the Docker runtime using
[the CPU recipe](cpu-benchmark.md). Mount the Linux Python environment at
`artifacts/docker/tooling` with CPU Torch, gguf, safetensors, Hugging Face Hub,
PyYAML and pytest. Place the complete source checkpoint and `policykit-source.json`
under `artifacts/docker/sources/smolvla`.

```bash
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m pytest -q
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m policykit.screen --reps 5 --timeout 900
```

Choose a new run name with `--run smolvla-screen-v2` to write an independent
output directory. `--reps` selects repeated calls and `--timeout` selects the
per-command deadline. Read `REPORT.md`, `results.json`, models and logs beneath
`artifacts/docker/runs/<run>/`.

For packed-runtime calls, follow the [patch recipe](../patches/README.md).
