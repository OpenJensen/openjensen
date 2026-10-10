# Run SO101 dataset preflight

Use the [GGUF worker environment](../README.md). Select
`configs/evaluation.so101.yaml`, which pins `codywang/so101_pickup_test` revision
`ecef85bc07005f771ad86deeff1427f9d72953ed`.

Cache metadata and tabular files under `artifacts/datasets/so101_pickup_test`
and record their hashes in `download-manifest.json`. Supply the existing policy
config and choose a new output directory:

```bash
uv run python -m policykit.dataset_preflight \
  --profile configs/evaluation.so101.yaml \
  --model-config artifacts/docker/sources/smolvla/config.json \
  --out artifacts/docker/runs/so101-preflight
```

Read the generated preflight report in `artifacts/docker/runs/so101-preflight`.
For recorded targets, use `policykit.output_fidelity --pairs` with matched
floating/candidate observations, preprocessing and noise.
