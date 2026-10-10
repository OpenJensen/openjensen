# Run the LIBERO Spatial workflow

## Validate local assets before GPU work

Prepare the pinned `lerobot/smolvla_libero` policy, native BF16 weights, floating F32/BF16 GGUF, tokenizer/processors, normalization, observation fixtures and simulator assets. Use a Python 3.11 native-policy/simulator environment.

Run [bundle preparation](../workers/vla_cpp/docs/spatial-application.md#prepare-exact-local-assets) into a new output directory. With the worker's `quantize` and `test` extras installed, check the local bundle and floating GGUF using their expected SHA-256 values:

```sh
uv run --offline --frozen --project workers/vla_cpp --extra quantize --extra test \
  python -m policykit.spatial_preflight \
  --bundle /local/spatial-bundle \
  --bundle-sha256 EXPECTED_SPATIAL_ASSETS_JSON_SHA256 \
  --gguf /local/smolvla-bf16.gguf \
  --gguf-sha256 EXPECTED_FLOATING_GGUF_SHA256
```

Use assets with 8-state/7-action normalization, 50×32 padded actions, ten denoising steps and the declared image layout. Keep the input inventory unchanged while checking it.

## Configure and submit

Copy [runtime.spatial.example.json](runtime.spatial.example.json), replace its absolute paths and digests, and set `FIREBIRD_RUNTIME_CONFIG` before starting the server. Supply `evaluation_python`, or `evaluation_image` for the same worker inside Docker.

Set the source task to `libero_spatial`, its local `evaluation_bundle` to the prepared bundle, and the expected hash to its `spatial-assets.json` SHA-256. Supply the floating GGUF's separate hash.

1. Open **Settings & diagnostics → Workflow settings**.
2. Choose **LIBERO Spatial**, task IDs and paired search/final states.
3. Use LIBERO mode, the full 280-step horizon and 50 replayed actions.
4. Enter `evaluation.parity_limits` with `profile`, `max_rmse` and `max_abs_error` for the fixed observations.
5. Choose Q8 or explicit Q4/vision candidates. For automatic selection, also supply quality, latency and memory `limits` and disjoint final states.
6. Review the recipe, select the configured runtime and start.

Omit `task_ids` to use all ten tasks (0–9), or supply the desired IDs. Use the same GPU and complete task/state/fixture inventory for the paired runs.

## Follow and download

Open the saved workflow to follow reference, candidate, final and package stages. Download the registered package from its completed result. Inspect the failed stage's logs or asset/parity message before changing the recipe and starting a new request. Cancel the selected job to stop its worker.
