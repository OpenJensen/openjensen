# Run the LIBERO Spatial workflow

Compare quantized SmolVLA candidates with the floating policy on LIBERO Spatial tasks, and package a candidate that meets your limits. Check the assets offline first, before you spend GPU time.

## What you need

- A Linux CUDA GPU and a Python 3.11 native-policy/simulator environment.
- The pinned `lerobot/smolvla_libero` policy, native BF16 weights, a floating F32/BF16 GGUF, tokenizer/processors, normalization, observation fixtures and simulator assets.
- Assets with 8-state/7-action normalization, 50×32 padded actions, ten denoising steps and the declared image layout.

## 1. Prepare and check the bundle

Run [bundle preparation](../workers/vla_cpp/docs/spatial-application.md#prepare-exact-local-assets) into a new output directory. Then, with the worker's `quantize` and `test` extras installed, check the bundle and the floating GGUF against their expected SHA-256 values:

```sh
uv run --offline --frozen --project workers/vla_cpp --extra quantize --extra test \
  python -m policykit.spatial_preflight \
  --bundle /local/spatial-bundle \
  --bundle-sha256 EXPECTED_SPATIAL_ASSETS_JSON_SHA256 \
  --gguf /local/smolvla-bf16.gguf \
  --gguf-sha256 EXPECTED_FLOATING_GGUF_SHA256
```

Keep the input inventory unchanged while it runs. The check covers hashes, the declared GGUF layout, normalization and fixture arrays only. It does not load the model or check the GPU, runtime, episodes or action parity.

## 2. Register the runtime and source

Copy [runtime.spatial.example.json](runtime.spatial.example.json), replace its absolute paths and digests, and set `JENSEN_RUNTIME_CONFIG` before starting the server.

| Key | Value |
| --- | --- |
| Runtime `device` | `cuda` |
| Runtime `evaluation_python` | The prepared evaluation interpreter. Required. |
| Runtime `evaluation_image` | Optional: run `evaluation_python` from this Docker image. |
| Runtime `simulator_lane` | A directory with the LIBERO `config.yaml` and `assets.json` |
| Source `task` | `libero_spatial` |
| Source `path`, `sha256` | The floating GGUF and its own SHA-256 |
| Source `evaluation_bundle` | The prepared bundle directory |
| Source `evaluation_bundle_sha256` | The SHA-256 of the bundle's `spatial-assets.json` |

## 3. Set the protocol

In **Settings → Workflow settings**:

1. Set **Task suite** to **LIBERO Spatial (pinned SmolVLA policy)**. This fixes **Protocol** to **Paired LIBERO episodes** and the full 280-step horizon; each chunk replays 50 actions.
2. Enter **Spatial task IDs** (all ten, `0,1,2,3,4,5,6,7,8,9`, by default), **Search initial states** and **Unused final initial states**. Search and final states must not overlap.
3. Enter **Approved parity profile**, **Maximum action RMSE** and **Maximum absolute action error** for the fixed observations. These are required; there is no default.
4. Choose the candidates under **Quantization recipe**: **LM Q8**, or opt in to **LM Q4 (experimental)**, **Compare Q8 and Q4 (experimental)** or **Also quantize vision to Q8 (experimental)**.
5. For automatic selection, tick **Select the smallest candidate that meets my limits** and set **Minimum task success (%)**, **Maximum p95 latency (ms)** and **Maximum measured target memory (MiB)**.

Run every paired measurement on the same GPU with the complete task, state and fixture inventory.

## 4. Start the workflow

In the web app, with the floating Spatial policy saved in the project, open **Quantize**, select **Start a new quantization**, choose that policy and the configured runtime, and select **Run quantization workflow**. To evaluate one Spatial GGUF without candidates, use **Evaluate** or **Settings → Diagnostics**.

From the CLI, submit a recipe with `jensen policy submit PROJECT_ID recipe.json`:

| Recipe key | Value |
| --- | --- |
| `operation`, `runtime_id` | `policy.workflow`, the configured Spatial runtime |
| `source_id` | The registered Spatial source |
| `candidates` | Up to four, such as `[{"language": "Q8_0"}]`. `Q4_0` and `"vision": "Q8_0"` are explicit choices. |
| `evaluation.mode`, `evaluation.suite` | `libero`, `libero_spatial` |
| `evaluation.task_ids` | Omit for all ten tasks (0–9), or list the IDs. |
| `evaluation.initial_states`, `evaluation.final_states` | Disjoint search and final state IDs |
| `evaluation.parity_limits` | `profile`, `max_rmse` and `max_abs_error` |
| `limits` | `min_success_rate`, `max_success_drop`, `max_p95_ms` and `max_peak_device_mib`. Needed for automatic selection. |

## 5. Follow and download

Open the saved workflow to follow its stages:

| Stage | What happens |
| --- | --- |
| `native-reference`, `baseline` | Measures the native BF16 policy and the floating GGUF. |
| `float-parity` | Compares their actions on the fixed observations. If parity fails, compression does not start. |
| `quantize-N`, `evaluate-N` | Builds and measures each candidate. |
| `selection` | Without `limits`, the run ends here with diagnostic evidence only. Otherwise it freezes the smallest candidate that meets them. |
| `final-reference`, `final-control`, `final-evaluation` | Repeats the checks on the unused final states. |
| `package-and-reload` | Packages the candidate and reruns it exactly. |

Download the registered package from the completed result. If a stage fails, read its logs or the asset/parity message before you change the recipe and start a new request. Select **Cancel run** to stop the worker.
