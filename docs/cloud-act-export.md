# Export a cloud ACT checkpoint

## Operator configuration

Install the pinned [ACT consumer](../workers/act_optimizer/README.md) in a durable environment. Add this runtime to the operator runtime JSON, replacing the interpreter and source paths:

```json
{
  "id": "act-cpu-export",
  "label": "Local ACT CPU export",
  "export_only": true,
  "act_export_python": "/absolute/install/workers/act_optimizer/.venv/bin/python",
  "act_export_root": "/absolute/install/workers/act_optimizer"
}
```

Set `FIREBIRD_RUNTIME_CONFIG` before starting the server and enable local compute in Settings. Prepare the SkyPilot interpreter with access to the checkpoint's private GCS objects.

## Select and export

1. Open the completed native ACT run in **Fine-tune**.
2. Select its **Reload-verified bundle** with the complete native checkpoint and pinned Hugging Face dataset lineage.
3. Choose **Download checkpoint and export**.
4. Follow the `policy.export` job through download and CPU export.
5. Download the registered inference export from the result.

Use a checkpoint containing its config, weights, saved processors, statistics and verification files. Allow space for up to 4 GiB of payload and supply files of at most 2 GiB each.

## Follow or cancel

Open the saved job for stage events and logs. Cancel the selected job to stop download/export. If it fails, review the integrity or source-admission message, correct the checkpoint selection or runtime paths, and make a new explicit request. Local output remains in its job-owned workspace directory; the original checkpoint stays in GCS.
