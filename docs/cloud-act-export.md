# Export a cloud ACT checkpoint

Turn an ACT checkpoint trained on Google Cloud into a CPU inference package. OpenJensen downloads the checkpoint to the server and exports it with the server's CPU. The original checkpoint stays in Cloud Storage.

## What you need

- A completed native ACT run (**Native training**) on Google Cloud. Only its final checkpoint, marked **Reload-verified bundle**, can be exported.
- A checkpoint with its config, weights, saved processors, statistics and verification files, and dataset lineage pinned to a Hugging Face dataset revision.
- At most 4 GiB of payload in total, at most 2 GiB per file, and free disk on the server for the payload plus 64 MiB.
- The ACT export worker set up as below.

## Set up the export worker

Do this once on the server.

1. Install the pinned [ACT consumer](../workers/act_optimizer/README.md) in a durable environment.
2. Add this runtime to the operator runtime JSON, replacing the interpreter and source paths:

   ```json
   {
     "id": "act-cpu-export",
     "label": "Local ACT CPU export",
     "export_only": true,
     "act_export_python": "/absolute/install/workers/act_optimizer/.venv/bin/python",
     "act_export_root": "/absolute/install/workers/act_optimizer"
   }
   ```

3. Set `JENSEN_RUNTIME_CONFIG` to that file before starting the server.
4. In **Settings → Compute**, under **Local runs**, check **Enable local runs** and select **Save local settings**.
5. Make sure SkyPilot's Python environment can read the checkpoint's private Cloud Storage objects. The download runs through it with your application-default credentials (`gcloud auth application-default login`).

## Export

1. Open the completed ACT run in **Fine-tune**.
2. Under **Use a trained checkpoint**, pick the **Checkpoint** marked **Reload-verified bundle**.
3. If more than one export worker is set up, choose the **Export computer**.
4. Select **Download checkpoint and export**.
5. Follow the `policy.export` job: it downloads and checks the checkpoint, then exports it on the CPU. **Refresh export jobs** updates the status.
6. Select **Download ACT inference package**.

The export removes the training-only VAE and keeps FP32 weights. The package is for inference only; keep the original checkpoint to resume training. To go further, select **Quantize this package** ([pack it locally](native-act-quantization.md)) or **Use as ACT teacher** ([distill ACT](policy-workflow.md#distill-act)).

## Follow, cancel or fix

- The status line under the button shows the export job's state and ID. Read its stage events with `uv run --frozen jensen jobs events JOB_ID`.
- To stop the download or export, cancel that job: `uv run --frozen jensen jobs cancel JOB_ID`, or select it in the [terminal UI](terminal.md) and press F8.
- Exported files stay in the job's own workspace folder. The original checkpoint stays in Cloud Storage.
- If a submission's outcome is unknown, further exports pause. Select **Refresh export jobs**, check the recorded jobs, then select **I checked the export jobs; allow a new request**.

| Message | What to do |
| --- | --- |
| Only the final ACT checkpoint can be exported. | Wait for the run to succeed, then pick the **Reload-verified bundle**. |
| Ask the app operator to configure the local ACT export worker. | Complete [Set up the export worker](#set-up-the-export-worker). |
| ACT checkpoint exceeds the 4 GiB download limit. | This checkpoint can't be exported here. |
| Not enough local disk space for the completed ACT checkpoint. | Free disk on the server and request the export again. |
| Completed ACT checkpoint download failed. | Check private storage access, integrity and local disk space. |
| Another integrity or source-admission error | Correct the checkpoint selection or runtime paths, then make a new explicit request. |
