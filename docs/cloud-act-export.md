# Export a completed cloud ACT checkpoint

The Fine-tune checkpoint panel can explicitly **Download checkpoint and export**
for a completed native ACT run. Select its **Reload-verified bundle**; periodic
checkpoints remain separate choices and cannot use this route, even at the same
step. This invokes one `policy.export` job. It downloads existing private GCS
objects and runs the isolated CPU exporter; it does not allocate a GPU or retry a
failed submission automatically.

## Operator configuration

Install the pinned [ACT consumer](../workers/act_optimizer/README.md) in a durable
location. Add a local runtime to the existing operator runtime JSON, keeping any
existing training/engine runtimes:

```json
{
  "id": "act-cpu-export",
  "label": "Local ACT CPU export",
  "export_only": true,
  "act_export_python": "/absolute/install/workers/act_optimizer/.venv/bin/python",
  "act_export_root": "/absolute/install/workers/act_optimizer"
}
```

Enable local compute in Settings. Cloud downloads reuse the existing isolated
SkyPilot interpreter and its private storage access. Interpreter paths, object
URIs and file destinations are operator/server owned, never browser arguments.
The export-only runtime needs no GGUF vendor/build and does not advertise Run,
Evaluate, Quantize or training. Existing runtimes default to `export_only=false`.

## Identity and failure behavior

The source must belong to this project and a succeeded training job, and its
manifest must identify a full native ACT checkpoint with reload verification,
pinned Hugging Face lineage and the complete checkpoint/verification inventory.
ACT local-dataset exports remain unsupported. The CPU worker independently checks
the downloaded complete native bundle before exporting.

The app verifies the registered descriptor, immutable manifest hashes, GCS object
generations and exact payload bytes. Downloads are limited to 128 filesystem
entries, 2 GiB per file and 4 GiB of payload, with a disk-space preflight. It creates
a new job-owned directory and uses atomic no-replace publication. The original
cloud descriptor remains unchanged. A new local artifact retains the complete
source dataset profile and cloud source identity; the inference export is its
child and carries the supported dataset identity projection. Both source manifest
hashes remain part of the ancestry.

The export job's timeout also bounds download. Cancellation waits for the owned
subprocess to be reaped, then removes its staging directory. No local child or
export is registered on download, integrity or parity failure. A failed export
may leave unregistered diagnostic output in its job directory; it cannot appear
as a successful package. An explicit later request starts a separate job.

## What the download proves

This recipe removes training-only VAE tensors and retains FP32 inference weights
and saved normalization. It requires complete synthetic action-chunk parity and
fresh-process reload of the exact published policy package. The downloaded archive
contains that tested export and its provenance. It is inference only, not a resume
checkpoint, quantization result, calibrated robot policy, task-success evaluation,
GPU-memory measurement or speedup claim.

Offline API tests exercise a real supervised copy subprocess over fixture GCS
streams, cancellation/timeout ownership, corruption, byte limits, source mutation,
publication races, cross-project rejection and full-profile lineage. Browser
checks cover explicit final/periodic selection, local ancestry/download links and
export-only runtime filtering. These fixtures do not exercise live GCS permissions.
