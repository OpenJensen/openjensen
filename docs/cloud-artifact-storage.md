# Cloud checkpoint storage

Firebird serves the application on xbox-360. GCP workers download model and dataset
weights directly from their pinned sources. Checkpoints, exported policies and
quantized models stay in a private Google Cloud Storage bucket in the selected
project. They are not automatically downloaded to the Mac or application server.

Each run receives an isolated prefix under
`gs://firebird-artifacts-<project>/jobs/<job-id>/<stage>/`. The bucket requires
uniform access and public access prevention. Existing buckets with a different
ownership label are rejected. Application credentials remain in the configured
Google Cloud/SkyPilot environments; no credential is embedded in the job recipe.

The worker verifies file hashes and uploads all files before publishing an atomic
checkpoint index. The application polls only small metadata documents and stores
manifest descriptors, recipes, metrics and logs. Every committed checkpoint has
a stable project artifact ID, optimizer step and cloud location. Quantization and
resume resolve that ID, download its files on the new cloud worker and verify
both the registered manifest and every file hash before using the checkpoint.

The GPU workspace retains only the newest two published working checkpoints.
Earlier checkpoints remain in GCS and can be selected in Firebird. A final
successful training artifact additionally records fresh-process reload evidence.
An intermediate checkpoint is selectable after publication but is not labeled
reload-verified until the worker has actually performed that check.

The explicit Download action streams a checked TAR from GCS through the
application to the requesting browser. It does not cache a TAR or model weights
on xbox. The download is bounded in memory, checks object sizes and hashes, and
stops when the client disconnects. Merely viewing runs or selecting a checkpoint
does not download weights to the client.

GPU teardown occurs after result metadata has been collected; durable GCS data
survives teardown. Cancellation preserves checkpoints already published to GCS.
Storage objects are retained until the operator removes them; GPU autodown does
not erase checkpoints or eliminate GCS storage charges.

Legacy jobs from before this feature keep their original local transfer behavior
and evidence. New cloud jobs use GCS automatically. Retrying an old failed job
creates a new run with current storage behavior.
