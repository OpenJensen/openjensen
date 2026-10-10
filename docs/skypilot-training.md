# Google Cloud training through SkyPilot

Connect Google Cloud once in Settings, choose a model and GPU, and click **Start
fine-tuning**. OPEN JENSEN accepts the job promptly and performs preparation in the
background. The application remains on its host. SkyPilot starts an isolated GCP
worker and downloads the pinned model and dataset there.

The server needs SkyPilot 0.13 with GCP support, Google Cloud CLI and existing
application-default credentials. OPEN JENSEN verifies the selected project and
creates or reuses its own SkyPilot workspace without overwriting unrelated
settings. A queued run retains the selected project, region, GPU, machine type,
disk size, idle teardown interval, server endpoint and workspace. All launch,
monitoring and cleanup commands use that saved selection. Quota and live capacity
are verified by the provider during preparation/provisioning.

## Training adapters

SmolVLA has a dedicated Python 3.11 LoRA/QLoRA worker with pinned dependencies,
selected cameras, deterministic episode split, train-only normalization, optimizer
state, validation loss and a fresh-process reload check. Other LeRobot policies
use a separate Python 3.12 environment and a pinned upstream revision. Their
architecture-specific configuration is held in `lifecycle/native_profiles.py`,
with separate immutable model revisions. ACT, Diffusion Policy, Multi-Task DiT
and VQ-BeT initialize from their native baseline recipes; the other entries
initialize from released model checkpoints.

Psi-Zero uses its own upstream Python 3.11 environment and frozen vision-language
backbone. Its adapter checks dataset compatibility before training. The catalog
reports the native method, minimum GPU memory and camera constraints where
applicable. Gated model access also requires the user's Hugging Face account to
have accepted the model's access terms. A token alone cannot grant model access.

Catalog availability describes supported adapters and configuration requirements;
it does not establish a model's task quality.

## Checkpoints, results and quantization

New jobs store checkpoints and final artifacts in private GCS, not on the
application host. Only small manifests, recipes, metrics and logs are returned.
The default is five checkpoints across a run, including the final step. The
checkpoint picker labels the latest checkpoint and lets the user choose any
published step. An interrupted run can resume its last committed checkpoint.
See [cloud artifact storage](cloud-artifact-storage.md) for publication, retention,
integrity, downloads and recovery details.

Quantize accepts a selected SmolVLA checkpoint directly. A GCP job reloads its
trained weights, merges adapters, converts to floating GGUF and packs language
weights to Q4 or Q8 while preserving action weights. It verifies tensor types,
shapes, hashes and checkpoint lineage before publishing a downloadable GCS
artifact. The progress view distinguishes export, conversion, packing and
verification. This is a representation check; robot task success requires a
separate compatible evaluation.

Native non-SmolVLA checkpoints are available for training, resume and download.
The existing GGUF compiler supports SmolVLA; other architectures are rejected
before a quantization GPU is provisioned.

## Monitoring, cancellation and recovery

Metrics count completed optimizer steps, not gradient accumulation microbatches
or FP16 overflow skips. Preparation remains indeterminate until a worker reports
progress. The monitor persists phases, loss curves, learning rate, elapsed time,
checkpoint steps, worker output and observed-speed ETA. Reaching 100% optimizer
steps is followed by checkpoint publication, reload verification and cloud
cleanup before the job is marked successful.

Every run receives a unique OPEN JENSEN-owned cluster. Its remote worker has the
recorded deadline, and SkyPilot receives an idle autodown interval. Completion,
failure and cancellation perform scoped cleanup; startup retries unfinished
owned dispatches. `sky-state.json` records identity and cleanup status durably.
Unconfirmed cleanup is reported visibly with the exact cluster command; OPEN JENSEN
never treats a missing local acknowledgement as proof that billing stopped.

Hugging Face tokens are optional for public models. Saved tokens are owner-only,
masked in API responses and passed with SkyPilot's secret mechanism. They never
appear in saved recipes or command arguments, and streamed diagnostics redact
known tokens. Model and dataset revisions remain explicit in reproducibility
metadata.

The separate [Cloud runs view](cloud-runs.md) reads operator-published snapshots
for standalone cloud workloads, including upstream Isaac rollout runners. It does
not replace the training monitor or control OPEN JENSEN-managed jobs.

## Evaluate and Run on cloud GPUs

In the web app, select **Evaluate** or **Run → Check inference** for this engine path. **3D simulation** and **Replay observations** have separate requirements.

The application advertises `engine_evaluation` and `run` on supported GCP
runtimes. Evaluate accepts a saved SmolVLA GGUF or deployment package, retrieves
it on the worker, and measures real CUDA inference with a fresh reload, finite
actions, per-call latency samples and sampled GPU memory. It uses the model's
recorded camera count and reports the real action dimensions separately from
padded channels.

Run executes the selected GGUF in an independent process and publishes a
reload-verified package. Selecting an existing package executes that exact
package again without repacking it. The app keeps its control plane on the application host;
model files and packages remain in GCS.

These cloud actions currently support synthetic-input engine checks. They do
not run a simulator, establish task success, or drive a physical robot. LIBERO
and Spatial protocols continue to require a separately prepared compatible
native runtime. Requests for unsupported cloud simulation fail before allocation.
