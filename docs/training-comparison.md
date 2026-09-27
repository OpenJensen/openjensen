# Data and training comparison — 2026-09-26

Reviewed Kite's signed-in dataset → cameras → models → configuration flow and an
existing live run in Chrome, plus its [training API documentation](https://docs.kiteml.com/platform-api/training-runs)
and [OpenAPI schema](https://api.kiteml.com/v1/openapi.json). No Kite training run
was launched or changed during this review.

| Area | Kite observed behavior | Firebird behavior and changes |
| --- | --- | --- |
| Dataset entry | Nine starter cards; Hugging Face ID/URL or GCS path; inspect before training | Two immutable Hugging Face examples: SO-101 pickup and SO-100 pick & place. Actual episode previews verified for both starters. |
| Cameras | Inspect dimensions; map dataset camera names to policy slots. UI explicitly says unchecking is only a label and the trainer still consumes every camera. | Explicit selected-camera list is passed to the SmolVLA worker and preserved in the recipe. |
| Models | Fifteen selectable policies in the signed-in picker, two marked experimental. Multi-select creates separate GPU runs. | All fifteen observed policies have registered cloud worker routes: dedicated SmolVLA, thirteen native LeRobot profiles, and isolated Psi-Zero. GPU verification is recorded per actual run; catalog visibility is not certification. |
| Setup | Policy-specific defaults, GPU minimums, steps, batch size and checkpoint frequency. SmolVLA displayed 20,000 steps, batch 64 and automatic approximately five checkpoints. | Existing defaults match these. Advanced controls now expose seed, learning rate, accumulation and validation settings; resume retains its original recipe. |
| Monitoring | Live run percentage, checkpoint steps/downloads and loss sparklines. API documents phase, step/max steps/loss and logs. Numeric loss was absent in the particular checkpoint table inspected. | Persistent phase/action, optimizer-step percentage, training/validation loss curves, learning rate, elapsed time, observed-speed ETA, checkpoint list, activity and worker log tail. |
| Reproduction | Public create schema does not expose seed or immutable dataset/model revisions; that does not establish what Kite stores internally. | Download accepted request, immutable dataset metadata and revision, model/backbone revisions, resolved recipe, seed, episode split, available package/GPU/source evidence and checkpoint lineage. |

Kite's picker contained ACT, Diffusion Policy, EO-1, EVO-1, GR00T N1.7,
Multi-Task DiT, pi0, pi0-FAST, pi0.5, SmolVLA, VLA-JEPA, VQ-BeT, WALL-X,
XVLA (experimental) and Psi-Zero (experimental). These are observations of its
available UI choices, not an independent training certification.

## Reading a Firebird run

The monitor is on **Fine-tune**, below setup and run history. Selecting an older
run restores its persisted observations. Refreshing the page does not reset
progress. Events and training measurements come from supervised worker output,
with local metric-file fallback and bounded histories. Raw worker output exposes
GPU provisioning and dependency/download diagnostics as well as training output.

The percentage means completed optimizer steps divided by the recorded target.
Preparation stays indeterminate until a worker reports a step; 100% optimizer
completion can still be followed by checkpoint saving, a fresh-process reload
check, artifact transfer and cloud cleanup. Completion follows the application's
terminal job status. Loss is imitation loss, not robot task success. ETA uses
observed step timing and is not a promise about download or provisioning time.

Worker events describe data/statistics preparation, model/adapters loading,
optimization, held-out validation, saving and reload verification. Metrics
include the first step, configured logging intervals and validation results.
Non-finite telemetry is excluded, partial writes are tolerated, and a stopped
or failed run retains its recorded progress and error.

The JSON download distinguishes an accepted request from a worker-resolved recipe;
defaults are not retroactively invented for legacy runs. Checkpoints already
contain optimizer, scheduler, random-number states, the recipe and episode splits.
New runs additionally record package/GPU/precision details and worker source
hashes, including uncommitted code. Reproducibility remains dependent on the
recorded software and hardware; bitwise equality across environments is not
guaranteed.

API: `GET /api/v1/jobs/{id}/training` and
`GET /api/v1/jobs/{id}/training/reproducibility`.

## Training evidence and remaining work

The existing [GPU validation record](workflow-validation.md) documents a real
two-step SmolVLA QLoRA run, held-out evaluation, checkpoint save, fresh-process
reload and floating export on an RTX 3070. It does not establish convergence,
robot task success, or acceptance of every later code change.

During this review an existing A100 run encountered capacity shortages in three
GCP zones, acquired a VM in a fourth, installed dependencies and began loading
the dataset and model. It then reported step 10 with training loss
`0.1388608068227768`, gradient norm `0.7824709415435791`, no skipped optimizer
steps, BF16 compute and `143.99065` worker elapsed seconds, and saved checkpoints
at steps 5 and 10. These are direct observations of an existing LoRA run, not
convergence or final reload evidence. The old UI collapsed much of that work into
broad status labels.
The existing process and its GPU job were left running; its copied worker cannot
gain new telemetry retroactively. The new dashboard/backend require the updated
application process for full telemetry, and richer worker phase/environment events
apply to new runs. While an older server is still running, the new frontend falls
back to its persisted events and real reported step counts, displays a restart
notice and leaves unavailable metrics unclaimed. This was verified in the user's
live workspace at step 40/20,000 (0.2%), without interrupting the GPU job.

The subsequent native-training integration adds pinned worker routes for the
remaining Kite choices. See [native policy training](native-training.md) for
initialization, camera/data constraints, dependency isolation and the distinction
between configuration validation and actual GPU evidence. New runs keep
checkpoints in private GCS rather than downloading every checkpoint locally;
see [cloud artifact storage](cloud-artifact-storage.md).

Validation includes the full core suite (730 passed, one skipped), the lightweight
worker suite (110 passed, four CUDA checks skipped), all 168 desktop/mobile browser
checks, TypeScript, Ruff and a production build. Core/build checks used the
installed Python 3.14.6, Node 25.8.0 and pnpm 12.6.0; the repository's declared
runtime pins remain unchanged. The existing
isolated CPU reader was missing its dependency; installing its already-declared
`pyarrow==25.0.1` enabled the complete core run. The worker suite also passed on
its supported Python 3.11.14 with its declared pytest 8.4.2. Browser fixtures
exercise monitor behavior without launching paid GPU jobs; fixture losses and
progress are not model-performance evidence. No new GPU training was launched
for this comparison.
