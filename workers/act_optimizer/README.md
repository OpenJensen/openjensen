# ACT inference-only export

This isolated worker removes the training-only VAE encoder from a supported FP32
LeRobot ACT checkpoint. It copies retained tensors without numerical conversion,
preserves saved processors, and changes only `use_vae` to `false` in the policy
configuration. This is an inference export, not quantization, distillation, or a
training checkpoint. The application can export a registered, complete local ACT
training bundle through a separately configured CPU worker.

LeRobot's ACT evaluation path uses a zero latent and skips the VAE encoder. The
shared latent projection remains in the export. See the [pinned ACT source](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/modeling_act.py).

## Isolated installation

Use **uv 0.12.19** and Python **3.12** in this worker directory. The lockfile pins
LeRobot **0.6.1**, PyTorch **2.11.0**, torchvision **0.26.0**, safetensors **0.8.0**,
and their resolved dependencies. Linux selects the PyTorch CPU wheel index;
macOS selects its native wheels. Nothing is installed in the application environment.
Initial installation needs network access; export verification runs offline.

```sh
cd workers/act_optimizer
uv sync --locked --python 3.12 --extra cpu --extra dev
uv run --no-sync firebird-act-export /absolute/original-checkpoint /absolute/new-export \
  --timeout 120 > /absolute/export-receipt.json
```

The output must not exist or be inside the source. No original file is edited.
A repeated invocation with the same output fails instead of replacing it.

## Supported checkpoint and checks

The first recipe supports a full, non-PEFT ACT checkpoint with ResNet18, one RGB
camera, six state/action coordinates, one observation, independently saved prediction
and execution horizons (`1 <= execution <= prediction <= 1024`), MEAN_STD normalization, no temporal ensemble, no AMP,
ReLU and post-normalized transformer blocks. Bounded transformer dimensions
support the supplied checkpoint and a small real-policy test fixture. RGB dimensions
must be integers from 32 through 2048, with at most 2,073,600 pixels (1920 × 1080)
per image. The saved image shape is preserved exactly; export does not resize it.
Other architectures/configurations are rejected explicitly.

The checkpoint contains `config.json`, `model.safetensors`, saved pre/postprocessor
JSON and their referenced statistics files. `train_config.json` is optional,
preserved provenance only. Extra source files, arbitrary processor steps,
non-FP32 tensors, malformed/overlapping tensor ranges and unknown VAE keys fail.

For every normalized feature, mean/std statistics must exist with exact shapes;
values must be finite and standard deviations nonnegative. Known auxiliary
statistics have bounded shapes, positive integral counts and ordered ranges or
quantiles (float32 rounding tolerance: relative 1e-6 / absolute 1e-7 for ordering
only; action parity remains exact). Missing statistics cannot silently turn normalization into identity.

Source directory/file symlinks and nonregular files are rejected. Reads are
bounded to 512 MiB per file, with tighter JSON/statistics limits. Parent directories
are trusted operator-owned locations; this is not a hostile-filesystem sandbox.
Inputs are snapshotted and rehashed to detect changes. Retained tensor hashes,
source hashes, config change and complete output inventory are recorded.

## Three fresh CPU processes gate publication

1. Strictly load the original snapshot; compute two seeded synthetic observations.
2. Strictly load the transformed checkpoint; compare every raw and postprocessed
   action in both complete prediction-horizon × 6 chunks. Test two queue refills at
   the saved execution horizon and confirm reset forces another inference.
3. Assemble all package metadata, remove the owned original snapshot, then strictly
   reload the complete package. Block reads beneath the original source/snapshot
   locations through Python open auditing in that child and require the same outputs
   again. This check is not an OS filesystem sandbox.

Each child has its own finite deadline: 120 seconds by default, configurable from
1 to 600. A failure, mismatch or timeout prevents publication. No loose tolerance,
empty probe, incomplete chunk or non-finite value passes. Process exit errors retain
a bounded diagnostic tail. Temporary staging is cleaned on handled failure;
force-killing the parent can leave a hidden `.act-export-*` directory for operator
cleanup, but cannot publish a partial package.

The supported OPEN JENSEN loader explicitly overrides the device to CPU and
`pretrained_backbone_weights=None`, and loads saved processors locally. A generic
LeRobot loader without that override may request torchvision backbone weights.
Verification sets Hugging Face offline flags and blocks Python socket connect/DNS
audits. It is not an OS network namespace. CPU FP32 is the only verified execution
recipe; no CUDA or MPS execution is requested.

After successful checks, one OS no-replace rename publishes the complete directory,
including when another process races to create an empty destination. macOS publication
is locally tested. The additive native-worker CI job runs these checks on Linux;
check its result before claiming Linux acceptance. Native Windows publication is
implemented but unverified. File contents are flushed before
publication; no power-loss durability guarantee is made for directory metadata.

## Package and receipts

`manifest.json` binds every other file by size and SHA256. `recipe.json` records
source identities and exact removed/retained tensor inventory. `parity.json` records
synthetic fixture identities and action hashes; it explicitly leaves task success
unknown and calibration unverified. The command's external JSON receipt records
complete-package final reload and manifest hash, avoiding a self-referential hash.
A package's manifest provides integrity checking, not third-party authenticity.

To independently reload an existing published package in another process:

```sh
uv run --no-sync python -m firebird_act.probe /absolute/new-export /absolute/new-probe.json
```

The result path must be new and outside the package. Reload validates the package
inventory before accepting its weights. Keep receipts with the measured artifact.

## Scope of evidence

CPU parity on synthetic observations proves only the tested transformation's
inference equivalence. It does not establish calibration, pickup success, dataset
generalization, GPU memory, latency improvement or deployment on an 8GB device.
The policy is ACT with vision/state inputs; there is no language block to quantize.
The removed VAE is already bypassed during baseline inference, so size reduction
alone provides no evidence of faster inference.

The original source must remain available separately for original-objective training.
Exported training configuration is historical provenance, not a supported resume path.
See `evidence/local-cpu-parity.json` for the supplied checkpoint's measured software
receipt. Private model weights and fixture outputs are not committed.

## Tests

```sh
uv run --no-sync python -m pytest -q -rs
uv run --no-sync ruff check --target-version py312 src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy --config-file pyproject.toml src --follow-imports=silent
```

Tests generate a real small ACT policy and saved processors. They cover fresh-process
parity, strict loader failures, source mutation, missing/invalid statistics,
malformed proof reports, bounded process termination, package tampering and
no-replacement publication. CI does not need the private checkpoint or a GPU.


## Preparatory native-training source admission

`firebird_act.training_source.admit_training_source` performs a read-only,
point-in-time check of a completed application ACT training bundle. The caller
must supply its registered outer manifest SHA256 and artifact identity. It
verifies both complete manifest inventories, the saved ACT/full-training recipe
and upstream revision, required resume files, dataset/camera lineage, and the
existing exporter's config, processor and FP32 parsing rules. Bounded file reads
reject symlinks, special files, unsafe paths, unfinalized entries and source
changes detected during admission. Parent directories must remain trusted and
operator-owned; this is not a hostile-filesystem sandbox.

The saved positive integer counter in `training_state/training_step.json` must
match the inner manifest step. The [pinned upstream writer](https://github.com/huggingface/lerobot/blob/e595b7902714ba51f91e47523f66f89c5181b649/src/lerobot/common/train_utils.py#L140)
also stores batch, precision and topology metadata; these remain hashed provenance,
not an independently verified native-resume contract.

The full-bundle inventory permits at most 128 entries, 2 GiB per file and 4 GiB
total. These are inventory limits, not broader export support: the existing
exporter's tighter 512 MiB weight-file limit and JSON/statistics limits still apply.

The returned source is `checkpoint/pretrained_model`. Its receipt binds source
files, both manifests, optimizer step, upstream revision and dataset identity.
The original bundle, including optimizer/RNG state and saved probes, is unchanged.
Opaque training-state/probe files are hashed, never deserialized by this adapter.
Remote-only descriptors must first be materialized through a separately verified
path; this module performs no network calls.

Source admission itself neither exports nor loads a model. The returned path is a
point-in-time selection. The application bridge binds its selected file identities
to the exporter's actual snapshots and re-admits the whole bundle before publication.
The admission receipt remains explicitly admission-only; a separate export receipt
records the completed fresh-process checks.

## Application bridge

Configure both `act_export_python` and `act_export_root` in an operator-owned local
runtime. These select the pinned consumer interpreter and this worker directory;
the fixed module is `firebird_act.application`. The server never accepts executable
or source paths from the browser. This optional worker has no GPU or container
requirement and does not inherit a runtime's training image.

The fixed invocation is `python -m firebird_act.application REQUEST.json RESULT.json`.
The bounded schema1 request supplies `job_id`, `operation: "policy.export"`, an
absolute `output_dir`, and `artifact` with its registered `id`,
`format: "training_checkpoint"`, absolute local `path`, and `manifest_sha256`.
Other normal application fields are ignored. The result must be a new file directly
inside `output_dir`; it returns the same job identity and a registered
`format: "inference_export"` artifact, or a bounded error with a nonzero exit.

`inference-export/` is published atomically without replacement. It contains:

- `policy/`: the exact package tested by the existing three-process exporter;
- `lineage.json`: frozen complete-bundle admission and checkpoint identities;
- `verification.json`: the exporter's complete-package reload receipt;
- `manifest.json`: the application's SHA256 inventory and source lineage.

The envelope records the parent artifact and both source manifests, saved step,
model revision, dataset revision, camera and CPU-only verification scope. It does
not edit the tested inner package. The original resumable checkpoint is preserved.
The app downloads the envelope with the tested policy under `policy/policy/` in
its tar archive. Inference exports cannot be selected for training resume,
quantization or unimplemented simulator execution.

The Fine-tune checkpoint monitor offers **Export ACT inference package** for ACT.
It requires a ready project, enabled local export worker and complete native
checkpoint with pinned Hugging Face dataset lineage. Local dataset snapshot
lineage is explicitly unsupported by this export recipe. Cloud descriptors fail before worker launch; operators must first
materialize and register the complete verified checkpoint. No automatic model
or cloud download is started by this action.

## Native producer compatibility fixture

`scripts/native_checkpoint_fixture.py` is an optional, separate producer test
recipe. At native LeRobot0.6.2 revision
`e595b7902714ba51f91e47523f66f89c5181b649`, it makes a small real ACT policy, performs
a CPU optimizer update on seeded synthetic tensors, calls upstream
`save_checkpoint`, then in a fresh process uses upstream two-phase resume with
Accelerate1.14.0. Saved model, optimizer, RNG and the next optimizer update must
match exactly. Its final mode uses the actual OPEN JENSEN checkpoint bundler.
Run `generate`, `resume`, then `bundle` against the same new scratch directory
in the isolated producer environment. The ACT exporter stays on its unchanged
LeRobot0.6.1 lock and consumes only the resulting complete bundle.

The recorded macOS CPU proof also ran that bundle through the actual app API,
export subprocess, artifact registration and tar download; all archive members
matched the tested package, and source hashes remained unchanged. See
`evidence/native-application-parity.json`. This is a synthetic full-checkpoint
compatibility test, not validation of a trained policy or a production training
run. It establishes neither GPU/native-Windows support nor calibration, task
success, learned quality or dataset-loader resume correctness. The default
ACT CI tests remain on the pinned0.6.1 consumer; the optional0.6.2 producer is
not silently installed into that environment.

The recorded PushT checkpoint remains outside this exporter's six-coordinate recipe. Inference-only output must never replace or be presented
as the original resumable training checkpoint.


## Changed-horizon software acceptance

`native_checkpoint_fixture.py` accepts `--prediction-horizon 8 --execution-horizon 3`.
Run `generate`, `resume`, and `bundle` in order in the separately pinned native
0.6.2 producer environment. The generated batches pass through actual saved
normalizers, perform optimizer updates, save the upstream checkpoint, and verify
an exact next update after resume. This is generated software evidence, not a
recorded robotics dataset or learned task-quality measurement.

The complete application exporter now accepts the independently configured
horizons. New training recipes must include their checkpoint-root
`temporal-contract.json`; admission checks its exact sampling indices, FPS and
saved config. Export copies those original bytes into the inference policy.
Neither processors nor source checkpoints are rewritten. A supplied temporal
record that disagrees with the config fails even when manifests are rehashed.
Legacy coupled 100/100 checkpoints without this record remain supported.

Use the fixed `firebird_act.application` protocol for export, then
`firebird_quant.native_application` for INT8 or INT4 packing. Both manifests and
reports carry `prediction_horizon`, `execution_horizon` and the optional temporal
record hash. The packed model identity includes the record when present.

After packing, run the following in the existing pinned 0.6.1 CPU interpreter,
with `workers/act_optimizer/src:workers/firebird_quant/src:workers/isaac_sim` on
`PYTHONPATH` and offline/one-thread environment settings:

```sh
python workers/act_optimizer/scripts/temporal_http_fixture.py   /absolute/export/policy /absolute/packed/verification.json /absolute/new-float.json
python workers/act_optimizer/scripts/temporal_http_fixture.py   /absolute/packed/policy /absolute/packed/verification.json /absolute/new-packed.json --packed
python workers/act_optimizer/scripts/temporal_replay_fixture.py   /absolute/packed /absolute/new-replay-proof
```

The first two commands use fresh processes and real HTTP `/reset`/`/predict`
requests. They verify full predictions separately from the execution prefix and
exercise the actual queue refill count. The final command creates explicitly
synthetic observations, runs the real owned observation-replay worker and checks
its saved full predictions. It applies no robot or simulator actions. CPU proof
of 8/3 portability does not establish SmolVLA compatibility, GPU performance,
calibration, task quality, or a simulator rollout.
