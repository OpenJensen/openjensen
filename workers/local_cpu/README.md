# Persistent local ACT CPU runtimes

Use this explicit setup command for the application’s **ACT distillation** and
**packed ACT observation replay** lanes. It creates two separate Python3.12
environments beneath a new persistent directory:

- `act-model/`: LeRobot0.6.1, Torch2.11.0, torchvision0.26.0 and safetensors0.8.0,
  installed from the existing ACT worker’s frozen lock.
- `dataset-reader/`: LeRobot0.6.2 source at
  `e595b7902714ba51f91e47523f66f89c5181b649`, with separately pinned and hashed
  dataset dependencies. It does not borrow packages from another environment.

The application Python environment stays free of model/dataset dependencies.
These environments are persistent and independent of temporary proof directories.
The selected complete repository checkout must also remain available: the app
loads its reviewed worker modules from that checkout. This is not an installer
for a packaged desktop app or an offline wheel bundle.

## Prerequisites and supported scope

- Apple Silicon macOS or Linux x86_64. Windows and other architectures are deferred.
- An existing **Python3.12** interpreter and **uv0.12.19** executable. Supply their
  absolute paths. This command never installs or upgrades Python/uv itself.
- A complete checkout containing the model, dataset, replay and distillation workers.
- Network access and disk space for the initial explicit installation. The reader
  is several packages; do not assume it is a small download. Policy weights and
  datasets are not downloaded by setup.
- FFmpeg/ffprobe are separate prerequisites for application video intake and
  teaching conversion. Setup does not install system packages. The runtime probe
  checks the PyAV H.264 decoder, not system FFmpeg availability.

Linux uses the exact CPU Torch/torchvision wheel URLs and SHA256 identities from
`workers/act_optimizer/uv.lock`. macOS uses native wheels and all jobs still select
CPU. The full reader dependency closure is version/hash locked; LeRobot’s source
archive is independently hash pinned. No floating branch or fallback model fetch
is used. The native reader’s critical source files are checked again after install.

## Plan first: no writes or subprocesses

From the repository root, supply real absolute paths (examples below must be
replaced for your machine). Choose a new versioned installation directory whose
parent you have explicitly created. Prefer a persistent user-owned location such
as `$HOME/.local/share/openjensen/cpu-20260927`; do not use `/tmp`.

```sh
python3 workers/local_cpu/manage.py plan \
  --root /absolute/persistent/cpu-20260927 \
  --python /absolute/existing/python3.12 \
  --uv /absolute/existing/uv
```

`plan` prints JSON containing exact commands and input hashes. `install` without
`--execute` also only prints this plan. No current runtime configuration, cloud
connection, environment variable, shell profile or service is changed.

## Explicit installation

After reviewing the plan:

```sh
python3 workers/local_cpu/manage.py install --execute \
  --root /absolute/persistent/cpu-20260927 \
  --python /absolute/existing/python3.12 \
  --uv /absolute/existing/uv
```

The destination must not exist. An existing or partial environment is never
upgraded, repaired or replaced in place. A failed or interrupted installation
retains its plan and local log for diagnosis, and does not publish a successful
`installation.json`. Choose a new destination after fixing the cause; cleanup of
old directories remains an explicit operator action.

Each installation step has a15-minute timeout and its own process group. Signals
request termination and cleanup; no automatic retry is made. A successful receipt
requires both dependency checks and fresh offline probes. Probes verify exact
runtime versions, CPU Torch, native reader source hashes, worker import paths,
PyArrow and the PyAV decoder. They do not load model weights or evaluate a task.

```sh
python3 workers/local_cpu/manage.py verify --root /absolute/persistent/cpu-20260927
```

## Generate a new configuration; preserve the old one

```sh
python3 workers/local_cpu/manage.py config \
  --root /absolute/persistent/cpu-20260927 \
  --base /absolute/existing/runtimes.json \
  --output /absolute/new/runtimes-with-local-cpu.json
```

The output parent must exist and output file must be new. `--base` is optional for
a new workspace; when supplied, all existing runtime/source entries are preserved.
ID collisions fail without replacing either file. The generated file has owner-only
permissions and contains no newly discovered credentials. Existing operator data in
a base registry is copied to the requested new file but never printed to the console.
Do not commit a private registry containing credentials.

The two added IDs are `local-act-distillation-cpu` and `local-act-replay-cpu`.
Both are dedicated local CPU lanes; they do not enable generic engine Run/Evaluate,
SmolVLA distillation, native CUDA or Isaac. The command **does not activate the file**.
Review it, then explicitly launch/restart the app with `FIREBIRD_RUNTIME_CONFIG`
pointing to the new path. Preserve the existing data directory and other settings.
Do not restart while an owned job is active. Google Cloud settings are stored and
managed separately; this command does not read or modify them.

Distillation still requires an admitted complete ACT teacher and explicit disjoint
train/validation/final episode groups. Replay still requires a registered complete
INT4/INT8 ACT artifact, an immutable matching dataset, explicit selected frames and
coordinate attestation. Setup cannot make an incompatible model/data pair compatible.
Replay returns full predicted actions; it never executes them on a robot/simulator.
Neither lane claims task quality or calibration from successful setup.

## Reproduce and review dependency locks

`reader-*.in` records the exact dependency closure selected from the validated
reader metadata; every native LeRobot dataset requirement was checked against it.
`reader-*.lock` adds distribution hashes using uv0.12.19 public package metadata.
The platform-specific CPU URLs are reused from the existing reviewed ACT lock.
`lock-provenance.json` records the input/output hashes and source pin. The reader
package itself is installed separately with `--no-deps --no-build-isolation` from
the exact archive; setuptools81.0.0 is already present in its locked dependencies.
The final `uv pip check` validates the installed full native dependency graph.

For maintainers, public-metadata resolution updates files but installs nothing:

```sh
uv pip compile workers/local_cpu/reader-macos-arm64.in \
  --python 3.12 --python-platform aarch64-apple-darwin \
  --generate-hashes --no-annotate --no-header \
  --output-file /absolute/new/reader-macos-arm64.lock
uv pip compile workers/local_cpu/reader-linux-x86_64.in \
  --python 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-annotate --no-header \
  --output-file /absolute/new/reader-linux-x86_64.lock
```

Only independently reviewed lock updates should replace repository locks. No
install or fresh-machine acceptance is implied by dependency resolution or a
mocked installer test. The initial implementation was tested with generated local
processes and offline ACT sync dry-run; new persistent installation remains an
explicit opt-in action after review.

Primary references: [uv PyTorch CPU index configuration](https://docs.astral.sh/uv/guides/integration/pytorch/),
[uv lock/sync behavior](https://docs.astral.sh/uv/concepts/projects/sync/),
[the pinned LeRobot source](https://github.com/huggingface/lerobot/tree/e595b7902714ba51f91e47523f66f89c5181b649).
See [Replay](../isaac_sim/NATIVE_REPLAY.md) and
[Distillation](../policy_distillation/README.md) for supported inputs and output limits.

Optional Bash completion: set `OPENJENSEN_REPO` to the absolute checkout path and
source `workers/local_cpu/completions.bash` in that shell. The wrapper
`openjensen-cpu-setup` exposes the same explicit commands. No profile is edited.
