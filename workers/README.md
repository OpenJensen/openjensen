# Native worker boundary

These workers belong to OPEN JENSEN. Existing `firebird` command names,
`firebird_*` Python packages, `FIREBIRD_*` settings and artifact identifiers remain
unchanged for compatibility.

Workers execute native operations; the Python application owns project/job
metadata. Intake contracts live in `vla_platform.contracts`; policy job contracts
live in `vla_platform.lifecycle.contracts`. Protocol version 1 uses fixed operation
names, validated JSON request/result files, and registered subprocess entry points.
A worker never receives an arbitrary shell command from a dataset.

Metadata inspection runs via
`python -m vla_platform.datasets.worker REQUEST RESULT`. It uses the small
application environment and does not import Torch. It writes a result atomically
and leaves state transitions to the supervisor. Requests/results and bounded
errors survive under the application workspace's job directory. Intake has a
90-second job deadline; cancellation terminates the process before returning.

The candidate manifests under `workers/lerobot` and `workers/openvla_oft` are
**not resolved locks or supported capability declarations**. Resolve and verify
each environment before exposing additional operations. Native ML dependencies must not
be added to core. The reserved LeRobot environment is separate from the
implemented [SmolVLA training worker](smolvla_qlora/README.md). Simulator drivers
and SkyPilot use their supported separate runtime environments.

## Installing an isolated worker

Native worker projects under `workers/*` are excluded from the application uv
workspace. Run each worker's install and test commands from its own directory,
using its own Python version and environment. Installing the core must not install
worker ML dependencies. See each implementation's README for its entry points,
validation scope and remaining integration work. A standalone worker does not
change the application's advertised capabilities until its adapter is integrated.

## Integrated policy jobs

`vla_platform.lifecycle` supervises the registered `policykit.application` and
`firebird_vla.application` entry points through the same job owner. Workers return
hashed artifacts and measurement reports; only the core publishes project state.
See [workflow setup and contracts](../docs/policy-workflow.md). Capabilities require
an operator-configured runtime. The supported ACT → ACT256 distillation lane uses
its [isolated CPU worker](policy_distillation/README.md); other model families
remain unavailable. The standalone worker CLI
interfaces remain available for experiments. Registered operations require
model-specific runtime, hardware and task-quality verification before deployment.

## Standalone Isaac simulation and SkyPilot

[`isaac_sim`](isaac_sim/README.md) contains the YAML-driven Isaac recording worker.
[`skypilot`](skypilot/README.md) provisions GCP L4 clusters and submits recording
jobs. The worker publishes MP4s and terminal result manifests to GCS; the launcher
checks completion and the video checksum.

The [SO101 scene](isaac_sim/scenes/so101-pickup/README.md) is an approximate
reconstruction. Its calibration is unverified, so experimental motion must be
explicitly selected. Joint and speed limits can clip actions; a completed motion
run does not establish pickup success or replay fidelity.

These remain standalone execution tools with no application job submission,
execution capability or job-database integration. The application now has a
separate [read-only cloud status/log monitor](../docs/cloud-runs.md), fed by an
isolated observer. They do not connect SO101 training to LIBERO evaluation.

The launcher template manages disposable clusters: it requests deletion after
15 minutes of job inactivity and installs a guest watchdog requesting deletion
48 hours after VM creation. These controls are not a spending cap; verify actual
VM and disk deletion after use. The template is not a
default setup path for a persistent shared L4 host; review its
[lifecycle and cleanup behavior](skypilot/README.md#results-and-cleanup) before use.

## Standalone ACT inference export

The [ACT optimizer worker](act_optimizer/README.md) is a separate Python 3.12
environment. Its first recipe removes verified training-only VAE tensors while
preserving retained FP32 weights and saved processors. Strict loading, required
normalization statistics, complete action-chunk parity and a separate full-package
reload gate publication. This is inference-only compression, not quantization.

The application supports [exporting a final native ACT training checkpoint](../docs/cloud-act-export.md).
Standalone export does not verify calibration, pickup quality, GPU memory or
inference speed. Existing GGUF recipes do not apply to ACT. Keep original
checkpoints and generated policy packages outside Git.

## Unified model quantization

[OPEN JENSEN Quant](firebird_quant/README.md) provides a shared packed 4/8-bit API
for dense PyTorch models and safetensors checkpoints, without model-family or
parameter-name allowlists. It supports portable dequantize-on-access inference,
explicit coverage audits, tied weights and save/reload checks. It is a
standalone library/CLI. Its bounded [native ACT packing operation](firebird_quant/NATIVE_ACT.md)
is integrated with the application; broader library support does not imply an
application adapter. Model-specific quality and optimized runtime deployment
remain separate gates.
