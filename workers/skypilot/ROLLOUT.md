# Isaac + VLA Job Group

For the configured ACT/SmolVLA experiment and dedicated service account, start with
[runner setup](RUNNER.md).

`launch-rollout.sh` submits two GPU tasks through SkyPilot 0.13.0:

```text
SkyPilot jobs controller (CPU VM)
  ├─ isaac: L4, g2-standard-16, on-demand
  │    Isaac → calibrated observations → private HTTP
  └─ vla: H100, a3-highgpu-1g, Spot
       ACT or SmolVLA → action targets → Isaac
```

Isaac is the primary task. SkyPilot terminates VLA after Isaac finishes. A managed
Job Group also needs a CPU controller VM; this is additional to the two GPU VMs.
No cloud resources are created by copying files or running `--validate-only`.

## Prerequisites

- An exported Kite ACT or SmolVLA checkpoint: `config.json`, `model.safetensors`, saved
  `policy_preprocessor.json` and `policy_postprocessor.json`, and their referenced
  statistics files. A training-run link alone is insufficient.
- A verified calibration file for this robot, joint order, action convention and
  gripper. Candidate calibration requires the explicit experimental mode below.
- L4 quota and one **Spot H100** quota in `us-central1`, plus available capacity.
- The immutable Isaac worker image used by the existing launcher.
- The existing [GCP setup](SETUP.md), extended as follows. These changes require
  your network administrator; the launcher never changes IAM or firewall rules.

### Network and controller configuration

Use the same VPC for both tasks and the controller. Add a `us-central1` subnet
with an unused CIDR, its regional IAP destination group, and your launcher user's
IAP access. Existing `configure.sh` is hardcoded to `us-east4`; setting an
environment variable does not configure the new region.

Merge these settings into local `config.yaml`, keeping its existing project,
identity, VPC and `us-east4` settings. Replace the project placeholder:

```yaml
gcp:
  subnet_names: [sim-us-east4, sim-us-central1]
  ssh_proxy_command:
    # Keep the existing us-east4 entry too.
    us-central1: >-
      gcloud compute start-iap-tunnel %h %p --listen-on-stdin
      --project=CHANGE_ME_PROJECT_ID
      --region=us-central1 --network=sim-network --dest-group=sim-ssh
jobs:
  force_disable_cloud_bucket: true
  controller:
    resources:
      infra: gcp/us-central1
      instance_type: e2-standard-4
```

Keep `gcp.use_internal_ips: true`, `gcp.remote_identity: SERVICE_ACCOUNT`, and
`gcp.capabilities: [compute]`. Local files stage through the controller instead
of a newly created SkyPilot bucket. Source and checkpoint copies therefore use
controller disk; size its disk for the exported model if needed.

The controller's service account must pass `sky check gcp`, independently of your
local account. SkyPilot 0.13 requires `serviceusage.services.use` and
`serviceusage.services.enable`; legacy worker roles can omit both. An administrator
can grant these through a narrow custom role. A controller-side `sky check gcp
--verbose` identifies missing permissions when jobs report `NoCloudAccessError`.

Allow **TCP 22 and 8080** from the central subnet to VMs using the simulation
service account. Port 22 lets the controller manage workers directly; SkyPilot
removes its IAP proxy after entering the VPC. Port 8080 carries policy traffic.
The existing IAP-only SSH rule does not allow controller-to-worker traffic.
For example, after verifying the subnet CIDR, an administrator can create:

```bash
SIM_CENTRAL_CIDR=10.43.0.0/24 # Replace with the actual, nonoverlapping subnet.
bash sky.sh gcloud compute firewall-rules create sim-rollout-internal \
  --project="$SIM_PROJECT_ID" --network=sim-network --direction=INGRESS \
  --allow=tcp:22,tcp:8080 --source-ranges="$SIM_CENTRAL_CIDR" \
  --target-service-accounts="skypilot-v1@$SIM_PROJECT_ID.iam.gserviceaccount.com"
```

Do not expose port 8080 with SkyPilot `resources.ports` or an internet ingress
rule. This v1 policy API has no authentication or TLS and belongs on this private
network. The subnet rule allows these ports to all simulation VMs in its scope.

The launcher labels both GPU VMs with a unique rollout ID. Isaac discovers exactly
one VLA VM through the Compute API and uses its private address. It does not rely
on Job Group DNS: native-cloud hostname discovery is not a supported contract in
the installed version, despite internal SSH host-mapping code.

## Prepare and launch

From `workers/skypilot`:

```bash
export SIM_PROJECT_ID=your-gcp-project
cp rollout.example.yaml rollout.local.yaml
cp ../isaac_sim/scenes/so101-pickup/rollout.example.yaml \
  ../isaac_sim/scenes/so101-pickup/rollout.local.yaml
```

Edit the two local files:

1. Inspect the checkpoint without loading model weights into a GPU:

   ```bash
   PYTHONPATH=../isaac_sim .venv/bin/python - <<'PY'
   from pathlib import Path
   from sim_worker.rollout.checkpoint import inspect_checkpoint
   print(inspect_checkpoint(Path("/path/to/exported-checkpoint")))
   PY
   ```

2. In `rollout.local.yaml`, set `SIM_IMAGE`, `SIM_RESULTS_URI`, the **local absolute
   checkpoint directory** mounted at `~/vla-checkpoint`, and `MODEL_ID`.
   Use the inspector's computed `model_id` fingerprint.
3. In the simulation manifest, set `policy.model_id` to that exact identifier,
   confirm camera/joints/task/control values, and select the verified calibration.
   Keep scene and calibration files inside `workers/isaac_sim`, with relative
   paths. Set capture width and height to the checkpoint's image dimensions.
4. Match `POLICY_STATE_DIM` to the number of joints. Keep
   `POLICY_ACTION_STEPS >= control.execute_steps`; start with both set to one.

After accepting NVIDIA's container license:

```bash
export ACCEPT_EULA=Y
bash launch-rollout.sh --validate-only
bash launch-rollout.sh
```

### Experimental policy motion

Use a separate local task file selecting the candidate calibration and scene.
Set `control.steps` to at most 300 in its simulation manifest:

```bash
bash launch-rollout.sh rollout.experimental.local.yaml --experimental --validate-only
bash launch-rollout.sh rollout.experimental.local.yaml --experimental
```

`--experimental` explicitly permits candidate calibration during validation and
simulation. The launcher forwards `SIM_EXPERIMENTAL=1` to the remote runner, which
passes the adapter flag. Ordinary launches still require verified calibration.
The mode cannot be combined with `--check-ready`.

### Select a checkpoint

Use the same scenario for either supported architecture:

```bash
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint ~/Downloads/act_step29000 --experimental --validate-only
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint ~/Downloads/smolvla_step20000 --experimental --validate-only
```

Remove `--validate-only` to launch. `--checkpoint` also works with standard
rollouts and `--check-ready`; their calibration and readiness checks still apply.
It overrides the checkpoint mount, model fingerprint, camera key, capture size,
state dimension and action horizon in an isolated manifest snapshot. It preserves
the original scenario and task files, including calibration and joint order.
Relative checkpoint paths resolve from the directory where the command is invoked.

By default, execution uses the smaller of the scenario's `execute_steps` and the
export's `n_action_steps` (or `chunk_size` when absent). Set `--execute-steps` with
`--checkpoint` to choose a shared replanning interval for comparisons. The value
must be positive and no larger than the model's chunk size.

The included ACT and SmolVLA exports share their dataset and normalization
statistics, but that does not verify the joint calibration. Compatible exports
must still match the scenario's robot, action convention and camera view. Other
architectures require an inference backend; matching dimensions alone is insufficient.

SmolVLA's VLM configuration and tokenizer are fetched on first startup. The full
export supplies the trained weights; keep its saved processors intact.

For capacity fallback, replace Isaac's `instance_type` with:

```yaml
ordered:
  - instance_type: g2-standard-32
  - instance_type: g2-standard-12
```

Keep shared `accelerators: L4:1`, region, image and disk settings. Experimental
mode accepts the single-L4 sizes `12`, `16` and `32`; each override may change only
`instance_type`. Standard mode retains `g2-standard-16`.
To pin an experimental Isaac zone, use `infra: gcp/us-central1/us-central1-c`;
do not add a separate `zone` field alongside `infra`.

Each test receives a unique `isaac-<policy-type>-test-` group name, disables retries after
task errors, and inherits the group cleanup. Keep the launcher running for its
two-hour provisioning and execution deadline; detached tests require manual
monitoring and cancellation. Results remain experimental even if execution succeeds.

### Check the VMs before calibration

```bash
bash launch-rollout.sh --check-ready --validate-only
bash launch-rollout.sh --check-ready
```

This provisions the same Job Group and loads the actual checkpoint on H100. The
L4 task uses the pinned Isaac image to send the saved first observation and front
camera frame, resized to the checkpoint dimensions, over the private network.
It validates health, reset and prediction, then prints a JSON readiness report.
It never loads the simulation or applies robot actions; verified calibration is
still mandatory for normal rollouts.

The readiness task has a 30-minute run limit. The local launcher waits at most two
hours for provisioning and execution, then requests cancellation of that exact job
ID if known, otherwise its unique group name. Keep the launcher running: SkyPilot
otherwise retries capacity or preemption indefinitely. Managed jobs attempt GPU VM
cleanup; verify resource inventory separately. The CPU controller autostops separately. Use this check to validate deployability, not to
leave idle GPU VMs running. Readiness JSON remains in the managed job logs.

The auxiliary `vla` task is expected to finish as `CANCELLED` after Isaac completes.
Confirm the group and primary `isaac` task are `SUCCEEDED`, the readiness JSON
says `ready`, and both GPU VMs are removed.

`--yes --detach-run` submits without confirmation or log streaming. The local
timeout then covers submission only. Monitor the printed unique group name and
cancel it explicitly if capacity stays unavailable; detached managed jobs can
otherwise keep retrying indefinitely.

### Attached completion and submission receipts

The launcher uses the pinned SkyPilot 0.13.0 SDK through the same isolated
`sky.sh` environment. It loads the prepared parallel Job Group using SkyPilot's
internal YAML loader and CLI defaults; it does not replace the scheduler.
Without `--yes`, SkyPilot's resource optimization and interactive confirmation
remain enabled. Declining returns failure before submission. `--detach-run`
returns after the job ID is received, with no log following or completion check;
its zero exit means **submitted**, not completed.

Every launch prints a private temporary receipt directory outside the repository.
`submission.json` records the prepared YAML digest, pinned SDK version, launch
request ID, returned job ID, group name and task IDs/roles. The request ID is saved
before waiting for submission, and the job ID before attaching logs. Both IDs are
also printed in case a later disk write fails. A lost response is an uncertain
submission: inspect that request; the launcher never automatically resubmits or
falls back to a latest-job lookup. Copy receipts to your experiment evidence
before your operating system clears temporary files.

SkyPilot can stop attached log following with exit `101` (`NOT_FINISHED`) while
the auxiliary is cancelling. Only that code triggers up to 120 seconds of final
status checks, every two seconds between completed reads, each read limited to
40 seconds or the remaining deadline. The readiness/experimental two-hour outer
limit still includes this check. Timeout stops owned local processes, with up to
five seconds of graceful shutdown followed by bounded forced cleanup. This local
cleanup precedes the cloud cancellation request: it is a supervised cancellation
guard, not an exact VM-runtime or billing cap. Other attached exit codes are preserved.

The check queries only the submitted job ID and validates the exact group,
task-ID/name bindings, parallel execution and explicit primary/auxiliary roles.
It never refreshes or restarts a stopped cloud controller. Exit 101 becomes zero
only when `isaac` is `SUCCEEDED` and `vla` is terminal `CANCELLED` or `SUCCEEDED`.
The group status is **derived from the single primary task**, matching the pinned
SkyPilot rule; it is not a separate status returned by the queue API. Cancellation
observed before primary success, failed tasks, missing metadata (including older
controller role fields), identity mismatches, unknown states, failed reads and
deadlines remain unresolved with exit 101. The receipt retains the original
attached exit and the first and last validated observations.

An interrupted/timed-out readiness or experimental launch requests cancellation
with its existing 120-second cancellation bound and records `cancellation.json`.
A successful cancellation request or terminal auxiliary `CANCELLED` is **not**
evidence that VMs were deleted or billing stopped; inspect cloud resources
separately. Queue rows also do not identify who initiated cancellation. These
outcomes concern execution only, not calibration, pickup success or model quality.

Validation checks local files, model fingerprint, policy type, state/action/image
dimensions, referenced processor statistics, calibration, resource choices and
network configuration before dispatching SkyPilot. It does not verify live IAM,
firewall rules, quota, GPU capacity, or successful tensor loading. The server loads
the checkpoint and its saved processors at startup.

The H100 uses a separate Python 3.12 environment with `lerobot[smolvla]==0.6.1`.
Isaac uses the existing image with the synced worker source mounted read-only;
the rollout code does not require an image rebuild. Changing dependencies does.

## Results and lifecycle

```bash
bash sky.sh jobs queue
bash sky.sh jobs logs JOB_ID isaac
bash sky.sh jobs logs JOB_ID vla
bash sky.sh jobs cancel JOB_ID
bash sky.sh status --refresh
```

Isaac uploads `job-result.json`, worker logs, `outputs/result.json`,
`outputs/trajectory.jsonl`, `outputs/video.mp4` and `outputs/final.ppm` beneath
`SIM_RESULTS_URI/<episode UUID>/`. `job-result.json` is published last. Confirm
both reports say `succeeded`; that means the rollout executed, not that the robot
completed the pickup task. On upload failure, preserve the printed VM artifact
directory before cancelling or deleting the job.

Setup, policy discovery and inference are bounded. The Isaac container has a
one-hour timeout; VLA has a 75-minute limit and is normally terminated sooner by
the group. Managed jobs attempt GPU cleanup; verify the resource inventory,
including failed cleanup or provisioning. Inspect `sky status` for the CPU controller and any failed provisioning before leaving; the controller is a
separate reusable resource and is not the auxiliary VLA task.

Spot interruption during an episode fails that episode after its request timeout.
There is no transparent replay or policy-state recovery. Submit a fresh group to
start a new episode with a reset simulation and empty action history.

## Local checks

```bash
SIM_REQUIRE_SKYPILOT=1 .venv/bin/python -m unittest discover -s tests -v
```

The launcher suite uses synthetic queue/submission responses and the real pinned
SDK's local YAML parser, typed queue records and declined-confirmation path with
network/submission calls blocked. Real local subprocess tests cover query and
outer-wall deadlines, creation-window interruption, repeated interruption and
owned-child cleanup. CI requires the SDK compatibility tests rather than silently
skipping them. These are offline contract tests: live reconciliation of a newly
submitted cloud job remains unverified until a separately authorized run.

On 2026-09-26, managed job 2 passed readiness in `us-central1-a`: L4
`10.43.0.3` sent a recorded observation to the ACT step-29000 checkpoint on H100
Spot `10.43.0.4:8080` and received a six-joint action over private HTTP. These addresses
belong to that run; discovery resolves each new run. No robot actions were applied.
This verifies deployment and inference, not Isaac rendering or learned control.
A normal trained-policy rollout requires verified calibration; experimental motion
must opt in explicitly.

References: [SkyPilot Job Groups](https://docs.skypilot.ai/en/latest/examples/job-groups.html),
[LeRobot SmolVLA](https://huggingface.co/docs/lerobot/smolvla).

### Import a complete policy folder or downloaded package

`--checkpoint` now admits a private, byte-preserving snapshot of a complete ACT
or SmolVLA export. Use `--checkpoint-archive` for a TAR (including gzip/bzip2/xz
compression) instead. These flags are mutually exclusive:

```bash
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint-archive /path/to/ACT-inference-package.tar \
  --experimental --validate-only
```

An OPEN JENSEN download contains an outer `policy/` envelope and an inner policy
folder; the resolver locates exactly one checkpoint without rewriting config,
weights or processors. External exports do **not** need an OPEN JENSEN manifest.
When any `manifest.json` is present, its complete inventory and hashes must match.
Imported parity, task-success or calibration claims never become verified merely
because the file hashes match. Bare weights are insufficient: the config, saved
processors and every required normalization statistic must accompany them.

Admission is CPU-only and does not load Torch, unpickle data, import custom model
code, download anything or start a cloud job. It verifies safetensors layout and
required finite normalization statistics. A successful strict runtime reload is
still required to prove that the tensor names/shapes form an executable model.
Only the built-in ACT/SmolVLA processor registries are admitted. SmolVLA can still
need its backbone config/tokenizer when the inference server starts, as described
above; this import step does not establish offline inference readiness.

Limits: 256 total members, 4 GiB archive/expanded payload, 2 GiB per file,
2 MiB JSON/statistics files, 16 MiB tensor headers, eight path levels and 512
characters per path. Links, devices, FIFOs, traversal, duplicate/case-colliding
paths, sparse files and ambiguous multiple policies are rejected. Unknown ordinary
extra files are retained but never executed. Directory snapshots require
POSIX directory-FD support (tested on macOS; Linux validation is recorded
separately). The private copy is removed after launcher submission/attached run,
including errors; originals stay unchanged. Job imports use an exclusive new
output directory and complete only after their receipt has been written.

Inspect without launching:

```bash
PYTHONPATH=workers/isaac_sim python -m sim_worker.rollout.checkpoint_package \
  --checkpoint-archive /path/to/package.tar --inspect-only
```

The fixed application bridge can persist an admitted copy using
`--source PATH [--archive] --output-dir NEW_DIR --json-output NEW_RECEIPT`.
Both output paths must be absent; the receipt is outside the payload. The JSON
contains `schema_version: 1`, a relative `directory` below `NEW_DIR`, the existing
`checkpoint` fields (`policy_type`, `model_id`, camera/dimensions/horizons), the
archive `source_sha256` (null for folders), verified manifest hashes, and exact
`files` with SHA-256/byte counts. `runtime_verified` and `calibration_verified`
remain false, and `task_success` remains null. The caller must verify this receipt
and inventory before registering or running the copied policy.

The current scene object is a **cup**. Some historical dataset/task strings say
“cube”; those recorded labels are preserved as evidence, not silently corrected.
Changing the live task prompt or admitting another policy does not establish
cup-pickup success or fix the still-unverified joint calibration.
