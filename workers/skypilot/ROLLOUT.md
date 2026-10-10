# Launch an Isaac and policy Job Group

Use the isolated SkyPilot **0.13.0** client installed by `install.sh`.
For the configured shared account follow [runner setup](RUNNER.md).

## Prerequisites

Prepare a complete ACT/SmolVLA export with `config.json`, `model.safetensors`,
saved `policy_preprocessor.json`/`policy_postprocessor.json` and all referenced
statistics. Supply a robot/joint/gripper calibration file, L4 and Spot H100
quota in `us-central1`, the immutable Isaac worker image and
[the GCP resources](SETUP.md).

The default task uses an L4 `g2-standard-16` Isaac VM, a Spot H100
`a3-highgpu-1g` policy VM and an `e2-standard-4` CPU controller. Prepare the
policy environment with Python **3.12** and `lerobot[smolvla]==0.6.1`.

### Network and controller configuration

Have the network administrator add an unused `us-central1` subnet, its IAP
destination group and launcher-user access in the same VPC as both workers.
Merge the following into local `config.yaml`, replacing the project placeholder
and retaining the existing `us-east4` entry:

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
`gcp.capabilities: [compute]`. Size the controller disk for staged checkpoints.
Check the controller identity with `sky check gcp --verbose`; provide
`serviceusage.services.use` and `serviceusage.services.enable` in its custom role.

Allow TCP 22 and 8080 from the actual central-subnet CIDR to the simulation VM
identity. After reviewing that CIDR, an administrator can run:

```bash
SIM_CENTRAL_CIDR=10.43.0.0/24 # Replace with the actual, nonoverlapping subnet.
bash sky.sh gcloud compute firewall-rules create sim-rollout-internal \
  --project="$SIM_PROJECT_ID" --network=sim-network --direction=INGRESS \
  --allow=tcp:22,tcp:8080 --source-ranges="$SIM_CENTRAL_CIDR" \
  --target-service-accounts="skypilot-v1@$SIM_PROJECT_ID.iam.gserviceaccount.com"
```

Keep policy port 8080 on that private subnet and omit public `resources.ports`.

## Prepare and launch

From `workers/skypilot`, prepare local task and simulation manifests:

```bash
export SIM_PROJECT_ID=your-gcp-project
cp rollout.example.yaml rollout.local.yaml
cp ../isaac_sim/scenes/so101-pickup/rollout.example.yaml \
  ../isaac_sim/scenes/so101-pickup/rollout.local.yaml
```

Inspect the checkpoint with the worker's isolated Python:

```bash
PYTHONPATH=../isaac_sim .venv/bin/python - <<'PY'
from pathlib import Path
from sim_worker.rollout.checkpoint import inspect_checkpoint
print(inspect_checkpoint(Path("/path/to/exported-checkpoint")))
PY
```

Fill these local inputs before launch:

1. In task `rollout.local.yaml`, set `SIM_IMAGE`, `SIM_RESULTS_URI`, the absolute
   checkpoint mount at `~/vla-checkpoint`, and the inspector's `MODEL_ID`.
2. In the simulation manifest, use that exact `policy.model_id`; set the scene,
   camera/joints/task/control values and selected calibration. Keep those files
   inside `workers/isaac_sim` and use relative references.
3. Match capture dimensions and `POLICY_STATE_DIM` to the checkpoint. Choose
   `POLICY_ACTION_STEPS >= control.execute_steps`; begin with one for each.
4. Accept NVIDIA's container license, then validate and launch:

```bash
export ACCEPT_EULA=Y
bash launch-rollout.sh --validate-only
bash launch-rollout.sh
```

### Experimental policy motion

Copy a separate task selecting its candidate scene/calibration. Set
`control.steps` to at most 300 and explicitly select experimental mode:

```bash
bash launch-rollout.sh rollout.experimental.local.yaml --experimental --validate-only
bash launch-rollout.sh rollout.experimental.local.yaml --experimental
```

Keep this mode separate from `--check-ready`. For normal launches select verified
calibration instead.

### Select a checkpoint

Use `--checkpoint` to fill model identity, camera dimensions, state dimension and
horizon from a complete export:

```bash
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint ~/Downloads/act_step29000 --experimental --validate-only
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint ~/Downloads/smolvla_step20000 --experimental --validate-only
```

Replace those example directories with your exports. Remove `--validate-only`
to launch. With `--checkpoint`, choose `--execute-steps` as a positive replanning
interval no larger than the export's chunk size. Keep its saved processors and
normalization files; permit the SmolVLA server to fetch its VLM config/tokenizer
on first startup.

For experimental single-L4 capacity alternatives set Isaac `instance_type` to:

```yaml
ordered:
  - instance_type: g2-standard-32
  - instance_type: g2-standard-12
```

Retain `accelerators: L4:1`, image, disk and region. For an experimental zone use
`infra: gcp/us-central1/us-central1-c` rather than a second `zone` field.
Keep the launcher attached for its two-hour provisioning/execution deadline.

### Check the VMs before calibration

```bash
bash launch-rollout.sh --check-ready --validate-only
bash launch-rollout.sh --check-ready
```

Use the first command for local validation. The second provisions workers and
runs health/reset/prediction with the saved observation. Read its JSON readiness
report in job logs. Check the submitted group, primary `isaac` status and terminal
auxiliary `vla` status, then inspect GPU resources and the CPU controller.

### Attached completion and submission receipts

Retain the printed unique group, request ID, job ID and private receipt directory.
Copy `submission.json` and `launch-context.json` to an operator-owned job archive
before temporary files expire. If submission returns no job ID, inspect the saved
request/group before another launch.

`--yes --detach-run` returns after submission. For detached work, follow the
printed job/group yourself and cancel it explicitly when finished or stalled.
For attached readiness/experimental work, keep the launcher running through its
final status check and save `cancellation.json` when interrupted.

## Separate packed ACT CPU profile

Copy `rollout.packed-cpu.example.yaml` to a private task file. Fill its worker
image, results prefix, scene manifest and complete ACT `firebird_quant` package.
Select the packed runtime explicitly:

```bash
bash launch-rollout.sh /absolute/path/to/packed-cpu.yaml \
  --policy-runtime packed-act-cpu --experimental --validate-only \
  --checkpoint /absolute/path/to/complete-packed-policy
```

Use `--check-ready` without `--validate-only` for its remote readiness run; retain
`--experimental` for an explicit motion run. The template selects an L4 Isaac VM,
one `n2-standard-8` policy CPU VM and the CPU controller.

Keep the fixed `firebird_quant/src` and `act_optimizer/src` source mounts with the
reviewed `remote` scripts. The policy setup uses Python **3.12**, LeRobot **0.6.1**,
Torch **2.11.0+cpu**, torchvision **0.26.0+cpu**, safetensors **0.8.0**, NumPy
**2.2.6** and PyYAML **6.0.3** from `remote/policy-cpu.requirements.txt`.
Keep `--device cpu`, disabled Hub access and the template's one-thread settings.

## Results and lifecycle

```bash
bash sky.sh jobs queue
bash sky.sh jobs logs JOB_ID isaac
bash sky.sh jobs logs JOB_ID vla
bash sky.sh jobs cancel JOB_ID
bash sky.sh status --refresh
```

Read the exact run prefix under `SIM_RESULTS_URI`: `job-result.json`, worker logs,
`outputs/result.json`, `outputs/trajectory.jsonl`, `outputs/video.mp4` and
`outputs/final.ppm`. Match terminal reports to the saved group/model/manifest IDs.
If uploads fail, preserve the printed VM artifact directory before teardown.

After cancellation/completion, inspect GPU resources and failed provisioning;
stop an unused CPU controller with `bash sky.sh down CONTROLLER_NAME`. Submit a
fresh group with reset simulation/policy state after an interrupted episode.

### Import a complete policy folder or downloaded package

Use `--checkpoint` for a complete policy directory or the mutually exclusive
`--checkpoint-archive` for a TAR, optionally gzip/bzip2/xz-compressed:

```bash
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint-archive /path/to/ACT-inference-package.tar \
  --experimental --validate-only
```

Include config, weights, processors and every normalization statistic. Keep any
manifest's full inventory/hashes consistent. Fit within 256 members, 4 GiB
archive/expanded payload, 2 GiB per file, 2 MiB JSON/statistics, 16 MiB tensor
headers, eight path levels and 512 characters per path. Use regular files and
noncolliding relative names.

From the repository root, inspect without launching:

```bash
PYTHONPATH=workers/isaac_sim python -m sim_worker.rollout.checkpoint_package \
  --checkpoint-archive /path/to/package.tar --inspect-only
```

For a persistent imported copy, invoke that module with
`--source PATH [--archive] --output-dir NEW_DIR --json-output NEW_RECEIPT`.
Use absent output paths and put the receipt outside the payload. Read its
relative policy directory, camera/dimensions/horizons, model ID, complete file
inventory and manifest hashes before selecting the copy.

## Local checks

From `workers/skypilot`:

```bash
SIM_REQUIRE_SKYPILOT=1 .venv/bin/python -m unittest discover -s tests -v
```

References: [SkyPilot Job Groups](https://docs.skypilot.ai/en/latest/examples/job-groups.html),
[LeRobot SmolVLA](https://huggingface.co/docs/lerobot/smolvla).
