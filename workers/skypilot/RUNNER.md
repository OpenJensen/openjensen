# Configure a simulation service account

## Prepare

Use macOS/Linux with Bash, Python **3.12**, Google Cloud CLI, Git, SSH and rsync.
Use your own Google Cloud project and provision a dedicated
`sim-rollout-runner@<your-project>.iam.gserviceaccount.com` identity using the
[account requirements below](#account-setup). Keep its key and complete policy
exports outside the checkout. Use an isolated OS account when another SkyPilot
API server already retains different credentials.

Store the key at `~/.config/isaac-act-runner/key.json`; create that directory
before applying the permissions below. Keep weights, saved processors and
normalization files together in each policy export.

```bash
bash
chmod 700 ~/.config/isaac-act-runner
chmod 600 ~/.config/isaac-act-runner/key.json

git clone https://github.com/OpenJensen/openjensen.git
cd openjensen/workers/skypilot
export SIM_PYTHON=python3.12
bash install.sh

export SIM_AUTH_PROJECT_ID=my-gcp-project
source ./auth.sh ~/.config/isaac-act-runner/key.json || exit 1
unset SKYPILOT_API_SERVER_ENDPOINT
cp config.runner.yaml config.yaml
cp rollout.experimental.example.yaml rollout.experimental.local.yaml
```

Replace `my-gcp-project` with your project before sourcing `auth.sh`. The helper
checks that the supplied key belongs to that project's dedicated runner identity
and selects an isolated gcloud profile. In each new Bash session export
`SIM_PYTHON` and `SIM_AUTH_PROJECT_ID`, source `auth.sh`, and unset
`SKYPILOT_API_SERVER_ENDPOINT` before using `sky.sh`.

Edit the copied templates before checking or submitting a job:

- Replace every project placeholder in `config.yaml` and match its VPC, subnets
  and IAP destination groups to your configured resources.
- Set `SIM_IMAGE` to your immutable Isaac image and `SIM_RESULTS_URI` to your
  private results prefix in `rollout.experimental.local.yaml`.
- Set the policy checkpoint mount and model identity, or use `--checkpoint`
  below to derive the identity, camera dimensions and horizons from a complete
  export. Replace any remaining placeholders before launch.

```bash
bash sky.sh check gcp
```

Use the networking and IAM configured for your project; reserve `configure.sh`
for an administrator. Review quota for the accelerators selected in your task.

## Run

Accept [NVIDIA's container license](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/license), then validate and launch:

```bash
export ACCEPT_EULA=Y
export CHECKPOINT_DIRECTORY=/absolute/path/to/complete-policy-export
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint "$CHECKPOINT_DIRECTORY" --experimental --validate-only
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint "$CHECKPOINT_DIRECTORY" --experimental
```

Supply a complete ACT or SmolVLA export. Keep the terminal attached for the
launcher's two-hour deadline and run one selected checkpoint at a time.

Use `--execute-steps` to select a positive replanning interval no larger than the
selected model's chunk size. Set the scene, control steps and FPS in your local
manifest. For normal mode supply verified calibration; select `--experimental`
explicitly when using a candidate calibration.

## Logs, results and cleanup

From another authenticated shell in this directory:

```bash
bash sky.sh jobs queue
bash sky.sh jobs logs JOB_ID isaac
bash sky.sh jobs logs JOB_ID vla
bash sky.sh gcloud storage cp --recursive \
  gs://YOUR_RESULTS_BUCKET/YOUR_RUN_PREFIX/RUN_UUID ./results/

# Cancel only if the job is still active.
bash sky.sh jobs cancel JOB_ID --yes
bash sky.sh status --refresh
```

Replace the results bucket/prefix, `JOB_ID` and `RUN_UUID` with your selected
run's values. Download outputs before cleanup, inspect both GPU resources after
the group stops, and delete an unused controller with
`bash sky.sh down CONTROLLER_NAME`.

## Account setup

Have an administrator provision and review these grants before using the key:

| Scope | Required configured grant |
| --- | --- |
| Your project | `simSkyLauncher`, defined in [runner-role.json](runner-role.json). |
| `skypilot-v1` service account | Service Account User for VM identity attachment. |
| Your configured `sim-ssh` IAP groups | IAP tunnel access. |
| Your simulation results bucket | Storage Object Viewer. |
| Your private checkpoint bucket | Storage Legacy Bucket Reader and [checkpoint object role](checkpoint-object-role.json). |

Use unique checkpoint object names. Check effective project, VM identity, IAP and
bucket permissions under this account before launch. Revoke a temporary key when
its work ends.

Keep existing deployment bucket names, object URIs and identities when upgrading
your application. The product/CLI rename does not provision replacement resources.
