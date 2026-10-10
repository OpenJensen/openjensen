# Run with the shared simulation account

## Prepare

Use macOS/Linux with Bash, Python **3.12**, Google Cloud CLI, Git, SSH and rsync.
Keep the service-account key and complete checkpoint exports outside the
checkout. Obtain repository access and the key separately. Use a fresh OS account
when a SkyPilot API server already retains another login's credentials.

Store the key at `~/.config/isaac-act-runner/key.json`; prepare that directory
before applying the permissions below. Keep weights, saved processors and
normalization files together at the checkpoint paths used in the commands.

```bash
bash
chmod 700 ~/.config/isaac-act-runner
chmod 600 ~/.config/isaac-act-runner/key.json

git clone https://github.com/OpenJensen/openjensen.git
cd openjensen/workers/skypilot
export SIM_PYTHON=python3.12
bash install.sh

source ./auth.sh ~/.config/isaac-act-runner/key.json || exit 1
unset SKYPILOT_API_SERVER_ENDPOINT
cp config.runner.yaml config.yaml
cp rollout.experimental.example.yaml rollout.experimental.local.yaml
bash sky.sh check gcp
```

The helper selects
`sim-rollout-runner@project-5693e83a-db3a-43e1-98c.iam.gserviceaccount.com` and its
isolated gcloud profile. In each new Bash session export `SIM_PYTHON=python3.12`,
source `auth.sh` and unset `SKYPILOT_API_SERVER_ENDPOINT` before using `sky.sh`.
Use the preconfigured networking/IAM; reserve `configure.sh` for an administrator.

## Run

Accept [NVIDIA's container license](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/license), then validate and launch:

```bash
export ACCEPT_EULA=Y
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint ~/Downloads/act_step29000 --experimental --validate-only
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint ~/Downloads/smolvla_step20000 --experimental --validate-only

# Launch the selected model after its preflight passes.
bash launch-rollout.sh rollout.experimental.local.yaml \
  --checkpoint ~/Downloads/smolvla_step20000 --experimental
```

Replace the example checkpoint directories with your complete ACT/SmolVLA
exports. Keep the terminal attached for the launcher's two-hour deadline. Run
one selected checkpoint at a time with the configured H100 quota.

Use `--execute-steps 10` to select a shared replanning interval. Choose a positive
value no larger than the selected model's chunk size. The example scenario uses
150 control steps at 30 Hz. For normal mode supply verified calibration; use the
explicit `--experimental` flag when selecting its candidate calibration.

## Logs, results and cleanup

From another authenticated shell in this directory:

```bash
bash sky.sh jobs queue
bash sky.sh jobs logs JOB_ID isaac
bash sky.sh jobs logs JOB_ID vla
bash sky.sh gcloud storage cp --recursive \
  gs://project-5693e83a-db3a-43e1-98c-sim-results/policy-motion-tests/RUN_UUID ./results/

# Cancel only if the job is still active.
bash sky.sh jobs cancel JOB_ID --yes
bash sky.sh status --refresh
```

Replace `JOB_ID` and `RUN_UUID` with the printed identifiers. Download outputs
before cleanup, inspect both GPU resources after the group stops, and delete an
unused controller with `bash sky.sh down CONTROLLER_NAME`.

## Account setup

Have an administrator provision and review these grants before sharing the key:

| Scope | Required configured grant |
| --- | --- |
| Project | `simSkyLauncher`, defined in [runner-role.json](runner-role.json). |
| `skypilot-v1` service account | Service Account User for VM identity attachment. |
| `sim-ssh` IAP groups in `us-central1`/`us-east4` | IAP tunnel access. |
| Simulation results bucket | Storage Object Viewer. |
| `firebird-artifacts-project-5693e83a-db3a-43e1-98c` | Storage Legacy Bucket Reader and [checkpoint object role](checkpoint-object-role.json). |

Use unique checkpoint object names. Check effective project, VM identity, IAP and
bucket permissions under this account before launch; ask the administrator to
revoke the shared key when its work ends.
