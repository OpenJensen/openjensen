# Run ACT or SmolVLA with Isaac

Code, login helper, configuration and scene live in this repository. Share the
service-account JSON key separately through a password manager's secure share.
Keep the key and checkpoint outside the checkout. The GCP account does not grant
GitHub access; the recipient also needs access to this repository.

## Prepare

Use macOS/Linux with Bash, Python 3.12, Google Cloud CLI, Git, SSH and rsync.
Use a fresh OS account if SkyPilot already runs under another login: its API
server retains its startup credentials, and a second port still shares state.
Authenticate below before starting SkyPilot.

Store the supplied key at `~/.config/isaac-act-runner/key.json`. Keep complete
checkpoint exports outside the checkout, including saved processors and normalization
files. The commands below use `~/Downloads/act_step29000` and
`~/Downloads/smolvla_step20000`. Model weights remain outside Git.

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

Select the export with `--checkpoint`; no model-specific YAML edits are needed.
The launcher derives its fingerprint, camera key, image size, state dimension and
action horizon. Scene, robot joint order and calibration remain explicit. Local
YAML edits are ignored by Git.

The helper selects only
`sim-rollout-runner@project-5693e83a-db3a-43e1-98c.iam.gserviceaccount.com`, uses a
separate gcloud profile and rejects keys inside the repository. No personal Google
login is required. In each new Bash session, export `SIM_PYTHON=python3.12`, source
`auth.sh` again and unset `SKYPILOT_API_SERVER_ENDPOINT` before using `sky.sh`.
Existing networking and IAM are configured; do not run `configure.sh` as this account.

## Run

After accepting the [NVIDIA container license](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/license):

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

Validation is local. Launch creates billable L4 and H100 Spot VMs plus a CPU
controller. Keep the terminal open for the launcher's two-hour deadline.
GPU capacity and Spot interruptions can prevent completion.

Each run selects one checkpoint. Run ACT and SmolVLA sequentially with the current
H100 quota. Use the ACT path in the same launch command to run ACT.
The default horizon is capped by the export's `n_action_steps`: 100 for this ACT
export and 50 for SmolVLA. Add `--execute-steps 10` to both runs for the same
replanning interval. Values above the model's chunk size are rejected.

Per-run manifest snapshots are mounted explicitly and cleaned up after submission;
the source templates are unchanged. SmolVLA needs its public VLM configuration and
tokenizer from Hugging Face on first startup.

This runs 150 steps, or five simulated seconds. Calibration is **unverified**;
the previous run demonstrated wrist/gripper motion, not a successful pickup.

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

Replace `JOB_ID` and `RUN_UUID` with the printed identifiers. Managed jobs remove
the GPU VMs; VLA normally finishes as `CANCELLED` after Isaac. The CPU controller
autostops after 10 idle minutes; its disk remains. Delete your idle controller with
`bash sky.sh down CONTROLLER_NAME` when no longer needed.

## Granted access

| Scope | Grant |
|---|---|
| Project | `simSkyLauncher`: the 28 permissions in [runner-role.json](runner-role.json), including project-wide VM management |
| `skypilot-v1` service account | Service Account User, to attach the VM identity |
| `sim-ssh` IAP groups in `us-central1` and `us-east4` | IAP tunnel access |
| Simulation results bucket | Storage Object Viewer |
| `firebird-artifacts-project-5693e83a-db3a-43e1-98c` bucket | Storage Legacy Bucket Reader + [`firebirdCheckpointObjects`](checkpoint-object-role.json): bucket metadata read; object read/list/create and metadata update |

The checkpoint grant cannot overwrite or delete existing objects. Use unique
checkpoint names; cleanup requires an administrator. Effective permissions and
bucket metadata reads passed as the runner on 2026-09-27. Private bucket settings
and simulation configuration are unchanged. No training launch was tested.

SkyPilot 0.13 uses the existing `skypilot-v1` VM identity. Its jobs inherit Compute
Admin, Storage Admin and project-wide Service Account User access. This account
is not restricted to its own VMs. Anyone holding its key has this access; ask an
administrator to revoke the key when no longer needed or if exposed.

Authentication, SkyPilot's Compute check, all 28 project permissions, VM identity
attachment, both IAP groups and result downloads passed on 2026-09-26. A full GPU
launch under this runner identity has not been tested.
