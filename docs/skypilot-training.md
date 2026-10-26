# Train on Google Cloud GPUs

Run **Fine-tune** jobs on a single Google Cloud GPU through SkyPilot. Each job gets its own VM, saves its checkpoints to a private Cloud Storage bucket in your project, and deletes the VM when it ends.

## What you need

- A Google Cloud project with billing enabled and GPU quota in the region you use.
- On the machine that runs the OpenJensen server (the API host): Google Cloud CLI and SkyPilot 0.13 with Google Cloud support.
- For a gated or private model: a Hugging Face access token, with the model's access terms accepted on the Hub.

You don't install training environments on the API host. Each job builds a pinned environment on its VM from the worker sources in your checkout: Python 3.11 for SmolVLA and Psi-Zero, Python 3.12 for native LeRobot models.

Each model has its own dataset, camera and GPU memory rules; see [model-specific inputs](native-training.md). The SmolVLA recipe is in the [SmolVLA training guide](../workers/smolvla_qlora/docs/training/smolvla-qlora.md).

## Set up the API host

### Install and sign in

1. Sign in to Google Cloud. The CLI login lets OpenJensen verify your project; application-default credentials let SkyPilot and Cloud Storage act for you.

   ```sh
   gcloud auth login
   gcloud auth application-default login
   ```

2. Install SkyPilot. OpenJensen finds `sky` on `PATH` or at `~/.local/bin/sky`.

   ```sh
   uv tool install --with pip "skypilot[gcp]==0.13.0"
   ```

### Connect and choose defaults

1. Open **Settings → Compute** and [connect Google Cloud](cloud-connections.md) with your project ID and region. Connecting turns on **Allow cloud training**.
2. Optional: expand **Training preferences** and pick the **Default GPU** (default A100). Under **Advanced**, set **Disk size (GB)** (100–2000, default 200) and **Idle shutdown (minutes)** (1–60, default 10). Select **Save preferences**.
3. Optional: save a **Hugging Face access token** in the **Hugging Face** section.
4. Optional: on the **Dashboard**, select **Check resources**. The **Google Cloud** card shows **Ready** once SkyPilot setup is verified.

You can skip the check. The first job prepares SkyPilot itself: it creates a SkyPilot workspace pinned to your project, and the checkpoint bucket. GPU quota, capacity and billing are only checked when a job launches.

## Start a training run

> **Cloud jobs cost money.** Google Cloud bills your project from the moment the VM is provisioned until it is deleted. The job deadline is not a spending cap. The terminal UI asks you to confirm each paid Google Cloud job before it submits.

1. Inspect a dataset, then open **Fine-tune**.
2. **Dataset:** choose the inspected dataset and its **Cameras**.
3. **Base model:** choose one or more models and a **Training method**: **LoRA**, **QLoRA** or **Native training**. Each model you select runs as its own job on its own GPU.
4. **Compute:** review the recipe, then choose a cloud **GPU**. Only GPUs with enough memory for the selected models appear. **Checkpoints** under **Advanced settings** sets how many checkpoints are saved.
5. Select **Start fine-tuning** (with two models, **Start 2 training jobs**).
6. Open the run to follow preparation, step progress, loss, events and checkpoints. **Training details** holds the activity log, the saved checkpoints and **Worker output**.

When you start, the job records its project, region, GPU, machine type, disk size, idle interval, SkyPilot server and workspace. If the Google Cloud connection changes before the VM launches, the job stops and asks you to start again.

Where the data goes:

- A Hugging Face model and dataset download on the VM. A local training copy (native LeRobot models only) uploads with the job.
- Checkpoints and final artifacts go to private Cloud Storage. See [checkpoint storage](cloud-artifact-storage.md).
- Manifests, recipes, logs and metrics come back to the workspace.

## GPUs

| GPU | Memory | Machine type | Runtime ID |
| --- | --- | --- | --- |
| L4 | 24 GB | `g2-standard-4` | `skypilot-gcp-L4` |
| T4 | 16 GB | `n1-highmem-4` | `skypilot-gcp-T4` |
| A100 | 40 GB | `a2-highgpu-1g` | `skypilot-gcp-A100` |
| A100-80GB | 80 GB | `a2-ultragpu-1g` | `skypilot-gcp-A100-80GB` |

**Fine-tune** lists both A100s as **A100**, each with its memory; **Default GPU** names the larger one **A100-80GB**. Every job uses one on-demand GPU, never spot. A GPU is available only if SkyPilot lists its machine type in your region. SmolVLA with **Native training** needs at least 24 GB.

## Use a checkpoint

The default recipe trains for 20,000 steps and saves five checkpoints: one every 4,000 steps, including the final step. Each checkpoint is copied to Cloud Storage as soon as it is written.

From the run:

- **Download:** open **Training details** and select the **Download** link for a checkpoint. The server streams it from Cloud Storage as a TAR.
- **Resume:** after the run stops, pick a **Checkpoint** under **Use a trained checkpoint** and select **Resume from checkpoint**. The dataset, model and timing stay bound to the saved checkpoint and its recipe. Select **Resume fine-tuning**.
- **Quantize (SmolVLA):** pick a **Checkpoint** and select **Quantize checkpoint**. In **Quantize**, choose **Language compression**: **8-bit** (Q8) or **4-bit** (Q4, experimental). Keep a cloud GPU under **Compute** and select **Start quantization**. When the job finishes, download the GGUF from **Output**. SmolVLA checkpoints from **Native training** can't be quantized to GGUF yet; download or resume them instead.
- **Export (ACT):** pick the final checkpoint and select **Download checkpoint and export**. See [ACT export](cloud-act-export.md).

## Cancel and clean up

- **Cancel run** stops the job, then OpenJensen deletes its VM. The activity log shows "Stopping Google Cloud resources for fb-…".
- Checkpoints already saved stay in Cloud Storage after a cancel, failure or timeout. Check which ones were saved before you resume.
- As a fallback, SkyPilot deletes the VM once it has been idle for the **Idle shutdown** interval.
- If deletion can't be confirmed, the job shows `Cloud cleanup needs attention. Run sky down -y <cluster> on this host.` Run that exact command on the API host, and check the project for leftover VMs.
- Deleting the VM does not delete stored checkpoints. Manage those [in the bucket](cloud-artifact-storage.md#delete-stored-checkpoints).

When the server restarts:

- It retries cleanup of saved cloud jobs for up to 10 seconds (at most 256 records), and adds a cleanup message to each job it can't confirm.
- The job's cluster name is in its `sky-state.json` under `<workspace>/jobs/<job-id>/`.
- While any cleanup is unconfirmed, OpenJensen refuses new cloud jobs and server updates. Stop the resources, then restart the server.

## Troubleshooting

Fix the credentials, quota or capacity problem a job reports before you start another one.

| What you see | What to do |
| --- | --- |
| Install SkyPilot with Google Cloud support first. | Install SkyPilot as in [Set up the API host](#set-up-the-api-host). |
| Google Cloud application credentials are missing or expired. | Run `gcloud auth application-default login`. |
| Required Google Cloud APIs are not enabled or visible. | Run `sky check --workspace WORKSPACE gcp`, where `WORKSPACE` is `gcp_status.workspace` from `GET /api/v1/compute-settings`. The project needs the Compute Engine, Cloud Resource Manager and IAM APIs. |
| SkyPilot does not offer this single-GPU machine in your region, or no supported machines are listed there. | Choose another GPU or connect another region. |
| Your project does not have enough quota to start the GPU. | Request more GPU quota in Google Cloud, or choose another GPU. |
| Google Cloud has no available capacity right now. | Choose another GPU or try again later. |
| Google Cloud denied access to start this GPU. | Check the connected account's access to the project. |
| The SkyPilot workspace points to a different project. | Fix the workspace in SkyPilot. OpenJensen never overwrites it. |
| Checkpoint storage cannot be prepared. | See [checkpoint storage](cloud-artifact-storage.md#if-storage-fails). |
| Google Cloud settings differ from this queued run. | Start the job again with the current settings. |
| Cloud worker setup or execution failed before publishing its result. | Open **Worker output** under **Training details**. |

## Technical notes

- **Deadline:** jobs started from **Fine-tune** run for at most 24 hours (86,400 seconds). Through the API the default is 7,200 seconds; the range is 30–86,400.
- **Hugging Face token:** passed to the VM as a SkyPilot secret and redacted from logs.
- **VM disk:** holds only the latest two checkpoint copies. Cloud Storage keeps every published checkpoint.
- **Names:** clusters are named `fb-<hash>-<id>` and the SkyPilot workspace `firebird-gcp-<hash>`. Don't rename them; saved jobs refer to them.

## Related guides

- [Test a policy on a cloud GPU](cloud-inference.md): **Evaluate** and **Run → Check inference** with a saved GGUF or package.
- [Monitor cloud runs](cloud-runs.md): follow cloud jobs and external rollouts.
