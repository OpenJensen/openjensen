# Cloud checkpoint storage

Cloud jobs save checkpoints and outputs in a private Cloud Storage bucket in your Google Cloud project. The workspace keeps only their manifests, recipes, metrics and logs, so the weights stay in Cloud Storage until you delete them.

## The bucket

Connect the Google Cloud project and prepare SkyPilot on the API host first; see [Set up the API host](skypilot-training.md#set-up-the-api-host).

The first cloud job creates the bucket `gs://firebird-artifacts-<project-id>` in the connected region. Each job then writes under `gs://firebird-artifacts-<project-id>/jobs/<job-id>/<stage>/`.

- The bucket uses uniform bucket-level access with public access prevention enforced, and carries the labels `application=firebird` and `firebird_project=<project-id>`. OpenJensen refuses a bucket of that name that isn't private or whose `firebird_project` label names another project.
- Access uses the connected account's application-default credentials through SkyPilot's Python environment. Keep credentials in that Google Cloud/SkyPilot environment.
- Keep the `firebird` bucket name and the stored object paths. Saved jobs and artifact records point to them, so renaming breaks downloads and resumes.

| Path under the job prefix | Contents |
| --- | --- |
| `checkpoints/checkpoint-<step>/` | One training checkpoint, for example `checkpoint-004000`: its files under `checkpoint/` plus `manifest.json`. |
| `checkpoints.json` | Index of the run's published checkpoints. |
| `artifact/` | The final output, such as a GGUF or package, with its `manifest.json`. |
| `evidence/` | Inference reports, timing logs and GPU readings. |
| `result.json` | The job's result. |

Each stored artifact holds at most 10,000 files and 100 GiB. Each artifact record in the workspace contains its OpenJensen project ID, optimizer step and cloud location.

## Use a stored checkpoint

1. Open the saved **Fine-tune** run.
2. Under **Use a trained checkpoint**, pick the exact **Checkpoint**.
3. Choose what to do with it:
   - **Download:** open **Training details** and select its **Download** link. The server streams a TAR (`policy/manifest.json` plus the checkpoint files) straight from Cloud Storage to the browser and checks every file's checksum. Nothing is cached on the server.
   - **Resume:** select **Resume from checkpoint**. The new VM downloads the checkpoint directly from Cloud Storage.
   - **Quantize (SmolVLA):** select **Quantize checkpoint**. Quantization runs on a cloud GPU from the cloud copy.
   - **Export (ACT):** follow [CPU export setup](cloud-act-export.md).

## Delete stored checkpoints

Deleting a job's VM never deletes its bucket objects; they stay, and are billed as storage, until you remove them. Follow each job through checkpoint publication and GPU cleanup first. If cleanup is uncertain, run the cluster command shown on that job; see [cancel and clean up](skypilot-training.md#cancel-and-clean-up).

Then manage retained objects in the bucket yourself. To see what is stored:

```sh
gcloud storage ls gs://firebird-artifacts-PROJECT_ID/jobs/
```

After you delete a job's objects, the workspace still lists its checkpoints, but downloads, resumes and quantization from them fail. Workspace backups don't include Cloud Storage objects; see [back up](upgrading.md#back-up).

## If storage fails

| Message | What to do |
| --- | --- |
| Checkpoint storage cannot be prepared. | Check that the connected account can create and read the bucket and its objects, that the bucket is private, and that its labels name your project. |
| Cloud checkpoints are stored in GCS; their local metadata needs synchronization. | The checkpoints are safe. The server retries the sync when it next starts. |
| Cloud artifact is unavailable or its stored integrity check failed. | The objects are missing or changed in the bucket. Choose another checkpoint. |
| Cloud artifact download preparation timed out. | Retry the download; check the server's network access to Cloud Storage. |
