# Cloud connections

## Provider setup

1. Install Google Cloud CLI on the API host.
2. Sign in with `gcloud auth login`.
3. Open **Settings & diagnostics → Compute** and enter the project ID and region.
4. Connect Google Cloud, save the GPU/disk preferences, and continue to [training](skypilot-training.md).

Inspect a project with `gcloud projects describe`. See the [login reference](https://docs.cloud.google.com/sdk/gcloud/reference/auth/login) and [project reference](https://docs.cloud.google.com/sdk/gcloud/reference/projects/describe).

## API

| Endpoint | Request |
| --- | --- |
| `GET /api/v1/cloud-connections` | Read saved provider records. |
| `POST /api/v1/cloud-connections/gcp/connect` | Send `{ "project_id": "my-project", "region": "us-central1" }`. |
| `POST /api/v1/cloud-connections/gcp/recheck` | Check the saved selection again. |
| `POST /api/v1/cloud-connections/gcp/disconnect` | Remove the workspace association. |

The association is saved in `FIREBIRD_DATA_DIR/cloud-connections.json`. After restarting, use **Recheck**. For `setup_required`, install the reported CLI or refresh credentials, then recheck. For `error`, read the message and correct the project/region selection.

Keep credentials in the Google Cloud CLI environment. Configure SkyPilot and application-default credentials through [compute setup](compute-settings.md).
