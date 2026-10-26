# Connect Google Cloud

Connect a Google Cloud project so OpenJensen can train, quantize and test policies on cloud GPUs. OpenJensen saves only the project ID and region. Your credentials stay with the Google Cloud CLI on the server.

## Connect a project

1. On the machine that runs the OpenJensen server, install [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install) and sign in:

   ```sh
   gcloud auth login
   ```

2. Optional: check that your account can see the project:

   ```sh
   gcloud projects describe PROJECT_ID
   ```

3. Open **Settings → Compute**. Under **Cloud providers**, select **Connect** on the **Google Cloud** card.
4. Enter the **Project ID** and **Region** (default `us-central1`), then select **Verify & save**.
5. Set up SkyPilot and application-default credentials, then train: see [Train on Google Cloud GPUs](skypilot-training.md).

The card then shows **Connected** with the project, region and signed-in account. Connecting also turns on **Allow cloud training** (under **Training preferences → Advanced**). Save your GPU and disk preferences in **Training preferences**; [compute settings](compute-settings.md) lists every value.

**Verify & save** runs two read-only checks, each limited to 15 seconds. `gcloud auth list` must show an active account, and `gcloud projects describe` must show the project as active. Nothing is created in your project.

- **Project ID:** 6–30 characters of lowercase letters, digits and hyphens. It starts with a letter and doesn't end with a hyphen.
- **Region:** any Google Cloud region, such as `us-central1`. The field suggests `us-central1`, `us-east1`, `us-west1`, `europe-west1`, `europe-west4` and `asia-east1`. Pick one that offers the GPUs you need.

Google references: [`gcloud auth login`](https://docs.cloud.google.com/sdk/gcloud/reference/auth/login) and [`gcloud projects describe`](https://docs.cloud.google.com/sdk/gcloud/reference/projects/describe).

## Manage the connection

| Action | What it does |
| --- | --- |
| **Recheck** | Runs the checks again for the saved project. Use it after a server restart, when the card shows **Verify connection**. |
| **Edit** | Changes the project or region. |
| **Disconnect** | Removes the saved project and region, and cloud GPUs disappear from job forms. It doesn't sign you out of the CLI or delete anything in your project. |

## Status and fixes

| Card status | API `status` | What to do |
| --- | --- | --- |
| **Not connected** | `disconnected` | Select **Connect**. |
| **Verify connection** | `unverified` | The connection is saved but not checked since the server started. Select **Recheck**. |
| **Connected** | `connected` | Nothing. |
| **Sign-in required** | `setup_required` | Install the CLI or refresh your sign-in, and choose an active project your account can access. In the dialog, open **Sign-in setup** for the commands (**Copy commands** copies them). Then recheck. |
| **Connection error** | `error` | Read the message and correct the project or region. Retry a timed-out check. If saved settings can't be read, connect again. |

## API

| Endpoint | Request |
| --- | --- |
| `GET /api/v1/cloud-connections` | Read saved provider records. |
| `POST /api/v1/cloud-connections/gcp/connect` | Send `{ "project_id": "my-project", "region": "us-central1" }`. |
| `POST /api/v1/cloud-connections/gcp/recheck` | Check the saved selection again. |
| `POST /api/v1/cloud-connections/gcp/disconnect` | Remove the workspace association. |

The connection is saved in `JENSEN_DATA_DIR/cloud-connections.json`. Keep credentials in the Google Cloud CLI environment; OpenJensen doesn't store Google Cloud tokens or keys.
