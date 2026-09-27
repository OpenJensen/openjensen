# Cloud connections

Settings can associate this OPEN JENSEN workspace with a Google Cloud project.
Connection checks use credentials already managed by the provider CLI on the
machine running the API. They verify account access only;
they do not create GPU workers, check GPU quota, grant permissions, or start jobs.

## Provider setup

For Google Cloud, install the Google Cloud CLI and sign in with
`gcloud auth login`. OPEN JENSEN reads the active account and verifies access to the
selected project with `gcloud projects describe`. It does not change the CLI's
default project or Application Default Credentials. See the
[Google login reference](https://docs.cloud.google.com/sdk/gcloud/reference/auth/login)
and [project reference](https://docs.cloud.google.com/sdk/gcloud/reference/projects/describe).

Each verification command has a 15-second process timeout. Commands use argument
arrays, no shell, and disabled interactive prompts. Provider diagnostics are not
returned to the browser.

## API

| Endpoint | Behavior |
| --- | --- |
| `GET /api/v1/cloud-connections` | Return the Google Cloud record without calling a provider CLI. |
| `POST /api/v1/cloud-connections/gcp/connect` | Verify and save `{ "project_id": "my-project", "region": "us-central1" }`. |
| `POST /api/v1/cloud-connections/gcp/recheck` | Verify the saved selection again. |
| `POST /api/v1/cloud-connections/gcp/disconnect` | Remove the workspace association; leave provider credentials intact. |

The list response is `{ "providers": [...] }`. Mutations return one provider
record. A record contains `provider`, `name`, `status`, `config`, `identity`,
`checked_at`, `message`, and `setup_commands`. Status is one of `disconnected`,
`unverified`, `connected`, `setup_required`, or `error`.

A failed connection attempt returns its failure without replacing a previous
working selection. A failed recheck updates the current in-memory status and
clears its verified identity. Missing CLIs and expired credentials produce
`setup_required`, with a short message and provider setup commands. Malformed
requests and extra fields, including credential fields, are rejected with 422.

Only successfully verified project/region selections are written to
`FIREBIRD_DATA_DIR/cloud-connections.json`, through an atomic replacement with
owner-only permissions. Tokens, keys, identities, and command output are not
stored. After restart, saved selections show `unverified` until rechecked.
Obsolete provider records are ignored when loading saved settings.

A successful connection enables cloud training and preserves the existing GPU
and disk preferences. Fine-tune's Start button queues the run and prepares the
selected provider automatically; the user does not need a separate setup check.
A saved account makes the GPU selectable, while live access, quota, and capacity
are checked during preparation and launch. See [compute settings](compute-settings.md)
and the [SkyPilot job lifecycle](skypilot-training.md).

The backend tests in `tests/test_cloud_connections.py` mock all provider commands.
They do not inspect the developer's credentials or call any real cloud account.
