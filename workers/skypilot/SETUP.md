# GCP prerequisites

Use a project with billing, regional L4 quota, and these APIs enabled:
Compute Engine, IAM, Cloud Resource Manager, Service Usage, Cloud Storage,
Artifact Registry, and IAP.

Create these resources before `bash sky.sh configure`:

| Resource | Default |
|---|---|
| VPC | `sim-network`, custom subnet mode |
| Subnet | `sim-us-east4`, region `us-east4`, CIDR `10.42.0.0/24` |
| Input bucket | `${SIM_PROJECT_ID}-sim-assets` |
| Results bucket | `${SIM_PROJECT_ID}-sim-results` |
| Docker repository | `simulation`, region `us-east4` |

Build the worker using the sibling [Isaac instructions](../isaac_sim/README.md),
then place its immutable image digest in your local `task.yaml`.
The operator needs authority to create service accounts, custom roles,
IAM bindings, firewall rules, and IAP destination groups, plus launch VMs and
attach their identity. Existing project administrators can perform setup.

```bash
export SIM_PROJECT_ID=your-gcp-project
export SIM_LAUNCHER=user:you@example.com
bash sky.sh configure
```

Review same-name resource settings and existing project grants on `skypilot-v1`
before configuration. Resolve any conflicts reported by the script, then rerun
`bash sky.sh configure`.

Setup creates:

- `skypilot-v1` with a custom project role matching the pinned SkyPilot release's
  minimum compute permissions.
- Self-scoped Service Account User, input-bucket object viewer, results-bucket
  object creator, and Artifact Registry reader on `simulation`.
- IAP destination group `sim-ssh` for the subnet, granting `SIM_LAUNCHER` tunnel
  access. A firewall rule permits `35.235.240.0/20 → TCP 22` to that VM identity.

Use IAP for SSH and the VM identity for worker credentials. Give the worker VMs
outbound access through their external IPs for image pulls and package downloads.

Upload outputs under unique run prefixes using create-only uploads.

## Optional Cloud Build setup

For `../isaac_sim/build.sh`, enable Cloud Build and Cloud Logging and create:

| Resource | Default |
|---|---|
| Source staging bucket | `${SIM_PROJECT_ID}-sim-build-source` |
| Build identity | `sim-builder@${SIM_PROJECT_ID}.iam.gserviceaccount.com` |

Grant the builder `roles/storage.objectViewer` on the staging bucket,
`roles/artifactregistry.writer` on the `simulation` repository, and
`roles/logging.logWriter` on the project. The submitting operator needs build
submission permission, source-upload access to that bucket, and
`iam.serviceAccounts.actAs` on the builder, such as through
`roles/iam.serviceAccountUser` scoped to that account.

Provision the builder, staging bucket and build grants before running
`../isaac_sim/build.sh`.

References: [GCP permissions](https://docs.skypilot.co/en/v0.13.0/cloud-setup/cloud-permissions/gcp.html),
[pinned permissions](https://github.com/skypilot-org/skypilot/blob/v0.13.0/sky/provision/gcp/constants.py),
[IAP firewall](https://docs.cloud.google.com/iap/docs/using-tcp-forwarding).
