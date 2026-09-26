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

The script checks same-name resources before granting access. It refuses
conflicting subnet, role, firewall, destination-group settings, and existing
broader project grants on `skypilot-v1`; review those grants manually rather
than weakening the checks. It creates no VM.

Setup creates:

- `skypilot-v1` with a custom project role matching the pinned SkyPilot release's
  minimum compute permissions, verified against the installed package.
- Self-scoped Service Account User, input-bucket object viewer, results-bucket
  object creator, and Artifact Registry reader on `simulation`.
- IAP destination group `sim-ssh` for the subnet, granting `SIM_LAUNCHER` tunnel
  access. A firewall rule permits `35.235.240.0/20 → TCP 22` to that VM identity.

The compute role permits project-wide VM lifecycle operations. Worker credentials
come from the VM identity; operator credentials and service-account keys are not
copied. SSH uses IAP; external IPs provide outbound access for image pulls and
package downloads. No public SSH rule or Cloud NAT is required by this profile.

Storage writes use unique run prefixes and create-only uploads. The worker
identity cannot overwrite results through these bucket grants. Existing grants
on other identities are left unchanged.

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

Provision these separately; `configure.sh` does not create the builder,
staging bucket, or build grants. An existing immutable worker image can be used
without Cloud Build.

References: [GCP permissions](https://docs.skypilot.co/en/v0.13.0/cloud-setup/cloud-permissions/gcp.html),
[pinned permissions](https://github.com/skypilot-org/skypilot/blob/v0.13.0/sky/provision/gcp/constants.py),
[IAP firewall](https://docs.cloud.google.com/iap/docs/using-tcp-forwarding).
