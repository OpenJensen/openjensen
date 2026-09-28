# GitHub Actions on Compute Engine

Issue: [#208](https://github.com/sobhanb-eth/firebird-hackathon-codebase/issues/208).

Zero to three **standard `e2-standard-2`** workers (2 vCPU, 8 GB RAM),
Ubuntu 24.04 x86-64, 50 GB balanced boot disks. One GitHub job per VM.
No Docker daemon or GPU stack on workers. Windows stays GitHub-hosted.

```text
GitHub job queue ← poll each minute ← Cloud Scheduler → private Cloud Run
        │                                    controller + GCS lease
        │                                             │
        └── JIT runner ← immutable image ← Compute Engine slots 0, 1, 2
                 │
              one job → power off → controller deletes VM + boot disk
```

The controller runs in GCP, so exhausted GitHub-hosted minutes cannot prevent
worker startup. Cloud Run requires a container internally; Cloud Build creates
it with buildpacks. Runner jobs execute directly on VMs.

## Repository dependencies

| Component | Image contents |
| --- | --- |
| Application | Node from `.node-version`; pnpm and Playwright from `package.json`; Python from `.python-version` |
| Python workers | uv 0.12.19; Python 3.11 and 3.12.14 |
| Browser checks | Pinned Playwright Chromium headless shell and native libraries |
| Video/simulation contracts | FFmpeg, Git, jq, build-essential, OpenGL/glib libraries |
| Runner | Official Linux x64 archive, version and SHA-256 in `image/versions.json` |

Existing workflows still install their locked Python/Node dependencies and run
all checks. Torch 2.2/2.11, NumPy 1/2, application Python 3.14, and worker Python
3.11/3.12 remain isolated. CUDA, Isaac Sim, Rust/Tauri, and macOS packaging are
outside current Linux CI; they are not installed merely because folders exist.
Python 3.11 resolves to the supported patch at image build time; the resolved
version and OS packages are recorded in `/opt/firebird-image`. Images are
immutable deployment artifacts, not byte-for-byte reproducible OS builds.

## Before deployment

Use a **dedicated CI project with billing enabled**. The controller's custom
Compute role is project-wide; project isolation confines its blast radius.
Choose a region/zone and confirm quota for six worker vCPUs, three external IPs,
and an additional temporary image builder. The image builder also costs money.

Install `gcloud`, Terraform 1.13.5+, and Packer 1.14.2+. Authenticate interactively:

```sh
gcloud auth login
gcloud auth application-default login
```

The deploying identity needs permission to enable APIs, manage project IAM,
service accounts, Compute networks/images/VMs, Cloud Run, Scheduler, Storage,
Artifact Registry, Secret Manager, and Cloud Build. Packer uses IAP SSH; grant
the operator `roles/iap.tunnelResourceAccessor` plus Compute permissions.
It does not need a downloaded GCP service-account key.

Create a GitHub App owned by the repository owner:

- Repository permissions: **Administration: read/write**, **Actions: read**.
- Disable webhooks; this controller polls.
- Install it on this repository only.
- Record its App ID and installation ID; generate a private key.

Do not commit the key or paste it into Terraform variables. Administration write
is required by GitHub's repository JIT-runner registration API. It is powerful;
keep this App limited to CI administration for this repository.

## 1. Bootstrap GCP resources

```sh
cd deploy/github-runners/terraform
cp terraform.tfvars.example terraform.tfvars
# Fill project_id, app_id, installation_id; adjust region/zone together if needed.
terraform init
terraform plan -out=bootstrap.tfplan
terraform apply bootstrap.tfplan
```

Defaults create the network, builder-only IAP firewall, service accounts,
restricted roles, registry, lease bucket, and **empty** secret. No controller,
Scheduler job, or worker exists yet. Keep Terraform state private and durable;
for team use configure a GCS backend before applying. Do not commit state/plans.

Upload the GitHub App key from its local file:

```sh
gcloud secrets versions add firebird-ci-github-key \
  --project YOUR_PROJECT --data-file=/absolute/path/to/github-app.pem
```

Only the controller can read this secret. Workers have no service account,
GCP credentials, GitHub App key, or reusable repository administration token.

## 2. Build the runner image

From the repository root, select an exact Ubuntu base image:

```sh
gcloud compute images list --project ubuntu-os-cloud \
  --filter='family=ubuntu-2404-lts-amd64' --format='table(name,creationTimestamp)'
```

Create an ignored `deploy/github-runners/image/local.pkrvars.hcl`:

```hcl
project_id  = "YOUR_PROJECT"
zone        = "us-central1-a"
network     = "firebird-ci"
subnetwork  = "firebird-ci"
source_image = "EXACT_UBUNTU_IMAGE_NAME"
image_name   = "firebird-ci-YYYYMMDD-COMMIT"
```

```sh
packer init deploy/github-runners/image/runner.pkr.hcl
packer validate -var-file=deploy/github-runners/image/local.pkrvars.hcl \
  deploy/github-runners/image/runner.pkr.hcl
packer build -var-file=deploy/github-runners/image/local.pkrvars.hcl \
  deploy/github-runners/image/runner.pkr.hcl
```

Packer's builder has an external IP for downloads and accepts SSH only through
IAP. Worker VMs have no inbound firewall allowance. The runner service is enabled
but never started on the image builder; registration happens on worker boot.
Inspect the manifest and builder output before using the image.

## 3. Build and deploy the controller

Create the controller container through Cloud Build (no local Docker required):

```sh
gcloud builds submit deploy/github-runners/controller --project YOUR_PROJECT \
  --pack 'image=us-central1-docker.pkg.dev/YOUR_PROJECT/firebird-ci/controller:REVISION,env=GOOGLE_PYTHON_VERSION=3.12'
gcloud artifacts docker images describe \
  us-central1-docker.pkg.dev/YOUR_PROJECT/firebird-ci/controller:REVISION \
  --project YOUR_PROJECT --format='value(image_summary.digest)'
```

The Cloud Build execution service account needs Artifact Registry writer on this
repository, access to its build source bucket, and permission to write build
logs. Configure that account according to your project's Cloud Build policy;
the runtime controller account deliberately cannot build or publish images.

Set these values in `terraform.tfvars`:

```hcl
runner_image      = "projects/YOUR_PROJECT/global/images/firebird-ci-YYYYMMDD-COMMIT"
controller_image  = "us-central1-docker.pkg.dev/YOUR_PROJECT/firebird-ci/controller@sha256:EXACT_DIGEST"
deploy_controller = true
enabled           = true
```

Then run `terraform plan -out=deploy.tfplan` and `terraform apply deploy.tfplan`.
Scheduler invokes the IAM-protected endpoint with OIDC. No anonymous invocation
binding is created. Google-managed Scheduler service-agent permissions must be
intact. Polling normally adds up to one minute plus VM boot time per new batch.

## 4. Accept the pool before routing CI

Merge the workflows before using manual dispatch. Leave the repository variable
`GCP_RUNNERS_ENABLED` unset while testing.

1. Dispatch **GCP runner acceptance**. It explicitly targets this pool.
2. Observe at most three `firebird-ci-0/1/2` VMs. Four independent jobs are queued;
   the fourth must wait. All four check tool versions, Chromium, and clean state.
3. Record four distinct GCE instance IDs in job logs, even when a slot name repeats.
4. After completion, confirm zero worker VMs and no stale pool runner registrations
   within two successful polling cycles. Disks must be deleted, too.
5. Cancel another acceptance run while workers boot. Confirm idle cleanup.
6. Invoke Scheduler twice close together. Confirm lease exclusion and the same cap.
7. Stop a test worker mid-job. Confirm cleanup and replacement for remaining queued
   work. The interrupted job requires an explicit rerun; it does not resume.
8. Set repository variable `GCP_RUNNERS_ENABLED=true`. Dispatch Application checks,
   Native worker contract checks, Simulation worker checks, and Runner infrastructure
   checks. Record real Linux results before considering deployment accepted.

The PR contains local tests, not a claim that this cloud acceptance has run.
Fork PRs retain hosted routing. Self-hosted labels are not a security boundary:
keep the repository private, limit collaborators, and do not enable untrusted
fork workflows on this pool. Jobs have passwordless sudo inside their disposable
VMs, so they must be trusted to execute there.

## Lifecycle and failures

- Three fixed names in one zone enforce the cap, even across controller revisions.
  A generation-checked GCS lease serializes polls and expires after five minutes.
- Reads complete before mutations. Rate limits, authorization failures, incomplete
  pagination, or missing configuration cannot be mistaken for an empty queue.
- Booting workers reserve capacity. Failed/offline boot attempts expire after
  15 minutes. Idle workers without demand retire after two minutes; a busy-runner
  retirement conflict preserves the VM.
- Runner exit powers off the VM; the next poll deletes it and its auto-delete disk.
  A one-hour GCE runtime limit deletes runaway running VMs even if the controller
  fails. Existing CI jobs have shorter timeouts. Never increase job timeouts past
  this limit without changing and testing the lifecycle policy.
- Stopped disks can remain if the controller is unavailable. Restore polling or
  remove the three named instances manually; inspect leftover disks afterward.
- Quota or capacity errors leave jobs queued. Async creation errors are recovered
  on later polls; no retry can create a fourth slot. There is no paid hosted fallback
  or Spot provisioning. A deleted worker does not resume its interrupted job.
- Controller logs expose counts and exception types, not response bodies or keys.
  Job logs remain in GitHub. Serial console logging captures worker startup output;
  never echo credentials in jobs. Full runner `_diag` archives are not retained.

## Operations and costs

Workers use ephemeral external IPv4 for outbound downloads; there is no Cloud NAT
fixed charge. Compute, disks while present, IPv4, internet egress, image storage,
Cloud Build, Cloud Run requests, Scheduler, secrets, and logs can all cost money.
Scale-to-zero refers to workers, not a zero total bill. Add a GCP budget alert;
it is an alert, not a spending cap.

Inspect the Cloud Run request/error logs, Scheduler execution status, GCE serial
logs, and GitHub queued jobs. For an idle fleet, there must be no worker instances.
Repeated HTTP 503 responses require investigation; no pass is inferred from a
successful Scheduler request alone. `status=locked` is an expected overlapping poll.

Rebuild images for runner/security updates and repo tool-pin changes. Verify the
runner release checksum in GitHub's release metadata. Existing setup actions still
honor workflow pins; the image is a startup optimization, not a replacement for
lockfiles. Pin the new image names/digests in Terraform; running VMs finish on their
old image. Rebuild before an outdated runner is refused by GitHub.

For rollback, unset `GCP_RUNNERS_ENABLED` and cancel/re-run jobs already queued for
the pool; changing the variable does not retarget existing jobs. Let active workers
finish and polling remove them, then set `enabled=false` and apply. Hosted jobs may
still be unavailable if the account's minutes remain exhausted.

Before changing project, zone, or prefix, drain and remove the old fleet to preserve
the three-worker limit. Before `terraform destroy`, disable routing, drain, pause
polling, explicitly remove worker instances/disks, and delete unused custom images.
Workers are controller-created and are not tracked in Terraform state. The lease
bucket must be empty before destruction. Never delete unrelated project resources.

## Local validation

```sh
uv venv /tmp/runner-tests
uv pip install --python /tmp/runner-tests/bin/python \
  -r deploy/github-runners/controller/requirements-dev.txt
/tmp/runner-tests/bin/python -m pytest -q deploy/github-runners/controller/tests tests/test_ci_scope.py
/tmp/runner-tests/bin/ruff check deploy/github-runners
/tmp/runner-tests/bin/ruff format --check deploy/github-runners
terraform -chdir=deploy/github-runners/terraform init -backend=false
terraform -chdir=deploy/github-runners/terraform validate
terraform -chdir=deploy/github-runners/terraform test
bash -n deploy/github-runners/image/provision.sh
bash -n deploy/github-runners/image/runner.sh
```

References: [JIT runners](https://docs.github.com/en/rest/actions/self-hosted-runners#create-configuration-for-a-just-in-time-runner-for-a-repository),
[VM runtime limits](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime),
[Packer GCP builder](https://developer.hashicorp.com/packer/integrations/hashicorp/googlecompute/latest/components/builder/googlecompute).
