# Runner infrastructure moved

Terraform, controller, image builds and infrastructure checks now live in
[OpenJensen/ci-infra](https://github.com/OpenJensen/ci-infra).
Application workflows, runner routing tests and `runner-smoke.yml` remain here.

Ignored local state, variables and provider files may still exist in this folder.
Preserve them. Use the infrastructure repository's migration instructions before
managing an existing deployment; never initialise a replacement state.

Create the GitHub App under OpenJensen and install it on **OpenJensen/openjensen**
only. The controller still registers repository-scoped runners. Its private key
belongs in GCP Secret Manager.
