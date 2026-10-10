# Set up CI runners

Follow the Terraform, controller and image-build instructions in [OpenJensen/ci-infra](https://github.com/OpenJensen/ci-infra). Use its migration steps when reusing existing deployment state.

Create the GitHub App under OpenJensen and install it on **OpenJensen/openjensen**. Store the private key in GCP Secret Manager.

Set the repository variable `CI_RUNNER` to `gcp` to select the configured self-hosted runners. Run `application.yml` from GitHub's Actions tab to check the application, and use [the local verification commands](../../docs/ci.md) for local checks.
