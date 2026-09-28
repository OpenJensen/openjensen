mock_provider "google" {}

variables {
  project_id      = "ci-validation-project"
  app_id          = "1234"
  installation_id = "5678"
}

run "bootstrap_has_no_workers" {
  command = plan
  assert {
    condition     = length(google_cloud_run_v2_service.controller) == 0 && length(google_cloud_scheduler_job.reconcile) == 0
    error_message = "Bootstrap must not start a controller or schedule workers."
  }
  assert {
    condition     = google_compute_network.runner.auto_create_subnetworks == false
    error_message = "Use an isolated network without default firewall rules."
  }
}

run "controller_is_bounded" {
  command = plan
  variables {
    deploy_controller = true
    runner_image      = "projects/ci-validation-project/global/images/firebird-ci-test"
    controller_image  = "us-central1-docker.pkg.dev/ci-validation-project/firebird-ci/controller@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
  assert {
    condition     = google_cloud_run_v2_service.controller[0].template[0].max_instance_request_concurrency == 1
    error_message = "One reconciliation request per controller instance."
  }
  assert {
    condition     = google_cloud_run_v2_service.controller[0].template[0].scaling[0].max_instance_count == 1
    error_message = "Bound controller capacity; the storage lease protects revision overlap."
  }
  assert {
    condition     = google_cloud_scheduler_job.reconcile[0].paused
    error_message = "Polling requires an explicit enabled flag."
  }
}

run "reject_missing_images" {
  command = plan
  variables {
    deploy_controller = true
  }
  expect_failures = [google_cloud_run_v2_service.controller]
}

run "builder_has_scoped_access" {
  command = plan
  assert {
    condition     = google_storage_bucket_iam_member.builder.role == "roles/storage.objectUser"
    error_message = "The build identity must read sources and write logs in its own bucket."
  }
  assert {
    condition     = google_artifact_registry_repository_iam_member.builder.role == "roles/artifactregistry.writer"
    error_message = "The build identity must publish only to the runner registry."
  }
}

run "builder_can_read_bucket" {
  command = plan
  assert {
    condition     = google_storage_bucket_iam_member.builder_metadata.role == "roles/storage.bucketViewer"
    error_message = "Cloud Build must also read bucket metadata, not only objects."
  }
}
