locals {
  name = "firebird-ci"
  apis = toset([
    "compute.googleapis.com", "run.googleapis.com", "cloudscheduler.googleapis.com",
    "secretmanager.googleapis.com", "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com", "storage.googleapis.com", "iap.googleapis.com",
    "iam.googleapis.com", "cloudresourcemanager.googleapis.com", "logging.googleapis.com",
  ])
}

resource "google_project_service" "api" {
  for_each           = local.apis
  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "runner" {
  name                    = local.name
  auto_create_subnetworks = false
  depends_on              = [google_project_service.api]
}
resource "google_compute_subnetwork" "runner" {
  name          = local.name
  region        = var.region
  network       = google_compute_network.runner.id
  ip_cidr_range = "10.87.0.0/24"
}
# Workers have no inbound rules. Only temporary Packer builders accept IAP SSH.
resource "google_compute_firewall" "builder" {
  name          = "${local.name}-builder-iap"
  network       = google_compute_network.runner.name
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["runner-image-builder"]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_service_account" "controller" {
  account_id   = "${local.name}-controller"
  display_name = "GitHub runner fleet controller"
  depends_on   = [google_project_service.api]
}
resource "google_service_account" "scheduler" {
  account_id   = "${local.name}-scheduler"
  display_name = "Invoke runner reconciliation"
  depends_on   = [google_project_service.api]
}
resource "google_project_iam_custom_role" "controller" {
  role_id = "firebirdRunnerController"
  title   = "Manage ephemeral runner VMs"
  permissions = [
    "compute.instances.get", "compute.instances.create", "compute.instances.delete",
    "compute.instances.setMetadata", "compute.instances.setLabels",
    "compute.disks.create", "compute.images.useReadOnly",
    "compute.subnetworks.use", "compute.subnetworks.useExternalIp",
    "compute.zoneOperations.get",
  ]
}
resource "google_project_iam_member" "controller" {
  project = var.project_id
  role    = google_project_iam_custom_role.controller.name
  member  = "serviceAccount:${google_service_account.controller.email}"
}

resource "google_secret_manager_secret" "github" {
  secret_id = "${local.name}-github-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.api]
}
# Upload key versions outside Terraform; private keys never enter its state.
resource "google_secret_manager_secret_iam_member" "controller" {
  secret_id = google_secret_manager_secret.github.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.controller.email}"
}
resource "google_storage_bucket" "lease" {
  name                        = "${var.project_id}-${local.name}-lease"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  depends_on                  = [google_project_service.api]
}
resource "google_storage_bucket_iam_member" "controller" {
  bucket = google_storage_bucket.lease.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.controller.email}"
}
resource "google_artifact_registry_repository" "controller" {
  location      = var.region
  repository_id = local.name
  format        = "DOCKER"
  depends_on    = [google_project_service.api]
}

resource "google_cloud_run_v2_service" "controller" {
  count               = var.deploy_controller ? 1 : 0
  name                = local.name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"
  lifecycle {
    precondition {
      condition     = var.runner_image != "" && var.controller_image != ""
      error_message = "Build and pin both images before deploying the controller."
    }
  }
  template {
    service_account                  = google_service_account.controller.email
    timeout                          = "180s"
    max_instance_request_concurrency = 1
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }
    containers {
      image = var.controller_image
      resources {
        limits   = { cpu = "1", memory = "512Mi" }
        cpu_idle = true
      }
      dynamic "env" {
        for_each = {
          PROJECT         = var.project_id
          ZONE            = var.zone
          IMAGE           = var.runner_image
          SUBNET          = google_compute_subnetwork.runner.id
          REPOSITORY      = var.repository
          APP_ID          = var.app_id
          INSTALLATION_ID = var.installation_id
          LEASE_BUCKET    = google_storage_bucket.lease.name
          MACHINE_TYPE    = var.machine_type
          DISK_GB         = tostring(var.disk_gb)
        }
        content {
          name  = env.key
          value = env.value
        }
      }
      volume_mounts {
        name       = "github-key"
        mount_path = "/secrets/github"
      }
    }
    volumes {
      name = "github-key"
      secret {
        secret = google_secret_manager_secret.github.secret_id
        items {
          version = "latest"
          path    = "private-key"
        }
      }
    }
  }
  depends_on = [
    google_project_service.api,
    google_secret_manager_secret_iam_member.controller,
    google_storage_bucket_iam_member.controller,
    google_project_iam_member.controller,
  ]
}
resource "google_cloud_run_v2_service_iam_member" "scheduler" {
  count    = var.deploy_controller ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.controller[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}
resource "google_cloud_scheduler_job" "reconcile" {
  count            = var.deploy_controller ? 1 : 0
  name             = local.name
  region           = var.region
  schedule         = "* * * * *"
  time_zone        = "Etc/UTC"
  paused           = !var.enabled
  attempt_deadline = "180s"
  http_target {
    uri         = "${google_cloud_run_v2_service.controller[0].uri}/reconcile"
    http_method = "POST"
    oidc_token {
      service_account_email = google_service_account.scheduler.email
      audience              = google_cloud_run_v2_service.controller[0].uri
    }
  }
  retry_config {
    retry_count = 1
  }
  depends_on = [google_cloud_run_v2_service_iam_member.scheduler]
}
