# Keep image-build permissions separate from the runtime controller.
resource "google_service_account" "builder" {
  account_id   = "${local.name}-builder"
  display_name = "Build runner controller images"
  depends_on   = [google_project_service.api]
}

resource "google_storage_bucket" "build" {
  name                        = "${var.project_id}-${local.name}-build"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }
  depends_on = [google_project_service.api]
}

resource "google_storage_bucket_iam_member" "builder" {
  bucket = google_storage_bucket.build.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.builder.email}"
}

resource "google_artifact_registry_repository_iam_member" "builder" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.controller.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.builder.email}"
}

resource "google_storage_bucket_iam_member" "builder_metadata" {
  bucket = google_storage_bucket.build.name
  role   = "roles/storage.bucketViewer"
  member = "serviceAccount:${google_service_account.builder.email}"
}
