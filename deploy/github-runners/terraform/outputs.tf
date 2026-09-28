output "network" { value = google_compute_network.runner.name }
output "subnetwork" { value = google_compute_subnetwork.runner.name }
output "controller_url" { value = try(google_cloud_run_v2_service.controller[0].uri, null) }
output "github_key_secret" { value = google_secret_manager_secret.github.secret_id }
output "image_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.controller.repository_id}"
}
