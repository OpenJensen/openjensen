variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "us-central1"
}
variable "zone" {
  type    = string
  default = "us-central1-a"
}
variable "repository" {
  type    = string
  default = "sobhanb-eth/firebird-hackathon-codebase"
}
variable "app_id" { type = string }
variable "installation_id" { type = string }
variable "runner_image" {
  default     = ""
  type        = string
  description = "Immutable projects/PROJECT/global/images/NAME; do not use an image family."
  validation {
    condition     = var.runner_image == "" || can(regex("^projects/[^/]+/global/images/[^/]+$", var.runner_image))
    error_message = "Specify an exact image resource."
  }
}
variable "controller_image" {
  default     = ""
  type        = string
  description = "Artifact Registry image pinned by sha256 digest."
  validation {
    condition     = var.controller_image == "" || can(regex("@sha256:[a-f0-9]{64}$", var.controller_image))
    error_message = "Pin the controller image by digest."
  }
}
variable "enabled" {
  type        = bool
  default     = false
  description = "Resume polling only after uploading the GitHub App key and building the image."
}
variable "machine_type" {
  type    = string
  default = "e2-standard-2"
}
variable "disk_gb" {
  type    = number
  default = 50
  validation {
    condition     = var.disk_gb >= 50 && var.disk_gb <= 100
    error_message = "Use 50–100 GB for Python wheels, browser binaries, and CI workspaces."
  }
}

variable "deploy_controller" {
  type        = bool
  default     = false
  description = "Create the controller after the image and GitHub key exist."
}
