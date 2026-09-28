packer {
  required_plugins {
    googlecompute = {
      source  = "github.com/hashicorp/googlecompute"
      version = "= 1.2.4"
    }
  }
}

variable "project_id" { type = string }
variable "zone" { type = string }
variable "network" { type = string }
variable "subnetwork" { type = string }
variable "image_name" { type = string }
variable "source_image" {
  type        = string
  description = "Exact Ubuntu 24.04 amd64 image name from ubuntu-os-cloud; never a floating family."
}

source "googlecompute" "runner" {
  project_id                      = var.project_id
  zone                            = var.zone
  network                         = var.network
  subnetwork                      = var.subnetwork
  source_image                    = var.source_image
  source_image_project_id         = ["ubuntu-os-cloud"]
  image_name                      = var.image_name
  image_family                    = "firebird-ci"
  image_description               = "One-job Firebird GitHub Actions runner; no credentials"
  machine_type                    = "e2-standard-2"
  disk_size                       = 50
  disk_type                       = "pd-balanced"
  ssh_username                    = "packer"
  use_iap                         = true
  omit_external_ip                = false
  disable_default_service_account = true
  tags                            = ["runner-image-builder"]
}

build {
  sources = ["source.googlecompute.runner"]

  provisioner "file" {
    source      = "${path.root}/../../../package.json"
    destination = "/tmp/workspace-package.json"
  }
  provisioner "file" {
    source      = "${path.root}/../../../.node-version"
    destination = "/tmp/node-version"
  }
  provisioner "file" {
    source      = "${path.root}/../../../.python-version"
    destination = "/tmp/python-version"
  }
  provisioner "file" {
    source      = "${path.root}/versions.json"
    destination = "/tmp/runner-versions.json"
  }
  provisioner "file" {
    source      = "${path.root}/runner.sh"
    destination = "/tmp/runner.sh"
  }
  provisioner "file" {
    source      = "${path.root}/runner.service"
    destination = "/tmp/runner.service"
  }
  provisioner "shell" {
    script          = "${path.root}/provision.sh"
    execute_command = "sudo -E bash '{{ .Path }}'"
  }
}
