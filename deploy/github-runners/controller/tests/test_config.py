"""Catch routing and image drift without needing cloud credentials."""

import json
from pathlib import Path

import hcl2
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "deploy/github-runners"
HCL_OPTIONS = hcl2.utils.SerializationOptions(strip_string_quotes=True)


def test_image_pins_match_workflows():
    versions = json.loads((DEPLOY / "image/versions.json").read_text())
    workflows = "\n".join(path.read_text() for path in (ROOT / ".github/workflows").glob("*.yml"))
    assert f"version: '{versions['uv']}'" in workflows
    assert "python-version: '" + (ROOT / ".python-version").read_text().strip() + "'" in workflows
    for version in versions["worker_python"]:
        assert f"python-version: '{version}'" in workflows
    assert len(versions["runner_sha256"]) == 64


@pytest.mark.parametrize("name", ["application", "native-workers", "simulation"])
def test_linux_routing_is_opt_in(name):
    document = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
    for job in document["jobs"].values():
        routing = job["runs-on"]
        assert "vars.GCP_RUNNERS_ENABLED == 'true'" in routing
        assert "github.event.pull_request.head.repo.full_name == github.repository" in routing
        assert "firebird-gcp" in routing
        assert "ubuntu-latest" in routing
    if name == "application":
        assert "matrix.os == 'ubuntu-latest'" in document["jobs"]["application"]["runs-on"]


def test_packer_uses_repo_files():
    with (DEPLOY / "image/runner.pkr.hcl").open() as source:
        document = hcl2.load(source, serialization_options=HCL_OPTIONS)
    files = document["build"][0]["provisioner"]
    assert sum("file" in item for item in files) == 6
    for provisioner in files:
        if "file" not in provisioner:
            continue
        path = provisioner["file"]["source"].replace("${path.root}", str(DEPLOY / "image"))
        assert Path(path).is_file(), path


def test_terraform_defaults():
    with (DEPLOY / "terraform/variables.tf").open() as source:
        variables = {
            name: value
            for entry in hcl2.load(source, serialization_options=HCL_OPTIONS)["variable"]
            for name, value in entry.items()
        }
    assert variables["enabled"]["default"] is False
    assert variables["deploy_controller"]["default"] is False
    assert variables["machine_type"]["default"] == "e2-standard-2"
    with (DEPLOY / "terraform/main.tf").open() as source:
        hcl2.load(source, serialization_options=HCL_OPTIONS)
