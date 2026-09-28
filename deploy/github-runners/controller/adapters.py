"""Provider APIs and authentication; no scheduling policy lives here."""

import json
import time
from http import HTTPStatus
from pathlib import Path

import google.auth
import jwt
import requests
from config import (
    GITHUB_API_VERSION,
    LEASE_SECONDS,
    MAX_LIFETIME_SECONDS,
    MAX_PAGES,
    PAGE_SIZE,
    POOL_LABEL,
    REQUEST_SECONDS,
    RUNNER_LABELS,
)
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

GITHUB_URL = "https://api.github.com"
COMPUTE_URL = "https://compute.googleapis.com/compute/v1"
ACTIVE_RUNS = ("queued", "in_progress", "waiting", "pending", "requested")
PRIVATE_KEY = Path("/secrets/github/private-key")
CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
JWT_BACKDATE_SECONDS = 60
JWT_LIFETIME_SECONDS = 540
DEFAULT_RUNNER_GROUP = 1


class ApiError(RuntimeError):
    """Never include response bodies, URLs, credentials or JIT configuration in logs."""


def _json(response):
    if not response.ok:
        raise ApiError(f"Provider request failed: HTTP {response.status_code}")
    if response.status_code == HTTPStatus.NO_CONTENT:
        return {}
    return response.json()


class GitHub:
    def __init__(self, settings):
        self._repo = settings.repository
        now = int(time.time())
        token = jwt.encode(
            {
                "iat": now - JWT_BACKDATE_SECONDS,
                "exp": now + JWT_LIFETIME_SECONDS,
                "iss": settings.app_id,
            },
            PRIVATE_KEY.read_text(),
            algorithm="RS256",
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
                "Authorization": f"Bearer {token}",
            }
        )
        installation = self._request(
            "POST",
            f"/app/installations/{settings.installation_id}/access_tokens",
            {
                "repositories": [self._repo.split("/")[1]],
                "permissions": {"administration": "write", "actions": "read"},
            },
        )
        self._session.headers["Authorization"] = f"Bearer {installation['token']}"

    def _request(self, method, path, body=None):
        return _json(
            self._session.request(
                method,
                GITHUB_URL + path,
                json=body,
                timeout=REQUEST_SECONDS,
            )
        )

    def _pages(self, path, key):
        separator = "&" if "?" in path else "?"
        items = []
        for page in range(1, MAX_PAGES + 1):
            result = self._request("GET", f"{path}{separator}per_page={PAGE_SIZE}&page={page}")
            items.extend(result[key])
            if len(result[key]) < PAGE_SIZE:
                if len(items) < result.get("total_count", len(items)):
                    raise ApiError("Provider returned a partial snapshot")
                return items
        raise ApiError("Pagination limit reached; refusing a partial snapshot")

    def runners(self):
        return self._pages(f"/repos/{self._repo}/actions/runners", "runners")

    def work(self):
        runs = {}
        for status in ACTIVE_RUNS:
            for run in self._pages(
                f"/repos/{self._repo}/actions/runs?status={status}",
                "workflow_runs",
            ):
                runs[run["id"]] = run
        jobs = {
            run_id: self._pages(
                f"/repos/{self._repo}/actions/runs/{run_id}/jobs?filter=latest",
                "jobs",
            )
            for run_id in runs
        }
        return list(runs.values()), jobs

    def register(self, name):
        return self._request(
            "POST",
            f"/repos/{self._repo}/actions/runners/generate-jitconfig",
            {
                "name": name,
                "runner_group_id": DEFAULT_RUNNER_GROUP,
                "labels": list(RUNNER_LABELS),
                "work_folder": "_work",
            },
        )

    def retire(self, runner_id):
        response = self._session.delete(
            f"{GITHUB_URL}/repos/{self._repo}/actions/runners/{runner_id}",
            timeout=REQUEST_SECONDS,
        )
        if response.status_code == HTTPStatus.NOT_FOUND:
            return True
        if response.status_code in {HTTPStatus.CONFLICT, HTTPStatus.UNPROCESSABLE_ENTITY}:
            return False
        _json(response)
        return True


class Compute:
    def __init__(self, settings):
        self._settings = settings
        credentials, _ = google.auth.default(scopes=[CLOUD_SCOPE])
        self._session = AuthorizedSession(credentials)
        self._base = f"{COMPUTE_URL}/projects/{settings.project}/zones/{settings.zone}"

    def instances(self):
        # Read every fixed slot, including pending/deleting resources, without filters.
        instances = []
        for name in self._settings.slots():
            response = self._session.get(f"{self._base}/instances/{name}", timeout=REQUEST_SECONDS)
            if response.status_code == HTTPStatus.NOT_FOUND:
                continue
            vm = _json(response)
            if vm.get("labels", {}).get("runner-pool") != POOL_LABEL:
                raise ApiError("Reserved slot contains an unmanaged VM")
            instances.append(vm)
        return instances

    def delete(self, name):
        response = self._session.delete(f"{self._base}/instances/{name}", timeout=REQUEST_SECONDS)
        if response.status_code != HTTPStatus.NOT_FOUND:
            _json(response)

    def create(self, slot, runner, jit):
        settings = self._settings
        payload = {
            "name": slot,
            "machineType": f"zones/{settings.zone}/machineTypes/{settings.machine_type}",
            "labels": {"runner-pool": POOL_LABEL},
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "initializeParams": {
                        "sourceImage": settings.image,
                        "diskSizeGb": settings.disk_gb,
                        "diskType": f"zones/{settings.zone}/diskTypes/pd-balanced",
                    },
                }
            ],
            "networkInterfaces": [
                {
                    "subnetwork": settings.subnet,
                    "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}],
                }
            ],
            "serviceAccounts": [],
            "scheduling": {
                "provisioningModel": "STANDARD",
                "automaticRestart": False,
                "onHostMaintenance": "MIGRATE",
                "maxRunDuration": {"seconds": str(MAX_LIFETIME_SECONDS)},
                "instanceTerminationAction": "DELETE",
            },
            "shieldedInstanceConfig": {
                "enableSecureBoot": True,
                "enableVtpm": True,
                "enableIntegrityMonitoring": True,
            },
            "metadata": {
                "items": [
                    {"key": "runner-name", "value": runner},
                    {"key": "runner-jit", "value": jit},
                    {"key": "block-project-ssh-keys", "value": "true"},
                    {"key": "serial-port-logging-enable", "value": "true"},
                ]
            },
        }
        _json(self._session.post(f"{self._base}/instances", json=payload, timeout=REQUEST_SECONDS))


class Lease:
    def __init__(self, bucket):
        self._blob = storage.Client().bucket(bucket).blob("reconcile.json")
        self._generation = None

    def acquire(self):
        generation = 0
        try:
            self._blob.reload(timeout=REQUEST_SECONDS)
            generation = int(self._blob.generation)
            lease = json.loads(
                self._blob.download_as_text(
                    if_generation_match=generation,
                    timeout=REQUEST_SECONDS,
                )
            )
            if lease["expires"] > time.time():
                return False
        except NotFound:
            pass
        except PreconditionFailed:
            return False
        try:
            self._blob.upload_from_string(
                json.dumps({"expires": time.time() + LEASE_SECONDS}),
                content_type="application/json",
                if_generation_match=generation,
                timeout=REQUEST_SECONDS,
                retry=None,
            )
        except PreconditionFailed:
            return False
        self._generation = int(self._blob.generation)
        return True

    def release(self):
        if self._generation is None:
            return
        try:
            self._blob.delete(if_generation_match=self._generation, timeout=REQUEST_SECONDS)
        except (NotFound, PreconditionFailed):
            pass
