"""Deployment settings shared by the fleet service and infrastructure adapters."""

import os
from dataclasses import dataclass

POOL_LABEL = "firebird-gcp"
RUNNER_LABELS = ("self-hosted", "Linux", "X64", POOL_LABEL)
MAX_WORKERS = 3
BOOT_GRACE_SECONDS = 900
IDLE_GRACE_SECONDS = 120
MAX_LIFETIME_SECONDS = 3600
LEASE_SECONDS = 300
REQUEST_SECONDS = 20
PAGE_SIZE = 100
MAX_PAGES = 20
GITHUB_API_VERSION = "2022-11-28"
DEFAULT_MACHINE = "e2-standard-2"
DEFAULT_DISK_GB = 50


@dataclass(frozen=True)
class Settings:
    project: str
    zone: str
    image: str
    subnet: str
    repository: str
    app_id: str
    installation_id: str
    lease_bucket: str
    machine_type: str = DEFAULT_MACHINE
    disk_gb: int = DEFAULT_DISK_GB
    prefix: str = "firebird-ci"

    @classmethod
    def from_env(cls):
        return cls(
            **{
                key: os.environ[key.upper()]
                for key in (
                    "project",
                    "zone",
                    "image",
                    "subnet",
                    "repository",
                    "app_id",
                    "installation_id",
                    "lease_bucket",
                )
            },
            machine_type=os.environ.get("MACHINE_TYPE", DEFAULT_MACHINE),
            disk_gb=int(os.environ.get("DISK_GB", str(DEFAULT_DISK_GB))),
        )

    def slots(self):
        return tuple(f"{self.prefix}-{index}" for index in range(MAX_WORKERS))
