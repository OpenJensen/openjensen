"""Optional workspace modules and whether this host has set them up.

Checks read saved configuration only. They never contact a companion, worker or
provider, so the workspace navigation can read them on every load.
"""

import os
from pathlib import Path
from typing import Literal

from vla_platform.cloud_connections import CloudConnections
from vla_platform.contracts import NonEmptyString, Record
from vla_platform.environment import getenv
from vla_platform.settings import Settings

ModuleId = Literal["robot-lab", "teaching", "augmentation", "decision-lab", "cloud-runs"]


class WorkspaceModule(Record):
    id: ModuleId
    name: NonEmptyString
    installed: bool
    setup: NonEmptyString


def _any_env(*names: str) -> bool:
    return any(getenv(name) for name in names)


def _cloud_connected(data_dir: Path) -> bool:
    return any(item.config for item in CloudConnections(data_dir).list().providers)


def workspace_modules(settings: Settings, robot_lab_config: Path) -> list[WorkspaceModule]:
    installed: dict[ModuleId, bool] = {
        "robot-lab": os.path.lexists(robot_lab_config),
        "teaching": _any_env("JENSEN_TEACHING_URL")
        or settings.teaching_session_config is not None
        or settings.recording_config is not None,
        "augmentation": _cloud_connected(settings.data_dir) or _any_env("GEMINI_API_KEY"),
        "decision-lab": _any_env(
            "JENSEN_DECISION_PYTHON", "JENSEN_DECISION_ROOT", "JENSEN_DECISION_MODEL_DIR"
        ),
        "cloud-runs": settings.cloud_runs_dir is not None,
    }
    # JENSEN_MODULES lists the modules to show and overrides detection; "all" shows every one.
    override = getenv("JENSEN_MODULES")
    if override is not None:
        chosen = {item.strip() for item in override.split(",")}
        installed = {key: "all" in chosen or key in chosen for key in installed}
    setup: dict[ModuleId, tuple[str, str]] = {
        "robot-lab": (
            "Robot Lab",
            "Install the robot profile on the computer with the arm and pair it with this host.",
        ),
        "teaching": (
            "Teaching",
            "Set JENSEN_TEACHING_URL, JENSEN_TEACHING_SESSION_CONFIG or JENSEN_RECORDING_CONFIG.",
        ),
        "augmentation": ("Augmentation", "Connect Google Cloud or set GEMINI_API_KEY."),
        "decision-lab": ("Decision lab", "Configure the local decision scorer on this host."),
        "cloud-runs": ("Cloud runs", "Set JENSEN_CLOUD_RUNS_DIR to the cloud run snapshots."),
    }
    return [
        WorkspaceModule(id=key, name=name, installed=installed[key], setup=hint)
        for key, (name, hint) in setup.items()
    ]
