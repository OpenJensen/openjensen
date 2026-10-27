"""Optional workspace modules and whether this host has set them up.

Checks read saved configuration only. They never contact a companion, worker or
provider, so the workspace navigation can read them on every load.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict

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
    configured: bool = False
    managed: bool = False
    setup_url: str = "/guide/"


class ModulePreference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool


def preferences(data_dir: Path) -> dict:
    path = data_dir / "workspace-modules.json"
    if not path.exists():
        return {}
    if path.is_symlink() or path.stat().st_size > 16384:
        raise ValueError("Module preferences need administrator attention.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or any(
        key not in {"robot-lab", "teaching", "augmentation", "decision-lab", "cloud-runs"}
        or type(enabled) is not bool
        for key, enabled in value.items()
    ):
        raise ValueError("Module preferences need administrator attention.")
    return value


def set_module_enabled(data_dir: Path, ident: ModuleId, enabled: bool):
    if getenv("JENSEN_MODULES") is not None:
        raise ValueError("Module visibility is managed by JENSEN_MODULES on this server.")
    data_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(data_dir / "workspace-modules.lock", timeout=5):
        value = preferences(data_dir)
        value[ident] = enabled
        if ident == "cloud-runs" and enabled:
            (data_dir / "cloud-runs").mkdir(mode=0o700, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".modules-", dir=data_dir)
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(value, handle)
            os.replace(temporary, data_dir / "workspace-modules.json")
        finally:
            Path(temporary).unlink(missing_ok=True)


def _any_env(*names: str) -> bool:
    return any(getenv(name) for name in names)


def _cloud_connected(data_dir: Path) -> bool:
    return any(item.config for item in CloudConnections(data_dir).list().providers)


def workspace_modules(settings: Settings, robot_lab_config: Path) -> list[WorkspaceModule]:
    installed: dict[ModuleId, bool] = {
        "robot-lab": os.path.lexists(robot_lab_config),
        "teaching": os.path.lexists(robot_lab_config)
        or (settings.data_dir / "agent-providers.json").exists()
        or _any_env("JENSEN_TEACHING_URL")
        or settings.teaching_session_config is not None
        or settings.recording_config is not None,
        "augmentation": _cloud_connected(settings.data_dir) or _any_env("GEMINI_API_KEY"),
        "decision-lab": _any_env(
            "JENSEN_DECISION_PYTHON", "JENSEN_DECISION_ROOT", "JENSEN_DECISION_MODEL_DIR"
        ),
        "cloud-runs": settings.cloud_runs_dir is not None
        or (settings.data_dir / "cloud-runs").is_dir(),
    }
    configured = installed.copy()
    saved = preferences(settings.data_dir)
    installed.update(saved)
    # JENSEN_MODULES lists the modules to show and overrides detection; "all" shows every one.
    override = getenv("JENSEN_MODULES")
    if override is not None:
        chosen = {item.strip() for item in override.split(",")}
        installed = {key: "all" in chosen or key in chosen for key in installed}
    setup: dict[ModuleId, tuple[str, str]] = {
        "robot-lab": (
            "Initial setup",
            "Install the robot profile on the computer with the arm and pair it with this host.",
        ),
        "teaching": (
            "AI teleoperation",
            "Configure an AI provider, then select a simulator session or a calibrated robot.",
        ),
        "augmentation": ("Augmentation", "Connect Google Cloud or set GEMINI_API_KEY."),
        "decision-lab": ("Decision Lab", "Configure the local decision scorer on this host."),
        "cloud-runs": ("Cloud runs", "Add this module to create a local snapshots folder."),
    }
    guides = {
        "robot-lab": "/guide/#robot-lab",
        "teaching": "/guide/#teaching",
        "augmentation": "/settings/",
        "decision-lab": "/guide/#decision-lab",
        "cloud-runs": "/guide/#cloud-runs",
    }
    return [
        WorkspaceModule(
            id=key,
            name=name,
            installed=installed[key],
            setup=hint,
            configured=configured[key],
            managed=override is not None,
            setup_url=guides[key],
        )
        for key, (name, hint) in setup.items()
    ]
