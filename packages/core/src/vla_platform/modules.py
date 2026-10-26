"""Optional workspace modules and whether this host has set them up.

Checks read saved configuration and cached local runtime metadata only. They never
contact a companion, worker or provider, so navigation remains available offline.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from filelock import FileLock
from pydantic import BaseModel, ConfigDict

from vla_platform.cloud_connections import CloudConnections
from vla_platform.contracts import NonEmptyString, Record
from vla_platform.environment import getenv
from vla_platform.robot_lab_bridge import RobotLabBridge, RobotLabError
from vla_platform.settings import Settings

ModuleId = Literal["robot-lab", "teaching", "augmentation", "decision-lab", "cloud-runs"]


class WorkspaceModule(Record):
    id: ModuleId
    name: NonEmptyString
    installed: bool
    available: bool = False
    enabled: bool = False
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


def _cloud_connected(data_dir: Path) -> bool:
    try:
        return any(item.config for item in CloudConnections(data_dir).list().providers)
    except OSError, ValueError:
        return False


def _paired(config_path: Path) -> bool:
    try:
        RobotLabBridge.read_config(config_path)
        return True
    except RobotLabError:
        return False


def _decision_configured() -> bool:
    from vla_platform.decision_api import configuration

    try:
        configuration()
        return True
    except HTTPException:
        return False


def _teaching_configured(settings: Settings) -> bool:
    from vla_platform.teaching_api import configuration

    try:
        configuration()
        return True
    except HTTPException:
        pass
    return any(
        path is not None and path.is_file()
        for path in (settings.teaching_session_config, settings.recording_config)
    )


def _agent_configured(data_dir: Path) -> bool:
    try:
        from vla_platform.agent_control.settings import Profiles

        return bool(Profiles(data_dir).read())
    except ImportError, OSError, ValueError:
        return False


def workspace_modules(
    settings: Settings, robot_lab_config: Path, runtime_ready: dict[str, bool] | None = None
) -> list[WorkspaceModule]:
    runtimes = runtime_ready or {}
    paired = _paired(robot_lab_config)
    cloud_runs = (settings.cloud_runs_dir is not None and settings.cloud_runs_dir.is_dir()) or (
        settings.data_dir / "cloud-runs"
    ).is_dir()
    configured: dict[ModuleId, bool] = {
        "robot-lab": paired,
        "teaching": paired
        or _teaching_configured(settings)
        or _agent_configured(settings.data_dir),
        "augmentation": _cloud_connected(settings.data_dir) or bool(getenv("GEMINI_API_KEY")),
        "decision-lab": _decision_configured(),
        "cloud-runs": cloud_runs,
    }
    available = configured.copy()
    available["robot-lab"] |= runtimes.get("robot", False)
    available["decision-lab"] |= runtimes.get("decision", False)
    # Snapshot viewing is bundled; adding it creates its local storage folder.
    available["cloud-runs"] = True
    enabled = {key: configured[key] or available[key] for key in configured}
    enabled["cloud-runs"] = cloud_runs
    enabled.update(preferences(settings.data_dir))
    # An operator may choose visibility, but cannot fabricate missing prerequisites.
    override = getenv("JENSEN_MODULES")
    if override is not None:
        chosen = {item.strip() for item in override.split(",")}
        enabled = {key: "all" in chosen or key in chosen for key in enabled}
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
            installed=available[key] and enabled[key],
            available=available[key],
            enabled=enabled[key],
            setup=hint,
            configured=configured[key],
            managed=override is not None,
            setup_url=guides[key],
        )
        for key, (name, hint) in setup.items()
    ]
