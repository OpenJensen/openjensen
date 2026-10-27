"""Installed workspace features, visibility and independent runtime readiness.

The registry stores feature registration only. Reading it never contacts a
provider, downloads a runtime or changes saved projects and credentials.
"""

import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, model_validator

from vla_platform.cloud_connections import CloudConnections
from vla_platform.contracts import NonEmptyString, Record
from vla_platform.environment import getenv
from vla_platform.huggingface_connection import HuggingFaceConnection
from vla_platform.robot_lab_bridge import RobotLabBridge, RobotLabError
from vla_platform.settings import Settings

ModuleId = Literal[
    "datasets",
    "models",
    "robot-lab",
    "teaching",
    "augmentation",
    "training",
    "distillation",
    "quantization",
    "evaluation",
    "simulation",
    "decision-lab",
    "cloud-runs",
    "google-cloud",
    "openai",
    "gemini",
    "huggingface",
]
ModuleGroup = Literal["Robot Lab", "Data", "Train", "Test", "Workspace", "Cloud & APIs"]
DEFAULT_MODULES = frozenset({"datasets", "models"})
LEGACY_CORE_MODULES = frozenset(
    {
        "datasets",
        "models",
        "training",
        "distillation",
        "quantization",
        "evaluation",
        "simulation",
    }
)
REGISTRY_FILE = "workspace-module-registry.json"
REGISTRY_ERROR = "The saved module registry needs administrator attention."

# Feature sources ship with Core; installing registers one without downloading dependencies.
CATALOG = (
    (
        "robot-lab",
        "Initial setup",
        "Robot Lab",
        "Pair the companion on the USB computer.",
        "/guide/#robot-lab",
    ),
    (
        "teaching",
        "AI teleoperation",
        "Robot Lab",
        "Configure an AI provider, then select a simulator session or a calibrated robot.",
        "/guide/#teaching",
    ),
    (
        "datasets",
        "Datasets",
        "Data",
        "Install dataset readers when needed for local conversions.",
        "/guide/#datasets",
    ),
    (
        "augmentation",
        "Augmentation",
        "Data",
        "Install and configure Google Cloud or Gemini.",
        "/cloud-apis/",
    ),
    (
        "training",
        "Fine-tuning",
        "Train",
        "Configure a training worker before starting a job.",
        "/guide/#training",
    ),
    (
        "distillation",
        "Distillation",
        "Train",
        "Configure a distillation worker before starting a job.",
        "/guide/#distillation",
    ),
    (
        "quantization",
        "Quantization",
        "Train",
        "Configure a quantization worker before starting a job.",
        "/guide/#quantization",
    ),
    (
        "evaluation",
        "Evaluate",
        "Test",
        "Configure an evaluation worker before starting a job.",
        "/guide/#evaluation",
    ),
    (
        "simulation",
        "Run",
        "Test",
        "Configure a simulator worker before starting a session.",
        "/guide/#simulation",
    ),
    (
        "decision-lab",
        "Decision Lab",
        "Test",
        "Configure the local scorer, model and license before scoring.",
        "/guide/#decision-lab",
    ),
    (
        "models",
        "Models",
        "Workspace",
        "Install Hugging Face to configure authenticated model downloads.",
        "/guide/#models",
    ),
    (
        "cloud-runs",
        "Cloud runs",
        "Workspace",
        "View cloud run snapshots from the local snapshots folder.",
        "/guide/#cloud-runs",
    ),
    (
        "google-cloud",
        "Google Cloud",
        "Cloud & APIs",
        "Connect a Google Cloud project and verify CLI access.",
        "/cloud-apis/",
    ),
    ("openai", "OpenAI", "Cloud & APIs", "Save an OpenAI API key.", "/cloud-apis/"),
    ("gemini", "Gemini Omni", "Cloud & APIs", "Save a Gemini API key.", "/cloud-apis/"),
    (
        "huggingface",
        "Hugging Face",
        "Cloud & APIs",
        "Save a Hugging Face access token.",
        "/cloud-apis/",
    ),
)
MODULE_IDS = frozenset(item[0] for item in CATALOG)


class WorkspaceModule(Record):
    id: ModuleId
    name: NonEmptyString
    group: ModuleGroup
    default_installed: bool = False
    installed: bool
    available: bool = True
    enabled: bool = False
    setup: NonEmptyString
    configured: bool = False
    managed: bool = False
    setup_url: str = "/guide/"


class ModulePreference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool


class ModuleRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    origin: Literal["fresh", "migrated"]
    installed: list[ModuleId]
    enabled: dict[ModuleId, bool]

    @model_validator(mode="after")
    def valid_membership(self):
        if len(set(self.installed)) != len(self.installed) or set(self.enabled) != set(
            self.installed
        ):
            raise ValueError("Invalid module membership")
        return self


def _unique_json_pairs(entries):
    value = {}
    for key, item in entries:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _read_json(path: Path, maximum: int, *, private: bool):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        if path.is_symlink():
            raise ValueError
        with os.fdopen(os.open(path, flags), "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise ValueError
            if (
                private
                and os.name == "posix"
                and (info.st_uid != os.getuid() or info.st_mode & 0o077)
            ):
                raise ValueError
            content = handle.read(maximum + 1)
            if len(content) > maximum:
                raise ValueError
            return json.loads(content, object_pairs_hook=_unique_json_pairs)
    except FileNotFoundError:
        raise
    except OSError, ValueError, TypeError:
        raise ValueError(REGISTRY_ERROR) from None


def preferences(data_dir: Path) -> dict:
    """Read the original visibility file for one-time migration; never rewrite it."""
    try:
        value = _read_json(data_dir / "workspace-modules.json", 16384, private=False)
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict) or any(
        key not in MODULE_IDS or type(enabled) is not bool for key, enabled in value.items()
    ):
        raise ValueError(REGISTRY_ERROR)
    return value


def read_registry(data_dir: Path) -> ModuleRegistry | None:
    try:
        value = _read_json(data_dir / REGISTRY_FILE, 65536, private=True)
        if not isinstance(value, dict) or type(value.get("version")) is not int:
            raise ValueError
        return ModuleRegistry.model_validate(value)
    except FileNotFoundError:
        return None
    except ValueError:
        raise ValueError(REGISTRY_ERROR) from None


def _write_registry(data_dir: Path, value: ModuleRegistry):
    descriptor, temporary = tempfile.mkstemp(prefix=".modules-", dir=data_dir)
    try:
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(value.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, data_dir / REGISTRY_FILE)
        if os.name == "posix":
            folder = os.open(data_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(folder)
            finally:
                os.close(folder)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _registry_lock(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "workspace-module-registry.lock"
    if path.is_symlink():
        raise ValueError(REGISTRY_ERROR)
    return FileLock(path, timeout=5)


def module_workspace_exists(data_dir: Path) -> bool:
    """Capture this before creating the first database or workspace lock."""
    return any(
        (data_dir / name).exists()
        for name in (
            REGISTRY_FILE,
            "workspace.sqlite3",
            "workspace-modules.json",
            "projects",
            "datasets",
            "models",
            "module-environments",
            "robot-lab-connection.json",
            "cloud-connections.json",
            "huggingface-credential.json",
            "agent-providers.json",
        )
    )


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
        for path in (
            settings.teaching_session_config,
            settings.recording_config,
        )
    )


def _agent_configured(data_dir: Path) -> bool:
    try:
        from vla_platform.agent_control.settings import Profiles

        return bool(Profiles(data_dir).read())
    except ImportError, OSError, ValueError:
        return False


def _worker_paths_ready(python, root):
    if not python or not root or not Path(root).is_dir():
        return False
    executable = Path(python) if Path(python).is_absolute() else shutil.which(python)
    return bool(executable and Path(executable).is_file() and os.access(executable, os.X_OK))


def runtime_feature_readiness(catalog, settings: Settings | None = None) -> dict[str, bool]:
    """Read existing worker declarations and local paths; never execute a worker."""
    public = {item["id"]: item for item in catalog.public()["runtimes"]}
    result = {
        key: False
        for key in (
            "training",
            "distillation",
            "quantization",
            "evaluation",
            "simulation",
        )
    }
    for runtime in catalog.runtimes:
        details = public[runtime.id]
        engine = _worker_paths_ready(runtime.python, runtime.worker_root)
        result["training"] |= bool(details.get("training")) and _worker_paths_ready(
            runtime.training_python,
            runtime.training_root,
        )
        result["distillation"] |= bool(details.get("native_distillation"))
        result["quantization"] |= bool(details.get("native_quantization")) or (
            engine and bool(details.get("run"))
        )
        result["evaluation"] |= bool(details.get("native_replay")) or (
            bool(details.get("engine_evaluation"))
            and _worker_paths_ready(
                runtime.evaluation_python or runtime.python,
                runtime.worker_root,
            )
        )
        result["simulation"] |= engine and bool(details.get("run") or details.get("simulation"))
    if settings is not None:
        from vla_platform.lifecycle.simulation import options

        result["simulation"] |= bool(options(settings)["profiles"])
    return result


def _configured(settings: Settings, robot_lab_config: Path, runtimes: dict, credentials=None):
    if credentials is None:
        from vla_platform.api_connections import ApiConnections

        credentials = ApiConnections(settings.data_dir).configured()
    paired = _paired(robot_lab_config)
    cloud = _cloud_connected(settings.data_dir)
    gemini = bool(credentials.get("gemini"))
    result = {ident: False for ident in MODULE_IDS}
    result.update(
        {
            "datasets": True,
            "models": True,
            "robot-lab": paired,
            "teaching": paired
            or _teaching_configured(settings)
            or _agent_configured(settings.data_dir),
            "augmentation": cloud or gemini,
            "decision-lab": _decision_configured(),
            "cloud-runs": (settings.cloud_runs_dir is not None and settings.cloud_runs_dir.is_dir())
            or (settings.data_dir / "cloud-runs").is_dir(),
            "google-cloud": cloud,
            "openai": bool(credentials.get("openai")),
            "gemini": gemini,
            "huggingface": HuggingFaceConnection(settings.data_dir).status().configured,
        }
    )
    result.update(
        {
            ident: bool(runtimes.get(ident, False))
            for ident in (
                "training",
                "distillation",
                "quantization",
                "evaluation",
                "simulation",
            )
        }
    )
    return result


def initialize_modules(
    settings: Settings,
    robot_lab_config: Path,
    *,
    existing: bool,
    runtime_ready: dict[str, bool] | None = None,
    credentials: dict[str, bool] | None = None,
) -> ModuleRegistry:
    with _registry_lock(settings.data_dir):
        value = read_registry(settings.data_dir)
        if value is not None:
            return value
        chosen = set(DEFAULT_MODULES)
        enabled = {}
        if existing:
            configured = _configured(settings, robot_lab_config, runtime_ready or {}, credentials)
            chosen.update(LEGACY_CORE_MODULES)
            chosen.update(key for key, ready in configured.items() if ready)
            if (runtime_ready or {}).get("robot"):
                chosen.add("robot-lab")
            if (runtime_ready or {}).get("decision"):
                chosen.add("decision-lab")
            saved = preferences(settings.data_dir)
            chosen.update(key for key, visible in saved.items() if visible)
            enabled.update(saved)
        value = ModuleRegistry(
            origin="migrated" if existing else "fresh",
            installed=[item[0] for item in CATALOG if item[0] in chosen],
            enabled={key: enabled.get(key, True) for key in chosen},
        )
        _write_registry(settings.data_dir, value)
        return value


def module_installed(data_dir: Path, ident: str) -> bool:
    """Read registration without changing it, running metadata probes or contacting providers."""
    value = read_registry(data_dir)
    return ident in (value.installed if value is not None else DEFAULT_MODULES)


def _editable():
    if getenv("JENSEN_MODULES") is not None:
        raise ValueError("Modules are managed by JENSEN_MODULES on this server.")


def set_module_installed(data_dir: Path, ident: ModuleId, installed: bool):
    _editable()
    if ident not in MODULE_IDS:
        raise ValueError("Unknown workspace module.")
    with _registry_lock(data_dir):
        value = read_registry(data_dir)
        if value is None:
            raise ValueError("The module registry has not been initialized.")
        if installed:
            if ident not in value.installed:
                value.installed.append(ident)
                value.enabled[ident] = True
            if ident == "cloud-runs":
                (data_dir / "cloud-runs").mkdir(mode=0o700, exist_ok=True)
        else:
            value.installed = [key for key in value.installed if key != ident]
            value.enabled.pop(ident, None)
        _write_registry(data_dir, value)


def set_module_enabled(data_dir: Path, ident: ModuleId, enabled: bool):
    _editable()
    if ident not in MODULE_IDS:
        raise ValueError("Unknown workspace module.")
    with _registry_lock(data_dir):
        value = read_registry(data_dir)
        if value is None or ident not in value.installed:
            raise ValueError("Install this module before changing its visibility.")
        value.enabled[ident] = enabled
        _write_registry(data_dir, value)


def workspace_modules(
    settings: Settings,
    robot_lab_config: Path,
    runtime_ready: dict[str, bool] | None = None,
    credentials: dict[str, bool] | None = None,
) -> list[WorkspaceModule]:
    configured = _configured(settings, robot_lab_config, runtime_ready or {}, credentials)
    registry = read_registry(settings.data_dir)
    registered = set(registry.installed if registry is not None else DEFAULT_MODULES)
    configured["augmentation"] = ("google-cloud" in registered and configured["google-cloud"]) or (
        "gemini" in registered and configured["gemini"]
    )
    enabled = registry.enabled.copy() if registry is not None else {key: True for key in registered}
    override = getenv("JENSEN_MODULES")
    if override is not None:
        selected = {item.strip() for item in override.split(",")}
        enabled = {key: "all" in selected or key in selected for key in registered}
    return [
        WorkspaceModule(
            id=ident,
            name=name,
            group=group,
            default_installed=ident in DEFAULT_MODULES,
            installed=ident in registered,
            enabled=ident in registered and enabled.get(ident, False),
            available=True,
            configured=configured[ident],
            setup=hint,
            setup_url=guide,
            managed=override is not None,
        )
        for ident, name, group, hint, guide in CATALOG
    ]
