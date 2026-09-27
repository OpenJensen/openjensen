"""Operator-selected local profiles; browser requests never choose paths or commands."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vla_platform.datasets import recordings

MAX_CONFIG_BYTES = 65536
MAX_SOURCE_BYTES = 8 * 1024**2


class TeachingError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


class LocalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    label: str = Field(min_length=1, max_length=100)
    project_id: str = Field(min_length=1, max_length=200)
    isaac_python: str = Field(min_length=1, max_length=4096)
    worker_root: str = Field(min_length=1, max_length=4096)
    settings_path: str = Field(min_length=1, max_length=4096)
    lease_path: str = Field(min_length=1, max_length=4096)
    control_port: int = Field(ge=1024, le=65535)
    accept_eula: Literal[True]
    max_seconds: int = Field(default=300, ge=1, le=3600)
    max_capture_bytes: int = Field(default=512 * 1024**2, ge=1024, le=8 * 1024**3)

    @field_validator("accept_eula", mode="before")
    @classmethod
    def explicit_license(cls, value):
        if value is not True:
            raise ValueError("Local Isaac requires explicit license acceptance")
        return value


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    profiles: list[LocalProfile] = Field(min_length=1, max_length=32)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Teaching configuration schema must be integer 1")
        return value

    @model_validator(mode="after")
    def identities(self):
        if len({profile.id for profile in self.profiles}) != len(self.profiles):
            raise ValueError("Teaching profile identities must be unique")
        return self


@dataclass(frozen=True)
class LoadedProfile:
    value: LocalProfile
    identity: str
    recording: recordings.Loaded
    settings_sha256: str
    scene_sha256: str
    source_sha256: dict[str, str]

    @property
    def worker(self) -> Path:
        return Path(self.value.worker_root) / "firebird_teaching"

    def environment(self, token_path: Path) -> dict[str, str]:
        result = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"}
        }
        result.update(
            PYTHONPATH=os.pathsep.join(
                [self.value.worker_root, str(Path(self.value.worker_root).parent / "isaac_sim")]
            ),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            ACCEPT_EULA="Y",
            FIREBIRD_TEACHING_CONTROL_TOKEN_FILE=str(token_path),
        )
        return result


def load(config_path: Path | None, recording_path: Path | None, profile_id: str, project_id: str):
    if config_path is None or recording_path is None:
        raise TeachingError("Managed teaching is not configured on this application host", 503)
    raw = recordings.read(config_path, MAX_CONFIG_BYTES)
    value = Configuration.model_validate(recordings.decode(raw))
    profile = next((p for p in value.profiles if p.id == profile_id), None)
    if profile is None or profile.project_id != project_id:
        raise TeachingError("No teaching profile belongs to this project", 404)
    recording = recordings.load_config(recording_path)
    capture_root = recording.root(project_id)
    settings_path = recordings.path(profile.settings_path)
    settings_raw = recordings.read(settings_path, 16384)
    # The fixed worker authoritatively validates settings and scene before SDK startup.
    settings_value = recordings.decode(settings_raw)
    scene_value = settings_value.get("scene")
    if not isinstance(scene_value, str) or not scene_value or len(scene_value) > 2048:
        raise TeachingError("Teaching settings need a local scene")
    scene = recordings.path(settings_path.parent / scene_value)
    scene_hash = recordings.digest(recordings.read(scene, 256 * 1024**2))
    lease = recordings.path(profile.lease_path)
    with recordings.directory(lease.parent):
        pass
    if lease == settings_path or lease.is_relative_to(capture_root):
        raise TeachingError("Teaching lease must be separate from configuration and captures")
    root = recordings.path(profile.worker_root)
    sources = {}
    total = 0
    for prefix in (root / "firebird_teaching", root.parent / "isaac_sim" / "sim_worker"):
        with recordings.directory(prefix):
            pass
        for source in sorted(prefix.rglob("*.py")):
            if len(sources) >= 256:
                raise TeachingError("Teaching source exceeds its file bound")
            data = recordings.read(source, min(MAX_SOURCE_BYTES - total, 2 * 1024**2))
            total += len(data)
            sources[str(source)] = recordings.digest(data)
    required = [
        root / "firebird_teaching" / name for name in ("isaac.py", "managed.py", "dataset.py")
    ]
    if any(str(source) not in sources for source in required):
        raise TeachingError("Managed teaching worker is incomplete", 503)
    settings_hash = recordings.digest(settings_raw)
    identity = recordings.digest(
        recordings.canonical(
            {
                "configuration": recordings.digest(raw),
                "recordings": recording.identity,
                "profile": profile.model_dump(),
                "settings": settings_hash,
                "scene_root": scene_hash,
                "sources": sources,
                "executable": recordings.executable_identity(profile.isaac_python),
            }
        )
    )
    return LoadedProfile(profile, identity, recording, settings_hash, scene_hash, sources)


def options(config_path: Path | None, recording_path: Path | None, project_id: str) -> dict:
    if config_path is None or recording_path is None:
        return {
            "configured": False,
            "available": False,
            "profiles": [],
            "message": "Managed local Isaac teaching is not configured.",
        }
    try:
        config = Configuration.model_validate(
            recordings.decode(recordings.read(config_path, MAX_CONFIG_BYTES))
        )
        found = []
        for profile in config.profiles:
            if profile.project_id != project_id:
                continue
            loaded = load(config_path, recording_path, profile.id, project_id)
            found.append(
                {
                    "id": profile.id,
                    "label": profile.label,
                    "profile_sha256": loaded.identity,
                    "max_seconds": profile.max_seconds,
                    "max_capture_bytes": profile.max_capture_bytes,
                    "runtime_verified": False,
                    "transport": "local_owned_process",
                }
            )
        supported = sys.platform.startswith("linux")
        return {
            "configured": bool(found),
            "available": bool(found) and supported,
            "profiles": found,
            "message": "Isaac readiness is checked only after an explicit start."
            if supported
            else (
                "This local Isaac runner requires a configured Linux host; "
                "no remote runner is enabled."
            ),
        }
    except OSError, ValueError:
        return {
            "configured": False,
            "available": False,
            "profiles": [],
            "message": "Managed teaching configuration is unavailable or invalid.",
        }
