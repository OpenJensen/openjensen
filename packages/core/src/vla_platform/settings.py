from dataclasses import dataclass
from pathlib import Path

from vla_platform.environment import getenv


def workspace_directory() -> Path:
    configured = getenv("JENSEN_DATA_DIR")
    if configured is not None:
        return Path(configured).expanduser().resolve()
    # Reuse alpha data instead of silently creating an empty replacement workspace.
    legacy = Path(".firebird")
    return (
        legacy if legacy.exists() and not Path(".jensen").exists() else Path(".jensen")
    ).resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    local_root: Path | None = None
    static_dir: Path | None = None
    runtime_config: Path | None = None
    cloud_runs_dir: Path | None = None
    simulation_config: Path | None = None
    recording_config: Path | None = None
    teaching_session_config: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        def path(name: str, *, resolve: bool = False) -> Path | None:
            raw = getenv(name)
            if not raw:
                return None
            value = Path(raw).expanduser()
            return value.resolve() if resolve else value.absolute()

        static = Path(getenv("JENSEN_WEB_DIR", "apps/web/out")).expanduser().resolve()
        return cls(
            data_dir=workspace_directory(),
            local_root=path("JENSEN_LOCAL_DATA_ROOT", resolve=True),
            static_dir=static if static.is_dir() else None,
            runtime_config=path("JENSEN_RUNTIME_CONFIG", resolve=True),
            cloud_runs_dir=path("JENSEN_CLOUD_RUNS_DIR"),
            simulation_config=path("JENSEN_SIMULATION_CONFIG", resolve=True),
            recording_config=path("JENSEN_RECORDING_CONFIG"),
            teaching_session_config=path("JENSEN_TEACHING_SESSION_CONFIG"),
        )
