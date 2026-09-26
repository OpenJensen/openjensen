import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    local_root: Path | None = None
    static_dir: Path | None = None
    runtime_config: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        local_root = os.getenv("FIREBIRD_LOCAL_DATA_ROOT")
        static = Path(os.getenv("FIREBIRD_WEB_DIR", "apps/web/out")).resolve()
        return cls(
            data_dir=Path(os.getenv("FIREBIRD_DATA_DIR", ".firebird")).resolve(),
            local_root=Path(local_root).resolve() if local_root else None,
            static_dir=static if static.is_dir() else None,
            runtime_config=Path(os.environ["FIREBIRD_RUNTIME_CONFIG"]).resolve()
            if os.getenv("FIREBIRD_RUNTIME_CONFIG")
            else None,
        )
