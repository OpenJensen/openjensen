from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def model(self, name: str) -> dict[str, Any]:
        try:
            return self.data["models"][name]
        except KeyError as exc:
            raise ValueError(f"Unknown model: {name}") from exc

    def preset(self, name: str) -> dict[str, Any]:
        try:
            preset = self.data["presets"][name]
        except KeyError as exc:
            raise ValueError(f"Unknown preset: {name}") from exc
        # `null` remains the concise manifest spelling for the BF16 reference.
        if preset is None:
            return {"type": None, "vision": False}
        if isinstance(preset, str):
            return {"type": preset, "vision": False}
        if not isinstance(preset, dict) or "type" not in preset:
            raise ValueError(f"Preset {name} must be null, a quantization type, or a mapping with type")
        return {"type": preset["type"], "vision": bool(preset.get("vision", False))}

    def matrix(self) -> list[tuple[str, str]]:
        return [(model, preset) for model in self.data["models"] for preset in self.data["presets"]]


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).resolve()
    with config_path.open() as stream:
        data = yaml.safe_load(stream)
    required = {"version", "runtime", "models", "presets", "evaluation", "machine_label"}
    missing = required - set(data or {})
    if missing:
        raise ValueError(f"Manifest missing required keys: {', '.join(sorted(missing))}")
    if not data["models"] or not data["presets"]:
        raise ValueError("Manifest must define at least one model and one preset")
    return BenchmarkConfig(config_path, data)
