"""Image-capable Codex model choices; catalog metadata never establishes account access."""

import json
import os
import re
from pathlib import Path

from .contracts import AgentSettings

EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
# Fresh installations can choose explicitly before Codex has cached its catalog.
FALLBACK = (
    ("gpt-6.1-sol", "GPT-6.1 Sol", EFFORTS),
    ("gpt-6-astra", "GPT-6 Astra", EFFORTS),
    ("gpt-6.1-terra", "GPT-6.1 Terra", EFFORTS),
    ("gpt-6-sol", "GPT-6 Sol", EFFORTS),
    ("gpt-6-luna", "GPT-6 Luna", EFFORTS[:-1]),
    ("gpt-5.6-sol", "GPT-5.6 Sol", EFFORTS),
    ("gpt-5.6-terra", "GPT-5.6 Terra", EFFORTS),
    ("gpt-5.6-luna", "GPT-5.6 Luna", EFFORTS[:-1]),
)


def catalog():
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    path = home / "models_cache.json"
    found = []
    try:
        if path.stat().st_size > 2 * 1024**2:
            raise ValueError("Catalog exceeds limit")
        entries = json.loads(path.read_bytes())["models"]
        if not isinstance(entries, list):
            raise ValueError("Invalid model catalog")
        for item in entries[:128]:
            if not isinstance(item, dict):
                continue
            ident = item.get("slug", "")
            if (
                not isinstance(ident, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", ident)
                or item.get("visibility") != "list"
                or item.get("supported_in_api") is not True
                or "image" not in (item.get("input_modalities") or [])
                or any(model["id"] == ident for model in found)
            ):
                continue
            levels = item.get("supported_reasoning_levels") or []
            if not isinstance(levels, list):
                continue
            advertised = {level.get("effort") for level in levels if isinstance(level, dict)}
            efforts = [effort for effort in EFFORTS if effort in advertised]
            if not efforts:
                continue
            label = item.get("display_name")
            found.append(choice(ident, label if isinstance(label, str) else ident, efforts))
    except OSError, ValueError, KeyError, TypeError:
        found = []
    source = "codex_catalog" if found else "bundled"
    models = found or [choice(ident, label, list(efforts)) for ident, label, efforts in FALLBACK]
    default = os.environ.get("JENSEN_SCENE_MODEL")
    if not any(item["id"] == default for item in models):
        default = models[0]["id"]
    return {"models": models, "default_model": default, "model_catalog_source": source}


def choice(ident, label, efforts):
    return {
        "id": ident,
        "label": label[:128],
        "reasoning_efforts": efforts,
        "default_reasoning_effort": "medium" if "medium" in efforts else efforts[0],
    }


def resolve(request, options):
    model = request.model or options["default_model"]
    selected = next((item for item in options["models"] if item["id"] == model), None)
    if selected is None:
        raise ValueError("Choose a model from the scene model list. Refresh it and try again.")
    if request.reasoning_effort not in selected["reasoning_efforts"]:
        raise ValueError("This model does not support the selected power. Choose a listed level.")
    return AgentSettings(model=model, reasoning_effort=request.reasoning_effort)
