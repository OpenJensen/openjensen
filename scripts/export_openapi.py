"""Export contracts without creating a workspace, starting jobs or importing Torch."""

import json
from pathlib import Path

from vla_platform.api import create_app
from vla_platform.settings import Settings

root = Path(__file__).resolve().parents[1]
schema = create_app(Settings(data_dir=root / ".firebird")).openapi()
(root / "packages/core/openapi.json").write_text(
    json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
