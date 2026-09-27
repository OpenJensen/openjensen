"""Fixed application-owned subprocess modes for source and frozen installations."""

import sys
from pathlib import Path


def intake_command(request: Path, result: Path) -> list[str]:
    """A frozen executable is not a general Python interpreter."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "intake-worker", str(request), str(result)]
    return [sys.executable, "-m", "vla_platform.datasets.worker", str(request), str(result)]
