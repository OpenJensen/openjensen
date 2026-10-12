"""Compatibility import for alpha installations; use jensen_act."""

from importlib import import_module
from pathlib import Path
from typing import Any

__path__ = [str(Path(__file__).resolve().parent.parent / "jensen_act")]


def __getattr__(name: str) -> Any:
    return getattr(import_module("jensen_act"), name)
