"""Compatibility import for alpha installations; use jensen_act."""

from importlib import import_module
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "jensen_act")]


def __getattr__(name):
    return getattr(import_module("jensen_act"), name)
