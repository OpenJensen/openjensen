"""Canonical OpenJensen settings with read compatibility for alpha deployments."""

import os
from collections.abc import Mapping


def getenv(name: str, default: str | None = None) -> str | None:
    """Prefer JENSEN_* even when a legacy caller asks for FIREBIRD_*.

    Read at call time so changing configuration in a test or worker does not
    mutate global environment state. Explicit empty canonical values win too.
    """
    if name.startswith("FIREBIRD_"):
        canonical = "JENSEN_" + name.removeprefix("FIREBIRD_")
        legacy = name
    elif name.startswith("JENSEN_"):
        canonical = name
        legacy = "FIREBIRD_" + name.removeprefix("JENSEN_")
    else:
        return os.environ.get(name, default)
    return os.environ.get(canonical, os.environ.get(legacy, default))


def canonical_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Copy source so each FIREBIRD_* value is also readable as JENSEN_*.

    An explicit JENSEN_* value, even an empty one, wins over its alias.
    """
    result = dict(source)
    for name, value in source.items():
        if name.startswith("FIREBIRD_"):
            result.setdefault("JENSEN_" + name.removeprefix("FIREBIRD_"), value)
    return result


def required_env(name: str) -> str:
    value = getenv(name)
    if value is None:
        raise KeyError(name)
    return value
