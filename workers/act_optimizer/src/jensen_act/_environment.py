"""Canonical OPEN JENSEN settings with read compatibility for alpha deployments."""

import os


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


def required_env(name: str) -> str:
    value = getenv(name)
    if value is None:
        raise KeyError(name)
    return value
