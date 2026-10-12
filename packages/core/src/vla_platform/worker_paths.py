"""Resolve installed worker source layouts without changing persisted configuration."""

from pathlib import Path

_PACKAGES = {
    "jensen_vla": "firebird_vla",
    "jensen_act": "firebird_act",
    "jensen_distill": "firebird_distill",
    "jensen_decision": "firebird_decision",
    "jensen_teaching": "firebird_teaching",
    "jensen_quant": "firebird_quant",
}
_CANONICAL = {legacy: current for current, legacy in _PACKAGES.items()}


def canonical_module(module: str) -> str:
    package, separator, suffix = module.partition(".")
    return _CANONICAL.get(package, package) + separator + suffix


def worker_project(workers: Path, name: str) -> Path:
    """Only the quant worker directory changed; other project names are stable."""
    if name in {"jensen_quant", "firebird_quant"}:
        for candidate in (workers / "jensen_quant", workers / "firebird_quant"):
            if candidate.is_dir():
                return candidate
        return workers / "jensen_quant"
    return workers / name


def worker_package(root: Path, package: str, *, layout: str = "src") -> Path:
    """Prefer canonical source, falling back to an unchanged installed alpha tree."""
    canonical = canonical_module(package)
    bases = [root / layout, root] if layout == "src" else [root / layout if layout else root]
    for name in (canonical, _PACKAGES.get(canonical, canonical)):
        for base in bases:
            candidate = base / name
            if candidate.is_dir():
                return candidate
    return bases[0] / canonical


def worker_module(root: Path, module: str, *, layout: str = "src") -> str:
    """Resolve a registered entry point in this tree; custom operator modules stay intact."""
    canonical = canonical_module(module)
    package, separator, suffix = canonical.partition(".")
    if package not in _PACKAGES:
        return module
    if not root.exists():
        # Command construction can precede mounted/image source availability.
        return module
    bases = [root / layout, root] if layout == "src" else [root / layout if layout else root]
    relative = Path(*suffix.split(".")).with_suffix(".py") if separator else Path("__main__.py")
    for name in (package, _PACKAGES[package]):
        for base in bases:
            if (base / name / relative).is_file():
                return name + separator + suffix
    raise ValueError(
        "Installed worker source is missing the configured entry point. "
        "Upgrade that worker source tree to the matching OPEN JENSEN release while "
        "preserving its environment and data, then check the worker again."
    )
