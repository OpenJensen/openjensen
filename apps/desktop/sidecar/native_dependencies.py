"""Inspect the relocated Mach-O closure with a fixed local tool; never load binaries."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SYSTEM_ROOTS = (Path("/usr/lib"), Path("/System/Library"))


def _tool(path: Path, option: str, limit: int) -> str:
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(
            ["/usr/bin/otool", option, str(path)],
            stdout=output,
            stderr=subprocess.STDOUT,
            env={"PATH": "/usr/bin:/bin"},
            timeout=10,
            check=False,
        )
        output.seek(0)
        raw = output.read(limit + 1)
    if result.returncode or len(raw) > limit:
        raise ValueError("Native dependency inspection failed or exceeded its output limit")
    return raw.decode("utf-8", errors="strict")


def parse_dependencies(text: str) -> list[str]:
    values = []
    for line in text.splitlines()[1:]:
        match = re.fullmatch(r"\s+(.+) \(compatibility version .+\)", line)
        if match:
            values.append(match.group(1))
        elif line.strip():
            raise ValueError("Unrecognized otool dependency output")
    return values


def parse_rpaths(text: str) -> list[str]:
    values, awaiting = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("cmd "):
            if awaiting:
                raise ValueError("Missing LC_RPATH path")
            awaiting = stripped == "cmd LC_RPATH"
        if awaiting and stripped.startswith("path "):
            match = re.fullmatch(r"path (.+) \(offset [0-9]+\)", stripped)
            if not match:
                raise ValueError("Unrecognized LC_RPATH path")
            values.append(match.group(1))
            awaiting = False
    if awaiting:
        raise ValueError("Missing LC_RPATH path")
    return values


def _system(path: Path) -> bool:
    return (
        path.is_absolute()
        and ".." not in path.parts
        and any(path.is_relative_to(root) for root in SYSTEM_ROOTS)
    )


def _location(value: str, binary: Path, executable: Path, root: Path) -> Path:
    for prefix, parent in (
        ("@loader_path", binary.parent),
        ("@executable_path", executable.parent),
    ):
        if value == prefix or value.startswith(prefix + "/"):
            candidate = (parent / value[len(prefix) :].lstrip("/")).resolve(strict=True)
            if not candidate.is_relative_to(root):
                raise ValueError("Native reference escapes the relocated payload")
            return candidate
    candidate = Path(value)
    if _system(candidate):
        return candidate
    # An absolute payload path would break another relocation and is not accepted.
    raise ValueError("Native reference is neither relocatable nor an allowed system path")


def inspect(root: Path, recorded: dict[str, Any]) -> dict[str, Any]:
    """Resolve each recorded Mach-O dependency within payload or fixed macOS roots.

    This is static dependency evidence, not a clean-machine or deployment-target proof.
    """
    root = root.resolve(strict=True)
    executable = root / "firebird-sidecar"
    binaries = [
        root / name
        for name, item in recorded["entries"].items()
        if item["kind"] == "file" and item["macho_cpu_types"] is not None
    ]
    if executable not in binaries:
        raise ValueError("Main executable is missing from Mach-O inventory")
    raw: dict[Path, dict[str, Any]] = {}
    for path in binaries:
        libraries = _tool(path, "-L", 65536)
        commands = _tool(path, "-l", 256 * 1024)
        raw[path] = {
            "dependencies": parse_dependencies(libraries),
            "rpaths": parse_rpaths(commands),
            "otool_L": libraries,
            "otool_l": commands,
        }
    resolved: dict[str, Any] = {}
    for path, item in raw.items():
        rpaths = [_location(value, path, executable, root) for value in item["rpaths"]]
        if path != executable:
            rpaths += [
                _location(value, executable, executable, root)
                for value in raw[executable]["rpaths"]
            ]
        dependencies = []
        for dependency in item["dependencies"]:
            if dependency.startswith("@rpath/"):
                suffix = dependency[len("@rpath/") :]
                candidates = [base / suffix for base in rpaths]
                matched = []
                for candidate in candidates:
                    if _system(candidate):
                        matched.append(str(candidate))
                    elif candidate.exists():
                        target = candidate.resolve(strict=True)
                        if target.is_relative_to(root) and target.is_file():
                            matched.append(target.relative_to(root).as_posix())
                if not matched:
                    raise ValueError(f"Unresolved relocated native dependency: {dependency}")
                dependencies.append({"reference": dependency, "resolved": sorted(set(matched))})
            else:
                target = _location(dependency, path, executable, root)
                if not _system(target) and not target.is_file():
                    raise ValueError("Native dependency does not resolve to a file")
                dependencies.append(
                    {
                        "reference": dependency,
                        "resolved": [
                            str(target) if _system(target) else target.relative_to(root).as_posix()
                        ],
                    }
                )
        resolved[path.relative_to(root).as_posix()] = {
            **item,
            "resolved_dependencies": dependencies,
        }
    return {
        "schema_version": 1,
        "native_files": resolved,
        "native_file_count": len(resolved),
        "private_absolute_dependencies": False,
        "static_relocation_check": True,
        "clean_machine_acceptance": False,
    }
