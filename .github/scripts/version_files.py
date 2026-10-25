"""Patch-only application versions; independent worker package versions are untouched."""

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path

FILES = (
    "pyproject.toml",
    "packages/core/pyproject.toml",
    "packages/core/src/vla_platform/__init__.py",
    "apps/web/package.json",
    "apps/desktop/package.json",
    "apps/desktop/src-tauri/Cargo.toml",
    "apps/desktop/src-tauri/Cargo.lock",
    "apps/desktop/src-tauri/tauri.conf.json",
    "uv.lock",
    "packages/core/openapi.json",
)
TOML = {
    "pyproject.toml": "project",
    "packages/core/pyproject.toml": "project",
    "apps/desktop/src-tauri/Cargo.toml": "package",
}
LOCKS = {
    "uv.lock": {"openjensen-workspace", "vla-platform"},
    "apps/desktop/src-tauri/Cargo.lock": {"openjensen-desktop"},
}
INIT = "packages/core/src/vla_platform/__init__.py"


def parse(version):
    if not isinstance(version, str) or not re.fullmatch(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version
    ):
        raise ValueError("Use a stable major.minor.patch version.")
    return tuple(int(part) for part in version.split("."))


def next_patch(version):
    major, minor, patch = parse(version)
    return f"{major}.{minor}.{patch + 1}"


def current(contents):
    value = tomllib.loads(contents["pyproject.toml"])["project"]["version"]
    parse(value)
    return value


def versions(contents):
    found = {}
    for path in FILES:
        text = contents[path]
        if path in TOML:
            found[path] = tomllib.loads(text)[TOML[path]]["version"]
        elif path in LOCKS:
            packages = tomllib.loads(text)["package"]
            for name in LOCKS[path]:
                values = [p["version"] for p in packages if p["name"] == name]
                if len(values) != 1:
                    raise ValueError(f"{path} must contain exactly one {name} package.")
                found[f"{path}:{name}"] = values[0]
        elif path == INIT:
            match = re.search(r"""(?m)^__version__\s*=\s*["']([^"']+)["']""", text)
            if not match:
                raise ValueError("Core __version__ is missing.")
            found[path] = match[1]
        else:
            value = json.loads(text)
            found[path] = (
                value["info"]["version"] if path.endswith("openapi.json") else value["version"]
            )
    return found


def validate(contents, base=None, manual=False):
    version = current(contents)
    if any(value != version for value in versions(contents).values()):
        raise ValueError("Application version manifests disagree.")
    if base is not None:
        if manual:
            if parse(version) <= parse(base):
                raise ValueError("A manual version must be newer than main.")
        elif version != next_patch(base):
            raise ValueError(f"Expected patch version {next_patch(base)}; got {version}.")
    return version


def assignment(text, name, version):
    pattern = rf"""(?m)^({re.escape(name)}\s*=\s*)(["']).*?\2(.*)$"""
    text, count = re.subn(pattern, lambda m: m[1] + json.dumps(version) + m[3], text)
    if count != 1:
        raise ValueError(f"Expected exactly one {name} assignment.")
    return text


def rewrite(contents, version):
    parse(version)
    changed = dict(contents)
    for path in FILES:
        text = contents[path]
        if path in TOML:
            section = TOML[path]
            pattern = rf"(?ms)^\[{section}\][^\n]*\n.*?(?=^\[|\Z)"
            match = re.search(pattern, text)
            if not match:
                raise ValueError(f"Missing {section} in {path}.")
            changed[path] = (
                text[: match.start()]
                + assignment(match[0], "version", version)
                + text[match.end() :]
            )
        elif path in LOCKS:
            seen = set()

            def replace(match):
                block = match[0]
                package = tomllib.loads(block)["package"][0]
                if package["name"] in LOCKS[path]:
                    if package["name"] in seen:
                        raise ValueError("Duplicate local package in lockfile.")
                    seen.add(package["name"])
                    return assignment(block, "version", version)
                return block

            changed[path] = re.sub(
                r"(?ms)^\[\[package\]\][^\n]*\n.*?(?=^\[\[package\]\]|\Z)", replace, text
            )
            if seen != LOCKS[path]:
                raise ValueError(f"Missing local packages in {path}.")
        elif path == INIT:
            changed[path] = assignment(text, "__version__", version)
        else:
            value = json.loads(text)
            if path.endswith("openapi.json"):
                value["info"]["version"] = version
            else:
                value["version"] = version
            changed[path] = (
                json.dumps(value, indent=2, sort_keys=path.endswith("openapi.json")) + "\n"
            )
    validate(changed)
    return changed


def load(root):
    values = {}
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Version manifest must be a regular file: {name}")
        values[name] = path.read_text(encoding="utf-8")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["current", "next", "apply", "check"])
    parser.add_argument("--root", type=Path, default=Path("."))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--base")
    group.add_argument("--base-ref")
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()
    contents = load(args.root)
    base = args.base
    if args.base_ref:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", args.base_ref):
            raise ValueError("Invalid base Git ref.")
        text = subprocess.check_output(
            ["git", "show", args.base_ref + ":pyproject.toml"], cwd=args.root, text=True
        )
        base = tomllib.loads(text)["project"]["version"]
    if args.command == "next":
        print(next_patch(base or current(contents)))
    elif args.command == "apply":
        if not base:
            raise ValueError("Supply --base or --base-ref.")
        result = rewrite(contents, next_patch(base))
        for name, text in result.items():
            if text != contents[name]:
                (args.root / name).write_text(text, encoding="utf-8")
        print(validate(result, base))
    else:
        print(validate(contents, base, args.manual))


if __name__ == "__main__":
    main()
