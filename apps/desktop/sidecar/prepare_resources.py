"""Build-time only: copy a completed static export into a new bounded resource directory."""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from sidecar_resources import (
    MAX_FILES,
    MAX_STATIC_BYTES,
    absolute_directory,
    file_identity,
    load_resources,
)


def prepare(web: Path, output: Path, build_id: str) -> Path:
    web = absolute_directory(web)
    output = absolute_directory(output, existing=False)
    if output.exists() or not re.fullmatch(r"[a-f0-9]{40}", build_id):
        raise ValueError("Use a new output directory and an exact source commit")
    if output in web.parents or web in output.parents:
        raise ValueError("Resource output and source must be disjoint")
    inventory, total = {}, 0
    for directory, dirs, files in os.walk(web, followlinks=False):
        for name in [*dirs, *files]:
            mode = (Path(directory) / name).lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("Static export must contain only real files and directories")
        for name in files:
            path = Path(directory) / name
            size = path.stat().st_size
            total += size
            if total > MAX_STATIC_BYTES or len(inventory) >= MAX_FILES:
                raise ValueError("Static export exceeds its build-time bound")
            inventory[path.relative_to(web).as_posix()] = {
                "bytes": size,
                "sha256": file_identity(path, size),
            }
    if "index.html" not in inventory:
        raise ValueError("A completed static export with index.html is required")
    # Exclusive new output; a failed build leaves its partial directory for inspection.
    # Never replace or recursively remove an existing resource directory.
    output.mkdir()
    target = output / "web"
    target.mkdir()
    for name, info in inventory.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            web / name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(fd, "rb") as source, destination.open("xb") as result:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("Static source changed type")
            seen, digest = 0, hashlib.sha256()
            while chunk := source.read(min(65536, info["bytes"] + 1 - seen)):
                seen += len(chunk)
                if seen > info["bytes"]:
                    raise ValueError("Static source grew while copying")
                result.write(chunk)
                digest.update(chunk)
            if seen != info["bytes"] or digest.hexdigest() != info["sha256"]:
                raise ValueError("Static source changed while copying")
    manifest = {
        "schema_version": 1,
        "app_version": "0.1.0",
        "build_id": build_id,
        "files": inventory,
    }
    with (output / "resources.json").open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
    load_resources(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    args = parser.parse_args()
    prepare(args.web, args.output, args.build_id)


if __name__ == "__main__":
    main()
