#!/usr/bin/env python3
"""Install a closed native checkpoint for offline inference, without motor permissions."""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers/robot_lab"))
from policy_contract import validate_policy_config, validate_processors


def install(checkpoint, data_dir, ident, name, source=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ident) or not name.strip() or len(name) > 100:
        raise ValueError("Choose a bounded policy identifier and name")
    config = validate_policy_config(checkpoint)
    validate_processors(checkpoint, config["type"])
    images = [
        key
        for key, value in config["input_features"].items()
        if value.get("type") == "VISUAL"
        or key.startswith(("observation.images.", "observation.image"))
    ]
    root = data_dir / "policies"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("Policy registry cannot use a symbolic link")
    destination = root / ident
    if destination.exists() or destination.is_symlink():
        raise ValueError("This policy identifier is already installed")
    with tempfile.TemporaryDirectory(prefix=".native-policy-", dir=data_dir) as temporary:
        staging = Path(temporary) / ident
        target = staging / "pretrained_model"
        target.mkdir(parents=True, mode=0o700)
        os.chmod(staging, 0o700)

        def copy_file(item, output):
            if item.is_symlink() or not item.is_file():
                raise ValueError("Native checkpoints accept regular data files only")
            try:
                os.link(item, output)
            except OSError:
                shutil.copy2(item, output)

        for item in checkpoint.iterdir():
            if item.name == "assets" and config["type"] == "smolvla":
                assets = target / "assets/smolvlm"
                assets.mkdir(parents=True, mode=0o700)
                for file in (item / "smolvlm").iterdir():
                    copy_file(file, assets / file.name)
            elif (
                item.is_file()
                and not item.is_symlink()
                and (
                    item.suffix in {".json", ".safetensors", ".md", ".txt"}
                    or item.name == "LICENSE"
                )
            ):
                copy_file(item, target / item.name)
            else:
                raise ValueError("Unsupported native checkpoint payload")
        validate_policy_config(target)
        validate_processors(target, config["type"])
        manifest = {
            "schema_version": 1,
            "name": name.strip(),
            "policy_type": config["type"],
            "checkpoint": "pretrained_model",
            "offline_only": True,
            "fps": None,
            "camera_key": images[0],
            "joint_names": None,
            "state_units": None,
            "action_units": None,
            "source": source or {"kind": "local_checkpoint"},
        }
        path = staging / "run_policy.json"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.chmod(path, 0o600)
        staging.rename(destination)
    return {"id": ident, "name": name.strip(), "policy_type": config["type"], "offline_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-repo")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    data = args.data_dir.expanduser().resolve()
    if not data.is_dir():
        parser.error("Choose the existing companion data directory")
    source = None
    if args.source_repo or args.source_revision:
        if not args.source_repo or not re.fullmatch(r"[a-f0-9]{40}", args.source_revision or ""):
            parser.error("Supply the source repository and its immutable revision")
        source = {
            "kind": "pretrained",
            "repository": args.source_repo,
            "revision": args.source_revision,
        }
    print(json.dumps(install(args.checkpoint.expanduser(), data, args.id, args.name, source)))


if __name__ == "__main__":
    main()
