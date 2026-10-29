"""Run under the isolated OpenUSD Python, never the API environment."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


def build(template, output, proposal):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("robot", "textures"):
        shutil.copytree(template / name, output / name, dirs_exist_ok=True)
    for name in ("LICENSE", "NOTICE.md", "capture.yaml", "smoke_test.py"):
        shutil.copyfile(template / name, output / name)
    config = json.loads((template / "scene_config.json").read_text())
    for key in ("camera", "robot", "cup", "box"):
        config[key].update(proposal[key])
    (output / "scene_config.json").write_text(json.dumps(config, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("scene_builder", template / "build_scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = output
    module.CONFIG = output / "scene_config.json"
    module.SCENE = output / "scene.usda"
    # Existing texture crops retain their source provenance and coordinates.
    module._textures = lambda: None
    module._build()
    spec = importlib.util.spec_from_file_location("scene_validator", template / "validate.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    from pxr import Usd

    stage = Usd.Stage.Open(str(output / "scene.usda"))
    errors = validator._check_stage(stage, output / "scene.usda")
    for check in (
        validator._check_physics,
        validator._check_bodies,
        validator._check_joints,
        validator._check_geometry,
        validator._check_cameras,
    ):
        errors.extend(check(stage))
    receipt = {
        "passed": not errors,
        "errors": errors,
        "scope": (
            "OpenUSD structure only; Isaac runtime, image similarity and "
            "calibrated replay are unverified"
        ),
    }
    (output / "structure.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("proposal", type=Path)
    args = parser.parse_args()
    result = build(args.template, args.output, json.loads(args.proposal.read_text()))
    raise SystemExit(0 if result["passed"] else 1)
