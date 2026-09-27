"""Build an episode-specific reconstruction without changing the baseline scene."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from pxr import Sdf, Usd, UsdGeom, UsdShade

_ROOT = Path(__file__).resolve().parent
_CUP = "/World/Props/Cup"
_BOX = "/World/Props/Box"
_CALIBRATION = "calibration.experimental.yaml"
_STAGE_METADATA = ("metersPerUnit", "upAxis", "startTimeCode")
_JOINT_COUNT = 6


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(profile):
    cup = profile["cup"]
    fields = ("height", "radius_bottom", "radius_top", "thickness", "mass")
    dimensions = [cup[key] for key in fields]
    if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
        raise ValueError("Cup dimensions and mass must be finite and positive")
    if cup["thickness"] >= min(cup["radius_bottom"], cup["radius_top"]):
        raise ValueError("Cup walls must leave an open interior")
    if (len(profile["initial_state"]) != _JOINT_COUNT
            or not np.isfinite(profile["initial_state"]).all()):
        raise ValueError("Initial state must contain six finite joint values")
    if len(cup["position"]) != 3 or not np.isfinite(cup["position"]).all():
        raise ValueError("Cup position must contain three finite coordinates")


def _props(stage, profile, builder):
    # Regenerate contact meshes and inertia together with visible dimensions.
    generated = Usd.Stage.CreateInMemory()
    names = ("Paper", "CupPrint", "Cardboard", "BoxPrint")
    materials = {
        name: UsdShade.Material.Define(generated, f"/World/Materials/{name}")
        for name in names
    }
    builder._cup(generated, profile["cup"], materials)
    UsdGeom.Xform.Define(stage, "/World/Props")
    Sdf.CopySpec(generated.GetRootLayer(), _CUP, stage.GetRootLayer(), _CUP)
    if "box" not in profile:
        return

    builder._box(generated, profile["box"], materials)
    Sdf.CopySpec(generated.GetRootLayer(), _BOX, stage.GetRootLayer(), _BOX)


def _manifest(output, name, profile):
    manifest = yaml.safe_load((_ROOT / "rollout.experimental.yaml").read_text())
    manifest["scene"]["uri"] = f"scene.{name}.usda"
    manifest["control"] = profile["control"]
    (output / f"rollout.{name}.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    if output != _ROOT:
        shutil.copyfile(_ROOT / _CALIBRATION, output / _CALIBRATION)


def _build(profile_path, output):
    profile_path, output = profile_path.resolve(), output.resolve()
    profile = json.loads(profile_path.read_text())
    _check(profile)
    output.mkdir(parents=True, exist_ok=True)
    name = f"episode-{profile['episode']:03d}"
    source_path = _ROOT / "scene.usda"
    source = Usd.Stage.Open(str(source_path))
    stage = Usd.Stage.CreateNew(str(output / f"scene.{name}.usda"))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(source_path, output)]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    for key in _STAGE_METADATA:
        stage.SetMetadata(key, source.GetMetadata(key))
    stage.SetFramesPerSecond(profile["control"]["fps"])
    stage.SetTimeCodesPerSecond(profile["control"]["fps"])
    stage.SetEndTimeCode(profile["control"]["steps"])
    stage.GetRootLayer().customLayerData = {
        "profile": profile_path.name,
        "episode": profile["episode"],
        "calibration": "unverified",
        "fidelity": "RGB fit under assumed metric scale; not a measured replica",
    }

    _props(stage, profile, _module("build_scene"))
    experiment = _module("build_experiment")
    mapping = yaml.safe_load((_ROOT / _CALIBRATION).read_text())
    robot_path = _ROOT / "robot/source/so101_new_calib.urdf"
    robot = ET.parse(robot_path).getroot()
    angles, initial = experiment._initial(robot, mapping, profile["initial_state"])
    experiment._robot_pose(stage, robot, angles)
    stage.GetRootLayer().Save()
    scene_path = Path(stage.GetRootLayer().identifier)
    scene_path.write_text(scene_path.read_text().rstrip() + "\n")
    _manifest(output, name, profile)

    report = {
        "status": "experimental_unverified",
        "episode": profile["episode"],
        "source": profile["source"],
        "profile_sha256": _digest(profile_path),
        "source_scene_sha256": _digest(source_path),
        "calibration_sha256": _digest(_ROOT / _CALIBRATION),
        "urdf_sha256": _digest(robot_path),
        "initial_joints": initial,
        "cup": profile["cup"],
        "control": profile["control"],
        "camera_source": "scene.usda; unchanged shelf-fit camera",
        "urdf_limits_changed": False,
        "limitations": [
            "Metric dimensions depend on an assumed shelf width, not measured scale.",
            "Initial joints use the unverified map; clipped joints are reported explicitly.",
            "Mass, friction, contact surfaces, and lighting remain estimates.",
        ],
    }
    if "box" in profile:
        report["box"] = profile["box"]
    (output / f"{name}-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=_ROOT / "episode-001.json")
    parser.add_argument("--output", type=Path, default=_ROOT)
    args = parser.parse_args()
    report = _build(args.profile, args.output)
    print(json.dumps({"status": report["status"], "episode": report["episode"]}))


if __name__ == "__main__":
    _main()
