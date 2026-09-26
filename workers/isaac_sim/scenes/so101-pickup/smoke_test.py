"""Record the scene and verify live Isaac physics with the production adapters."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import faulthandler
import hashlib
import json
import logging
from pathlib import Path
import signal
import sys
import time

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT.parents[1]))

from sim_worker.adapters import isaac
from sim_worker.adapters.manifest import load
from sim_worker.adapters.video import Recorder

_OUTPUT = Path("/probe-output")
_FRAMES = 150
_WIDTH = 1280
_HEIGHT = 720
_RGB_CHANNELS = 3
_JOINT_COUNT = 6
_BASE_PATH = "/World/Robot/base_link"
_ROBOT_PATTERN = "/World/Robot/joints/root_joint"
_CUP_PATH = "/World/Props/Cup"
_SHELF_PATH = "/World/Environment/Shelf/Board0"
_STACK_INTERVAL_SECONDS = 60
_ASSET_WAIT_SECONDS = 60.0
_ASSET_WAIT_SETTING = "/exts/omni.replicator.core/maxAssetLoadingTime"
_BASE_DRIFT_METERS = 0.0001
_CUP_DRIFT_METERS = 0.05
_CUP_HEIGHT_METERS = 0.01
_JOINT_LIMIT_DEGREES = 1.0
_MIN_PIXEL_MEAN = 1.0
_MIN_PIXEL_STD = 1.0


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@contextmanager
def _stacks(output):
    # Preserve Python stacks if shader warmup or physics cooking stalls.
    with (output / "stacks.log").open("w") as stream:
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=False)
        faulthandler.dump_traceback_later(_STACK_INTERVAL_SECONDS, repeat=True, file=stream)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.unregister(signal.SIGUSR1)


def _reference(scene):
    from pxr import Usd, UsdGeom, UsdPhysics

    # Read the saved stage separately; live transforms may change after Play.
    stage = Usd.Stage.Open(scene)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    shelf = bounds.ComputeWorldBound(stage.GetPrimAtPath(_SHELF_PATH)).ComputeAlignedRange()
    return {
        "base_position": list(transforms.GetLocalToWorldTransform(
            stage.GetPrimAtPath(_BASE_PATH)).ExtractTranslation()),
        "cup_position": list(transforms.GetLocalToWorldTransform(
            stage.GetPrimAtPath(_CUP_PATH)).ExtractTranslation()),
        "shelf_top": float(shelf.GetMax()[2]),
        "q_degrees": {
            prim.GetName(): prim.GetAttribute("state:angular:physics:position").Get()
            for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)
        },
    }


def _array(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=np.float64)


class _Physics:
    """Read live tensors, including when Fabric bypasses USD transform updates."""

    def __init__(self):
        from isaacsim.core.simulation_manager import SimulationManager

        # Isaac owns the active stage's physics context; a new default view lacks it.
        self._engine = SimulationManager.get_active_physics_engine()
        self._view = SimulationManager.get_physics_simulation_view()
        if self._view is None:
            raise RuntimeError(f"Isaac has no initialized {self._engine} simulation view")
        self._robot = self._view.create_articulation_view(_ROBOT_PATTERN)
        self._cup = self._view.create_rigid_body_view(_CUP_PATH)
        if self._robot.count != 1 or self._robot.max_dofs != _JOINT_COUNT:
            raise RuntimeError("Expected one live articulation with six DOFs")
        if self._cup.count != 1:
            raise RuntimeError("Cup was not created as a live rigid body")

    def _metadata(self):
        return {
            "dof_names": list(self._robot.shared_metatype.dof_names),
            "q_limits_degrees": np.degrees(_array(self._robot.get_dof_limits())[0]).tolist(),
            "q_targets_degrees": np.degrees(_array(self._robot.get_dof_position_targets())[0]).tolist(),
            "articulation_count": self._robot.count,
            "link_count": self._robot.max_links,
            "engine": self._engine,
            "state_source": "Isaac SimulationManager live tensor view, world coordinates",
        }

    def _snapshot(self):
        arrays = {
            "q_degrees": np.degrees(_array(self._robot.get_dof_positions())[0]),
            "qd_degrees_per_second": np.degrees(_array(self._robot.get_dof_velocities())[0]),
            "base_pose": _array(self._robot.get_root_transforms())[0],
            "cup_pose": _array(self._cup.get_transforms())[0],
        }
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Live physics state contains nonfinite values")
        return {key: value.tolist() for key, value in arrays.items()}


def _record(index, frame, spec, started, simulation):
    import omni.timeline

    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(spec.height, spec.width, _RGB_CHANNELS)
    record = {
        "frame": index,
        "time": float(omni.timeline.get_timeline_interface().get_current_time()),
        "simulation_time": float(simulation.get_simulation_time()),
        "elapsed_seconds": time.monotonic() - started,
        "pixel_min": int(pixels.min()),
        "pixel_max": int(pixels.max()),
        "pixel_mean": float(pixels.mean()),
        "pixel_std": float(pixels.std()),
    }
    return record, pixels


def _assess(records, reference, metadata, spec):
    base = np.asarray([record["base_pose"][:3] for record in records])
    cup = np.asarray([record["cup_pose"][:3] for record in records])
    q = np.asarray([record["q_degrees"] for record in records])
    limits = np.asarray(metadata["q_limits_degrees"])
    max_base_drift = float(np.linalg.norm(base - reference["base_position"], axis=1).max())
    cup_drift = float(np.linalg.norm(cup[-1, :2] - np.asarray(reference["cup_position"])[:2]))
    cup_height = float(cup[-1, 2] - reference["shelf_top"])
    checks = {
        "frame_count": len(records) == spec.frames,
        "finite_joint_state": bool(np.isfinite(q).all() and q.shape == (spec.frames, _JOINT_COUNT)),
        "joint_limits": bool(np.all(q >= limits[:, 0] - _JOINT_LIMIT_DEGREES)
                             and np.all(q <= limits[:, 1] + _JOINT_LIMIT_DEGREES)),
        "fixed_base": max_base_drift <= _BASE_DRIFT_METERS,
        "cup_on_shelf": cup_drift <= _CUP_DRIFT_METERS and abs(cup_height) <= _CUP_HEIGHT_METERS,
        "visible_frames": all(record["pixel_mean"] > _MIN_PIXEL_MEAN
                              and record["pixel_std"] > _MIN_PIXEL_STD for record in records),
        "timeline_advanced": records[-1]["time"] > records[0]["time"],
        "physics_advanced": records[-1]["simulation_time"] > records[0]["simulation_time"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "max_base_drift_meters": max_base_drift,
        "final_cup_planar_drift_meters": cup_drift,
        "final_cup_height_above_shelf_meters": cup_height,
        "max_target_error_degrees": np.max(np.abs(q - metadata["q_targets_degrees"]), axis=0).tolist(),
        "initial": records[0],
        "final": records[-1],
        "thresholds": {
            "base_drift_meters": _BASE_DRIFT_METERS,
            "cup_planar_drift_meters": _CUP_DRIFT_METERS,
            "cup_height_meters": _CUP_HEIGHT_METERS,
            "joint_limit_slack_degrees": _JOINT_LIMIT_DEGREES,
        },
    }


def _capture(spec, output, capture, report):
    import carb.settings
    from isaacsim.core.simulation_manager import SimulationManager

    # Replicator reads this limit in seconds during its initial loading loop.
    settings = carb.settings.get_settings()
    report["asset_loading"] = {"previous_seconds": settings.get(_ASSET_WAIT_SETTING),
                               "limit_seconds": _ASSET_WAIT_SECONDS}
    settings.set(_ASSET_WAIT_SETTING, _ASSET_WAIT_SECONDS)
    _write(output / "result.json", report)

    records = []
    physics = None
    physics_error = None
    started = time.monotonic()
    selected = {0, spec.frames // 2, spec.frames - 1}
    video_path = output / "video.mp4"
    with Recorder(video_path, spec) as recorder:
        for index, frame in enumerate(capture):
            record, pixels = _record(index, frame, spec, started, SimulationManager)
            if physics_error is None:
                try:
                    if physics is None:
                        physics = _Physics()
                        report["physics"] = physics._metadata()
                    record.update(physics._snapshot())
                except Exception as error:
                    # Preserve the recording even when physics validation fails.
                    physics_error = str(error)
                    report["physics_error"] = physics_error
                    logging.exception("Live physics probe failed")
            records.append(record)
            recorder.append(frame)
            _write(output / "stats.json", records)
            if index in selected:
                Image.fromarray(pixels).save(output / f"frame-{index:03}.png")
            if index in selected or (index + 1) % spec.fps == 0:
                print("PROBE " + json.dumps(record), flush=True)

    if not records:
        raise RuntimeError("Isaac returned no frames")
    if physics_error is None:
        report.update(_assess(records, report["authored"], report["physics"], spec))
    else:
        report.update(status="failed", checks={"live_physics_available": False},
                      initial=records[0], final=records[-1])
    report["video"] = {"path": str(video_path), "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest()}
    _write(output / "result.json", report)
    print("RESULT " + json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Scene acceptance failed; inspect result.json")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT)
    parser.add_argument("--frames", type=int, default=_FRAMES)
    parser.add_argument("--width", type=int, default=_WIDTH)
    parser.add_argument("--height", type=int, default=_HEIGHT)
    args = parser.parse_args()
    if args.frames < 2 or args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("Use at least two frames and positive even image dimensions")

    spec = replace(load(_ROOT / "capture.yaml"), frames=args.frames,
                   width=args.width, height=args.height)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scene": spec.scene, "camera": spec.camera,
              "width": spec.width, "height": spec.height, "fps": spec.fps, "frames": spec.frames,
              "scope": "Passive pose-hold smoke test; no pickup controller or task-success claim."}
    _write(output / "result.json", report)
    _write(output / "stats.json", [])
    logging.basicConfig(level=logging.INFO)
    with _stacks(output), isaac.frames(spec) as capture:
        try:
            report["authored"] = _reference(spec.scene)
            _capture(spec, output, capture, report)
        except Exception as error:
            # Kit shutdown can exit Python; publish evidence before leaving its context.
            report.update(status="failed", error=str(error))
            _write(output / "result.json", report)
            raise


if __name__ == "__main__":
    _main()
