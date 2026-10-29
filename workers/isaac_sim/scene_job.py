"""Trusted scene previews and recorded-state replay inside Isaac Sim.

The dataset-to-articulation mapping is explicitly uncalibrated. Rendering a
recorded joint-state trajectory does not establish grasp or transfer success.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from sim_worker.adapters.video import Recorder
from sim_worker.contracts import RunSpec
from sim_worker.rollout.contracts import Frame, SimSpec
from sim_worker.rollout.isaac import IsaacSim

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MAPPING = "uncalibrated_dataset_normalized"
MAX_ROWS = 18_000
MAX_SECONDS = 600
MAX_TRAJECTORY_BYTES = 8 * 1024**2
CAMERA = "/World/Cameras/Front"
ARTICULATION = "/World/Robot/joints/root_joint"
SCOPE = (
    "Recorded dataset joint-state replay with uncalibrated normalized-to-joint-limit "
    "mapping. Joint targets are simulated, not measured physical calibration. "
    "Object interaction, grasp success and real-world transfer are unverified."
)


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def trajectory(value):
    """Validate complete, uniformly timed recorded rows without synthesizing motion."""
    if not isinstance(value, dict) or set(value) != {
        "episode_index",
        "fps",
        "joint_names",
        "frames",
        "mapping_status",
    }:
        raise ValueError("Trajectory fields do not match the recorded-state contract")
    episode, fps = value["episode_index"], value["fps"]
    if type(episode) is not int or not 0 <= episode < 100_000:
        raise ValueError("Invalid source episode")
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("Source FPS must be an integer from 1 to 60")
    if value["mapping_status"] != MAPPING:
        raise ValueError("Only the explicit uncalibrated normalized mapping is supported")
    names = value["joint_names"]
    if (
        not isinstance(names, list)
        or len(names) != len(JOINTS)
        or not all(isinstance(name, str) for name in names)
    ):
        raise ValueError("Recorded trajectory requires six named joints")
    canonical = tuple(name.removesuffix(".pos") for name in names)
    if len(set(canonical)) != len(JOINTS) or set(canonical) != set(JOINTS):
        raise ValueError("Recorded joints differ from the registered SO101 articulation")
    rows = value["frames"]
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_ROWS:
        raise ValueError("Replay requires 2 to 18000 complete recorded frames")
    if len(rows) / fps > MAX_SECONDS:
        raise ValueError("Recorded replay exceeds 600 seconds")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"timestamp", "positions"}:
            raise ValueError("Each recorded frame requires timestamp and positions")
        stamp = _number(row["timestamp"], "Recorded timestamp")
        if abs(stamp - index / fps) > 0.002:
            raise ValueError("Recorded timestamps must be contiguous from zero at source FPS")
        positions = row["positions"]
        if not isinstance(positions, list) or len(positions) != len(JOINTS):
            raise ValueError("Each recorded frame requires six joint positions")
        ordered = tuple(
            _number(positions[canonical.index(name)], "Joint position") for name in JOINTS
        )
        if any(abs(position) > 200 for position in ordered):
            raise ValueError("Normalized source positions exceed the bounded replay range")
        normalized.append({"timestamp": stamp, "positions": ordered})
    return {"episode_index": episode, "fps": fps, "frames": normalized, "joint_names": JOINTS}


def targets(positions, limits):
    """Map known normalized coordinates into live simulator limits, recording clamps."""
    if len(positions) != len(JOINTS) or len(limits) != len(JOINTS):
        raise ValueError("Joint positions and live limits must contain six entries")
    values, clipped = [], []
    for name, value, bounds in zip(JOINTS, positions, limits, strict=True):
        value = _number(value, "Source position")
        if abs(value) > 200:
            raise ValueError("Source position exceeds replay bounds")
        if len(bounds) != 2:
            raise ValueError("Live joint limits must contain two bounds")
        low, high = (_number(bound, "Live joint limit") for bound in bounds)
        if not low < high:
            raise ValueError("Live joint limits must have positive extent")
        minimum, maximum = (0.0, 100.0) if name == "gripper" else (-100.0, 100.0)
        bounded = max(minimum, min(maximum, value))
        clipped.append(bounded != value)
        fraction = (bounded - minimum) / (maximum - minimum)
        values.append(low + fraction * (high - low))
    return tuple(values), tuple(clipped)


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            result.update(part)
    return result.hexdigest()


def _write(path, value):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".scene-result-", delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            file.write((json.dumps(value, allow_nan=False, indent=2) + "\n").encode())
            file.flush()
            os.fsync(file.fileno())
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_trajectory(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_TRAJECTORY_BYTES:
        raise ValueError("Trajectory must be a bounded regular JSON file")
    return trajectory(json.loads(path.read_bytes()))


def _visible(frame):
    import numpy as np

    if not isinstance(frame, Frame) or len(frame.rgb) != frame.width * frame.height * 3:
        raise ValueError("Isaac returned an invalid RGB frame")
    pixels = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    if pixels.mean() <= 1 or pixels.std() <= 1:
        raise RuntimeError("Isaac returned a blank or uniform frame")
    return pixels


def _image(path, frame):
    from PIL import Image

    Image.fromarray(_visible(frame)).save(path)
    if path.stat().st_size <= 100:
        raise RuntimeError("Environment image is incomplete")


class SceneSimulation(IsaacSim):
    """Extend the reviewed live driver only with static environment viewpoints."""

    def environment_views(self, config):
        from pxr import Gf, UsdGeom

        stage = self._context.get_stage()
        center = tuple(
            (config["robot"]["position"][index] + config["cup"]["position"][index]) / 2
            + (0.15 if index == 2 else 0)
            for index in range(3)
        )
        yield "front", self.observe().frame
        for name, offset in [("overview", (-0.9, -1.1, 1.1)), ("side", (1.2, -0.4, 0.65))]:
            path = "/World/Cameras/SceneJob" + name.title()
            camera = UsdGeom.Camera.Define(stage, path)
            eye = tuple(value + delta for value, delta in zip(center, offset, strict=True))
            transform = (
                Gf.Matrix4d()
                .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0, 0, 1))
                .GetInverse()
            )
            camera.AddTransformOp().Set(transform)
            camera.CreateFocalLengthAttr(28)
            camera.CreateHorizontalApertureAttr(36)
            camera.CreateVerticalApertureAttr(36 * self._spec.height / self._spec.width)
            camera.CreateClippingRangeAttr((0.01, 100))
            product = self._rep.create.render_product(path, (self._spec.width, self._spec.height))
            annotator = self._rep.annotators.get("rgb")
            try:
                annotator.attach(product)
                self._render(8)
                pixels = annotator.get_data()
                if (
                    pixels.ndim != 3
                    or pixels.shape[:2] != (self._spec.height, self._spec.width)
                    or pixels.shape[2] not in (3, 4)
                    or str(pixels.dtype) != "uint8"
                ):
                    raise RuntimeError("Isaac returned an invalid environment camera frame")
                yield (
                    name,
                    Frame(self._spec.width, self._spec.height, pixels[:, :, :3].tobytes(order="C")),
                )
            finally:
                annotator.detach()
                product.destroy()


@contextmanager
def _runtime(simulation, spec, output, report):
    with simulation(spec) as driver:
        try:
            yield driver
        except BaseException as error:
            # Kit can terminate Python during close; persist failure before leaving it.
            report.update(status="failed", error_type=type(error).__name__)
            report.pop("isaac_runtime", None)
            _write(output / "result.json", report)
            raise


def run(bundle, trajectory_path, output, *, width=960, height=540, simulation=SceneSimulation):
    if (
        type(width) is not int
        or type(height) is not int
        or not 2 <= width <= 1920
        or not 2 <= height <= 1080
        or width % 2
        or height % 2
    ):
        raise ValueError("Capture dimensions must be positive even values within 1920x1080")
    bundle, trajectory_path, output = Path(bundle), Path(trajectory_path), Path(output)
    scene, config_path = bundle / "scene.usda", bundle / "scene_config.json"
    if any(path.is_symlink() or not path.is_file() for path in (scene, config_path)):
        raise ValueError("Generated scene and configuration must be regular files")
    if config_path.stat().st_size > 64 * 1024:
        raise ValueError("Scene configuration exceeds the worker limit")
    recorded = _read_trajectory(trajectory_path)
    config = json.loads(config_path.read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    owned = (
        "result.json",
        "replay.mp4",
        "replay-trace.jsonl",
        "environment-front.png",
        "environment-overview.png",
        "environment-side.png",
    )
    if any((output / name).exists() or (output / name).is_symlink() for name in owned):
        raise ValueError("Replay output already exists; existing evidence was retained")
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": SCOPE,
        "scene_sha256": _sha(scene),
        "trajectory_sha256": _sha(trajectory_path),
        "episode_index": recorded["episode_index"],
        "source_frames": len(recorded["frames"]),
        "fps": recorded["fps"],
        "mapping_status": MAPPING,
        "joint_names": list(JOINTS),
        "mapping": {
            "arm_normalized_range": [-100, 100],
            "gripper_normalized_range": [0, 100],
            "destination": "live_articulation_joint_limits_radians",
            "clipping_counts": {name: 0 for name in JOINTS},
        },
        "calibration_verified": False,
        "task_success": None,
        "media": [],
    }
    _write(output / "result.json", report)
    spec = SimSpec(
        str(scene.resolve()),
        CAMERA,
        ARTICULATION,
        JOINTS,
        width,
        height,
        recorded["fps"],
        recorded["fps"] * 4,
    )
    capture = RunSpec(spec.scene, spec.camera, width, height, spec.fps, len(recorded["frames"]), "")
    try:
        with _runtime(simulation, spec, output, report) as driver:
            driver.reset("recorded-episode-" + str(recorded["episode_index"]))
            for name, frame in driver.environment_views(config):
                target = output / ("environment-" + name + ".png")
                _image(target, frame)
                report["media"].append(
                    {
                        "role": "environment_image",
                        "path": target.name,
                        "label": name.title(),
                        "sha256": _sha(target),
                        "bytes": target.stat().st_size,
                    }
                )
            first, _ = targets(recorded["frames"][0]["positions"], driver.joint_limits)
            driver.apply(first)
            # Settle the first recorded target before the video, preserving every source row.
            for _ in range(spec.fps):
                driver.step()
            trace_path = output / "replay-trace.jsonl"
            origin = None
            with (
                Recorder(output / "replay.mp4", capture) as recorder,
                trace_path.open("x") as trace,
            ):
                for index, row in enumerate(recorded["frames"]):
                    target, clipped = targets(row["positions"], driver.joint_limits)
                    driver.apply(target)
                    driver.step()
                    observed = driver.observe()
                    if origin is None:
                        origin = observed.sim_time
                    if abs(observed.sim_time - origin - index / spec.fps) > 1e-5:
                        raise RuntimeError("Isaac replay clock did not follow the recorded cadence")
                    _visible(observed.frame)
                    recorder.append(observed.frame.rgb)
                    for name, was_clipped in zip(JOINTS, clipped, strict=True):
                        report["mapping"]["clipping_counts"][name] += int(was_clipped)
                    trace.write(
                        json.dumps(
                            {
                                "source_frame": index,
                                "source_timestamp": row["timestamp"],
                                "source_positions": row["positions"],
                                "targets_radians": target,
                                "observed_radians": observed.state,
                                "simulation_time": observed.sim_time,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    if (index + 1) % spec.fps == 0 or index + 1 == capture.frames:
                        print(
                            json.dumps(
                                {"stage": "replaying", "frames": index + 1, "total": capture.frames}
                            ),
                            flush=True,
                        )
            video = output / "replay.mp4"
            report["media"].append(
                {
                    "role": "episode_replay",
                    "path": video.name,
                    "label": "Episode " + str(recorded["episode_index"]),
                    "episode_index": recorded["episode_index"],
                    "sha256": _sha(video),
                    "bytes": video.stat().st_size,
                }
            )
            report.update(
                status="succeeded",
                rendered_frames=capture.frames,
                isaac_runtime=True,
                calibrated_replay=False,
                trace={"path": trace_path.name, "sha256": _sha(trace_path)},
            )
            if any(report["mapping"]["clipping_counts"].values()):
                report["warning"] = (
                    "Some recorded normalized positions exceeded nominal limits and were "
                    "clamped before mapping. See per-joint clipping counts."
                )
            _write(output / "result.json", report)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        report.pop("isaac_runtime", None)
        _write(output / "result.json", report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    run(args.bundle, args.trajectory, args.output, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
