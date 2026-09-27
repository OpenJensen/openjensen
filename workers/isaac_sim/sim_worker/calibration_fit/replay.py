"""Recorded-state projections using frozen scene geometry, without simulation."""

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim_worker.rollout.calibration import CalibrationUse, JointMap

_DEFAULT_CAMERA = "/World/Cameras/Front"
_DEFAULT_ROBOT = "/World/Robot"
_MIN_DEPTH = 1e-6
_MATRIX_SIZE = 4
_LIMIT_TOLERANCE = 1e-6
_TRANSFORM_TOLERANCE = 1e-6
_DEFAULT_FPS = 30
_PANEL_WIDTH = 960
_HEADER_HEIGHT = 64
_VIDEO_CODEC = "libx264"
_VIDEO_FORMAT = "yuv420p"


class LimitMode(Enum):
    REQUESTED = "requested"
    CLIPPED = "clipped"


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {size} finite values")
    return result.copy()


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _origin(element):
    result = np.eye(_MATRIX_SIZE)
    origin = element.find("origin")
    if origin is None:
        return result

    xyz = _vector(origin.get("xyz", "0 0 0").split(), 3, "origin.xyz")
    rpy = _vector(origin.get("rpy", "0 0 0").split(), 3, "origin.rpy")
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    kind: str


class _Kinematics:
    def __init__(self, path, names, landmarks):
        content = Path(path).read_bytes()
        self._digest = hashlib.sha256(content).hexdigest()
        robot = ET.fromstring(content)
        links = [element.get("name") for element in robot.findall("link")]
        if not links or None in links or len(links) != len(set(links)):
            raise ValueError("URDF links must have unique names")
        pending, children, seen, limits = [], set(), set(), {}
        for element in robot.findall("joint"):
            name, kind = element.get("name"), element.get("type")
            parent_node, child_node = element.find("parent"), element.find("child")
            if parent_node is None or child_node is None:
                raise ValueError("URDF joint requires parent and child")
            parent, child = parent_node.get("link"), child_node.get("link")
            if parent not in links or child not in links or child in children or name in seen:
                raise ValueError("URDF contains invalid or duplicate joint topology")
            if kind not in {"fixed", "revolute"}:
                raise ValueError(f"Replay requires bounded revolute joints: {kind}")
            axis_node = None if kind == "fixed" else element.find("axis")
            axis = "1 0 0" if axis_node is None else axis_node.get("xyz", "1 0 0")
            axis = _vector(axis.split(), 3, "joint.axis")
            length = np.linalg.norm(axis)
            if length == 0:
                raise ValueError("Joint axis cannot be zero")
            if kind == "revolute":
                limit = element.find("limit")
                if name not in names or limit is None:
                    raise ValueError(f"Moving joint lacks expected name or limits: {name}")
                bounds = _vector([limit.get("lower"), limit.get("upper")], 2, "joint limits")
                if bounds[0] >= bounds[1]:
                    raise ValueError("Joint limits must increase")
                limits[name] = bounds
            pending.append(_Joint(name, parent, child, _origin(element), axis / length, kind))
            seen.add(name)
            children.add(child)
        if set(limits) != set(names):
            raise ValueError("URDF moving joints differ from requested joint order")
        roots = set(links) - children
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root")
        self._root = roots.pop()
        self._joints = []
        reached = {self._root}
        while pending:
            ready = [joint for joint in pending if joint.parent in reached]
            if not ready:
                raise ValueError("URDF contains a disconnected or cyclic tree")
            for joint in ready:
                self._joints.append(joint)
                reached.add(joint.child)
                pending.remove(joint)
        self._names = names
        self._limits = np.array([limits[name] for name in names])
        self._landmarks = {}
        if not landmarks:
            raise ValueError("Replay needs at least one landmark")
        for name, landmark in landmarks.items():
            if landmark["link"] not in links:
                raise ValueError(f"Unknown landmark link: {landmark['link']}")
            self._landmarks[name] = (landmark["link"], _vector(landmark["xyz"], 3, f"{name}.xyz"))

    def clamp(self, angles):
        return np.clip(angles, self._limits[:, 0], self._limits[:, 1])

    def points(self, angles):
        values = dict(zip(self._names, angles, strict=True))
        poses = {self._root: np.eye(_MATRIX_SIZE)}
        for joint in self._joints:
            rotation = np.eye(_MATRIX_SIZE)
            if joint.kind == "revolute":
                rotation[:3, :3] = Rotation.from_rotvec(joint.axis * values[joint.name]).as_matrix()
            poses[joint.child] = poses[joint.parent] @ joint.origin @ rotation
        return {
            name: poses[link][:3, :3] @ point + poses[link][:3, 3]
            for name, (link, point) in self._landmarks.items()
        }

    def provenance(self):
        return {
            "urdf_sha256": self._digest,
            "joint_names": list(self._names),
            "limits_rad": self._limits.tolist(),
            "landmarks": {
                name: {"link": link, "xyz": point.tolist()}
                for name, (link, point) in self._landmarks.items()
            },
        }


def _rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (_MATRIX_SIZE, _MATRIX_SIZE) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} requires a finite transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_TRANSFORM_TOLERANCE):
        raise ValueError(f"{name} must not contain scaling or shear")
    if not math.isclose(np.linalg.det(rotation), 1, abs_tol=_TRANSFORM_TOLERANCE):
        raise ValueError(f"{name} must preserve handedness")
    return matrix.copy()


class _SceneProjection:
    def __init__(self, path, camera_path, robot_path, image_size):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError("Cannot open replay scene")
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise ValueError("Replay requires scene units in meters")
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        robot = stage.GetPrimAtPath(robot_path)
        if not camera or not robot:
            raise ValueError("Scene camera or robot base is missing")
        if camera.GetProjectionAttr().Get() != "perspective":
            raise ValueError("Replay requires a perspective camera")
        size = _vector(image_size, 2, "image_size")
        if np.any(size <= 0):
            raise ValueError("Image dimensions must be positive")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        camera_world = _rigid_matrix(
            np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T, "camera"
        )
        robot_world = _rigid_matrix(
            np.asarray(cache.GetLocalToWorldTransform(robot)).T, "robot base"
        )
        aperture = _vector(
            [camera.GetHorizontalApertureAttr().Get(), camera.GetVerticalApertureAttr().Get()],
            2,
            "camera aperture",
        )
        focal = _number(camera.GetFocalLengthAttr().Get(), "focal length")
        if focal <= 0 or np.any(aperture <= 0):
            raise ValueError("Camera focal length and aperture must be positive")
        if not math.isclose(size[0] / size[1], aperture[0] / aperture[1], rel_tol=1e-5):
            raise ValueError("Image aspect must match USD camera aperture")
        offsets = _vector(
            [
                camera.GetHorizontalApertureOffsetAttr().Get(),
                camera.GetVerticalApertureOffsetAttr().Get(),
            ],
            2,
            "camera aperture offset",
        )
        self._robot_camera = np.linalg.inv(camera_world) @ robot_world
        self._focal = focal * size / aperture
        self._center = size / 2 + offsets * size / aperture * [-1, 1]
        self._metadata = {
            "scene": str(Path(path).resolve()),
            "camera": camera_path,
            "robot": robot_path,
            "image_size": size.tolist(),
            "focal_pixels": self._focal.tolist(),
            "principal_point": self._center.tolist(),
            "camera_to_world": camera_world.tolist(),
            "robot_to_world": robot_world.tolist(),
            "scene_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "layer_sha256": {
                str(Path(layer.realPath).resolve()): hashlib.sha256(
                    Path(layer.realPath).read_bytes()
                ).hexdigest()
                for layer in stage.GetUsedLayers()
                if layer.realPath
            },
            "projection": "USD perspective: +X right, +Y up, -Z forward; image +Y down",
        }

    def project(self, points):
        result = {}
        for name, point in points.items():
            local = self._robot_camera @ np.r_[point, 1]
            depth = -local[2]
            if depth <= _MIN_DEPTH:
                raise ValueError(f"Landmark {name} must lie in front of the camera")
            result[name] = (self._center + self._focal * local[:2] / depth * [1, -1]).tolist()
        return result

    def provenance(self):
        # Return independent containers so callers cannot change the frozen camera.
        import copy

        return copy.deepcopy(self._metadata)


class KinematicReplay:
    """Project recorded states; no renderer, physics, drives, or policy inference."""

    def __init__(
        self,
        *,
        scene: Path,
        urdf: Path,
        calibration: Path,
        landmarks: dict,
        image_size: tuple,
        joints: tuple,
        camera=_DEFAULT_CAMERA,
        robot=_DEFAULT_ROBOT,
    ):
        if not joints or len(joints) != len(set(joints)):
            raise ValueError("Replay needs a unique joint order")
        self._names = tuple(joints)
        self._mapping = JointMap(calibration, self._names, CalibrationUse.EXPERIMENTAL)
        self._kinematics = _Kinematics(urdf, self._names, landmarks)
        self._camera = _SceneProjection(scene, camera, robot, image_size)

    def project_angles(self, angles, mode: LimitMode):
        """Angles are radians; the selected limit behavior is always explicit."""
        angles = _vector(angles, len(self._names), "angles")
        if not isinstance(mode, LimitMode):
            raise ValueError("Projection needs an explicit LimitMode")
        if mode is LimitMode.CLIPPED:
            angles = self._kinematics.clamp(angles)
        return self._camera.project(self._kinematics.points(angles))

    def project(self, state):
        requested = np.asarray(self._mapping.to_sim(tuple(state)))
        clipped = self._kinematics.clamp(requested)
        return {
            "requested_rad": requested.tolist(),
            "clipped_rad": clipped.tolist(),
            "limit_clipped": (np.abs(requested - clipped) > _LIMIT_TOLERANCE).tolist(),
            "policy_extrapolated": list(self._mapping.action_outside(tuple(state))),
            "requested_pixels": self.project_angles(requested, LimitMode.REQUESTED),
            "clipped_pixels": self.project_angles(clipped, LimitMode.CLIPPED),
        }

    def audit_limits(self, states):
        """Audit every recorded vector, independently of camera visibility."""
        requested = np.array([self._mapping.to_sim(tuple(state)) for state in states])
        if requested.ndim != 2 or len(requested) == 0:
            raise ValueError("Limit audit requires recorded states")
        clipped = self._kinematics.clamp(requested)
        delta = np.abs(requested - clipped)
        conflicts = delta > _LIMIT_TOLERANCE
        return {
            "rows": len(requested),
            "rows_with_limit_conflict": int(np.any(conflicts, axis=1).sum()),
            "joints": {
                name: {
                    "limit_conflicts": int(conflicts[:, index].sum()),
                    "requested_min_rad": float(requested[:, index].min()),
                    "requested_max_rad": float(requested[:, index].max()),
                    "maximum_clip_rad": float(delta[:, index].max()),
                }
                for index, name in enumerate(self._names)
            },
        }

    def provenance(self):
        return {
            **self._kinematics.provenance(),
            **self._camera.provenance(),
            "calibration_sha256": self._mapping.digest,
            "calibration_status": self._mapping.status,
            "mode": "offline kinematic projection; no physics or learned policy",
            "clipping": "URDF position bounds only; no speed/drive guard or solver dynamics",
        }


class RecordedStates:
    """Interpolate local frame coordinates without crossing an episode boundary."""

    def __init__(self, rows, joints):
        if not joints or len(joints) != len(set(joints)):
            raise ValueError("Recorded states need a unique joint order")
        self._episodes = {}
        global_indices = set()
        for row in rows:
            episode = _integer(row["episode_index"], "episode_index")
            frame = _integer(row["frame_index"], "frame_index")
            index = _integer(row["index"], "index")
            states = self._episodes.setdefault(episode, {})
            if frame in states or index in global_indices:
                raise ValueError("Recorded frames and global indices must be unique")
            states[frame] = (index, _vector(row["observation.state"], len(joints), "state"))
            global_indices.add(index)
        if not self._episodes:
            raise ValueError("Recorded states cannot be empty")
        for states in self._episodes.values():
            if sorted(states) != list(range(len(states))):
                raise ValueError("Episode frames must be contiguous and start at zero")
            indices = [states[frame][0] for frame in range(len(states))]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError("Episode global indices must be contiguous")

    @property
    def episodes(self):
        return tuple(sorted(self._episodes))

    def frames(self, episode):
        return len(self._episode(episode))

    def local_frame(self, episode, global_index):
        states = self._episode(episode)
        index = _integer(global_index, "global_index")
        frame = index - states[0][0]
        if frame not in states or states[frame][0] != index:
            raise ValueError("Global index is outside the selected episode")
        return frame

    def sample(self, episode, frame, lag_frames=0.0):
        states = self._episode(episode)
        frame = _number(frame, "frame") + _number(lag_frames, "lag_frames")
        if frame < 0 or frame > len(states) - 1:
            raise ValueError("Aligned frame is outside the selected episode")
        lower, upper = math.floor(frame), math.ceil(frame)
        weight = frame - lower
        result = states[lower][1] * (1 - weight) + states[upper][1] * weight
        return tuple(result.tolist())

    def _episode(self, episode):
        episode = _integer(episode, "episode")
        if episode not in self._episodes:
            raise ValueError(f"Unknown episode: {episode}")
        return self._episodes[episode]


def _metrics(errors):
    if not errors:
        return None
    values = np.asarray(errors)
    return {
        "points": len(values),
        "rms_px": float(np.sqrt(np.mean(values**2))),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(values.max()),
    }


def _evaluate(model, records, observations, episodes, lag):
    rows, errors = [], []
    for row in observations["samples"]:
        if row["episode"] not in episodes:
            continue
        local = records.local_frame(row["episode"], row["frame"])
        try:
            state = records.sample(row["episode"], local, lag)
            predicted = model.project(state)
        except ValueError as error:
            errors.append({"episode": row["episode"], "frame": row["frame"], "error": str(error)})
            continue
        distances = {}
        for mode in ("requested", "clipped"):
            distances[mode] = {
                name: float(
                    np.linalg.norm(
                        np.asarray(predicted[f"{mode}_pixels"][name]) - _vector(point, 2, "label")
                    )
                )
                for name, point in row["points"].items()
            }
        rows.append(
            {
                "episode": row["episode"],
                "frame": row["frame"],
                "local_frame": local,
                "state_frame": local + lag,
                "split": row["split"],
                "state": list(state),
                "observed_pixels": row["points"],
                **predicted,
                "errors_px": distances,
            }
        )
    if not rows:
        raise ValueError("No labeled frames could be evaluated")
    groups = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        groups[split] = {
            "frames": len(selected),
            "episodes": sorted({row["episode"] for row in selected}),
            **{
                mode: _metrics(
                    [error for row in selected for error in row["errors_px"][mode].values()]
                )
                for mode in ("requested", "clipped")
            },
            "per_landmark": {
                name: {
                    mode: _metrics(
                        [
                            row["errors_px"][mode][name]
                            for row in selected
                            if name in row["errors_px"][mode]
                        ]
                    )
                    for mode in ("requested", "clipped")
                }
                for name in observations["landmarks"]
            },
        }
    return {"groups": groups, "rows": rows, "errors": errors}


def _episode_window(rows, episode, fps):
    selected = [row for row in rows if row["episode_index"] == episode]
    if len(selected) != 1:
        raise ValueError("Episode metadata must identify exactly one episode")
    row = selected[0]
    prefix = "videos/observation.images.front/"
    start = _number(row[prefix + "from_timestamp"], "video start") * fps
    end = _number(row[prefix + "to_timestamp"], "video end") * fps
    if not math.isclose(start, round(start), abs_tol=1e-5):
        raise ValueError("Episode starts between video frames")
    if not math.isclose(end - start, row["length"], abs_tol=1e-5):
        raise ValueError("Episode duration disagrees with its frame count")
    return int(round(start)), int(row["length"]), row


def _overlay_panel(frame, projection, mode, scale, title):
    from PIL import Image, ImageDraw, ImageFont

    height, width = frame.shape[:2]
    panel = Image.fromarray(frame).resize((round(width * scale), round(height * scale)))
    color = (255, 200, 0) if mode == "requested" else (0, 200, 255)
    draw = ImageDraw.Draw(panel)
    if projection is not None:
        for name, point in projection[f"{mode}_pixels"].items():
            x, y = (round(value * scale) for value in point)
            draw.line((x - 5, y, x + 5, y), fill=color, width=2)
            draw.line((x, y - 5, x, y + 5), fill=color, width=2)
            draw.text((x + 5, y - 12), name, fill=color)
    band = Image.new("RGB", (panel.width, _HEADER_HEIGHT))
    draw = ImageDraw.Draw(band)
    draw.text((10, 7), title, fill=color, font=ImageFont.load_default(size=18))
    note = "Recorded state / kinematics only / no physics"
    if projection is None:
        note = "No aligned state inside this episode; projection omitted"
    draw.text((10, 36), note, fill="white", font=ImageFont.load_default(size=14))
    return np.vstack((np.asarray(band), np.asarray(panel)))


def _overlay_video(model, records, video, window, episode, lag, fps, output, size):
    import av

    start, count, _ = window
    scale = min(1.0, _PANEL_WIDTH / size[0])
    width, height = 2 * round(size[0] * scale), round(size[1] * scale) + _HEADER_HEIGHT
    source = av.open(str(video))
    destination = av.open(str(output / f"episode-{episode:03d}-overlay.mp4"), mode="w")
    stream = destination.add_stream(_VIDEO_CODEC, rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, _VIDEO_FORMAT
    stream.options = {"crf": "19", "preset": "fast"}
    trace = []
    try:
        for index, frame in enumerate(source.decode(video=0)):
            if index < start:
                continue
            local = index - start
            if local >= count:
                break
            image = frame.to_ndarray(format="rgb24")
            if image.shape[:2] != (size[1], size[0]):
                raise ValueError("Source video size differs from annotated image size")
            projection, error = None, None
            try:
                state = records.sample(episode, local, lag)
                projection = model.project(state)
            except ValueError as exception:
                error = str(exception)
            trace.append(
                {
                    "frame": local,
                    "source_frame": index,
                    "time_s": local / fps,
                    "state_frame": local + lag,
                    "error": error,
                    "projection": projection,
                }
            )
            requested = _overlay_panel(
                image,
                projection,
                "requested",
                scale,
                f"Episode {episode}, frame {local}: requested",
            )
            clipped = _overlay_panel(image, projection, "clipped", scale, "URDF-clipped")
            composed = av.VideoFrame.from_ndarray(np.hstack((requested, clipped)), format="rgb24")
            for packet in stream.encode(composed):
                destination.mux(packet)
        for packet in stream.encode():
            destination.mux(packet)
    finally:
        source.close()
        destination.close()
    if len(trace) != count:
        raise ValueError(f"Source video ended before episode completed: {len(trace)}/{count}")
    return trace


def _main():
    import argparse
    import json

    import pyarrow.parquet as parquet

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "urdf", "calibration", "observations", "data", "episodes", "video"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--lag-frames", required=True, type=float)
    parser.add_argument("--evaluation-episodes", nargs="+", type=int)
    parser.add_argument("--joints", nargs="+", required=True)
    parser.add_argument("--fps", type=int, default=_DEFAULT_FPS)
    parser.add_argument("--camera", default=_DEFAULT_CAMERA)
    parser.add_argument("--robot", default=_DEFAULT_ROBOT)
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("Frame rate must be positive")
    observations = json.loads(args.observations.read_text())
    data = parquet.read_table(args.data).to_pylist()
    records = RecordedStates(data, tuple(args.joints))
    model = KinematicReplay(
        scene=args.scene,
        urdf=args.urdf,
        calibration=args.calibration,
        landmarks=observations["landmarks"],
        image_size=tuple(observations["image_size"]),
        joints=tuple(args.joints),
        camera=args.camera,
        robot=args.robot,
    )
    window = _episode_window(parquet.read_table(args.episodes).to_pylist(), args.episode, args.fps)
    if window[1] != records.frames(args.episode):
        raise ValueError("Episode metadata and recorded states differ in length")
    selected = args.evaluation_episodes or [args.episode]
    evaluation = _evaluate(model, records, observations, selected, args.lag_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = _overlay_video(
        model,
        records,
        args.video,
        window,
        args.episode,
        args.lag_frames,
        args.fps,
        args.output_dir,
        tuple(observations["image_size"]),
    )
    report = {
        "status": "diagnostic_only",
        "physics_started": False,
        "provenance": model.provenance(),
        "inputs_sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("data", "episodes", "observations", "video")
        },
        "episode": args.episode,
        "video_first_frame": window[0],
        "video_frames": len(trace),
        "video_file_index": window[2]["videos/observation.images.front/file_index"],
        "data_indices_half_open": [
            window[2]["dataset_from_index"],
            window[2]["dataset_to_index"],
        ],
        "lag_frames": args.lag_frames,
        "lag_seconds": args.lag_frames / args.fps,
        "alignment": "Video frame i uses episode-local state i + lag; no boundary crossing",
        "unprojected_frames": [row["frame"] for row in trace if row["error"]],
        "evaluation": evaluation,
        "full_dataset_limits": {
            feature: model.audit_limits([row[feature] for row in data])
            for feature in ("observation.state", "action")
        },
        "episode_limits": {
            feature: model.audit_limits(
                [row[feature] for row in data if row["episode_index"] == args.episode]
            )
            for feature in ("observation.state", "action")
        },
        "limitations": [
            "No physics, policy inference, contact, or task-success assessment.",
            "Requested projections can violate URDF limits; clipped projections omit dynamics.",
            "Nominal CAD landmark geometry, camera scale, signs and time offset remain uncertain.",
            "This evaluates the frozen current map; labels do not refit or verify it.",
        ],
    }
    for name, content in (("report.json", report), ("frames.json", trace)):
        (args.output_dir / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "groups": {
                    name: {key: group[key] for key in ("frames", "requested", "clipped")}
                    for name, group in evaluation["groups"].items()
                },
                "unprojected_frames": report["unprojected_frames"],
            },
            indent=2,
        )
    )
    return 1 if evaluation["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
