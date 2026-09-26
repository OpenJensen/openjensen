"""Single-threaded Isaac driver; only step() advances simulation time."""

import logging
import math
import os
import threading

from sim_worker.rollout.contracts import RGB_CHANNELS, Frame, Observation, SimSpec

_RENDERER = "RealTimePathTracing"
_AA_DLSS = 3
_DLSS_QUALITY = 2
_DLSS_MODE = "/rtx/post/dlss/execMode"
_DENOISING = "/rtx/dldenoiser/responsiveDenoising"
_WARMUP_SUBFRAMES = 8
_CAPTURE_SUBFRAMES = 1
_RGBA_CHANNELS = 4
_CLOCK_TOLERANCE = 1e-7
_SUCCESS_EXIT = 0
_FAILURE_EXIT = 1


def _main_thread() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Isaac must run on the main thread")


def _array(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _target_buffers(values, reference):
    import numpy as np

    # Match Isaac's active tensor frontend without creating a second physics view.
    if isinstance(reference, np.ndarray):
        return np.asarray(values, dtype=np.float32), np.asarray([0], dtype=np.int32)
    if hasattr(reference, "detach"):
        import torch

        return (
            torch.tensor(values, dtype=torch.float32, device=reference.device),
            torch.tensor([0], dtype=torch.int32, device=reference.device),
        )
    if type(reference).__module__.split(".")[0] == "warp":
        import warp as wp

        return (
            wp.array(values, dtype=wp.float32, device=reference.device),
            wp.array([0], dtype=wp.int32, device=reference.device),
        )
    raise RuntimeError(f"Unsupported Isaac tensor type: {type(reference).__name__}")


class IsaacSim:
    def __init__(self, spec: SimSpec):
        if any(
            type(value) is not int or value <= 0
            for value in (spec.width, spec.height, spec.fps, spec.physics_hz)
        ):
            raise ValueError("Image dimensions and simulation rates must be positive integers")
        if spec.physics_hz % spec.fps:
            raise ValueError("physics_hz must be an exact multiple of fps")
        if not spec.joints or len(set(spec.joints)) != len(spec.joints):
            raise ValueError("Configured joints must be nonempty and unique")

        self._spec = spec
        self._app = None
        self._timeline = None
        self._manager = None
        self._rep = None
        self._context = None
        self._geom = None
        self._robot = None
        self._view = None
        self._product = None
        self._annotator = None
        self._order = ()
        self._limits = ()
        self._episode = None
        self._step = 0
        self._origin = 0.0

    def __enter__(self):
        _main_thread()
        if self._app is not None:
            raise RuntimeError("Isaac session is already open")
        if os.environ.get("ACCEPT_EULA") != "Y":
            raise RuntimeError("Set ACCEPT_EULA=Y after accepting NVIDIA's container license")

        # Kit must start before loading any Omniverse SDK modules.
        from isaacsim import SimulationApp

        self._app = SimulationApp(
            {
                "headless": True,
                "width": self._spec.width,
                "height": self._spec.height,
                "renderer": _RENDERER,
                "anti_aliasing": _AA_DLSS,
                "extra_args": [f"--{_DENOISING}=false", f"--{_DLSS_MODE}={_DLSS_QUALITY}"],
            }
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _main_thread()
        self._close(_FAILURE_EXIT if exc_type else _SUCCESS_EXIT)

    def _load_sdk(self) -> None:
        import carb.settings
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import UsdGeom

        self._rep = rep
        self._context = omni.usd.get_context()
        self._timeline = omni.timeline.get_timeline_interface()
        self._manager = SimulationManager
        self._geom = UsdGeom
        settings = carb.settings.get_settings()
        settings.set(_DENOISING, False)
        settings.set(_DLSS_MODE, _DLSS_QUALITY)
        self._rep.orchestrator.set_capture_on_play(False)

    def reset(self, episode_id: str) -> None:
        self._require_open()
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a nonempty string")

        # Report SDK startup failures from the episode body before Kit closes.
        if self._manager is None:
            self._load_sdk()

        self._release()
        self._context.close_stage()
        logging.info("Opening scene: %s", self._spec.scene)
        if not self._context.open_stage(self._spec.scene):
            raise RuntimeError(f"Cannot open USD scene: {self._spec.scene}")

        # Reload disk-authored state, including objects, velocities, and drives.
        stage = self._context.get_stage()
        stage.Reload()
        camera = stage.GetPrimAtPath(self._spec.camera)
        if not camera or not camera.IsA(self._geom.Camera):
            raise ValueError(f"Camera not found: {self._spec.camera}")
        self._timeline.set_current_time(0.0)
        self._timeline.set_looping(False)
        self._timeline.commit()
        logging.info("Initializing physics")
        self._manager.setup_simulation(dt=1.0 / self._spec.physics_hz)
        self._manager.initialize_physics()
        self._bind_robot()
        logging.info("Physics ready; warming renderer")

        self._product = self._rep.create.render_product(
            self._spec.camera, (self._spec.width, self._spec.height)
        )
        self._annotator = self._rep.annotators.get("rgb")
        self._annotator.attach(self._product)
        self._render(_WARMUP_SUBFRAMES)
        self._origin = self._time()
        self._step = 0
        self._episode = episode_id
        logging.info("Scene ready: %s", episode_id)

    def _bind_robot(self) -> None:
        import numpy as np

        self._view = self._manager.get_physics_simulation_view()
        if self._view is None:
            raise RuntimeError("Isaac did not initialize its physics view")
        self._robot = self._view.create_articulation_view(self._spec.articulation)
        if self._robot.count != 1:
            raise ValueError("Expected exactly one robot articulation")
        native = tuple(self._robot.shared_metatype.dof_names)
        if (
            len(native) != self._robot.max_dofs
            or len(set(native)) != len(native)
            or set(native) != set(self._spec.joints)
        ):
            raise ValueError(f"Articulation joints do not match configuration: {native}")
        self._order = tuple(native.index(name) for name in self._spec.joints)
        limits = _array(self._robot.get_dof_limits())
        if (
            limits.shape != (1, len(native), 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, :, 0] > limits[:, :, 1])
        ):
            raise RuntimeError("Invalid articulation joint limits")
        self._limits = tuple(
            tuple(float(value) for value in limits[0, index]) for index in self._order
        )

    def observe(self) -> Observation:
        self._require_episode()
        self._render(_CAPTURE_SUBFRAMES)
        state = _array(self._robot.get_dof_positions())
        if state.shape != (1, len(self._order)):
            raise RuntimeError("Invalid articulation state shape")
        positions = tuple(float(state[0, index]) for index in self._order)
        if not all(math.isfinite(value) for value in positions):
            raise RuntimeError("Articulation state contains nonfinite values")

        pixels = self._annotator.get_data()
        if (
            pixels.ndim != 3
            or pixels.shape[:2] != (self._spec.height, self._spec.width)
            or pixels.shape[2] not in (RGB_CHANNELS, _RGBA_CHANNELS)
            or str(pixels.dtype) != "uint8"
        ):
            raise RuntimeError(f"Invalid RGB frame: {pixels.shape}, {pixels.dtype}")
        frame = Frame(
            self._spec.width, self._spec.height, pixels[:, :, :RGB_CHANNELS].tobytes(order="C")
        )
        return Observation(self._episode, self._step, self._time() - self._origin, positions, frame)

    def apply(self, targets: tuple[float, ...]) -> None:
        self._require_episode()
        if len(targets) != len(self._order) or not all(math.isfinite(value) for value in targets):
            raise ValueError("Joint targets must contain one finite radian value per joint")
        for name, value, limits in zip(self._spec.joints, targets, self._limits, strict=True):
            if not limits[0] <= value <= limits[1]:
                raise ValueError(f"Joint target exceeds limits: {name}={value}")

        reference = self._robot.get_dof_position_targets()
        ordered = _array(reference).copy()
        for index, value in zip(self._order, targets, strict=True):
            ordered[0, index] = value
        data, indices = _target_buffers(ordered, reference)
        self._robot.set_dof_position_targets(data, indices)

    @property
    def joint_limits(self) -> tuple[tuple[float, float], ...]:
        self._require_episode()
        return self._limits

    def step(self) -> None:
        self._require_episode()
        before = self._time()
        # Physics advances by an integer number of substeps; inference never ticks Kit.
        self._manager.step(
            steps=self._spec.physics_hz // self._spec.fps,
            update_fabric=self._manager.is_fabric_enabled(),
        )
        self._check_time(before + 1.0 / self._spec.fps)
        self._step += 1

    def _render(self, subframes: int) -> None:
        before = self._time()
        self._rep.orchestrator.step(
            delta_time=0.0, rt_subframes=subframes, pause_timeline=True, wait_for_render=True
        )
        self._check_time(before)

    def _time(self) -> float:
        value = float(self._manager.get_simulation_time())
        if not math.isfinite(value):
            raise RuntimeError("Isaac returned an invalid simulation clock")
        return value

    def _check_time(self, expected: float) -> None:
        if not math.isclose(self._time(), expected, rel_tol=0.0, abs_tol=_CLOCK_TOLERANCE):
            raise RuntimeError("Isaac advanced outside the requested simulation timestep")

    def _require_open(self) -> None:
        _main_thread()
        if self._app is None:
            raise RuntimeError("Enter the Isaac session before using it")

    def _require_episode(self) -> None:
        self._require_open()
        if self._episode is None:
            raise RuntimeError("Reset Isaac before using an episode")

    def _release(self) -> None:
        self._episode = None
        if self._annotator is not None:
            self._annotator.detach()
            self._annotator = None
        if self._product is not None:
            self._product.destroy()
            self._product = None
        if self._timeline is not None:
            self._timeline.stop()
            self._timeline.commit()
        if self._view is not None:
            self._manager.invalidate_physics()
            self._view = None
            self._robot = None

    def _close(self, exit_code: int) -> None:
        app = self._app
        try:
            self._release()
        except BaseException:
            exit_code = _FAILURE_EXIT
            raise
        finally:
            self._app = None
            if app is not None:
                app.close(exit_code=exit_code)
