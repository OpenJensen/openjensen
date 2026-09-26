from collections.abc import Iterator
from contextlib import closing, contextmanager
import logging
import os
from pathlib import Path

from sim_worker.contracts import BUILTIN_SCENE, RGB_CHANNELS, RunSpec


_SCENE_DIR = Path(__file__).resolve().parents[2] / "scenes"
_WARMUP_SUBFRAMES = 8
_CAPTURE_SUBFRAMES = 4
_RENDERER = "PathTracing"
_SAMPLES_PER_PIXEL = 64
_AA_DISABLED = 0
_FULL_DENOISING = 0.0
_RGBA_CHANNELS = 4
_START_SECONDS = 0.0
_END_PADDING_FRAMES = 1
_SUCCESS_EXIT = 0
_FAILURE_EXIT = 1


@contextmanager
def frames(spec: RunSpec) -> Iterator[Iterator[bytes]]:
    if os.environ.get("ACCEPT_EULA") != "Y":
        raise RuntimeError("Set ACCEPT_EULA=Y after accepting NVIDIA's container license")

    # Kit must start before any Omniverse SDK import.
    from isaacsim import SimulationApp

    logging.info("Starting Isaac")
    # Offline recording uses sampled lighting and OptiX, without DLSS/NGX.
    app = SimulationApp({
        "headless": True,
        "width": spec.width,
        "height": spec.height,
        "renderer": _RENDERER,
        "samples_per_pixel_per_frame": _SAMPLES_PER_PIXEL,
        "denoiser": True,
        "anti_aliasing": _AA_DISABLED,
    })
    exit_code = _SUCCESS_EXIT
    try:
        # Keep Isaac alive until the caller has encoded and published the result.
        with closing(_capture(spec, app)) as capture:
            yield capture
    except BaseException:
        exit_code = _FAILURE_EXIT
        raise
    finally:
        logging.info("Closing Isaac, exit_code=%d", exit_code)
        app.close(exit_code=exit_code)


def _capture(spec: RunSpec, app) -> Iterator[bytes]:
    import carb.settings
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import UsdGeom

    scene = str(_SCENE_DIR / "falling-cube.usda") if spec.scene == BUILTIN_SCENE else spec.scene
    context = omni.usd.get_context()
    logging.info("Opening scene: %s", scene)
    if not context.open_stage(scene):
        raise RuntimeError(f"Cannot open USD scene: {scene}")

    # USD loading can overwrite render settings; restore the recording profile.
    app.reset_render_settings()
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/blendFactor", _FULL_DENOISING)
    logging.info("Renderer=%s, samples=%d, subframes=%d, OptiX denoising=full",
                 _RENDERER, _SAMPLES_PER_PIXEL, _CAPTURE_SUBFRAMES)
    stage = context.get_stage()
    camera = stage.GetPrimAtPath(spec.camera)
    if not camera or not camera.IsA(UsdGeom.Camera):
        raise ValueError(f"Camera not found: {spec.camera}")

    rep.orchestrator.set_capture_on_play(False)
    logging.info("Creating camera capture")
    product = rep.create.render_product(spec.camera, (spec.width, spec.height))
    annotator = rep.annotators.get("rgb")
    timeline = omni.timeline.get_timeline_interface()
    try:
        annotator.attach(product)
        _set_timing(stage, timeline, spec)
        timeline.play()

        # Warm shaders at time zero so the falling motion stays in the recording.
        logging.info("Warming renderer")
        rep.orchestrator.step(delta_time=0.0, rt_subframes=_WARMUP_SUBFRAMES, pause_timeline=False)
        logging.info("Renderer ready; capturing %d frames", spec.frames)
        has_visible_rgb = False
        for index in range(spec.frames):
            # Subframes refine the image while physics stays at this frame's time.
            rep.orchestrator.step(delta_time=1.0 / spec.fps, rt_subframes=_CAPTURE_SUBFRAMES,
                                  pause_timeline=False, wait_for_render=True)
            pixels = annotator.get_data()
            valid_shape = (
                pixels.ndim == 3
                and pixels.shape[:2] == (spec.height, spec.width)
                and pixels.shape[2] in (RGB_CHANNELS, _RGBA_CHANNELS)
            )
            if not valid_shape or str(pixels.dtype) != "uint8":
                raise RuntimeError(f"Invalid camera frame: {pixels.shape}, {pixels.dtype}")
            frame = pixels[:, :, :RGB_CHANNELS].tobytes(order="C")
            if spec.scene == BUILTIN_SCENE and not has_visible_rgb:
                has_visible_rgb = frame.count(0) < len(frame)
            yield frame
            count = index + 1
            if count == 1 or count % spec.fps == 0 or count == spec.frames:
                logging.info("Captured %d/%d frames", count, spec.frames)

        # The lit demo must show something; custom scenes may intentionally be dark.
        if spec.scene == BUILTIN_SCENE and not has_visible_rgb:
            raise RuntimeError("Builtin demo recording is entirely black")
    finally:
        timeline.stop()
        annotator.detach()
        product.destroy()


def _set_timing(stage, timeline, spec: RunSpec) -> None:
    # Extend playback past the last capture so it cannot wrap to the first frame.
    end_seconds = (spec.frames + _END_PADDING_FRAMES) / spec.fps
    root = stage.GetRootLayer()
    root.startTimeCode = _START_SECONDS
    root.endTimeCode = end_seconds * stage.GetTimeCodesPerSecond()
    timeline.set_start_time(_START_SECONDS)
    timeline.set_end_time(end_seconds)
    timeline.set_looping(False)
    timeline.set_current_time(_START_SECONDS)
