"""Verify production rendering and motion without encoding or cloud uploads."""

from dataclasses import replace
from contextlib import contextmanager
import faulthandler
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, "/opt/sim-worker")

import numpy as np
from PIL import Image

from sim_worker.adapters import isaac
from sim_worker.adapters.manifest import load


_OUTPUT = Path("/probe-output")
_FRAMES = 60
_CUBE = "/World/Cube"
_STACK_INTERVAL_SECONDS = 60
_MIN_MOTION_METERS = 0.1
_SETTING_PATHS = (
    "/rtx/rendermode",
    "/rtx/ecoMode/enabled",
    "/rtx/hydra/supportMultiTickRate",
    "/persistent/rtx/modes/rt2/enabled",
    "/app/asyncRendering",
    "/omni/replicator/asyncRendering",
    "/exts/isaacsim.core.throttling/enable_async",
    "/exts/omni.replicator.core/maxAssetLoadingTime",
    "/rtx/materialDb/syncLoads",
    "/rtx/hydra/materialSyncLoads",
    "/omni/kit/plugin/syncUsdLoads",
    "/rtx/dldenoiser/responsiveDenoising",
    "/rtx/post/aa/op",
    "/rtx/post/dlss/execMode",
)
_STATES = []


@contextmanager
def _stacks():
    # Keep Python stacks when rendering blocks before any frames are returned.
    with (_OUTPUT / "stacks.log").open("w") as output:
        faulthandler.register(signal.SIGUSR1, file=output, all_threads=False)
        faulthandler.dump_traceback_later(_STACK_INTERVAL_SECONDS, repeat=True, file=output)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.unregister(signal.SIGUSR1)


def _state(phase):
    import carb.settings
    import omni.usd

    settings = carb.settings.get_settings()
    record = {
        "phase": phase,
        "settings": {key: settings.get(key) for key in _SETTING_PATHS},
        "loading": omni.usd.get_context().get_stage_loading_status(),
    }
    print("STATE " + json.dumps(record), flush=True)
    _STATES.append(record)
    (_OUTPUT / "states.json").write_text(json.dumps(_STATES, indent=2))


def _cube_position():
    from omni.physx import get_physx_interface

    # Read physics directly; USD transforms may lag when Fabric is enabled.
    try:
        pose = get_physx_interface().get_rigidbody_transformation(_CUBE)
    except (AttributeError, RuntimeError) as error:
        return {"unavailable": str(error)}
    if not pose.get("ret_val"):
        return None
    return [float(value) for value in pose["position"]]


spec = replace(load(Path("/opt/sim-worker/demo.yaml")), width=640, height=360, frames=_FRAMES)
records = []
_OUTPUT.mkdir(parents=True, exist_ok=True)
(_OUTPUT / "stats.json").write_text("[]")
with _stacks(), isaac.frames(spec) as capture:
    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager

    _state("app_startup")
    started = time.monotonic()
    timeline = omni.timeline.get_timeline_interface()
    for index, frame in enumerate(capture):
        if index == 0:
            _state("first_frame")
        pixels = np.frombuffer(frame, dtype=np.uint8).reshape(spec.height, spec.width, 3)
        record = {"frame": index, "min": int(pixels.min()), "max": int(pixels.max()),
                  "mean": float(pixels.mean()), "nonzero": int(np.count_nonzero(pixels)),
                  "time": timeline.get_current_time(), "elapsed": time.monotonic() - started,
                  "simulation_time": SimulationManager.get_simulation_time(),
                  "cube_position": _cube_position()}
        print("PROBE " + json.dumps(record), flush=True)
        records.append(record)
        # Preserve evidence even if the worker's black-frame guard raises at the end.
        (_OUTPUT / "stats.json").write_text(json.dumps(records, indent=2))
        Image.fromarray(pixels).save(_OUTPUT / f"frame-{index}.png")
    _state("after_capture")
    assert len(records) == _FRAMES, "Capture returned the wrong frame count"
    assert all(record["max"] > 0 for record in records), "Demo capture contains black frames"
    assert records[-1]["time"] > records[0]["time"], "Timeline did not advance"
    assert records[-1]["simulation_time"] > records[0]["simulation_time"], "Physics time did not advance"

    positions = [record["cube_position"] for record in records]
    assert all(isinstance(position, list) and len(position) == 3 for position in positions), "Cube pose unavailable"
    displacement = np.linalg.norm(np.asarray(positions) - positions[0], axis=1)
    assert displacement.max() > _MIN_MOTION_METERS, "Cube did not move"
    print("PROBE PASSED", flush=True)
