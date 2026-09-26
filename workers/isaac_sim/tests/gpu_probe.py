"""Compare a short synchronous capture against the observed black recording."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, "/opt/sim-worker")

import isaacsim
import numpy as np
from PIL import Image

from sim_worker.adapters import isaac
from sim_worker.adapters.manifest import load


_OUTPUT = Path("/probe-output")
_FRAMES = 60
_PROFILE = os.environ.get("SIM_PROBE_PROFILE", "pathtracing")
_SYNC_ARGS = [
    "--/exts/isaacsim.core.throttling/enable_async=false",
    "--/app/asyncRendering=false",
    "--/omni/replicator/asyncRendering=false",
    "--/exts/omni.replicator.core/maxAssetLoadingTime=30",
]
_ORIGINAL_APP = isaacsim.SimulationApp


def _app(config):
    config["extra_args"] = _SYNC_ARGS
    if _PROFILE == "realtime":
        config["renderer"] = "RealTimePathTracing"
        config["anti_aliasing"] = 3
        config["extra_args"] += ["--/rtx/dldenoiser/responsiveDenoising=false", "--/rtx/post/dlss/execMode=2"]
    print("CONFIG " + json.dumps(config), flush=True)
    return _ORIGINAL_APP(config)


isaacsim.SimulationApp = _app
spec = replace(load(Path("/opt/sim-worker/demo.yaml")), width=640, height=360, frames=_FRAMES)
records = []
with isaac.frames(spec) as capture:
    import carb.settings
    import omni.usd
    import omni.timeline

    settings = carb.settings.get_settings()
    started = time.monotonic()
    timeline = omni.timeline.get_timeline_interface()
    for index, frame in enumerate(capture):
        pixels = np.frombuffer(frame, dtype=np.uint8).reshape(spec.height, spec.width, 3)
        record = {"frame": index, "min": int(pixels.min()), "max": int(pixels.max()),
                  "mean": float(pixels.mean()), "nonzero": int(np.count_nonzero(pixels)),
                  "time": timeline.get_current_time(), "elapsed": time.monotonic() - started}
        print("PROBE " + json.dumps(record), flush=True)
        records.append(record)
        Image.fromarray(pixels).save(_OUTPUT / f"frame-{index}.png")
    print("SETTINGS " + json.dumps({key: settings.get(key) for key in (
        "/app/asyncRendering", "/omni/replicator/asyncRendering",
        "/exts/isaacsim.core.throttling/enable_async", "/rtx/rendermode",
        "/exts/omni.replicator.core/maxAssetLoadingTime", "/rtx/pathtracing/spp",
        "/rtx/pathtracing/totalSpp", "/rtx/pathtracing/clampSpp",
        "/rtx/dldenoiser/responsiveDenoising",
    )}), flush=True)
    print("LOADING " + repr(omni.usd.get_context().get_stage_loading_status()), flush=True)
    (_OUTPUT / "stats.json").write_text(json.dumps(records, indent=2))
    assert all(record["max"] > 0 for record in records), "Demo capture is entirely black"
    print("PROBE PASSED", flush=True)
