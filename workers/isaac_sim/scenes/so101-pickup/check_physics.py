"""Check the live articulation lookup without waiting for camera rendering."""

import json
from pathlib import Path

from isaacsim import SimulationApp

_ROOT = Path(__file__).resolve().parent
_OUTPUT = Path("/probe-output/lookup-result.json")
_STEPS = 300
_FPS = 60
_LINK_COUNT = 7
_SUCCESS = 0
_FAILURE = 1


def _check():
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from smoke_test import _Physics

    # Use the same scene and probe as the rendered test, with manual physics steps.
    context = omni.usd.get_context()
    if not context.open_stage(str(_ROOT / "scene.usda")):
        raise RuntimeError("Cannot open the SO101 scene")
    SimulationManager.setup_simulation(dt=1.0 / _FPS)
    SimulationManager.initialize_physics()
    probe = _Physics()
    initial = probe._snapshot()
    started = SimulationManager.get_simulation_time()
    SimulationManager.step(steps=_STEPS)
    final = probe._snapshot()
    finished = SimulationManager.get_simulation_time()
    metadata = probe._metadata()
    if metadata["link_count"] != _LINK_COUNT or finished <= started:
        raise RuntimeError("Incomplete articulation or physics did not advance")
    return {"status": "passed", "physics": metadata, "initial": initial,
            "final": final, "simulation_seconds": finished - started,
            "scope": "Live articulation lookup and finite state; rendering tested separately."}


def _main():
    app = SimulationApp({"headless": True, "disable_viewport_updates": True})
    code = _SUCCESS
    try:
        report = _check()
    except Exception as error:
        code = _FAILURE
        report = {"status": "failed", "error": str(error)}
    finally:
        _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        _OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("LOOKUP_RESULT " + json.dumps(report), flush=True)
        app.close(exit_code=code)


if __name__ == "__main__":
    _main()
