"""Exercise live Isaac control through HTTP using a synthetic, non-ML policy."""

import argparse
import faulthandler
import json
import logging
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

# The mounted standalone script must import the adjacent worker package.
_WORKER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKER))

from sim_worker.rollout.calibration import JointMap  # noqa: E402
from sim_worker.rollout.client import RemotePolicy  # noqa: E402
from sim_worker.rollout.config import RolloutSpec  # noqa: E402
from sim_worker.rollout.contracts import CALIBRATION_VERSION, SimSpec  # noqa: E402
from sim_worker.rollout.isaac import IsaacSim  # noqa: E402
from sim_worker.rollout.server import create_server  # noqa: E402
from sim_worker.rollout.service import Rollout  # noqa: E402
from sim_worker.rollout.trace import Trace, save_result  # noqa: E402

_OUTPUT = Path("/probe-output")
_SCENE = _WORKER / "scenes" / "so101-pickup" / "scene.usda"
_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_WIDTH = 1280
_HEIGHT = 720
_FPS = 30
_PHYSICS_HZ = 120
_STEPS = 30
_NUDGE_RAD = 0.03
_MIN_MOVE_RAD = 0.005
_RESET_TOLERANCE_RAD = 0.001
_CLOCK_TOLERANCE = 1e-6
_MIN_PIXEL_MEAN = 1.0
_MIN_PIXEL_STD = 1.0
_REQUEST_TIMEOUT = 30.0
_STACK_INTERVAL = 60
_MODEL = "MOCK-ONLY-fixed-joint-nudge-v1"
_SCOPE = "MOCK ONLY: HTTP control, rendering, stepping and reset; no VLA or dataset calibration."


class _Nudge:
    def __init__(self):
        self._target = None

    def reset(self) -> None:
        self._target = None

    def predict(self, state, rgb, width, height, task):
        # Fix the target from the initial observation; repeated calls must not accumulate motion.
        if self._target is None:
            self._target = list(state)
            self._target[0] += _NUDGE_RAD
        return [list(self._target)]


def _mapping(path):
    data = {
        "api_version": CALIBRATION_VERSION,
        "status": "verified",
        "source": _SCOPE,
        "joints": {
            name: {"sim_rad": [-math.tau, math.tau], "policy": [-math.tau, math.tau]}
            for name in _JOINTS
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return JointMap(path, _JOINTS)


def _pixels(observation):
    import numpy as np

    values = np.frombuffer(observation.frame.rgb, dtype=np.uint8)
    stats = {"mean": float(values.mean()), "std": float(values.std())}
    if stats["mean"] <= _MIN_PIXEL_MEAN or stats["std"] <= _MIN_PIXEL_STD:
        raise RuntimeError(f"Camera image is dark or constant: {stats}")
    return stats


def _run(simulation, policy, mapping, spec, output, report):
    import numpy as np

    logging.info("Checking reset and repeated observations")
    simulation.reset("probe-initial")
    initial = simulation.observe()
    repeated = simulation.observe()
    checks = {
        "zero_initial_clock": initial.step == 0 and abs(initial.sim_time) <= _CLOCK_TOLERANCE,
        "observe_preserves_clock": initial.sim_time == repeated.sim_time,
        "observe_preserves_joints": initial.state == repeated.state,
    }
    report.update(checks=checks, initial_state_rad=initial.state,
                  initial_pixels=_pixels(initial))
    save_result(output, report)
    if not all(checks.values()):
        raise RuntimeError("Observing Isaac changed its state or clock")

    trace_dir = output / "rollout"
    trace_dir.mkdir()
    logging.info("Running %d HTTP control steps", spec.steps)
    with Trace(trace_dir, spec) as trace:
        rollout = Rollout(spec, simulation, policy, mapping).run(
            "probe-http", trace.append, trace.finish)
    final = simulation.observe()
    movement = final.state[0] - initial.state[0]
    checks.update(
        measured_joint_motion=_MIN_MOVE_RAD <= movement <= 2.0 * _NUDGE_RAD,
        exact_final_clock=abs(final.sim_time - spec.steps / spec.sim.fps) <= _CLOCK_TOLERANCE,
        exact_step_count=final.step == spec.steps,
        one_request_per_step=rollout["policy_requests"] == spec.steps,
    )
    report.update(rollout=rollout, final_pixels=_pixels(final),
                  final_state_rad=final.state, requested_nudge_rad=_NUDGE_RAD,
                  measured_nudge_rad=movement, trace_directory=str(trace_dir))
    save_result(output, report)

    logging.info("Checking post-control reset")
    simulation.reset("probe-reset")
    restored = simulation.observe()
    error = float(np.max(np.abs(np.asarray(restored.state) - initial.state)))
    checks.update(
        reset_restores_joints=error <= _RESET_TOLERANCE_RAD,
        reset_clears_clock=restored.step == 0 and abs(restored.sim_time) <= _CLOCK_TOLERANCE,
    )
    report.update(reset_state_rad=restored.state, reset_max_error_rad=error,
                  reset_tolerance_rad=_RESET_TOLERANCE_RAD)
    if not all(checks.values()):
        raise RuntimeError(f"GPU rollout checks failed: {checks}")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "probe.log")])
    report = {"status": "running", "scope": _SCOPE, "scene": str(_SCENE)}
    save_result(output, report)
    started = time.monotonic()

    # The daemon provides transport only; all Isaac calls stay on this main thread.
    server = create_server(("127.0.0.1", 0), _Nudge(), _MODEL, state_dim=len(_JOINTS))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    policy = RemotePolicy(endpoint, _MODEL, "MOCK ONLY: nudge shoulder_pan", _REQUEST_TIMEOUT)
    with tempfile.TemporaryDirectory(prefix="isaac-rollout-probe-") as config_dir:
        calibration = Path(config_dir) / "mock-calibration.json"
        mapping = _mapping(calibration)
        spec = RolloutSpec(
            SimSpec(str(_SCENE), "/World/Cameras/Front", "/World/Robot/joints/root_joint",
                    _JOINTS, _WIDTH, _HEIGHT, _FPS, _PHYSICS_HZ),
            _STEPS, 1, endpoint, _MODEL, "MOCK ONLY: nudge shoulder_pan",
            _REQUEST_TIMEOUT, calibration,
        )
        with (output / "stacks.log").open("w") as stacks:
            faulthandler.dump_traceback_later(_STACK_INTERVAL, repeat=True, file=stacks)
            policy.wait_ready(_REQUEST_TIMEOUT)
            with IsaacSim(spec.sim) as simulation:
                try:
                    _run(simulation, policy, mapping, spec, output, report)
                    report.update(status="passed", elapsed_seconds=time.monotonic() - started)
                    save_result(output, report)
                    logging.info("GPU rollout passed; nudge=%f rad", report["measured_nudge_rad"])
                    print("ROLLOUT_RESULT " + json.dumps(report), flush=True)
                except BaseException as error:
                    report.update(status="failed", error=str(error),
                                  elapsed_seconds=time.monotonic() - started)
                    save_result(output, report)
                    logging.exception("GPU rollout failed")
                    print("ROLLOUT_RESULT " + json.dumps(report), flush=True)
                    raise
                finally:
                    # Persist every result before Kit shutdown can terminate Python.
                    faulthandler.cancel_dump_traceback_later()
                    server.shutdown()
                    server.server_close()


if __name__ == "__main__":
    _main()
