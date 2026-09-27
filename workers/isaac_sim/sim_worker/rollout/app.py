import hashlib
import logging
import uuid
from pathlib import Path

from sim_worker.rollout.calibration import CalibrationUse, JointMap
from sim_worker.rollout.client import RemotePolicy
from sim_worker.rollout.config import load
from sim_worker.rollout.experimental import MotionGuard
from sim_worker.rollout.service import Rollout
from sim_worker.rollout.trace import Trace, save_result


def validate(path: Path, use: CalibrationUse = CalibrationUse.VERIFIED) -> None:
    spec = load(path)
    JointMap(spec.calibration, spec.sim.joints, use)


def execute(
    path: Path,
    output_dir: Path,
    ready_timeout: float,
    use: CalibrationUse = CalibrationUse.VERIFIED,
) -> dict:
    # Validate before acquiring a GPU runtime or creating output files.
    spec = load(path)
    mapping = JointMap(spec.calibration, spec.sim.joints, use)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("Rollout output directory must be empty")
    episode_id = uuid.uuid4().hex
    identity = {
        "episode_id": episode_id,
        "model_id": spec.model_id,
        "calibration_sha256": mapping.digest,
        "calibration_status": mapping.status,
        "experimental": use is CalibrationUse.EXPERIMENTAL,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "endpoint": spec.endpoint,
    }
    save_result(output_dir, identity | {"status": "running"})
    policy = RemotePolicy(spec.endpoint, spec.model_id, spec.task, spec.timeout_seconds)
    failure = None
    try:
        policy.wait_ready(ready_timeout)
        # Import only after CPU preflight; Isaac's SDK remains in its driver.
        from sim_worker.rollout.isaac import IsaacSim

        with IsaacSim(spec.sim, evaluation=spec.evaluation) as simulation:
            try:
                guard = None
                if use is CalibrationUse.EXPERIMENTAL:
                    guard = MotionGuard(lambda: simulation.joint_limits, spec.sim.fps)
                with Trace(output_dir, spec) as trace:
                    result = Rollout(spec, simulation, policy, mapping, guard).run(
                        episode_id, trace.append, trace.finish
                    )
                result = identity | result
                # Kit may terminate Python during close; commit the result first.
                save_result(output_dir, result)
                return result
            except BaseException as error:
                failure = str(error)
                save_result(output_dir, identity | {"status": "failed", "error": str(error)})
                raise
    except BaseException as error:
        logging.exception("Rollout failed")
        # Do not overwrite a committed result when Kit performs a successful exit.
        if failure is None and (not isinstance(error, SystemExit) or error.code not in (None, 0)):
            save_result(output_dir, identity | {"status": "failed", "error": str(error)})
        raise
