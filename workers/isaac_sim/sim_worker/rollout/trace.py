import json
from contextlib import ExitStack
from pathlib import Path

from sim_worker.adapters.video import Recorder
from sim_worker.contracts import RunSpec
from sim_worker.rollout.config import RolloutSpec
from sim_worker.rollout.contracts import Observation


class Trace:
    def __init__(self, directory: Path, spec: RolloutSpec):
        self._directory = directory
        self._spec = spec
        self._stack = ExitStack()
        self._stream = None
        self._video = None

    def __enter__(self):
        sim = self._spec.sim
        recording = RunSpec(
            sim.scene,
            sim.camera,
            sim.width,
            sim.height,
            sim.fps,
            self._spec.steps,
            str(self._directory),
        )
        try:
            self._stream = self._stack.enter_context(
                (self._directory / "trajectory.jsonl").open("x")
            )
            self._video = self._stack.enter_context(
                Recorder(self._directory / "video.mp4", recording)
            )
        except BaseException:
            self._stack.close()
            raise
        return self

    def append(self, values: dict, observation: Observation) -> None:
        self._stream.write(json.dumps(values, allow_nan=False) + "\n")
        self._stream.flush()
        self._video.append(observation.frame.rgb)

    def finish(self, observation: Observation) -> None:
        frame = observation.frame
        header = f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii")
        (self._directory / "final.ppm").write_bytes(header + frame.rgb)

    def __exit__(self, error_type, error, traceback):
        return self._stack.__exit__(error_type, error, traceback)


def save_result(directory: Path, result: dict) -> None:
    pending = directory / "result.pending.json"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.replace(directory / "result.json")
