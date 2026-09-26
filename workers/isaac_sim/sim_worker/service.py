import logging
from contextlib import ExitStack
from pathlib import Path

from sim_worker.adapters import isaac, manifest
from sim_worker.adapters.artifacts import Artifacts
from sim_worker.adapters.video import Recorder


def validate(path: Path) -> None:
    manifest.load(path)


def execute(path: Path, output_dir: Path) -> dict:
    spec = manifest.load(path)
    artifacts = Artifacts(output_dir, spec)
    # SDK shutdown may terminate Python, so publish success or failure before it.
    with ExitStack() as session:
        try:
            artifacts.start()
            frames = session.enter_context(isaac.frames(spec))
            with Recorder(artifacts.video_path, spec) as recorder:
                for frame in frames:
                    recorder.append(frame)
            logging.info("Recording verified; uploading video and result")
            result = artifacts.complete()
            logging.info("Result published: %s", result["result_uri"])
            return result
        except Exception as error:
            logging.exception("Simulation failed")
            try:
                artifacts.fail(str(error))
            except Exception:
                logging.exception("Could not upload failure record; inspect worker logs")
            raise
