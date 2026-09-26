"""Check remote inference using a recorded observation; never apply robot actions."""

import argparse
import json
import math
import time
import uuid
from pathlib import Path

from sim_worker.rollout.client import RemotePolicy
from sim_worker.rollout.config import load
from sim_worker.rollout.contracts import Frame, Observation

_READY_SECONDS = 900
_DATASET_FILE = "dataset.json"
_IMAGE_FILE = "front-frame-000.jpg"
_STATE_KEY = "observation.state"


def _observation(manifest, spec, episode):
    from PIL import Image

    evidence = manifest.parent / "evidence"
    record = json.loads((evidence / _DATASET_FILE).read_text())
    state = record["first_frame_raw"][_STATE_KEY]
    if len(state) != len(spec.sim.joints) or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in state
    ):
        raise ValueError("Recorded observation state must match the checkpoint joint count")
    with Image.open(evidence / _IMAGE_FILE) as image:
        rgb = image.convert("RGB").resize(
            (spec.sim.width, spec.sim.height), Image.Resampling.BILINEAR
        )
        frame = Frame(spec.sim.width, spec.sim.height, rgb.tobytes())
    return Observation(episode, 0, 0.0, tuple(state), frame)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    spec = load(args.manifest)
    policy = RemotePolicy(args.endpoint, spec.model_id, spec.task, spec.timeout_seconds)
    policy.wait_ready(_READY_SECONDS)
    episode = uuid.uuid4().hex
    observation = _observation(args.manifest, spec, episode)
    policy.reset(episode)
    started = time.monotonic()
    chunk = policy.predict(observation)
    result = {
        "status": "ready",
        "model_id": spec.model_id,
        "endpoint": args.endpoint,
        "input": "recorded_first_frame_resized_bilinear",
        "width": spec.sim.width,
        "height": spec.sim.height,
        "inference_seconds": time.monotonic() - started,
        "actions": chunk.actions,
        "robot_actions_applied": 0,
    }
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    _main()
