"""Re-score a complete recorded object trajectory without starting Isaac."""

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

from sim_worker.rollout.config import load
from sim_worker.rollout.contracts import Frame, ObjectState, Observation
from sim_worker.rollout.evaluation import EpisodeEvaluator


def _body(value):
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(ObjectState)}:
        raise ValueError("Trace requires a complete object state, not inferred video geometry")
    return ObjectState(
        **{key: tuple(item) if isinstance(item, list) else item for key, item in value.items()}
    )


def evaluate(spec, records):
    if spec.evaluation is None:
        raise ValueError("The manifest must explicitly declare evaluation criteria")
    evaluator = EpisodeEvaluator(spec.evaluation, fps=spec.sim.fps, steps=spec.steps)
    previous = None
    for index, row in enumerate(records):
        if index >= spec.steps or not isinstance(row, dict):
            raise ValueError("Trace exceeds its declared horizon or contains an invalid record")
        if type(row.get("step")) is not int or row["step"] != index:
            raise ValueError("Trace must contain every control step exactly once")
        try:
            before, after = _body(row["object_state"]), _body(row["next_object_state"])
            observation = Observation(
                row["episode_id"], index, row["sim_time"], (), Frame(0, 0, b""), before
            )
            following = Observation(
                row["episode_id"], index + 1, row["next_sim_time"], (), Frame(0, 0, b""), after
            )
        except KeyError as error:
            raise ValueError("Trace is missing its ordered object evidence") from error
        if previous is None:
            evaluator.observe(observation)
        elif observation != previous:
            raise ValueError("Trace object states or timestamps are discontinuous")
        evaluator.observe(following)
        previous = following
    return evaluator.finish()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Trace contains duplicate JSON fields")
        result[key] = value
    return result


def score_file(spec, path):
    digest = hashlib.sha256()
    total = 0

    def records():
        nonlocal total
        with path.open("rb") as stream:
            while line := stream.readline(65537):
                total += len(line)
                if len(line) > 65536 or total > 64 * 1024**2:
                    raise ValueError("Trace exceeds its bounded record or file size")
                digest.update(line)
                yield json.loads(line, object_pairs_hook=_pairs)

    result = evaluate(spec, records())
    return result | {
        "evidence_kind": "trajectory_replay",
        "trajectory_sha256": digest.hexdigest(),
        "trajectory_bytes": total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_file(load(args.manifest), args.trajectory)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
