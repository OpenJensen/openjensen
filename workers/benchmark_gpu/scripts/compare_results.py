"""Compare completed runs only after checking their fixture and episode contracts."""

import argparse
import json
import statistics
from pathlib import Path


def paired_quality(reference, candidate):
    keys = ("task", "init_state", "seed", "noise_seed")

    def index(rows):
        result = {tuple(row[key] for key in keys): bool(row["success"]) for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate episode identity")
        return result

    ref = index(reference)
    cand = index(candidate)
    if not ref or ref.keys() != cand.keys():
        raise ValueError("Paired quality requires identical nonempty episode identities")
    return {
        "episodes": len(ref),
        "reference_successes": sum(ref.values()),
        "candidate_successes": sum(cand.values()),
        "both_succeed": sum(ref[k] and cand[k] for k in ref),
        "reference_only": sum(ref[k] and not cand[k] for k in ref),
        "candidate_only": sum(not ref[k] and cand[k] for k in ref),
        "both_fail": sum(not ref[k] and not cand[k] for k in ref),
    }


def compare(reference_dir, candidate_dir):
    import numpy as np

    ref = json.loads((reference_dir / "result.json").read_text())
    cand = json.loads((candidate_dir / "result.json").read_text())
    if ref.get("status") != "passed" or cand.get("status") != "passed":
        raise ValueError("Both runs must be complete and passed")
    for key in ("checkpoint", "revision", "contract"):
        if key not in ref or ref[key] != cand.get(key):
            raise ValueError(f"Mismatched {key}")

    def hashes(result):
        return {f["name"]: f["sha256"] for f in result["fixtures"]}

    if not hashes(ref) or hashes(ref) != hashes(cand):
        raise ValueError("Fixture identities differ")
    errors, signs, per_fixture = [], [], []
    for name in hashes(ref):
        filename = Path(name).stem + ".actions.npy"
        a, b = np.load(reference_dir / filename), np.load(candidate_dir / filename)
        if a.shape != (1, 50, 7) or b.shape != a.shape:
            raise ValueError("Invalid action shape")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite actions")
        delta = b.astype(np.float64) - a.astype(np.float64)
        errors.append(delta)
        signs.append(np.sign(a[..., -1]) == np.sign(b[..., -1]))
        per_fixture.append(
            {
                "fixture": name,
                "mae": float(np.abs(delta).mean()),
                "max_abs": float(np.abs(delta).max()),
            }
        )
    all_errors = np.concatenate(errors)
    result = {
        "reference": reference_dir.name,
        "candidate": candidate_dir.name,
        "fixture_count": len(errors),
        "mae": float(np.abs(all_errors).mean()),
        "rmse": float(np.sqrt(np.square(all_errors).mean())),
        "max_abs": float(np.abs(all_errors).max()),
        "gripper_sign_agreement": float(np.concatenate(signs).mean()),
        "median_fixture_p50_ms": statistics.median(f["p50_ms"] for f in cand["fixtures"]),
        "fixtures": per_fixture,
    }
    if ref.get("episodes") or cand.get("episodes"):
        result["paired_quality"] = paired_quality(ref.get("episodes", []), cand.get("episodes", []))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
