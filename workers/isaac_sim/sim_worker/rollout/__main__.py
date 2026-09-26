import argparse
import json
import logging
from pathlib import Path

from sim_worker.rollout.app import execute, validate
from sim_worker.rollout.calibration import CalibrationUse

_READY_TIMEOUT_SECONDS = 900


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [rollout] %(message)s")
    parser = argparse.ArgumentParser(description="Run a calibrated remote-policy Isaac episode")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rollout-results"))
    parser.add_argument("--ready-timeout", type=float, default=_READY_TIMEOUT_SECONDS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Use an unverified map with logged extrapolation and bounded joint targets",
    )
    args = parser.parse_args()
    use = CalibrationUse.EXPERIMENTAL if args.experimental else CalibrationUse.VERIFIED
    if args.validate_only:
        validate(args.manifest, use)
        print(f"Rollout manifest accepted for {use.value} calibration use")
        return
    print(json.dumps(execute(args.manifest, args.output_dir, args.ready_timeout, use)), flush=True)


if __name__ == "__main__":
    _main()
