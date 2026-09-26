import argparse
import json
import logging
from pathlib import Path

from sim_worker.service import execute, validate


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [sim] %(message)s")
    parser = argparse.ArgumentParser(description="Run a manifest-driven Isaac recording")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sim-results"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.manifest)
        print("Manifest valid")
        return

    result = execute(args.manifest, args.output_dir)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    _main()
