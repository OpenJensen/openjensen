"""Record package versions, GPU identity and harness hashes without host credentials."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": props.name,
        "total_vram_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor],
        "torch_cuda": torch.version.cuda,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            )
        ),
        "script_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
