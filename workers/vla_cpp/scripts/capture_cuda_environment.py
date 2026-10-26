"""Record explicit environment provenance on the remote Docker host (no credentials)."""

import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cuda-image", default="jensen-quant-cuda:local")
parser.add_argument("--sim-image", default="jensen-quant-sim:local")
parser.add_argument("--modelopt-image", default="jensen-modelopt:local")
parser.add_argument("--output", type=Path, default=Path("artifacts/cuda/setup"))
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)


def command(args):
    return subprocess.check_output(args, text=True).strip()


result = {
    "kernel": command(["uname", "-srmo"]),
    "docker": command(["docker", "--version"]),
    "gpu": command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ]
    ),
    "images": [],
}
for name, tag, python in [
    ("cuda", args.cuda_image, "python3"),
    ("sim", args.sim_image, "python3"),
    ("modelopt", args.modelopt_image, "python"),
]:
    data = json.loads(command(["docker", "image", "inspect", tag]))[0]
    result["images"].append(
        {
            "tag": tag,
            "id": data["Id"],
            "created": data["Created"],
            "architecture": data["Architecture"],
        }
    )
    (out / (name + "-packages.txt")).write_text(
        command(["docker", "run", "--rm", "--entrypoint", python, tag, "-m", "pip", "freeze"])
        + "\n"
    )
(out / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
