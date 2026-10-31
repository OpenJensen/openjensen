#!/usr/bin/env python3
"""Install the optional native ACT runtime on the computer hosting the USB companion."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

LEROBOT_REVISION = "e595b7902714ba51f91e47523f66f89c5181b649"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Existing Robot Lab companion data directory"
    )
    parser.add_argument(
        "--runtime", type=Path, required=True, help="Persistent, separately managed ACT environment"
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    data, runtime = args.data_dir.expanduser().resolve(), args.runtime.expanduser().resolve()
    if not data.is_dir():
        parser.error("--data-dir must be the existing companion data directory")
    if args.device == "mps" and platform.system() != "Darwin":
        parser.error("MPS is available only on macOS")
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv before preparing the optional ACT runtime")
    if shutil.disk_usage(data).free < 14 * 1024**3:
        parser.error("ACT runtime preparation needs 14 GiB free, retaining 10 GiB headroom")
    settings = data / "policy-runtime.json"
    if settings.is_symlink():
        parser.error("Policy runtime settings cannot be a symbolic link")
    python = runtime / "bin/python"
    if not python.is_file():
        subprocess.run([uv, "venv", "--python", "3.12", str(runtime)], check=True)
    torch = [uv, "pip", "install", "--python", str(python), "torch==2.11.0", "torchvision==0.26.0"]
    if platform.system() == "Linux":
        torch += ["--index-url", "https://download.pytorch.org/whl/cpu"]
    subprocess.run(torch, check=True)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            f"lerobot @ git+https://github.com/huggingface/lerobot.git@{LEROBOT_REVISION}",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            "from lerobot.policies.act.modeling_act import ACTPolicy; "
            "from lerobot.policies.factory import make_pre_post_processors",
        ],
        check=True,
    )
    handle, temporary = tempfile.mkstemp(prefix="policy-runtime-", suffix=".json", dir=data)
    try:
        with os.fdopen(handle, "w") as output:
            json.dump({"python": str(python), "device": args.device}, output)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, settings)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("Native ACT runtime configured. Check camera and inference rate before enabling motion.")


if __name__ == "__main__":
    main()
