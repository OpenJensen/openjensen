"""Cloud-only FlashAttention installer with pinned upstream wheel hashes."""

import os
import platform
import shutil
import subprocess
import sys

VERSION = "2.7.4.post1"
WHEEL_HASHES = {
    False: "42b38a57cb3e80b0db0b879279371e7552bd63da3b80d4614e7ccee8b9834e27",
    True: "22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113",
}


def wheel_url(*, python_version, system, machine, torch_version, cuda_version, cxx11_abi):
    if (
        python_version != (3, 11)
        or system != "Linux"
        or machine != "x86_64"
        or torch_version.split("+")[0].split(".")[:2] != ["2", "7"]
        or not cuda_version
        or cuda_version.split(".")[0] != "12"
    ):
        return None
    abi = "TRUE" if cxx11_abi else "FALSE"
    return (
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        f"v{VERSION}/flash_attn-{VERSION}%2Bcu12torch2.7cxx11abi{abi}"
        "-cp311-cp311-linux_x86_64.whl#sha256=" + WHEEL_HASHES[bool(cxx11_abi)]
    )


def main():
    import torch

    url = wheel_url(
        python_version=sys.version_info[:2],
        system=platform.system(),
        machine=platform.machine(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    )
    command = ["python3", "-m", "uv", "pip", "install", "--python", sys.executable, "--no-deps"]
    if url:
        print("Installing the hash-pinned official FlashAttention CUDA/Torch wheel", flush=True)
        subprocess.run([*command, url], check=True)
    else:
        if not shutil.which("nvcc"):
            raise RuntimeError(
                "No matching pinned FlashAttention wheel for this environment; "
                "a CUDA toolkit with nvcc is required for the source-build fallback"
            )
        print(
            "Building the pinned FlashAttention source for this CUDA/Torch environment", flush=True
        )
        subprocess.run(
            [*command, f"flash_attn=={VERSION}", "--no-build-isolation"],
            env={**os.environ, "MAX_JOBS": "2"},
            check=True,
        )


if __name__ == "__main__":
    main()
