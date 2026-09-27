"""Pinned native CUDA inference setup, shared by cloud Evaluate and Run."""

import shutil
from pathlib import Path

from .sky_quantization import stage_native_worker

WORKER_MODULE = "policykit.cloud_inference"
CUDA_ARCHITECTURES = {"T4": "75", "L4": "89", "A100": "80"}
KEYRING_SHA256 = {
    "ubuntu2204": "d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df",
    "ubuntu2404": "d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba",
}


def stage_inference(bundle: Path, training_root: Path, accelerator: str) -> list[str]:
    if accelerator not in CUDA_ARCHITECTURES:
        raise ValueError("This GPU has no prepared native CUDA inference build profile")
    source = training_root.parent / "vla_cpp"
    if not (source / "policykit/cloud_inference.py").is_file():
        raise ValueError("The bundled cloud inference worker is missing")
    common = stage_native_worker(bundle, training_root)
    shutil.copyfile(
        source / "scripts/instrument_cuda_bench.py", bundle / "instrument_cuda_bench.py"
    )
    return [
        "python3 -m uv pip install --python .venv/bin/python "
        "numpy==2.2.6 PyYAML==6.0.3 cmake==4.1.3 psutil==7.2.2",
        *common,
        # CUDA packages are installed on the isolated VM, never on the application host.
        ". /etc/os-release",
        'firebird_cuda_distro="${ID}${VERSION_ID//./}"',
        'case "$firebird_cuda_distro" in '
        f"ubuntu2204) firebird_keyring_sha={KEYRING_SHA256['ubuntu2204']} ;; "
        f"ubuntu2404) firebird_keyring_sha={KEYRING_SHA256['ubuntu2404']} ;; "
        '*) echo "Native CUDA inference requires Ubuntu 22.04 or 24.04" >&2; exit 1 ;; esac',
        "curl --fail --location --retry 3 "
        '"https://developer.download.nvidia.com/compute/cuda/repos/${firebird_cuda_distro}/x86_64/'
        'cuda-keyring_1.1-1_all.deb" -o cuda-keyring.deb',
        'echo "$firebird_keyring_sha  cuda-keyring.deb" | sha256sum --check -',
        "sudo dpkg -i cuda-keyring.deb",
        "sudo apt-get update -qq",
        "sudo apt-get install -y --no-install-recommends "
        "cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8",
        "export PATH=/usr/local/cuda-12.8/bin:$PATH",
        "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}",
        ".venv/bin/python instrument_cuda_bench.py vendor/vla.cpp/src/serving/vla-bench.cpp",
        ".venv/bin/cmake -S vendor/vla.cpp -B native-inference "
        "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_METAL=OFF "
        "-DGGML_NATIVE=OFF -DVLA_BUILD_TESTS=ON -DLLAMA_CURL=OFF "
        "-DCUDAToolkit_ROOT=/usr/local/cuda-12.8 "
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc "
        f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES[accelerator]}",
        ".venv/bin/cmake --build native-inference --target vla-bench vla_predict_check -j2",
    ]
