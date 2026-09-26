"""Fixed cloud quantization setup; model weights never enter the dispatch bundle."""

import shutil
from pathlib import Path

WORKER_MODULE = "policykit.cloud_quantize"
VENDOR_COMMIT = "52439f7c6c362d7bee218b400b9080cc32d75cc3"
CPPZMQ_SHA256 = "1f8b641161dcf12641ae4951c2c49552de425be88d66c988ec5e85046f1320f6"


def stage_quantization(bundle: Path, training_root: Path) -> list[str]:
    """Stage source only and return setup lines to append after trainer installation."""
    source = training_root.parent / "vla_cpp"
    package = source / "policykit"
    if not (package / "cloud_quantize.py").is_file():
        raise ValueError("The bundled quantization worker is missing on the application server")
    destination = bundle / "quantization-worker"
    destination.mkdir()
    shutil.copytree(
        package,
        destination / "policykit",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(source / "pyproject.toml", destination / "pyproject.toml")
    return [
        # The pinned root CMake configures its serving targets unconditionally,
        # so their Protobuf/ZeroMQ headers are required even for the CPU probe.
        "sudo apt-get install -y build-essential cmake git curl pkg-config "
        "libprotobuf-dev protobuf-compiler libzmq3-dev "
        "libssl-dev libcurl4-openssl-dev",
        # GCP's image omits the apt repository containing cppzmq-dev. Use the
        # same pinned upstream header as our established Ubuntu CUDA image.
        "curl --fail --location --retry 3 "
        "https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp -o cppzmq.hpp",
        "echo '" + CPPZMQ_SHA256 + "  cppzmq.hpp' | sha256sum --check -",
        "sudo install -m 0644 cppzmq.hpp /usr/local/include/zmq.hpp",
        "python3 -m uv pip install --python .venv/bin/python gguf==0.17.1",
        "python3 -m uv pip install --python .venv/bin/python "
        "--no-deps --no-build-isolation -e quantization-worker",
        "git init vendor/vla.cpp",
        "git -C vendor/vla.cpp fetch --depth 1 "
        "https://github.com/VinRobotics/vla.cpp.git " + VENDOR_COMMIT,
        "git -C vendor/vla.cpp checkout --detach FETCH_HEAD",
        "git -C vendor/vla.cpp apply "
        '"$PWD/quantization-worker/policykit/patches/vla-cpp-smolvla-packed.patch"',
        ".venv/bin/python -m policykit.worker --describe-runtime vendor/vla.cpp",
        ".venv/bin/cmake -S vendor/vla.cpp -B native-smoke "
        "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF "
        "-DGGML_NATIVE=OFF -DVLA_BUILD_TESTS=ON -DLLAMA_CURL=OFF",
        ".venv/bin/cmake --build native-smoke --target vla_predict_check -j2",
    ]
