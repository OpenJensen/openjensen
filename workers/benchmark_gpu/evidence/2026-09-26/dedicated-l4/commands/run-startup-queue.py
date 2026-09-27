import json
import os
import pathlib
import subprocess
import time

root = pathlib.Path.home() / "smolvla-benchmark"
os.chdir(root)
scripts = root / "workers/benchmark_gpu/scripts"
out = root / "comparison"
while not (out / "complete.json").exists():
    time.sleep(5)
rows = []
backends = [
    "native-bf16",
    "cpp-bf16",
    "native-fp16",
    "cpp-Q8_0",
    "native-int8",
    "cpp-Q4_0",
    "native-nf4",
    "cpp-Q8_0-vision",
    "vllm-bf16",
    "trtllm-fp16",
    "native-bf16-repeat",
]
for name in backends:
    backend = name.removesuffix("-repeat")
    engine = (
        "vllm"
        if backend.startswith("vllm")
        else "trtllm"
        if backend.startswith("trtllm")
        else "native"
    )
    python = root / (".venv-" + engine) / "bin/python"
    libdir = subprocess.check_output(
        [str(python), "-c", 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'],
        text=True,
    ).strip()
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "OMP_NUM_THREADS": "4",
        "VLA_N_THREADS": "4",
        "VLLM_USE_V1": "0",
        "PYTHONPATH": str(scripts),
        "LIBERO_CONFIG_PATH": str(root / ("libero-config-" + engine)),
        "LD_LIBRARY_PATH": ":".join(
            [
                libdir,
                str(root / "vla.cpp/build-cuda"),
                str(root / "vla.cpp/build-cuda/bin"),
                str(root / "cuda-build/lib"),
            ]
        ),
    }
    command = [
        str(python),
        str(scripts / "measure_startup.py"),
        "--backend",
        backend,
        "--root",
        str(root),
        "--fixture",
        str(out / "fixtures/task0-init0-chunk0.npz"),
        "--output",
        str(out / "startup" / name),
    ]
    started = time.time()
    with (out / ("startup-" + name + ".log")).open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=340)
    row = {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "wall_s": time.time() - started,
    }
    rows.append(row)
    (out / "startup-queue.json").write_text(json.dumps(rows, indent=2))
    print(row, flush=True)
    if result.returncode:
        raise RuntimeError(name + " startup measurement failed")
(out / "startup-complete.json").write_text(
    json.dumps({"completed": time.time(), "runs": len(rows)})
)
