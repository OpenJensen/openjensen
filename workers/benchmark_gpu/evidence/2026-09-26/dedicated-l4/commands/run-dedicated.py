import json
import os
import pathlib
import subprocess
import time

root = pathlib.Path.home() / "smolvla-benchmark"
os.chdir(root)
scripts = root / "workers/benchmark_gpu/scripts"
out = root / "comparison"
queue = json.loads((out / "queue.json").read_text()) if (out / "queue.json").exists() else []


def run(name, command, environment, timeout=1800):
    if any(record["name"] == name and record["returncode"] == 0 for record in queue):
        print("Already complete:", name, flush=True)
        return
    started = time.time()
    record = {"name": name, "command": list(map(str, command)), "started": started}
    with (out / (name + ".log")).open("w") as log:
        process = subprocess.run(
            record["command"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    record.update(returncode=process.returncode, wall_s=time.time() - started)
    queue.append(record)
    (out / "queue.json").write_text(json.dumps(queue, indent=2))
    print(record, flush=True)
    if process.returncode:
        raise RuntimeError(name + " failed; inspect its log before continuing")


for marker in (
    "native-installed",
    "vllm-installed",
    "trtllm-installed",
    "cpp-transferred",
    "models-transferred",
):
    while not (root / "setup" / marker).exists():
        time.sleep(5)

environments = {}
for engine in ("native", "vllm", "trtllm"):
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
    environments[engine] = (python, env)
    run(
        "prepare-" + engine,
        [python, scripts / "prepare_libero_assets.py", "--config-dir", env["LIBERO_CONFIG_PATH"]],
        env,
    )
    run(
        "environment-" + engine,
        [
            python,
            scripts / "record_environment.py",
            "--output",
            out / ("environment-" + engine + ".json"),
        ],
        env,
    )
python, env = environments["native"]
run("regression-tests", [python, "-m", "pytest", "-q", root / "workers/benchmark_gpu/tests"], env)

for engine in ("vllm", "trtllm"):
    python, env = environments[engine]
    run(
        "probe-" + engine,
        [
            python,
            scripts / ("probe_" + engine + "_smolvla.py"),
            "--output",
            out / ("probe-" + engine),
            "--fixtures",
            out / "fixtures",
        ],
        env,
    )

idle = []
for _ in range(10):
    idle.append(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    time.sleep(1)
(out / "idle-before-timing.json").write_text(
    json.dumps(
        {
            "samples": idle,
            "processes": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
        },
        indent=2,
    )
)
if any(int(sample.split(",")[0]) > 2 for sample in idle[-5:]):
    raise RuntimeError("GPU not idle before timing")

for engine, backends, repeat in [
    (
        "native",
        [
            "native-bf16",
            "cpp-bf16",
            "native-fp16",
            "cpp-Q8_0",
            "native-int8",
            "cpp-Q4_0",
            "native-nf4",
            "cpp-Q8_0-vision",
        ],
        "native-bf16",
    ),
    ("vllm", ["native-bf16", "vllm-bf16"], "native-bf16"),
    ("trtllm", ["native-fp16", "trtllm-fp16"], "native-fp16"),
]:
    python, env = environments[engine]
    run(
        "latency-" + engine,
        [
            python,
            scripts / "run_cross_stack_matrix.py",
            "--output",
            out / ("latency-" + engine),
            "--fixtures",
            out / "fixtures",
            "--samples",
            "5",
            "--backends",
            *backends,
            "--repeat-backend",
            repeat,
        ],
        env,
    )

for backend in (
    "native-bf16",
    "native-fp16",
    "native-int8",
    "native-nf4",
    "cpp-bf16",
    "cpp-Q8_0",
    "cpp-Q4_0",
    "cpp-Q8_0-vision",
    "vllm-bf16",
    "trtllm-fp16",
):
    engine = (
        "vllm"
        if backend.startswith("vllm")
        else "trtllm"
        if backend.startswith("trtllm")
        else "native"
    )
    python, env = environments[engine]
    run(
        "quality-" + backend,
        [
            python,
            scripts / "smolvla_cross_stack.py",
            "--backend",
            backend,
            "--output",
            out / ("quality-" + backend),
            "--fixtures",
            out / "fixtures",
            "--samples",
            "3",
            "--quality",
            "--task-ids",
            *map(str, range(10)),
            "--episodes",
            "2",
        ],
        env,
    )
(out / "complete.json").write_text(json.dumps({"completed": time.time(), "runs": len(queue)}))
