from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .config import BenchmarkConfig


class CommandError(RuntimeError):
    pass


def run(command: Iterable[str], *, cwd: Path | None = None, timeout: int | None = None) -> str:
    command = [str(part) for part in command]
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise CommandError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}")
    return completed.stdout


def hardware_fingerprint() -> dict[str, str]:
    fingerprint = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count() or "unknown"),
    }
    if platform.system() == "Darwin":
        try:
            fingerprint["memory_bytes"] = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return fingerprint


class VlaCpp:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        runtime = config.data["runtime"]
        self.source_dir = config.root / runtime["source_dir"]
        self.build_dir = config.root / runtime["build_dir"]
        self.cache_dir = config.root / runtime["cache_dir"]

    @property
    def cli(self) -> Path:
        return self.build_dir / "vla-cli"

    @property
    def bench(self) -> Path:
        return self.build_dir / "vla-bench"

    @property
    def server(self) -> Path:
        return self.build_dir / "vla-server"

    @property
    def libero_python(self) -> Path:
        return self.source_dir / "eval/sim/libero/libero_uv/.venv/bin/python"

    def ensure_prepared(self) -> None:
        if not self.source_dir.exists():
            print("[prepare] Cloning vla.cpp …", flush=True)
            self.source_dir.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--branch", self.config.data["runtime"]["tag"], "--depth", "1", self.config.data["runtime"]["repository"], str(self.source_dir)])
        self.apply_patches()
        self.build_dir.mkdir(parents=True, exist_ok=True)
        print("[prepare] Configuring vla.cpp …", flush=True)
        run(["cmake", "-S", str(self.source_dir), "-B", str(self.build_dir), "-DCMAKE_BUILD_TYPE=Release"])
        print("[prepare] Building vla.cpp (first build also fetches llama.cpp; this can take several minutes) …", flush=True)
        run(["cmake", "--build", str(self.build_dir), "--config", "Release", "-j"])
        print("[prepare] Installing conversion dependencies …", flush=True)
        run([sys.executable, "-m", "pip", "install", "-e", ".[convert]"], cwd=self.source_dir)
        if not self.libero_python.exists():
            print("[prepare] Bootstrapping the isolated LIBERO environment …", flush=True)
            run(["bash", "eval/sim/libero/setup_libero.sh"], cwd=self.source_dir)

    def apply_patches(self) -> None:
        for relative in self.config.data["runtime"].get("patches", []):
            patch = self.config.root / relative
            already = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                                     cwd=self.source_dir, capture_output=True)
            if already.returncode == 0:
                continue
            run(["git", "apply", "--check", str(patch)], cwd=self.source_dir)
            run(["git", "apply", str(patch)], cwd=self.source_dir)

    def quantize_command(self, source: Path, destination: Path, preset: dict[str, object]) -> list[str]:
        command = [sys.executable, str(Path(__file__).with_name("quantization.py")),
                   "--vendor-script", str(self.source_dir / "scripts/quantize_gguf.py"),
                   "--in", str(source), "--out", str(destination), "--type", str(preset["type"])]
        if preset.get("vision"):
            command.extend(["--vision-type", "Q8_0"])
        return command

    def converter_command(self, converter: str, checkpoint: Path, destination: Path) -> list[str]:
        return [sys.executable, str(self.source_dir / converter), "--ckpt", str(checkpoint), "--out", str(destination)]

    def bench_command(self, model_path: Path, repetitions: int) -> list[str]:
        return [str(self.bench), "--ckpt", str(model_path), "--images", "2", "--size", "512", "--reps", str(repetitions), "--markdown"]

    def check_tools(self) -> None:
        missing = [tool for tool in ("git", "cmake", "bash") if not shutil.which(tool)]
        if missing:
            raise RuntimeError(
                f"Missing required tools: {', '.join(missing)}. "
                "Run `UV_HTTP_TIMEOUT=300 uv sync --extra prepare` from the repository root."
            )

    def require_prepared(self) -> None:
        if not self.source_dir.exists() or not self.build_dir.exists():
            raise RuntimeError(
                "vla.cpp is not prepared. Run `uv run policykit prepare` successfully "
                "before quantizing a model."
            )

    def write_lock(self, lock: dict) -> Path:
        destination = self.config.artifacts / "prepared-lock.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        return destination
