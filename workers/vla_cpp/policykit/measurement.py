"""Bounded process measurements shared by native policy application jobs."""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from pathlib import Path


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(command: list[str], log: Path, timeout: int = 600) -> dict:
    telemetry = log.with_suffix('.gpu.csv')
    before = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'
    ], text=True).strip()
    started = time.monotonic()
    peak_rss_kib = 0
    env = {**os.environ, 'VLA_N_THREADS': '4', 'OMP_NUM_THREADS': '4', 'VLA_IMG_SIZE': '512'}
    with log.open('w') as output, telemetry.open('w') as gpu:
        monitor = subprocess.Popen([
            'nvidia-smi', '-i', '0', '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits', '-lms', '100'
        ], stdout=gpu, stderr=subprocess.STDOUT)
        process = None
        try:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
            while process.poll() is None:
                try:
                    status = Path(f'/proc/{process.pid}/status').read_text()
                    sizes = re.findall(r'Vm(?:RSS|HWM):\s+(\d+)', status)
                    peak_rss_kib = max([peak_rss_kib, *map(int, sizes)])
                except FileNotFoundError:
                    pass
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f'Exceeded {timeout} seconds: {command[0]}')
                time.sleep(.05)
            if process.returncode:
                raise RuntimeError(f'Exit {process.returncode}; see {log.name}')
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    used = []
    for line in telemetry.read_text().splitlines():
        fields = line.split(',')
        try:
            used.append(float(fields[1]))
        except (IndexError, ValueError):
            continue
    peak = max(used) if used else None
    return {
        'wall_seconds': time.monotonic() - started,
        'sampled_peak_rss_mib': peak_rss_kib / 1024,
        'device_used_mib_before': float(before), 'sampled_peak_device_used_mib': peak,
        'sampled_device_increase_mib': None if peak is None else max(0, peak - float(before)),
        'gpu_telemetry': str(telemetry), 'gpu_samples': len(used),
        'memory_scope': 'Whole GPU, including desktop/other processes; 100ms sampling. Not per-process allocated VRAM.',
    }
