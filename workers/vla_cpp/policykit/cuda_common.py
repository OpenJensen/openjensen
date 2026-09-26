"""Small, dependency-free helpers for the standalone CUDA experiments."""
import math
import os
import platform
import re
import subprocess


def hardware_fingerprint() -> dict[str, str]:
    fingerprint = {
        'system': platform.system(), 'release': platform.release(),
        'machine': platform.machine(), 'processor': platform.processor() or 'unknown',
        'cpu_count': str(os.cpu_count() or 'unknown'),
    }
    if platform.system() == 'Darwin':
        try:
            fingerprint['memory_bytes'] = subprocess.check_output(
                ['sysctl', '-n', 'hw.memsize'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return fingerprint


def parse_actions(text: str) -> list[float]:
    match = re.search(r'action_len=(\d+)\n', text)
    if match is None or int(match[1]) != 1600:
        raise ValueError('Expected a declared 50 x 32 action vector')
    values = [float(v) for v in text[match.end():].splitlines()[:1600]]
    if len(values) != 1600 or not all(map(math.isfinite, values)):
        raise ValueError('Expected 50 x 32 finite SmolVLA action values')
    # Check all padding for finiteness but score only real action channels.
    return [values[step*32+dim] for step in range(50) for dim in range(7)]
