"""Fresh-process CUDA engine screen of the immutable CPU-screen artifacts."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from .worker import sha256 as file_hash
from .cuda_common import parse_actions
from .cuda_common import hardware_fingerprint


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def samples_from_log(text: str, count: int) -> list[float]:
    if 'backend = CUDA' not in text or 'falling back to CPU' in text:
        raise ValueError('CUDA execution was not established')
    match = re.search(r'^vla-bench: samples_ms=([^\n]+)', text, re.M)
    values = [] if not match else [float(x) for x in match[1].split(',')]
    if len(values) != count or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('Missing, nonfinite or nonpositive latency samples')
    return values


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


def report(payload: dict, output: Path) -> None:
    lines = ['# RTX 3070 SmolVLA CUDA engine screen', '',
             '**Deployment decision: pending closed-loop quality and package validation.**', '',
             'Two synthetic 512px camera images, fixed state/language/noise, four CPU threads. '
             'Three warmups per fresh process. Latencies include the native predict call, '
             'not camera capture, external preprocessing or robot I/O.', '',
             '| Candidate | Status | GGUF MiB | GPU weight buffer MiB | Pooled p50 ms | Pooled p95 ms | Timed calls |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in payload['results']:
        samples = [x for batch in row['batches'] for x in batch['samples_ms']]
        row['timed_calls'] = len(samples)
        row['p50_ms'] = percentile(samples, 50) if samples else None
        row['p95_ms'] = percentile(samples, 95) if samples else None
        keys = ('preset', 'status', 'size_mib', 'weight_buffer_mib', 'p50_ms', 'p95_ms', 'timed_calls')
        lines.append('| ' + ' | '.join(str(row.get(k, '—')) for k in keys) + ' |')
    lines.extend(['', 'Raw per-call samples and separate batch percentiles are in results.json. '
                  'GPU telemetry measures total device use, including other applications; '
                  'RSS measures host memory. Synthetic action differences are diagnostics, not task success.',
                  '', '## Failures', ''])
    for row in payload['results']:
        if row.get('error'):
            lines.append(f"- {row['preset']}: {row['error']}")
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='smolvla-cuda-v1')
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in {'.', '..'} or min(args.reps, args.rounds) < 1:
        parser.error('Use a single run-directory name and positive counts')
    root = Path.cwd()
    output = root / 'artifacts/cuda/runs' / args.run
    output.mkdir(parents=True, exist_ok=False)
    prior = root / 'artifacts/docker/runs/smolvla-screen-v1'
    old = json.loads((prior / 'results.json').read_text())
    build = root / 'artifacts/cuda/build'
    bench, probe = build / 'vla-bench', build / 'tests/vla_predict_check'
    payload = {
        'run_id': args.run, 'scope': 'CUDA synthetic engine diagnostics; no task-success claim',
        'hardware': hardware_fingerprint(), 'gpu': subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version,compute_cap',
            '--format=csv,noheader'], text=True).strip(),
        'source': old['source'], 'runtime_base_commit': old['runtime_commit'],
        'bench_sha256': file_hash(bench), 'probe_sha256': file_hash(probe),
        'runtime_patch_sha256': file_hash(root / 'artifacts/cuda/setup/runtime.patch'),
        'repetitions_per_batch': args.reps, 'rounds': args.rounds, 'warmups': 3,
        'task_success': None, 'results': [], 'run_order': [],
    }

    def save():
        report(payload, output)
        temporary = output / 'results.partial.json'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(output / 'results.json')

    baseline = None
    for original in old['results']:
        name = original['preset']
        model = prior / 'models' / Path(original['artifact']).name
        row = {'preset': name, 'status': 'verifying', 'artifact': str(model),
               'sha256': original['sha256'], 'size_mib': model.stat().st_size / 2**20,
               'task_success': None, 'batches': []}
        payload['results'].append(row)
        save()
        try:
            if file_hash(model) != row['sha256']:
                raise ValueError('Model hash differs from the original CPU screen')
            command = [str(probe), str(model)]
            row['action_command'] = command
            row['action_measurement'] = measure(command, output / f'{name}-actions.log')
            log = (output / f'{name}-actions.log').read_text()
            if 'backend = CUDA' not in log or 'falling back to CPU' in log:
                raise ValueError('Probe did not use CUDA')
            actions = parse_actions(log)
            if name == 'float_reference':
                baseline = actions
            if baseline is None:
                raise ValueError('A working floating CUDA reference is required')
            differences = [abs(a - b) for a, b in zip(actions, baseline)]
            row['action_mae'] = sum(differences) / len(differences)
            row['action_max_abs'] = max(differences)
            (output / f'{name}-actions.json').write_text(json.dumps(actions) + '\n')
            packed = re.search(r'packed resident matrices: lm=(\d+) vision=(\d+)', log)
            expected = (0, 0) if name == 'float_reference' else (224, 72 if 'vision' in name else 0)
            if packed is None or tuple(map(int, packed.groups())) != expected:
                raise ValueError(f'Packed resident matrix count differs from {expected}')
            row['packed_lm'], row['packed_vision'] = expected
            row['weight_buffer_mib'] = float(re.search(r'weight_buf = ([\d.]+) MiB', log)[1])
            row['status'] = 'actions_verified'
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save()
            if name == 'float_reference':
                raise
        save()
        print(name + ': ' + row['status'], flush=True)

    for round_id in range(args.rounds):
        rows = payload['results'] if round_id % 2 == 0 else list(reversed(payload['results']))
        for row in rows:
            if row['status'] == 'failed':
                continue
            name = row['preset']
            command = [str(bench), '--ckpt', row['artifact'], '--images', '2', '--size', '512',
                       '--warmup', '3', '--reps', str(args.reps), '--markdown']
            log = output / f'round{round_id}-{name}.log'
            payload['run_order'].append({'round': round_id, 'preset': name})
            try:
                batch = measure(command, log)
                text = log.read_text()
                samples = samples_from_log(text, args.reps)
                batch.update(round=round_id, command=command, samples_ms=samples,
                             p50_ms=percentile(samples, 50),
                             p95_ms=percentile(samples, 95),
                             load_ms=float(re.search(r'load_ms=([\d.]+)', text)[1]))
                row['batches'].append(batch)
                row['status'] = 'engine_verified' if len(row['batches']) == args.rounds else 'benchmarking'
                print(f"round {round_id} {name}: p50={batch['p50_ms']:.2f}ms", flush=True)
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save()
    print(output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
