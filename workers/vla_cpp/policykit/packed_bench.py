"""Recheck existing artifacts against the patched SmolVLA CPU runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from importlib.resources import files

from .benchmark import file_hash, parse_bench
from .runtime import hardware_fingerprint
from .output_fidelity import compare_actions


def runtime_patch_sha256():
    patch = files('policykit').joinpath('patches/vla-cpp-smolvla-packed.patch')
    return hashlib.sha256(patch.read_bytes()).hexdigest()


def measure(command, log, timeout=900):
    start = time.monotonic()
    peak_kib = 0
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'VLA_N_THREADS': '4', 'OMP_NUM_THREADS': '4', 'VLA_IMG_SIZE': '512'})
        try:
            while process.poll() is None:
                try:
                    status = Path(f'/proc/{process.pid}/status').read_text()
                    values = re.findall(r'Vm(?:RSS|HWM):\s+(\d+)', status)
                    peak_kib = max([peak_kib, *map(int, values)])
                except FileNotFoundError:
                    pass
                if time.monotonic()-start > timeout:
                    raise TimeoutError(f'Process exceeded {timeout}s')
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(f'Exit {process.returncode}; see {log.name}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return {'wall_seconds': round(time.monotonic()-start, 3),
            'sampled_peak_rss_mib': round(peak_kib/1024, 2)}


def parse_actions(text):
    match = re.search(r'action_len=(\d+)\n', text)
    if not match:
        raise ValueError('Missing action vector')
    values = [float(v) for v in text[match.end():].splitlines()[:int(match[1])]]
    if len(values) != 1600 or not all(map(math.isfinite, values)):
        raise ValueError('Expected 50 x 32 finite SmolVLA action values')
    # Only seven channels are real; do not dilute error with 25 padding channels.
    return [values[step*32+dim] for step in range(50) for dim in range(7)]


def report(payload, output):
    lines = ['# SmolVLA packed-weight runtime benchmark', '',
             '**Product decision: pending closed-loop task-success evaluation.**', '',
             f"Docker CPU, four inference threads, two synthetic 512px images, three warmups and {payload['repetitions']} timed calls per candidate. Latency excludes camera capture, preprocessing and robot I/O.", '',
             '| Candidate | Status | File MiB | Weight buffer MiB | Sampled peak RSS MiB | Load ms | p50 ms | p90 ms |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    keys = ['preset','status','size_mib','weight_buffer_mib','sampled_peak_rss_mib','load_ms','p50_ms','p90_ms']
    for row in payload['results']:
        lines.append('| ' + ' | '.join(str(row.get(k, '—')) for k in keys) + ' |')
    lines += ['', '## Numerical smoke check', '',
              'Fixed synthetic images, language, state and noise; 350 real action values per candidate. These errors are not task success or safety metrics. Channels have different meanings/units.', '',
              '| Candidate | LM packed matrices | Vision packed matrices | Mean absolute action difference | Maximum absolute difference |',
              '|---|---:|---:|---:|---:|']
    for row in payload['results']:
        lines.append('| ' + ' | '.join(str(row.get(k, '—')) for k in
            ['preset','packed_lm','packed_vision','action_mae','action_max_abs']) + ' |')
    lines += ['', 'BF16 regression: ' + str(payload.get('bf16_regression', 'pending')), '',
              'Packed matrix counts come from runtime allocations. Weight-buffer size is distinct from total process memory. RSS/HWM is sampled from Linux /proc every 50 ms; it is not GPU memory.',
              'The original artifacts and failed v1 results are preserved. Exact commands, artifact hashes, runtime base commit, patch hash, executable hash, action arrays and logs accompany this report.', '', '## Failures', '']
    for row in payload['results']:
        if row.get('error'):
            lines.append(f"- {row['preset']}: {row['error']}")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--run', default='smolvla-packed-v2')
    parser.add_argument('--reference-only', action='store_true', help='Repeat the reference to check run-order drift')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in {'.', '..'} or args.reps < 1:
        parser.error('Use a single directory name and a positive repetition count')
    lane = Path.cwd()/'artifacts/docker'
    vendor = lane/'vendor/vla.cpp'
    previous = lane/'runs/smolvla-screen-v1'
    output = lane/'runs'/args.run
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise RuntimeError('This run already exists; preserve it before starting another run')
    old = json.loads((previous/'results.json').read_text())
    payload = {'repetitions':args.reps, 'hardware':hardware_fingerprint(), 'source':old['source'],
               'runtime_base_commit':old['runtime_commit'], 'patch_sha256':runtime_patch_sha256(),
               'binary_sha256':file_hash(vendor/'build/vla-bench'), 'task_success':None, 'results':[]}
    # Record actual vendor changes, not only the intended patch.
    diff = subprocess.check_output(['git','-C',str(vendor),'diff','--','src/models/smolvla.cpp','src/serving/vla-bench.cpp'])
    (output/'runtime.patch').write_bytes(diff)
    if hashlib.sha256(diff).hexdigest() != payload['patch_sha256']:
        raise RuntimeError('Built source diff does not match the recorded patch')
    def save():
        (output/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
        report(payload,output)
    baseline = None
    for source_row in (old['results'][:1] if args.reference_only else old['results']):
        name = source_row['preset']
        path = previous/'models'/Path(source_row['artifact']).name
        row = {k:source_row[k] for k in ['preset','size_mib','sha256','quantize_seconds']}
        row.update(status='running', artifact=str(path))
        payload['results'].append(row)
        save()
        print('Starting '+name,flush=True)
        try:
            if file_hash(path) != row['sha256']:
                raise RuntimeError('Artifact hash changed since v1')
            actions_cmd = [str(vendor/'build/tests/vla_predict_check'),str(path)]
            row['action_command'] = actions_cmd
            measure(actions_cmd,output/f'{name}-actions.log')
            text = (output/f'{name}-actions.log').read_text()
            actions = parse_actions(text)
            (output/f'{name}-actions.json').write_text(json.dumps(actions)+'\n')
            if baseline is None:
                baseline = actions
                unpatched = parse_actions((lane/'unpatched-float-actions.log').read_text())
                if actions != unpatched:
                    raise RuntimeError('BF16 action regression against unpatched runtime')
                payload['bf16_regression'] = 'All 350 real action values identical to unpatched runtime at printed precision.'
            row['output_fidelity'] = compare_actions(
                [baseline[i:i+7] for i in range(0,350,7)],
                [actions[i:i+7] for i in range(0,350,7)])
            errors = [abs(a-b) for a,b in zip(actions,baseline)]
            row.update(action_mae=sum(errors)/len(errors), action_max_abs=max(errors),
                       action_mae_per_channel=[sum(errors[d::7])/50 for d in range(7)])
            packed = re.search(r'packed resident matrices: lm=(\d+) vision=(\d+)',text)
            if not packed:
                raise RuntimeError('Missing resident tensor audit')
            row.update(packed_lm=int(packed[1]),packed_vision=int(packed[2]))
            expected = (0,0) if name=='float_reference' else (224,72 if 'vision' in name else 0)
            if (row['packed_lm'],row['packed_vision']) != expected:
                raise RuntimeError(f'Resident tensor audit differs from {expected}')
            command = [str(vendor/'build/vla-bench'),'--ckpt',str(path),'--images','2','--size','512',
                       '--warmup','3','--reps',str(args.reps),'--markdown']
            row['benchmark_command'] = command
            row.update(measure(command,output/f'{name}-bench.log'))
            text = (output/f'{name}-bench.log').read_text()
            row['p50_ms'],row['p90_ms'] = parse_bench(text)
            if row['p50_ms'] is None:
                raise RuntimeError('Missing latency')
            row['weight_buffer_mib'] = float(re.search(r'weight_buf = ([\d.]+) MiB',text)[1])
            row['load_ms'] = float(re.search(r'load_ms=([\d.]+)',text)[1])
            row['status'] = 'engine_verified'
        except Exception as exc:
            row.update(status='failed',error=str(exc))
            if name=='float_reference':
                save()
                raise
        save()
        print(json.dumps(row),flush=True)
    print(output/'REPORT.md',flush=True)


if __name__ == '__main__':
    main()
