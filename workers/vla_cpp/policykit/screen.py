"""Measured CPU hypothesis screen. Never promotes an engine result to task quality."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .benchmark import file_hash, parse_bench
from .quantization import quantize
from .runtime import hardware_fingerprint


def render_report(payload, output):
    lines = ['# SmolVLA quantization hypothesis screen', '', payload['scope'], '',
             f"**Product decision: {payload['product_decision']}**", '',
             f"Each candidate attempts {payload['repetitions']} measured calls after 1 warmup, 2 cameras, 512px synthetic images, 4 CPU threads.", '',
             '| Candidate | Status | MiB | Quantize seconds | p50 ms | p90 ms |',
             '|---|---|---:|---:|---:|---:|']
    for result in payload['results']:
        lines.append('| ' + ' | '.join(str(result.get(k, '—')) for k in
                     ['preset','status','size_mib','quantize_seconds','p50_ms','p90_ms']) + ' |')
    lines += ['', '## Load failures', '']
    for result in payload['results']:
        if result.get('error'):
            lines.append(f"- `{result['preset']}`: {result.get('diagnostic', result['error'])}")
    lines += ['', '## Interpretation', '',
              'Engine-only results do not establish task success. Five samples are an exploratory screen, not a stable tail-latency estimate.',
              'The floating reference preserves source tensor precision; inspect its audit before calling it BF16.',
              'Action expert, action/state projections, embeddings, and multimodal projector retain source precision.',
              'Quantization time excludes checkpoint download and initial conversion. Latency excludes camera capture, preprocessing and robot I/O.',
              'Raw logs, tensor audits, exact commands, hashes and source revision accompany results.json.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='smolvla-screen-v1', help='Fresh evidence directory name')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if not args.run or args.run in {'.', '..'} or Path(args.run).name != args.run:
        parser.error('--run must be a single directory name')
    root = Path.cwd()
    lane = root / 'artifacts/docker'
    vendor = lane / 'vendor/vla.cpp'
    output = lane / 'runs' / args.run
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'Evidence directory already exists: {output}. Choose a fresh --run name.')
    models = output / 'models'
    models.mkdir(exist_ok=True)
    payload = {'run_id': output.name, 'hardware': hardware_fingerprint(),
               'scope': 'Docker CPU synthetic observation-to-action inference; no task-success evidence',
               'repetitions': args.reps, 'warmups': 1, 'images': 2, 'image_size': 512, 'cpu_threads': 4,
               'runtime_commit': subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip(),
               'source': json.loads((lane / 'sources/smolvla/policykit-source.json').read_text()),
               'product_decision': 'BLOCKED: closed-loop quality evaluation required', 'results': []}

    def save():
        (output / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
        render_report(payload, output)

    reference = models / 'smolvla-float.gguf'
    save()
    if not reference.exists():
        temporary = reference.with_suffix('.partial.gguf')
        with (output / 'conversion.log').open('w') as log:
            subprocess.run([sys.executable, str(vendor / 'scripts/convert_smolvla_to_gguf.py'),
                            '--ckpt', str(lane / 'sources/smolvla'), '--out', str(temporary)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        temporary.replace(reference)
    import gguf
    counts = Counter(t.tensor_type.name for t in gguf.GGUFReader(reference).tensors)
    (reference.with_suffix('.audit.json')).write_text(json.dumps({'type_counts': dict(counts)}, indent=2))
    candidates = [('float_reference', None, None), ('lm_q8', 'Q8_0', None),
                  ('lm_q4', 'Q4_0', None), ('lm_q8_vision_q8', 'Q8_0', 'Q8_0'),
                  ('lm_q4_vision_q8', 'Q4_0', 'Q8_0')]
    for name, language, vision in candidates:
        result = {'preset': name, 'status': 'running', 'task_success': None}
        payload['results'].append(result)
        save()
        print(f'Starting {name}', flush=True)
        try:
            path = reference if language is None else models / f'{name}.gguf'
            start = time.monotonic()
            if language:
                quantize(reference, path, vendor / 'scripts/quantize_gguf.py', language, vision)
            result.update(quantize_seconds=round(time.monotonic()-start, 2),
                          size_mib=round(path.stat().st_size / 2**20, 2), sha256=file_hash(path), artifact=str(path))
            command = [str(vendor / 'build/vla-bench'), '--ckpt', str(path), '--images', '2', '--size', '512',
                       '--warmup', '1', '--reps', str(args.reps), '--markdown']
            result['command'] = command
            save()
            with (output / f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True,
                               env={**os.environ, 'OMP_NUM_THREADS': '4', 'VLA_N_THREADS': '4'})
            p50, p90 = parse_bench((output / f'{name}.log').read_text())
            if p50 is None or p90 is None:
                raise RuntimeError('Missing measured latency in benchmark output')
            result.update(p50_ms=p50, p90_ms=p90, status='engine_measured')
        except Exception as exc:
            result.update(status='failed', error=str(exc))
            log_path = output / f'{name}.log'
            if log_path.exists():
                diagnostics = [line for line in log_path.read_text().splitlines()
                               if 'unsupported dtype' in line or 'model_load failed' in line]
                if diagnostics:
                    result['diagnostic'] = '; '.join(diagnostics)
        save()
        print(json.dumps(result), flush=True)
    if all(result['status'] == 'failed' for result in payload['results'][1:]):
        payload['product_decision'] = 'BLOCKED: every quantized candidate failed execution; no compressed policy qualifies.'
        save()
    print(output / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
