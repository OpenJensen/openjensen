"""Summarize an existing run without modifying or rerunning its evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import comparisons
from .output_fidelity import digest


def build(payload: dict, source: Path, calls: int, fidelity: dict | None = None) -> dict:
    if fidelity is not None and fidelity.get('source_results_sha256') != digest(source):
        raise ValueError('Output fidelity does not match this results file; regenerate the comparison')
    fidelity_rows = {r['preset']:r for r in (fidelity or {}).get('results', [])}
    rows = []
    for candidate, comparison in zip(payload['results'], comparisons(payload['results'])):
        row = {**candidate, **comparison}
        if candidate['preset'] in fidelity_rows:
            paired = fidelity_rows[candidate['preset']]
            if paired.get('artifact_sha256') != candidate.get('sha256'):
                raise ValueError('Output fidelity artifact hash mismatch')
            row['output_fidelity'] = paired.get('output_fidelity')
        quantization = row.get('quantize_seconds')
        latency = row.get('p50_ms')
        measured = row.get('status') in {'engine_verified', 'engine_measured', 'engine_benchmarked', 'complete'}
        row['estimated_quantize_plus_inference_seconds'] = (
            quantization + calls * latency / 1000
            if measured and quantization is not None and latency is not None else None)
        rows.append(row)
    return {'source_results': str(source.resolve()), 'inference_calls': calls,
            'hardware': payload.get('hardware'), 'source': payload.get('source'),
            'scope': 'Quantization plus warm model inference; excludes download, conversion, loading, evaluation and robot I/O. Totals are estimates using p50, not measured batch durations.',
            'results': rows}


def add_control(payload: dict, original: dict, control: dict) -> None:
    for key in ('hardware', 'source', 'binary_sha256', 'patch_sha256'):
        if original.get(key) is None or original.get(key) != control.get(key):
            raise ValueError(f'Control run has missing or different {key}')
    def baseline(run):
        return next(r for r in run['results'] if r['preset'] == 'float_reference' and r.get('status') == 'engine_verified')
    first, repeated = baseline(original), baseline(control)
    if first.get('sha256') != repeated.get('sha256'):
        raise ValueError('Control run uses a different floating artifact')
    if not first.get('p50_ms') or not repeated.get('p50_ms'):
        raise ValueError('Control run lacks baseline latency')
    payload['baseline_control'] = {
        'initial_p50_ms': first['p50_ms'], 'repeated_p50_ms': repeated['p50_ms'],
        'change_pct': 100 * (repeated['p50_ms']/first['p50_ms']-1),
        'interpretation':'Run-order drift is a confounder. Candidate ratios are descriptive single-run comparisons, not established quantization speedups or slowdowns.'}


def render(payload: dict) -> str:
    def fmt(value, places=2):
        return '—' if value is None else f'{value:.{places}f}'
    lines = ['# Quantization performance and pipeline cost', '', payload['scope'], '',
             f"Evidence: `{payload['source_results']}`", '',
             f"Cost estimate: one quantization followed by {payload['inference_calls']:,} model calls.", '',
             '| Candidate | Status | Quantize s | Load ms | Inference p50 ms | p90 ms | Latency change | Speedup | Break-even calls | Estimated cost s | Task-success drop pp |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    control = payload.get('baseline_control')
    if control:
        lines[2:2] = [f"**Timing control:** repeating the floating baseline changed p50 from {control['initial_p50_ms']:.1f} to {control['repeated_p50_ms']:.1f} ms ({control['change_pct']:+.1f}%). {control['interpretation']}", '']
    for row in payload['results']:
        values = [row['preset'], row['status'], fmt(row.get('quantize_seconds')),
                  fmt(row.get('load_ms')), fmt(row.get('p50_ms')), fmt(row.get('p90_ms')),
                  fmt(row.get('latency_change_pct'))+'%' if row.get('latency_change_pct') is not None else '—',
                  fmt(row.get('speedup'))+'x' if row.get('speedup') is not None else '—',
                  fmt(row.get('quantization_break_even_calls'), 0),
                  fmt(row.get('estimated_quantize_plus_inference_seconds')),
                  fmt(row.get('success_drop_pp'))]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Output agreement with floating baseline', '',
              '| Candidate | MAE | RMSE | Maximum error | Exact values | Within tolerance |',
              '|---|---:|---:|---:|---:|---:|']
    for row in payload['results']:
        fidelity = row.get('output_fidelity')
        if not fidelity:
            lines.append(f"| {row['preset']} | — | — | — | — | — |")
            continue
        m = fidelity['agreement']
        lines.append(f"| {row['preset']} | {m['mae']:.6g} | {m['rmse']:.6g} | {m['max_abs_error']:.6g} | {m['exact_match_fraction']:.2%} | {m['within_tolerance_fraction']:.2%} |")
    first = next((r['output_fidelity'] for r in payload['results'] if r.get('output_fidelity')), None)
    if first:
        lines += ['', f"Absolute tolerance per channel: {first['atol_per_channel']}; relative tolerance: {first['rtol']}. These are numerical diagnostics, not calibrated robot acceptance limits. Errors use saved action precision; aggregate errors mix channel units. Per-channel results are in JSON."]
    lines += ['', 'Positive latency change means slower; speedup above 1 means faster. Break-even uses quantization time divided by median time saved per call. It is absent when inference is not faster or measurements are missing. Shared conversion and model loading are excluded from break-even.', '',
              'Task quality remains unmeasured until closed-loop evaluations complete on matching tasks and seeds. Synthetic action differences are numerical diagnostics, not percentages of robot performance lost. Twenty calls on one CPU run are exploratory; differences may include timing noise.', '',
              'For the full pipeline, add conversion (when not cached), model loading, calibration (if the method needs it), evaluation, and surrounding observation/action processing. Quantize once per new checkpoint and reuse the artifact for inference.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control-results', type=Path, help='A repeat of the same floating baseline to expose timing drift')
    parser.add_argument('--fidelity', type=Path, help='Matching output-fidelity.json, verified against the source results hash')
    parser.add_argument('--calls', type=int, default=1000)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error('--calls must be positive')
    try:
        fidelity = json.loads(args.fidelity.read_text()) if args.fidelity else None
        original = json.loads(args.results.read_text())
        payload = build(original, args.results, args.calls, fidelity)
        if args.control_results:
            add_control(payload, original, json.loads(args.control_results.read_text()))
    except (ValueError, StopIteration) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    for destination in (args.out/'performance.json', args.out/'PERFORMANCE.md'):
        if destination.resolve() == args.results.resolve():
            parser.error('Output must not overwrite source evidence')
    (args.out/'performance.json').write_text(json.dumps(payload, indent=2)+'\n')
    (args.out/'PERFORMANCE.md').write_text(render(payload))
    print(args.out/'PERFORMANCE.md')


if __name__ == '__main__':
    main()
