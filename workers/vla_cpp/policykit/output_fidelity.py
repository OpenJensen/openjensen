"""Compare paired VLA actions; baseline agreement is distinct from task accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(values, name):
    if not isinstance(values, list) or not values or not isinstance(values[0], list) or not values[0]:
        raise ValueError(f'{name} must be a nonempty [steps, real_action_channels] array')
    width = len(values[0])
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError(f'{name} has a ragged shape')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for row in values for v in row):
        raise ValueError(f'{name} must contain finite numeric actions')
    return values


def _summary(reference, candidate, atol, rtol):
    errors = [abs(a-b) for a, b in zip(reference, candidate)]
    tolerance = [abs(a-b) <= atol + rtol * abs(a) for a, b in zip(reference, candidate)]
    return {'mae': sum(errors)/len(errors),
            'rmse': math.sqrt(sum(e*e for e in errors)/len(errors)),
            'max_abs_error': max(errors),
            'p95_abs_error': sorted(errors)[math.ceil(.95*len(errors))-1],
            'exact_match_fraction': sum(a == b for a, b in zip(reference, candidate))/len(errors),
            'within_tolerance_fraction': sum(tolerance)/len(errors),
            'all_within_tolerance': all(tolerance)}


def compare_actions(reference, candidate, *, atol=0.001, rtol=0.01, targets=None):
    """Compare real channels in matching units; callers remove padding first.

    Scalar atol applies to all channels; pass a list for channel-specific units.
    Tolerance is abs(candidate-reference) <= atol[channel] + rtol*abs(reference).
    Default tolerances are diagnostics, not calibrated robot acceptance limits.
    """
    reference = _matrix(reference, 'reference')
    candidate = _matrix(candidate, 'candidate')
    channels = len(reference[0])
    if len(reference) != len(candidate) or channels != len(candidate[0]):
        raise ValueError('Reference and candidate action shapes differ')
    limits = list(atol) if isinstance(atol, (list, tuple)) else [atol]*channels
    if len(limits) != channels or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                    or not math.isfinite(v) or v < 0 for v in [*limits, rtol]):
        raise ValueError('Tolerances must be finite, nonnegative, and match channel count')
    per_channel = [_summary([row[d] for row in reference], [row[d] for row in candidate], limits[d], rtol)
                   for d in range(channels)]
    flat_ref = [v for row in reference for v in row]
    flat_candidate = [v for row in candidate for v in row]
    aggregate = _summary(flat_ref, flat_candidate, 0, 0)
    aggregate['within_tolerance_fraction'] = sum(m['within_tolerance_fraction'] for m in per_channel)/channels
    aggregate['all_within_tolerance'] = all(m['all_within_tolerance'] for m in per_channel)
    aggregate['all_exact'] = flat_ref == flat_candidate
    aggregate['steps_within_tolerance_fraction'] = sum(
        all(abs(a-b) <= limits[d]+rtol*abs(a) for d,(a,b) in enumerate(zip(ref, cand)))
        for ref,cand in zip(reference,candidate))/len(reference)
    result = {'steps':len(reference), 'real_action_channels':channels, 'values':len(flat_ref),
              'atol_per_channel':limits, 'rtol':rtol, 'agreement':aggregate,
              'per_channel':per_channel, 'target_error':None}
    if targets is not None:
        targets = _matrix(targets, 'targets')
        if len(targets) != len(reference) or len(targets[0]) != channels:
            raise ValueError('Target and action shapes differ')
        flat_targets = [v for row in targets for v in row]
        baseline_error = _summary(flat_targets, flat_ref, 0, 0)
        candidate_error = _summary(flat_targets, flat_candidate, 0, 0)
        result['target_error'] = {
            'baseline':{k:baseline_error[k] for k in ('mae','rmse','max_abs_error')},
            'candidate':{k:candidate_error[k] for k in ('mae','rmse','max_abs_error')},
            'mae_increase':candidate_error['mae']-baseline_error['mae'],
            'rmse_increase':candidate_error['rmse']-baseline_error['rmse'],
            'per_channel':[
                {label:{k:metrics[k] for k in ('mae','rmse','max_abs_error')}
                 for label, metrics in (
                     ('baseline', _summary([r[d] for r in targets], [r[d] for r in reference],0,0)),
                     ('candidate', _summary([r[d] for r in targets], [r[d] for r in candidate],0,0)))}
                for d in range(channels)]}
    return result


def from_run(path: Path, atol, rtol):
    run = json.loads(path.read_text())
    baseline = next(row for row in run['results'] if row['preset'] == 'float_reference')
    def actions(row):
        action_file = path.parent/f"{row['preset']}-actions.json"
        values = json.loads(action_file.read_text())
        if not isinstance(values, list) or len(values) != 350:
            raise ValueError('Expected saved SmolVLA 50 x 7 real actions with padding removed')
        return [values[start:start+7] for start in range(0,350,7)], digest(action_file)
    ref, ref_hash = actions(baseline)
    compare_actions(ref, ref, atol=atol, rtol=rtol)  # validate baseline and tolerances before candidates
    rows = []
    for candidate in run['results']:
        row = {'preset':candidate['preset'], 'artifact_sha256':candidate.get('sha256'),
               'quantize_seconds':candidate.get('quantize_seconds'),
               'p50_ms':candidate.get('p50_ms'), 'p90_ms':candidate.get('p90_ms')}
        try:
            values, action_hash = actions(candidate)
            row.update(status='compared', actions_sha256=action_hash,
                       output_fidelity=compare_actions(ref, values, atol=atol, rtol=rtol))
        except (OSError, ValueError) as exc:
            row.update(status='unavailable', error=str(exc))
        rows.append(row)
    return {'schema_version':1, 'source_results':str(path.resolve()), 'source_results_sha256':digest(path),
            'baseline_artifact_sha256':baseline['sha256'], 'baseline_actions_sha256':ref_hash,
            'scope':'One fixed synthetic observation and fixed noise; 50 steps x 7 real action channels. Comparison uses saved actions at printed precision (9 significant digits).',
            'task_success':None, 'ground_truth_accuracy':None, 'results':rows}


def from_pairs(path: Path, atol, rtol):
    """JSON cases carry matched inputs by ID, reference/candidate actions, optional targets."""
    payload = json.loads(path.read_text())
    cases = payload.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Paired dataset must contain a nonempty cases list')
    seen = set()
    results = []
    for case in cases:
        key = case['id']
        if not isinstance(key,str) or not key or key in seen:
            raise ValueError('Each paired case needs a unique nonempty string id')
        seen.add(key)
        results.append({'case_id':key, 'output_fidelity':compare_actions(
            case['reference'],case['candidate'],atol=atol,rtol=rtol,targets=case.get('targets'))})
    return {'schema_version':1, 'source_pairs_sha256':digest(path),
            'scope':'Caller-supplied paired actions; caller must match observations, preprocessing, state, prompt, noise, units and checkpoint lineage.',
            'cases':results, 'task_success':None}


def render(payload):
    lines = ['# VLA output agreement', '', payload['scope'], '',
             'Tolerance is a configurable numerical diagnostic, not a robot-task acceptance threshold. Exact equality is evaluated at the saved output precision. Aggregate errors mix channel units; inspect per-channel metrics in JSON.', '',
             '| Candidate/case | MAE | RMSE | Max error | Exact values | Within tolerance | All values within tolerance |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in payload.get('results', payload.get('cases', [])):
        name = row.get('preset', row.get('case_id'))
        fidelity = row.get('output_fidelity')
        if fidelity is None:
            lines.append(f'| {name} | unavailable | — | — | — | — | — |')
            continue
        m = fidelity['agreement']
        lines.append(f"| {name} | {m['mae']:.6g} | {m['rmse']:.6g} | {m['max_abs_error']:.6g} | {m['exact_match_fraction']:.2%} | {m['within_tolerance_fraction']:.2%} | {m['all_within_tolerance']} |")
    target_rows = [r for r in payload.get('results', payload.get('cases', []))
                   if (r.get('output_fidelity') or {}).get('target_error')]
    if target_rows:
        lines += ['', '## Error against supplied target actions', '',
                  '| Case | Baseline MAE | Candidate MAE | Baseline RMSE | Candidate RMSE |',
                  '|---|---:|---:|---:|---:|']
        for row in target_rows:
            m = row['output_fidelity']['target_error']
            lines.append(f"| {row.get('case_id', row.get('preset'))} | {m['baseline']['mae']:.6g} | {m['candidate']['mae']:.6g} | {m['baseline']['rmse']:.6g} | {m['candidate']['rmse']:.6g} |")
    first = next((row['output_fidelity'] for row in payload.get('results', payload.get('cases', [])) if row.get('output_fidelity')), None)
    if first:
        lines += ['', f"Absolute tolerance per channel: {first['atol_per_channel']}; relative tolerance: {first['rtol']}."]
    lines += ['', 'The JSON includes tolerances, p95 absolute error, per-channel errors, and the fraction of whole action steps within tolerance. Ground-truth action error is included only when targets are supplied. Task success requires separate closed-loop evaluation.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-results', type=Path)
    source.add_argument('--pairs', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--atol', type=float, nargs='+', default=[0.001])
    parser.add_argument('--rtol', type=float, default=0.01)
    args = parser.parse_args()
    atol = args.atol[0] if len(args.atol) == 1 else args.atol
    try:
        payload = from_run(args.run_results,atol,args.rtol) if args.run_results else from_pairs(args.pairs,atol,args.rtol)
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True,exist_ok=True)
    output = args.out/'output-fidelity.json'
    if output.resolve() == (args.run_results or args.pairs).resolve():
        parser.error('Output must not overwrite input evidence')
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    (args.out/'OUTPUT_FIDELITY.md').write_text(render(payload))
    print(args.out/'OUTPUT_FIDELITY.md')


if __name__ == '__main__':
    main()
