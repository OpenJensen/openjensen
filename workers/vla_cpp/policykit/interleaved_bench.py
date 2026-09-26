"""Balanced run order for CPU latency finalists; task quality stays unmeasured."""
import json
from pathlib import Path

from .benchmark import file_hash, parse_bench
from .packed_bench import measure


def main():
    lane = Path.cwd()/'artifacts/docker'
    parent = lane/'runs/smolvla-packed-v2'
    previous = json.loads((parent/'results.json').read_text())
    out = lane/'runs/smolvla-balanced-v3'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():
        raise RuntimeError('Preserve the previous run before repeating')
    binary = lane/'vendor/vla.cpp/build/vla-bench'
    if file_hash(binary) != previous['binary_sha256']:
        raise RuntimeError('Runtime changed since v2')
    indexed = {row['preset']:row for row in previous['results']}
    order = ['float_reference','lm_q8','lm_q4_vision_q8','lm_q4_vision_q8','lm_q8','float_reference']
    data = {'scope':'Synthetic CPU latency, balanced ABC-CBA order; no task success',
            'parent_run':'smolvla-packed-v2','binary_sha256':previous['binary_sha256'],
            'cpu_threads':4,'warmups':2,'repetitions_per_batch':5,'results':[]}
    def save():
        (out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
        lines = ['# Balanced CPU latency check','',data['scope'],'',
                 '| Batch | Candidate | p50 ms | p90 ms | Sampled peak RSS MiB |',
                 '|---|---|---:|---:|---:|']
        for row in data['results']:
            lines.append('| '+' | '.join(str(row.get(k,'—')) for k in
                ['batch','preset','p50_ms','p90_ms','sampled_peak_rss_mib'])+' |')
        lines += ['', 'Two independent processes per candidate, five timed calls after two warmups per process. This balances linear run-order effects but does not eliminate host-load or thermal noise. Individual batch percentiles are not pooled percentiles.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save()
    for i,name in enumerate(order):
        model = Path(indexed[name]['artifact'])
        if file_hash(model) != indexed[name]['sha256']:
            raise RuntimeError('Model changed since v2')
        command = [str(binary),'--ckpt',str(model),'--images','2','--size','512',
                   '--warmup','2','--reps','5','--markdown']
        log = out/f'{i+1}-{name}.log'
        print(f'Batch {i+1}: {name}',flush=True)
        row = {'batch':i+1,'preset':name,'command':command,'sha256':indexed[name]['sha256']}
        row.update(measure(command,log))
        row['p50_ms'],row['p90_ms'] = parse_bench(log.read_text())
        if row['p50_ms'] is None:
            raise RuntimeError('Missing latency')
        data['results'].append(row)
        save()
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
