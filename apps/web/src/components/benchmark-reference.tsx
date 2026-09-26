'use client';

import { useState } from 'react';
import reference from '@/lib/benchmark-reference.json';

// Recorded examples are pinned to evidence, never inserted into project jobs.
const source = `https://github.com/sobhanb-eth/firebird-hackathon-codebase/tree/${reference.sourceCommit}/${reference.sourcePath}`;
const worker = `https://github.com/sobhanb-eth/firebird-hackathon-codebase/blob/${reference.sourceCommit}/workers/benchmark_gpu/README.md`;
const number = (value: number | null) => value === null ? '—' : value.toFixed(2);

export function BenchmarkReference() {
  const [target, setTarget] = useState<keyof typeof reference.hosts>('dedicatedL4');
  const selected = reference.hosts[target];
  return <section className="benchmark-reference" aria-labelledby="benchmark-reference-title">
    <div className="reference-benchmark-heading">
      <div><p className="eyebrow">Reference results · {reference.recordedOn}</p><h2 id="benchmark-reference-title">Recorded benchmark comparison</h2></div>
      <a className="text-link" href={`${source}/${selected.reportPath}`} target="_blank" rel="noreferrer">View source report ↗</a>
    </div>
    <p className="muted">Measured SmolVLA reference examples. Your project’s own measurements appear above under Diagnostic runs.</p>
    <label className="reference-hardware">Reference hardware
      <select value={target} onChange={event => setTarget(event.target.value as keyof typeof reference.hosts)}>
        {Object.entries(reference.hosts).map(([id, item]) => <option key={id} value={id}>{item.label}</option>)}
      </select>
    </label>
    <p id="benchmark-reference-scope" className="muted">{selected.scope}</p>
    <div className="table-scroll" role="region" aria-label="Reference benchmark measurements" tabIndex={0}>
      <table className="feature-table workflow-metrics" aria-describedby="benchmark-reference-scope benchmark-reference-method">
        <caption>{selected.label} · recorded reference results</caption>
        <thead><tr>
          <th scope="col">Candidate</th><th scope="col">Median fixture p50 (ms/chunk)</th>
          {selected.fullStartup && <th scope="col">Process to first action (s)</th>}
          <th scope="col">Cached runtime initialization (s)</th>
          <th scope="col">Sampled process peak VRAM (MiB)</th><th scope="col">LIBERO task success</th>
          <th scope="col">Reference successes lost / new successes gained</th>
        </tr></thead>
        <tbody>{selected.rows.map(row => <tr key={row.candidate}>
          <th scope="row"><code>{row.candidate}</code></th>
          <td>{number(row.medianP50Ms)}</td>
          {selected.fullStartup && <td>{number(row.firstActionSeconds)}</td>}
          <td>{number(row.runtimeInitSeconds)}</td><td>{row.processPeakMiB ?? '—'}</td>
          <td>{row.quality ? `${row.quality.succeeded}/${row.quality.total}` : '—'}</td>
          <td>{row.quality ? `${row.quality.lost} lost / ${row.quality.gained} gained` : '—'}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <p className="reference-benchmark-finding">{selected.finding}</p>
    <p className="muted">On L4, custom vLLM and TensorRT-LLM run the complete policy using native PyTorch kernels inside their workers. They showed no latency advantage over their own native controls. Optimized engine kernels and engine-native INT8/INT4 remain unvalidated.</p>
    <details className="provenance">
      <summary>Measurement scope &amp; reproduction</summary>
      <p id="benchmark-reference-method">Latency is the median of 14 fixture p50 values for a complete 50 × 7 action chunk, including preprocessing and transfers, excluding simulator stepping. Cached runtime initialization excludes preceding backend imports and first inference. Process-to-first-action includes both; model assets are cached. VRAM sampling may miss short peaks. Twenty paired episodes are a small behavior-retention sample, not general robot acceptance.</p>
      <p>Project diagnostics evaluate one native policy with your configured settings. The application's simulator uses LIBERO Object; the reference comparison uses LIBERO Spatial. To reproduce this full ten-candidate comparison, use the <a className="text-link" href={worker} target="_blank" rel="noreferrer">GPU benchmark setup and commands ↗</a>. Its fixtures, timing and memory scope also differ from the application's engine checks.</p>
      <dl>
        <div><dt>Checkpoint</dt><dd><code>{reference.checkpoint}</code></dd></div>
        <div><dt>Checkpoint revision</dt><dd><code>{reference.checkpointRevision}</code></dd></div>
        <div><dt>Evidence snapshot</dt><dd><a className="text-link" href={source} target="_blank" rel="noreferrer"><code>{reference.sourceCommit.slice(0, 7)}</code> · raw JSON and checksums ↗</a></dd></div>
      </dl>
    </details>
  </section>;
}
