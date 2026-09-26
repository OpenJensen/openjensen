'use client';

import { useState } from 'react';
import reference from '@/lib/benchmark-reference.json';

const hardware = {
  l4: {
    label: 'NVIDIA L4',
    scope: 'Timing was collected while the other GPU workload was paused. Peak VRAM covers the benchmark process and its descendants, sampled every 100 ms. Task success comes from separate paired runs on the same GPU.',
    finding: 'Packed-candidate task scores were not collected in this snapshot. Lower latency alone does not establish retained task quality.',
  },
  rtx3070: {
    label: 'NVIDIA RTX 3070',
    scope: 'Recorded on a shared WSL desktop. Process peak VRAM was not reliably collected. Each candidate used the same 20 LIBERO Spatial episodes.',
    finding: 'C++ Q4 completed 8/20 tasks versus the native BF16 reference’s 15/20. C++ Q8 matched the reference on all 20 episode outcomes.',
  },
} as const;

// This snapshot is extracted from the recorded JSON, not from application jobs.
// Pin source links so later benchmark collection cannot change its provenance.
const source = `https://github.com/sobhanb-eth/firebird-hackathon-codebase/tree/${reference.sourceCommit}/${reference.sourcePath}`;

export function BenchmarkReference() {
  const [target, setTarget] = useState<keyof typeof hardware>('l4');
  const selected = hardware[target];
  return <section className="benchmark-reference" aria-labelledby="benchmark-reference-title">
    <div className="reference-benchmark-heading">
      <div><p className="eyebrow">Reference results · 26 Sep 2026</p><h2 id="benchmark-reference-title">Recorded benchmark comparison</h2></div>
      <a className="text-link" href={`${source}/REPORT.md`} target="_blank" rel="noreferrer">View source report ↗</a>
    </div>
    <p className="muted">Previously recorded SmolVLA measurements from our benchmark hosts. These are reference examples; this project’s own results appear under Run diagnostics.</p>
    <label className="reference-hardware">Reference hardware
      <select value={target} onChange={event => setTarget(event.target.value as keyof typeof hardware)}>
        {Object.entries(hardware).map(([id, item]) => <option key={id} value={id}>{item.label}</option>)}
      </select>
    </label>
    <p id="benchmark-reference-scope" className="muted">{selected.scope}</p>
    <div className="table-scroll" role="region" aria-label="Reference benchmark measurements" tabIndex={0}>
      <table className="feature-table workflow-metrics" aria-describedby="benchmark-reference-scope benchmark-reference-method">
        <caption>{selected.label} · recorded reference results</caption>
        <thead><tr>
          <th scope="col">Candidate</th><th scope="col">Median fixture p50 (ms/chunk)</th><th scope="col">Cached startup (s)</th>
          {target === 'l4' && <th scope="col">Sampled process peak VRAM (MiB)</th>}
          <th scope="col">LIBERO task success</th>
        </tr></thead>
        <tbody>{reference[target].map(row => <tr key={row.candidate}>
          <th scope="row"><code>{row.candidate}</code></th>
          <td>{row.medianP50Ms.toFixed(2)}</td><td>{row.startupSeconds.toFixed(2)}</td>
          {target === 'l4' && <td>{row.processPeakMiB ?? 'Not recorded'}</td>}
          <td>{row.quality ? `${row.quality.succeeded}/${row.quality.total}` : 'Not in snapshot'}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <p className="reference-benchmark-finding">{selected.finding}</p>
    <details className="provenance">
      <summary>Measurement scope &amp; sources</summary>
      <p id="benchmark-reference-method">Latency is the median of 14 fixture p50 values for a complete 50 × 7 action chunk, including preprocessing and transfers, excluding simulator stepping. Cached startup excludes downloads, process launch and inference warmup. Twenty paired episodes are a small behavior-retention sample, not general robot acceptance.</p>
      <dl>
        <div><dt>Checkpoint</dt><dd><code>{reference.checkpoint}</code></dd></div>
        <div><dt>Checkpoint revision</dt><dd><code>{reference.checkpointRevision}</code></dd></div>
        <div><dt>Evidence snapshot</dt><dd><a className="text-link" href={source} target="_blank" rel="noreferrer"><code>{reference.sourceCommit.slice(0, 7)}</code> · raw JSON and checksums ↗</a></dd></div>
      </dl>
    </details>
  </section>;
}
