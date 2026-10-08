'use client';

import { useEffect, useRef, useState } from 'react';
import type { Job } from '@/lib/api';
import { pooledActionRmse, quantizationComparison, type QuantizationComparison as Comparison } from '@/lib/quantization-comparison';
import { TrainingHelp } from './training-help';
import './quantization-comparison.css';

const metric = (value: number) => value !== 0 && value < 0.0001 ? value.toExponential(3) : value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
const bytes = (value: number) => value >= 1024 ** 2 ? `${(value / 1024 ** 2).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB` : `${value.toLocaleString()} bytes`;

function ActionChart({ comparison }: { comparison: Comparison }) {
  const ref = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(240, entries[0].contentRect.width)));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const top = Math.max(...comparison.samples.map(sample => sample.rmse), 0.000001) * 1.2;
  const y = (value: number) => 172 - (value / top) * 146;
  const originalX = 66 + (width - 90) * .2;
  const packedX = 66 + (width - 90) * .8;
  return <figure ref={ref} className="quantization-action-chart">
    <svg viewBox={`0 0 ${width} 214`} role="img" aria-label="Action difference from original model on fixed inputs">
      {[0, .5, 1].map(fraction => <g key={fraction}>
        <line x1="62" x2={width - 20} y1={y(top * fraction)} y2={y(top * fraction)} className="quantization-chart-grid" />
        <text x="54" y={y(top * fraction) + 4} textAnchor="end">{metric(top * fraction)}</text>
      </g>)}
      <circle cx={originalX} cy={y(0)} r="4" className="quantization-chart-original"><title>Original compared with itself: RMSE 0</title></circle>
      {comparison.samples.map((sample, index) => <circle key={sample.label} cx={packedX + (index - (comparison.samples.length - 1) / 2) * 10} cy={y(sample.rmse)} r="4" className="quantization-chart-packed"><title>{sample.label}: action RMSE {metric(sample.rmse)}</title></circle>)}
      <text x={originalX} y="194" textAnchor="middle">Original</text>
      <text x={packedX} y="194" textAnchor="middle">Quantized</text>
      <text x={width / 2} y="211" textAnchor="middle">Action RMSE · lower is closer to original</text>
    </svg>
    <figcaption><span className="quantization-legend-original">Original · zero reference</span><span className="quantization-legend-packed">Quantized · {comparison.samples.length} fixed {comparison.samples.length === 1 ? 'input' : 'inputs'}</span></figcaption>
  </figure>;
}

export function QuantizationComparison({ job, reports = [] }: { job: Job; reports?: Record<string, unknown>[] }) {
  const comparison = quantizationComparison(job, reports);
  const active = job.status === 'queued' || job.status === 'running';
  if (!comparison) return <section className="quantization-comparison" aria-label="Comparison with original model">
    <h3>Compared with original</h3>
    <p className="quantization-comparison-empty" role="status">{active ? 'The paired prediction comparison appears after quantization and verification finish.' : job.status === 'succeeded' ? 'This job did not record a paired comparison. New supported quantization jobs compare the original and quantized predictions on identical fixed inputs.' : 'No completed paired comparison is available for this job.'}</p>
  </section>;
  const saved = (1 - comparison.packedBytes / comparison.sourceBytes) * 100;
  const largest = Math.max(comparison.sourceBytes, comparison.packedBytes);
  return <section className="quantization-comparison" aria-label="Comparison with original model">
    <header><h3>Compared with original</h3><TrainingHelp label="Quantization comparison" wide>{comparison.description} RMSE is the root mean square difference between the two model outputs, rather than loss against dataset actions. Original is zero by definition. These fixed-input checks do not measure held-out validation loss or robot task success. Stored size compares matching {comparison.sizeLabel.toLowerCase()}, not GPU memory or adapter-only checkpoint size.</TrainingHelp></header>
    <dl className="quantization-comparison-facts">
      <div><dt>Action RMSE</dt><dd>{metric(pooledActionRmse(comparison))}</dd><span>Difference from original</span></div>
      <div><dt>Largest action difference</dt><dd>{metric(Math.max(...comparison.samples.map(sample => sample.maximum)))}</dd><span>Native output coordinates</span></div>
      <div><dt>Stored size</dt><dd>{metric(Math.abs(saved))}% {saved >= 0 ? 'smaller' : 'larger'}</dd><span>{comparison.sizeLabel}</span></div>
    </dl>
    <ActionChart comparison={comparison} />
    <p className="quantization-comparison-scope">{comparison.description} Validation loss and robot task success were not measured.</p>
    <figure className="quantization-size-chart" aria-label="Original and quantized stored size">
      {[{ label: comparison.reference, value: comparison.sourceBytes, kind: 'original' }, { label: 'Quantized', value: comparison.packedBytes, kind: 'packed' }].map(row => <div key={row.kind}>
        <span>{row.label}</span><span className="quantization-size-track"><span className={`quantization-size-${row.kind}`} style={{ width: `${row.value / largest * 100}%` }} /></span><strong>{bytes(row.value)}</strong>
      </div>)}
      <figcaption>{comparison.sizeLabel} · stored size, rather than runtime memory.</figcaption>
    </figure>
  </section>;
}
