import type { ReactNode } from 'react';
import type { Tone } from './notice';

/** A short state label such as Connected, Not installed or Failed. */
export function StatusBadge({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`status-badge status-badge-${tone}`}>{children}</span>;
}
