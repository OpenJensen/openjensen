'use client';

import { useId, type ReactNode } from 'react';

export function TrainingHelp({ label, children, id }: { label: string; children: ReactNode; id?: string }) {
  const generated = useId();
  const description = id ?? `training-help-${generated}`;
  return <span className="training-help">
    <button type="button" aria-label={`Help for ${label}`} aria-describedby={description}>?</button>
    <span id={description} role="tooltip">{children}</span>
  </span>;
}
