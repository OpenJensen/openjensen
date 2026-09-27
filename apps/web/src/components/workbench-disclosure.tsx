'use client';

import { useState, type ReactNode, type SyntheticEvent } from 'react';

/** Disclosure state belongs to the selected record, not changing poll results. */
export function WorkbenchDisclosure({ title, initiallyOpen = false, children }: { title: string; initiallyOpen?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(initiallyOpen);
  function update(event: SyntheticEvent<HTMLDetailsElement>) { setOpen(event.currentTarget.open); }
  return <details className="workbench-disclosure" open={open} onToggle={update}>
    <summary>{title}</summary>
    <div className="workbench-disclosure-body">{children}</div>
  </details>;
}
