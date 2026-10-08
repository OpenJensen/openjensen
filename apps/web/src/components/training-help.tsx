'use client';

import { useId, useRef, useState, type ReactNode } from 'react';

export function TrainingHelp({ label, children, id }: { label: string; children: ReactNode; id?: string }) {
  const generated = useId();
  const anchor = useRef<HTMLSpanElement>(null);
  const [left, setLeft] = useState<number | undefined>();
  function position() {
    if (!anchor.current) return;
    const x = anchor.current.getBoundingClientRect().left;
    const width = Math.min(260, window.innerWidth * 0.65);
    setLeft(Math.min(Math.max(-20, 12 - x), window.innerWidth - 12 - width - x));
  }
  const description = id ?? `training-help-${generated}`;
  return <span className="training-help" ref={anchor} onMouseEnter={position} onFocusCapture={position}>
    <button type="button" aria-label={`Help for ${label}`} aria-describedby={description}>?</button>
    <span id={description} role="tooltip" style={left === undefined ? undefined : { left, right: "auto" }}>{children}</span>
  </span>;
}
