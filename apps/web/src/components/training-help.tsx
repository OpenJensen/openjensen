'use client';

import { useId, useRef, type ReactNode } from 'react';
import './training-help.css';
import { Icon } from './icon';

export function TrainingHelp({ label, children, id, wide = false }: { label: string; children: ReactNode; id?: string; wide?: boolean }) {
  const generated = useId();
  const anchor = useRef<HTMLSpanElement>(null);
  const tooltip = useRef<HTMLSpanElement>(null);
  function position() {
    if (!anchor.current || !tooltip.current) return;
    const x = anchor.current.getBoundingClientRect().left;
    // Mobile innerWidth can include the tooltip's own overflowing content.
    // Use the layout viewport and position synchronously before its first paint.
    const viewport = document.documentElement.clientWidth;
    const width = Math.min(wide ? 360 : 260, viewport * (wide ? 0.85 : 0.65));
    tooltip.current.style.left = `${Math.min(Math.max(-20, 12 - x), viewport - 12 - width - x)}px`;
    tooltip.current.style.right = "auto";
  }
  const description = id ?? `training-help-${generated}`;
  return <span className={`training-help${wide ? " wide" : ""}`} ref={anchor} onMouseEnter={position} onFocusCapture={position}>
    <button type="button" aria-label={`Help for ${label}`} aria-describedby={description}><Icon name="help" size={20} /></button>
    <span id={description} role="tooltip" ref={tooltip}>{children}</span>
  </span>;
}
