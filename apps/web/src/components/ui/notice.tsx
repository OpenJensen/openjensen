import type { ReactNode } from 'react';

export type Tone = 'neutral' | 'info' | 'success' | 'warning' | 'danger';

/**
 * One message box for every page. Lead with a short title, put the next step in
 * the body, and keep raw technical output in `details`, collapsed by default.
 */
export function Notice({
  tone = 'info',
  title,
  children,
  details,
  actions,
  live,
  label,
}: {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
  /** Raw output such as driver or server errors, shown monospaced on request. */
  details?: string;
  actions?: ReactNode;
  /** Announce changes: `alert` for failures that need attention, `status` for progress. */
  live?: 'alert' | 'status';
  /** Accessible name, when several notices of one kind can appear together. */
  label?: string;
}) {
  return (
    <div className={`notice notice-${tone}`} role={live} aria-label={label}>
      {title && <strong className="notice-title">{title}</strong>}
      {children && <div className="notice-body">{children}</div>}
      {details && (
        <details className="notice-details">
          <summary>Details</summary>
          <pre>{details}</pre>
        </details>
      )}
      {actions && <div className="notice-actions">{actions}</div>}
    </div>
  );
}
