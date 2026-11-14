'use client';

import { useEffect, useRef } from 'react';
import type { RobotLabState } from './robot-lab-state';

export function useCameraWorkspace(open: boolean, close: () => void) {
  const root = useRef<HTMLDivElement>(null);
  const closeRef = useRef(close);
  useEffect(() => {
    closeRef.current = close;
  }, [close]);
  useEffect(() => {
    if (!open || !root.current) return;
    const panel = root.current;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const exit = panel.querySelector<HTMLButtonElement>('[data-camera-workspace-exit]');
    exit?.focus();
    const focusable = () =>
      Array.from(
        panel.querySelectorAll<HTMLElement>(
          'button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary, [tabindex="0"]',
        ),
      ).filter((element) => element.getClientRects().length && !element.closest('[hidden]'));
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        closeRef.current();
      }
      if (event.key === 'Tab') {
        const items = focusable(),
          first = items[0],
          last = items.at(-1);
        if (event.shiftKey && (document.activeElement === first || !panel.contains(document.activeElement))) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !panel.contains(document.activeElement))) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    const onFocus = (event: FocusEvent) => {
      if (!panel.contains(event.target as Node)) exit?.focus();
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('focusin', onFocus);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('focusin', onFocus);
      previous?.focus();
    };
  }, [open]);
  return root;
}

export function CameraRecordingStatus({ lab }: { lab: RobotLabState }) {
  const recording = lab.recording;
  const collection = recording?.collection?.enabled ? recording.collection : null;
  const labels = {
    idle: 'Ready to record',
    countdown: 'Get ready',
    recording: 'Recording',
    saving: 'Saving episode',
    reset: 'Reset the scene',
    paused: 'Collection paused',
    complete: 'Collection complete',
  };
  return (
    <div className="robot-lab-camera-recording-status" aria-label="Camera view recording status">
      <strong>
        {recording?.error
          ? 'Recording needs attention'
          : recording?.saving
            ? 'Saving episode'
            : collection
              ? labels[collection.phase]
              : recording?.episode_active
                ? 'Recording'
                : recording?.active
                  ? 'Ready for next episode'
                  : 'Preview only — not recording'}
      </strong>
      {recording?.active && (
        <span>
          {recording.dataset_id} · {recording.episodes} episodes saved · {recording.frames} frames
        </span>
      )}
      {collection && ['countdown', 'recording', 'reset'].includes(collection.phase) && (
        <output aria-label="Camera view time remaining">{Math.max(0, Math.ceil(collection.remaining_s))}s</output>
      )}
    </div>
  );
}
