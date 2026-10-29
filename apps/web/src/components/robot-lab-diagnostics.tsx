'use client';

import { useState } from 'react';
import { robotApi, type RobotDiagnostics } from '@/lib/robot-api';

export function RobotLabDiagnostics({ diagnostics }: { diagnostics?: RobotDiagnostics }) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!diagnostics) return null;
  const last = diagnostics.last_stop;
  async function download() {
    setDownloading(true);
    setError(null);
    try {
      const result = await robotApi.diagnostics();
      const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = 'openjensen-robot-logs.json';
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not download robot logs.');
    } finally {
      setDownloading(false);
    }
  }
  return (
    <details className="robot-lab-diagnostics">
      <summary>
        Robot logs · {diagnostics.error ? 'Saving needs attention' : 'Always on'}
        {last ? ' · Last stop recorded' : ''}
      </summary>
      <p>
        Control starts, stops and recent joint readings are saved automatically on the robot computer. History survives
        reconnects and restarts.
      </p>
      {diagnostics.error && <p role="alert">{diagnostics.error}</p>}
      {last ? (
        <div className="robot-lab-log-stop">
          <strong>
            Last stop · <time dateTime={last.time}>{new Date(last.time).toLocaleString()}</time>
          </strong>
          <p>{last.reason}</p>
          {last.context?.fault?.source && <small>Encoder reading from the {last.context.fault.source} arm.</small>}
          {last.cleanup_errors && last.cleanup_errors.length > 0 && (
            <p role="alert">Motor stop acknowledgement failed: {last.cleanup_errors.join('; ')}</p>
          )}
        </div>
      ) : (
        <p>No stop has been recorded yet.</p>
      )}
      <ol aria-label="Recent robot events">
        {diagnostics.events
          .slice()
          .reverse()
          .map((event) => (
            <li key={event.id}>
              <time dateTime={event.time}>{new Date(event.time).toLocaleTimeString()}</time>
              {' · '}
              {event.reason ?? event.message ?? event.event.replaceAll('_', ' ')}
            </li>
          ))}
      </ol>
      <button className="text-button" onClick={() => void download()} disabled={downloading}>
        {downloading ? 'Downloading…' : 'Download robot logs'}
      </button>
      {error && <p role="alert">{error}</p>}
      <small>
        Includes the last 100 events and up to 120 recent readings. Older entries rotate within a{' '}
        {Math.round(diagnostics.retention_bytes / 1024 / 1024)} MiB disk limit. The latest stop is kept separately.
      </small>
    </details>
  );
}
