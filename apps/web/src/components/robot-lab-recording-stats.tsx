'use client';

import type { RobotStatus } from '@/lib/robot-api';

type Recording = NonNullable<RobotStatus['recording']>;
type Outcome = 'saved' | 'saving' | 'discarded' | 'failed';

const outcomeNames: Record<Outcome, string> = {
  saved: 'saved',
  saving: 'saving',
  discarded: 'discarded',
  failed: 'not saved',
};

export function formatSeconds(seconds: number) {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${String(Math.floor(seconds % 60)).padStart(2, '0')}s`;
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}m`;
}

/** A running episode's clock: 0:07.4, 1:02.0. */
export function formatClock(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest < 10 ? '0' : ''}${rest.toFixed(1)}`;
}

function formatBytes(bytes: number) {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`;
}

function formatRemaining(seconds: number) {
  if (seconds >= 2 * 3600) return `${Math.floor(seconds / 3600)} hours`;
  if (seconds >= 120) return `${Math.floor(seconds / 60)} minutes`;
  return 'less than 2 minutes';
}

/**
 * What an operator watches while collecting: how many episodes are saved, how
 * many tries it took, how long episodes run, and how much recording fits on disk.
 * Each try is a bar, coloured by what happened to it, as tall as it was long.
 */
export function RecordingStats({ recording, liveSeconds }: { recording: Recording; liveSeconds: number | null }) {
  const stats = recording.stats;
  if (!stats) return null;
  const saved = recording.episodes;
  const saving = recording.saving_episodes ?? 0;
  const average = saved ? stats.saved_seconds / saved : null;
  const recent = stats.recent;
  const longest = Math.max(1, liveSeconds ?? 0, ...recent.map((take) => take.seconds));
  const firstShown = stats.tries - recent.length + 1;
  const counts = recent.reduce<Record<Outcome, number>>(
    (total, take) => ({ ...total, [take.outcome]: total[take.outcome] + 1 }),
    { saved: 0, saving: 0, discarded: 0, failed: 0 },
  );
  const summary = (Object.keys(counts) as Outcome[])
    .filter((outcome) => counts[outcome])
    .map((outcome) => `${counts[outcome]} ${outcomeNames[outcome]}`)
    .join(', ');
  const free = stats.disk_free_bytes;
  return (
    <section className="robot-lab-stats" aria-label="Recording progress">
      <dl className="robot-lab-stats-tiles">
        <div className="robot-lab-stat saved">
          <dt>Saved episodes</dt>
          <dd>
            {saved}
            {saving > 0 && <small className="robot-lab-stat-saving">+{saving} saving</small>}
          </dd>
        </div>
        <div className="robot-lab-stat discarded">
          <dt>Discarded</dt>
          <dd>{stats.discarded}</dd>
        </div>
        {stats.failed > 0 && (
          <div className="robot-lab-stat failed">
            <dt>Not saved</dt>
            <dd>{stats.failed}</dd>
          </div>
        )}
        <div className="robot-lab-stat">
          <dt>Tries</dt>
          <dd>{stats.tries}</dd>
        </div>
        <div className="robot-lab-stat">
          <dt>Average length</dt>
          <dd>{average === null ? '–' : formatSeconds(average)}</dd>
        </div>
        <div className="robot-lab-stat">
          <dt>Saved time</dt>
          <dd>{formatSeconds(stats.saved_seconds)}</dd>
        </div>
      </dl>
      {(recent.length > 0 || liveSeconds !== null) && (
        <div
          className="robot-lab-stats-tries"
          role="img"
          aria-label={`Tries ${recent.length ? `${firstShown} to ${stats.tries}: ${summary}` : 'none yet'}${
            liveSeconds !== null ? ', one recording now' : ''
          }`}
        >
          {recent.map((take, offset) => (
            <span
              key={firstShown + offset}
              className={take.outcome}
              style={{ height: `${Math.max(10, (take.seconds / longest) * 100)}%` }}
              title={`Try ${firstShown + offset}: ${outcomeNames[take.outcome]}, ${formatSeconds(take.seconds)}`}
            />
          ))}
          {liveSeconds !== null && (
            <span className="live" style={{ height: `${Math.max(10, (liveSeconds / longest) * 100)}%` }} />
          )}
        </div>
      )}
      {free !== null && (
        <p className={`robot-lab-stats-disk${free < 5 * 1024 ** 3 ? ' low' : ''}`}>
          {formatBytes(free)} free on the robot computer
          {stats.bytes_per_second > 0 && ` · room for about ${formatRemaining(free / stats.bytes_per_second)}`}
        </p>
      )}
    </section>
  );
}
