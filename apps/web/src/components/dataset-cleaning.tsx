'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { Icon } from '@/components/icon';
import { Segmented } from '@/components/ui/choice';
import { ErrorNotice } from '@/components/workspace-ui';
import type { LibraryDataset } from '@/lib/dataset-library';
import {
  datasetPipelines,
  episodeDisputed,
  episodeVerdict,
  pipelineActive,
  type DatasetPipeline,
  type PipelineEpisode,
} from '@/lib/dataset-pipelines';

type Filter = 'all' | 'check' | 'kept' | 'dropped';

/**
 * Find bad demonstrations, decide episode by episode, and save the rest as a new dataset.
 * Checking reads the gripper trace for missed grasps and, with an OpenAI key, asks a vision
 * model whether each episode reached its goal. Both are suggestions; the operator decides.
 */
export function DatasetCleaning({ entry }: { entry: LibraryDataset }) {
  const project = entry.project_id;
  const client = useQueryClient();
  const pipelines = useQuery({
    queryKey: ['dataset-pipelines', project],
    queryFn: ({ signal }) => datasetPipelines.list(project, signal),
    refetchInterval: (query) =>
      query.state.data?.pipelines.some((item) => item.dataset.id === entry.id && pipelineActive(item)) ? 2000 : false,
  });
  const latest = pipelines.data?.pipelines.find((item) => item.dataset.id === entry.id) ?? null;
  const check = useMutation({
    mutationFn: () => datasetPipelines.check(project, entry.id),
    onSuccess: () => client.invalidateQueries({ queryKey: ['dataset-pipelines', project] }),
  });

  return (
    <section className="panel dataset-cleaning" aria-labelledby="cleaning-title">
      <div className="dataset-cleaning-head">
        <div>
          <h3 id="cleaning-title">Clean this dataset</h3>
          <p>Find failed and retried demonstrations, choose what to keep, save the rest as a new dataset.</p>
        </div>
        {!pipelineActive(latest) && (
          <button
            className={latest ? 'secondary-button' : 'primary-button'}
            onClick={() => check.mutate()}
            disabled={check.isPending || entry.status !== 'ready'}
          >
            <Icon name="spark" size={16} />
            {latest ? 'Check again' : 'Check episodes'}
          </button>
        )}
      </div>
      <ErrorNotice error={pipelines.error ?? check.error} />
      {latest && <PipelineView key={latest.id} pipeline={latest} entry={entry} />}
    </section>
  );
}

/** "grasp 37/50 · lift 35/50 …": how many episodes reached each stage, in order. */
function stageRates(summary: Record<string, unknown> | undefined, total: number) {
  const stages = (summary?.waypoints as { in_order?: Record<string, number> } | undefined)?.in_order;
  if (!stages || !total) return null;
  return (
    <span className="dataset-cleaning-cost">
      {Object.entries(stages)
        .map(([name, count]) => `${name} ${count}/${total}`)
        .join(' · ')}
    </span>
  );
}

function PipelineView({ pipeline, entry }: { pipeline: DatasetPipeline; entry: LibraryDataset }) {
  const project = entry.project_id;
  const client = useQueryClient();
  const active = pipelineActive(pipeline);
  const episodes = useQuery({
    queryKey: ['dataset-pipeline-episodes', project, pipeline.id, pipeline.updated_at ?? pipeline.status],
    queryFn: ({ signal }) => datasetPipelines.episodes(project, pipeline.id, signal),
    enabled: !active,
  });
  const [filter, setFilter] = useState<Filter>('all');
  const [copies, setCopies] = useState(0);
  const [name, setName] = useState(`${entry.name}-cleaned`);
  const [decisions, setDecisions] = useState<Record<string, boolean>>({});
  const review = useMutation({
    mutationFn: (changes: Record<string, boolean>) => datasetPipelines.review(project, pipeline.id, changes),
  });
  const save = useMutation({
    mutationFn: async () => {
      if (Object.keys(decisions).length) await datasetPipelines.review(project, pipeline.id, decisions);
      return datasetPipelines.save(project, pipeline.id, name.trim(), copies);
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ['dataset-pipelines', project] }),
  });
  const resume = useMutation({
    mutationFn: () => datasetPipelines.resume(project, pipeline.id),
    onSuccess: () => client.invalidateQueries({ queryKey: ['dataset-pipelines', project] }),
  });

  const rows = useMemo(
    () =>
      (episodes.data?.episodes ?? [])
        .filter((row) => row.origin === 'recorded')
        .map((row) => ({ ...row, keep: decisions[row.id] ?? row.keep })),
    [episodes.data, decisions],
  );
  const kept = rows.filter((row) => row.keep).length;
  const shown = rows.filter((row) =>
    filter === 'kept'
      ? row.keep
      : filter === 'dropped'
        ? !row.keep
        : filter === 'check'
          ? episodeDisputed(row) || row.keep !== row.suggested
          : true,
  );
  const vision = pipeline.results.find((_, index) => pipeline.steps[index]?.processor === 'vision_review')?.summary;
  const counts = rows.reduce<Record<string, number>>((total, row) => {
    const label = episodeVerdict(row).label;
    total[label] = (total[label] ?? 0) + 1;
    return total;
  }, {});

  function toggle(row: PipelineEpisode & { keep: boolean }) {
    const change = { [row.id]: !row.keep };
    setDecisions((current) => ({ ...current, ...change }));
    review.mutate(change);
  }

  if (active)
    return (
      <div className="dataset-cleaning-progress" role="status">
        <span className="spinner" aria-hidden="true" />
        {pipeline.message}
      </div>
    );
  if (pipeline.status === 'failed' || pipeline.status === 'interrupted')
    return (
      <div className="warning-box" role="alert">
        <p>{pipeline.message}</p>
        <button className="secondary-button" onClick={() => resume.mutate()} disabled={resume.isPending}>
          Continue
        </button>
      </div>
    );
  return (
    <>
      <div className="dataset-cleaning-summary">
        {Object.entries(counts).map(([label, count]) => (
          <span
            key={label}
            className={`verdict-chip verdict-${label === 'Clean' ? 'good' : label === 'Failed' ? 'bad' : 'warn'}`}
          >
            {count} {label.toLowerCase()}
          </span>
        ))}
        {typeof vision?.cost_usd === 'number' && (
          <span className="dataset-cleaning-cost">
            Vision review {String(vision.model)} · ${vision.cost_usd.toFixed(2)}
          </span>
        )}
        {typeof vision?.skipped === 'string' && <span className="dataset-cleaning-cost">{vision.skipped}</span>}
        {stageRates(vision, rows.length)}
      </div>
      <Segmented<Filter>
        label="Episodes to show"
        value={filter}
        onChange={setFilter}
        options={[
          { value: 'all', label: `All ${rows.length}` },
          { value: 'check', label: 'Worth a look' },
          { value: 'kept', label: `Kept ${kept}` },
          { value: 'dropped', label: `Dropped ${rows.length - kept}` },
        ]}
      />
      <ErrorNotice error={episodes.error ?? review.error} />
      <ul className="episode-review-list">
        {shown.map((row) => {
          const verdict = episodeVerdict(row);
          return (
            <li key={row.id} className={`episode-review${row.keep ? '' : ' dropped'}`}>
              {row.strip && (
                <img
                  src={datasetPipelines.media(project, pipeline.id, row.strip)}
                  alt={`Episode ${row.episode_index} frames over time`}
                  loading="lazy"
                />
              )}
              <div className="episode-review-text">
                <strong>Episode {row.episode_index}</strong>
                <span className={`verdict-chip verdict-${verdict.tone}`}>{verdict.label}</span>
                {episodeDisputed(row) && <span className="verdict-chip verdict-warn">Checks disagree</span>}
                {row.waypoints && Object.keys(row.waypoints).length > 0 && (
                  <span className="waypoint-row" aria-label="Task stages reached">
                    {Object.entries(row.waypoints).map(([name, reached]) => (
                      <span key={name} className={`waypoint ${reached ? 'reached' : 'missed'}`}>
                        {reached ? '✓' : '✗'} {name}
                      </span>
                    ))}
                  </span>
                )}
                {row.vision?.notes && <p>{row.vision.notes}</p>}
                {!row.vision && row.gripper && (
                  <p>
                    {row.gripper.missed_grasps
                      ? `${row.gripper.missed_grasps} empty grasp${row.gripper.missed_grasps > 1 ? 's' : ''} first`
                      : 'Gripper held the object on the first close'}
                  </p>
                )}
              </div>
              <button
                type="button"
                className={row.keep ? 'secondary-button' : 'primary-button'}
                aria-pressed={!row.keep}
                onClick={() => toggle(row)}
              >
                {row.keep ? 'Drop' : 'Keep'}
              </button>
            </li>
          );
        })}
      </ul>
      <div className="dataset-cleaning-save">
        <label>
          New dataset name
          <input value={name} onChange={(event) => setName(event.target.value)} maxLength={100} />
        </label>
        <div>
          <span className="field-label">Augmented copies of each kept episode</span>
          <Segmented<number>
            label="Augmented copies of each kept episode"
            value={copies}
            onChange={setCopies}
            options={[0, 1, 2, 3].map((value) => ({ value, label: value ? `${value}` : 'None' }))}
          />
        </div>
        <button
          className="primary-button"
          onClick={() => save.mutate()}
          disabled={save.isPending || !kept || !name.trim()}
        >
          Save {kept * (copies + 1)} episodes as a new dataset
        </button>
      </div>
      <ErrorNotice error={save.error} />
      {pipeline.saved_datasets.map((saved) => (
        <p key={`${saved.dataset_id}-${saved.name}`} className="success-note" role="status">
          Saved {saved.name}: {saved.episodes} episodes{saved.augmented ? `, ${saved.augmented} of them augmented` : ''}
          . It is in your dataset list, ready to train.
        </p>
      ))}
    </>
  );
}
