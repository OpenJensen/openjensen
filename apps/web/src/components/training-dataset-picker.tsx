import { useEffect, useId, useRef, useState } from 'react';
import type { DatasetJob, DatasetProfile } from '@/lib/api';
import type { LibraryDataset } from '@/lib/dataset-library';
import { datasetStarters } from '@/lib/dataset-starters';
import { Icon } from './icon';
import './training-dataset-picker.css';

type InspectedDataset = DatasetJob & { result: DatasetProfile };
type Props = {
  datasets: InspectedDataset[];
  library: LibraryDataset[];
  selectedIds: string[];
  disabled: boolean;
  libraryLoading: boolean;
  libraryError?: string;
  onRetryLibrary: () => void;
  issue: (job: InspectedDataset, selectedIds: string[]) => string | null;
  onChange: (ids: string[]) => void;
  onImport: () => void;
};

function datasetName(job: InspectedDataset) {
  const starter = datasetStarters.find(
    (item) => item.repoId === job.result.repo_id && item.revision === job.result.revision,
  );
  return starter?.title ?? job.result.repo_id?.split('/').at(-1) ?? 'Local dataset';
}

function DatasetCard({
  job,
  selected,
  disabled,
  reason,
  onToggle,
  prefix,
}: {
  job: InspectedDataset;
  selected: boolean;
  disabled: boolean;
  reason: string | null;
  onToggle: () => void;
  prefix: string;
}) {
  const profile = job.result;
  const starter = datasetStarters.find((item) => item.repoId === profile.repo_id && item.revision === profile.revision);
  const cameras = Object.values(profile.features).filter((feature) =>
    ['image', 'video'].includes(String((feature as { dtype?: string }).dtype)),
  ).length;
  const reasonId = `${prefix}-${job.id}`;
  return (
    <label
      className={`training-dataset${disabled ? ' unavailable' : ''}`}
      title={profile.repo_id ?? 'Local dataset'}
      tabIndex={disabled && reason ? 0 : undefined}
      aria-describedby={reason ? reasonId : undefined}
    >
      <input
        type="checkbox"
        name={prefix === 'recent' ? 'training-dataset' : 'browse-training-dataset'}
        value={job.id}
        aria-label={profile.repo_id ?? 'Local dataset'}
        checked={selected}
        disabled={disabled}
        aria-describedby={reason ? reasonId : undefined}
        onChange={onToggle}
      />
      {starter ? (
        <img src={starter.poster} alt="" />
      ) : (
        <span className="training-dataset-icon">
          <Icon name="database" size={18} />
        </span>
      )}
      <span className="training-dataset-copy">
        <strong>{datasetName(job)}</strong>
        <small>
          {profile.total_episodes} episodes · {cameras} {cameras === 1 ? 'camera' : 'cameras'}
        </small>
        {prefix !== 'recent' && (
          <small>
            {profile.repo_id ?? 'Local training copy'} · {profile.format.replace('lerobot_', 'LeRobot ')}
          </small>
        )}
        {prefix !== 'recent' && reason && (
          <small className="training-picker-reason" id={reasonId}>
            {reason}
          </small>
        )}
      </span>
      {prefix === 'recent' && reason && (
        <span className="training-dataset-reason" id={reasonId} role="tooltip">
          {reason}
        </span>
      )}
    </label>
  );
}

export function TrainingDatasetPicker(props: Props) {
  const { datasets, library, selectedIds, disabled, issue, onChange } = props;
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<string[]>([]);
  const [search, setSearch] = useState('');
  const dialog = useRef<HTMLDialogElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const browseButton = useRef<HTMLButtonElement>(null);
  const id = useId();
  useEffect(() => {
    if (open) {
      dialog.current?.showModal();
      searchInput.current?.focus();
    }
  }, [open]);
  function closeBrowser() {
    dialog.current?.close();
    setOpen(false);
    browseButton.current?.focus();
  }
  // Include saved imports that have no completed inspection, without offering
  // them as trainable or duplicating an inspection of the same snapshot.
  const pending = library.filter(
    (entry) =>
      !datasets.some(
        (job) =>
          entry.job_id === job.id ||
          (entry.profile &&
            entry.profile.source === job.result.source &&
            entry.profile.repo_id === job.result.repo_id &&
            entry.profile.revision === job.result.revision),
      ),
  );
  const query = search.trim().toLocaleLowerCase();
  const matching = datasets.filter((job) =>
    `${datasetName(job)} ${job.result.repo_id ?? ''} ${job.result.source} ${job.result.format} ${job.result.robot_type ?? ''}`
      .toLocaleLowerCase()
      .includes(query),
  );
  const matchingPending = pending.filter((entry) =>
    `${entry.name} ${entry.source} ${entry.status}`.toLocaleLowerCase().includes(query),
  );
  const recent = datasets.slice(0, 3);
  const olderSelected = datasets.filter((job) => selectedIds.includes(job.id) && !recent.includes(job));
  const missing = draft.some((selected) => !datasets.some((job) => job.id === selected));
  function toggle(ids: string[], job: InspectedDataset) {
    return ids.includes(job.id) ? ids.filter((id) => id !== job.id) : [...ids, job.id];
  }
  function card(job: InspectedDataset, ids: string[], browse: boolean) {
    const selected = ids.includes(job.id);
    const reason = issue(job, ids);
    return (
      <DatasetCard
        key={job.id}
        job={job}
        selected={selected}
        reason={reason}
        disabled={disabled || (!!reason && !selected)}
        prefix={browse ? id : 'recent'}
        onToggle={() => (browse ? setDraft(toggle(ids, job)) : onChange(toggle(ids, job)))}
      />
    );
  }
  return (
    <>
      {datasets.length ? (
        <fieldset>
          <legend className="visually-hidden">Inspected dataset</legend>
          <section aria-label="Recent datasets">
            <div className="training-picker-heading">
              <h3>Latest datasets</h3>
              <span>Recently inspected in this project</span>
            </div>
            <div className="training-datasets">{recent.map((job) => card(job, selectedIds, false))}</div>
          </section>
          {!!olderSelected.length && (
            <section aria-label="Selected older datasets">
              <div className="training-picker-heading">
                <h3>Also selected for this job</h3>
              </div>
              <div className="training-datasets">{olderSelected.map((job) => card(job, selectedIds, false))}</div>
            </section>
          )}
        </fieldset>
      ) : (
        <div className="training-empty">
          <p>No inspected datasets yet.</p>
          <button type="button" className="secondary-button" onClick={props.onImport}>
            Import dataset <Icon name="arrow" size={15} />
          </button>
        </div>
      )}
      <button
        ref={browseButton}
        type="button"
        className="training-browse-datasets"
        aria-haspopup="dialog"
        disabled={disabled}
        onClick={() => {
          setDraft([...selectedIds]);
          setSearch('');
          setOpen(true);
        }}
      >
        <span className="training-browse-mark">
          <Icon name="database" size={20} />
        </span>
        <span>
          <strong>
            Browse all datasets <span className="training-picker-count">{datasets.length + pending.length}</span>
          </strong>
          <small>Choose multiple datasets for this training job</small>
        </span>
        <Icon name="arrow" size={18} />
      </button>
      <p className="training-picker-summary" role="status">
        {selectedIds.length} {selectedIds.length === 1 ? 'dataset' : 'datasets'} selected · Combine up to 8 compatible
        datasets
      </p>
      {open && (
        <dialog
          ref={dialog}
          className="training-dataset-dialog"
          aria-labelledby={`${id}-title`}
          aria-describedby={`${id}-description`}
          onCancel={(event) => {
            event.preventDefault();
            closeBrowser();
          }}
          onClose={closeBrowser}
        >
          <header>
            <div>
              <h2 id={`${id}-title`}>Choose datasets</h2>
              <p id={`${id}-description`}>Select up to 8 compatible datasets.</p>
            </div>
            <button type="button" className="text-button" aria-label="Close dataset browser" onClick={closeBrowser}>
              ×
            </button>
          </header>
          <label className="training-picker-search">
            <Icon name="search" size={18} />
            <span className="visually-hidden">Search datasets for training</span>
            <input
              ref={searchInput}
              type="search"
              value={search}
              placeholder="Search by name, source or format…"
              onChange={(event) => setSearch(event.target.value)}
            />
          </label>
          <div className="training-picker-results">
            {props.libraryLoading && <p role="status">Loading saved datasets…</p>}
            {props.libraryError && (
              <p role="alert">
                Saved imports could not load. Inspected datasets are shown below.{' '}
                <button type="button" className="text-button" onClick={props.onRetryLibrary}>
                  Retry
                </button>
              </p>
            )}
            <fieldset>
              <legend className="visually-hidden">Datasets for this training job</legend>
              <div className="training-picker-grid">
                {matching.map((job) => card(job, draft, true))}
                {matchingPending.map((entry) => (
                  <label key={entry.id} className="training-dataset unavailable">
                    <input
                      type="checkbox"
                      disabled
                      aria-label={entry.name}
                      aria-describedby={`${id}-${entry.id}-pending`}
                    />
                    <span className="training-dataset-icon">
                      <Icon name="database" size={18} />
                    </span>
                    <span className="training-dataset-copy">
                      <strong>{entry.name}</strong>
                      <small>
                        {entry.source === 'huggingface' ? 'Hugging Face' : 'Local files'} · {entry.status}
                      </small>
                      <small className="training-picker-reason" id={`${id}-${entry.id}-pending`}>
                        {entry.status === 'uploading'
                          ? 'Finish importing before training.'
                          : (entry.error ?? 'Inspect this dataset and prepare a training copy before selecting it.')}
                      </small>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
            {!matching.length && !matchingPending.length && (
              <p className="training-picker-no-results">
                {query ? 'No matching datasets. Try another search.' : 'No saved datasets in this project yet.'}
              </p>
            )}
          </div>
          {missing && (
            <p role="alert">
              A selected dataset is no longer available. Close and reopen the browser to refresh the selection.
            </p>
          )}
          <footer>
            <span role="status">{draft.length} of 8 selected</span>
            <div>
              <button type="button" className="secondary-button" onClick={closeBrowser}>
                Cancel
              </button>
              <button
                type="button"
                className="primary-button"
                disabled={disabled || missing}
                onClick={() => {
                  onChange(draft);
                  closeBrowser();
                }}
              >
                Use selected datasets
              </button>
            </div>
          </footer>
        </dialog>
      )}
    </>
  );
}
