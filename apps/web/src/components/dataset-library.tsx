'use client';

import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  datasetDownload,
  datasetLibrary,
  formatName,
  sampleImage,
  labelDraftKey,
  readLabelDraft,
  sameLabelDraft,
  isProjectDataset,
  type LabelDraft,
  type DatasetLabels,
  type LibraryDataset,
} from '@/lib/dataset-library';
import { api, apiMediaUrl } from '@/lib/api';
import { datasetPollingInterval, publishDatasetReceipt } from '@/lib/dataset-cache';
import { Icon } from './icon';
import './dataset-library.css';

export function DatasetLibrary({
  projectId,
  onOpen,
  active,
}: {
  active: boolean;
  projectId: string;
  onOpen: (entry: LibraryDataset) => void;
}) {
  const datasets = useQuery({
    queryKey: ['datasets', projectId],
    queryFn: ({ signal }) => datasetLibrary.list(projectId, signal),
    enabled: !!projectId && active,
    retry: false,
    refetchInterval: active ? 5000 : false,
  });
  const [search, setSearch] = useState('');
  const entries = (datasets.data ?? [])
    .filter((entry): entry is LibraryDataset => isProjectDataset(entry, projectId))
    .filter((entry) => entry.name.toLowerCase().includes(search.toLowerCase()));
  const repairNotices = (datasets.data ?? []).filter(
    (entry) => entry.status === 'unavailable' && entry.project_id === null,
  );
  return (
    <section className="dataset-library" aria-labelledby="dataset-library-title">
      <div className="library-heading">
        <h2 id="dataset-library-title">Saved datasets</h2>
        <div className="workbench-actions">
          {!!datasets.data?.length && (
            <label className="library-search">
              <Icon name="search" size={18} />
              <input
                type="search"
                aria-label="Search my datasets"
                placeholder="Search your datasets"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
          )}
          <button
            type="button"
            className="text-button"
            disabled={!projectId || datasets.isFetching}
            onClick={() => void datasets.refetch()}
          >
            Refresh datasets
          </button>
        </div>
      </div>
      {!!repairNotices.length && (
        <section role="alert" aria-label="Workspace dataset repair notice">
          <p>
            Some dataset records need repair: their project ownership could not be verified. Their files are preserved.
          </p>
          <ul>
            {repairNotices.map((entry) => (
              <li key={entry.id}>{entry.error}</li>
            ))}
          </ul>
        </section>
      )}
      {datasets.isError ? (
        <div role="alert">
          <p>Could not load your datasets.</p>
          <button className="secondary-button" onClick={() => void datasets.refetch()}>
            Retry datasets
          </button>
        </div>
      ) : datasets.isPending && projectId ? (
        <p role="status">Loading your datasets…</p>
      ) : !entries.length ? (
        <div className="dataset-empty">
          <Icon name="database" size={32} />
          <h3>{search ? 'No matching datasets' : 'No saved datasets yet'}</h3>
          <p>{search ? 'Try a different name.' : 'Import a dataset above to inspect it here.'}</p>
        </div>
      ) : (
        <div className="dataset-library-grid">
          {entries.map((entry) => (
            <DatasetCard key={entry.id} entry={entry} active={active} onOpen={() => onOpen(entry)} />
          ))}
        </div>
      )}
    </section>
  );
}

function DatasetCover({ entry, active }: { entry: LibraryDataset; active: boolean }) {
  const local = !entry.id.startsWith('inspection:') && entry.status === 'ready';
  const hub = entry.source === 'huggingface' && !!entry.job_id && entry.status === 'ready';
  const samples = useQuery({
    queryKey: ['dataset-samples', entry.id],
    queryFn: ({ signal }) => datasetLibrary.samples(entry.id, signal),
    enabled: active && local,
    retry: false,
  });
  const preview = useQuery({
    queryKey: ['dataset-cover', entry.job_id],
    queryFn: () => api.datasetCover(entry.job_id!),
    enabled: active && hub,
    staleTime: Infinity,
    retry: false,
  });
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const wantsPlayback = useRef(false);
  const [hovered, setHovered] = useState(false);
  const sample = samples.data?.[0];
  const camera = preview.data?.cameras[0];
  function stopPreview() {
    wantsPlayback.current = false;
    setHovered(false);
    video.current?.pause();
    if (video.current && camera && loaded) video.current.currentTime = camera.start_seconds;
  }
  function playPreview() {
    if (!active || failed) return;
    wantsPlayback.current = true;
    setHovered(true);
    const element = video.current;
    if (!element || !camera || !loaded) return;
    if (element.currentTime >= camera.end_seconds - 1 / camera.fps || element.currentTime < camera.start_seconds)
      element.currentTime = camera.start_seconds;
    void element
      .play()
      .then(() => {
        if (!wantsPlayback.current || !active) element.pause();
      })
      .catch(() => {});
  }
  useEffect(() => {
    if (!active) stopPreview();
  }, [active]);
  useEffect(() => {
    if (!hovered || !camera || !loaded || !active) return;
    playPreview();
    let frame = 0;
    const advance = () => {
      const element = video.current;
      if (!element || !wantsPlayback.current) return;
      // Chunked LeRobot videos may contain the next episode immediately after
      // this interval. Loop one frame before its exclusive end boundary.
      const end =
        Math.min(camera.end_seconds, Number.isFinite(element.duration) ? element.duration : Infinity) - 1 / camera.fps;
      if (element.currentTime >= Math.max(camera.start_seconds, end)) element.currentTime = camera.start_seconds;
      frame = requestAnimationFrame(advance);
    };
    frame = requestAnimationFrame(advance);
    return () => {
      cancelAnimationFrame(frame);
      video.current?.pause();
    };
  }, [hovered, camera, loaded, active]);
  useEffect(() => {
    const pause = () => {
      if (document.hidden) stopPreview();
    };
    document.addEventListener('visibilitychange', pause);
    return () => {
      document.removeEventListener('visibilitychange', pause);
      video.current?.pause();
    };
  }, []);
  const loading =
    !failed && ((local && samples.isPending) || (hub && preview.isPending) || (!!(sample || camera) && !loaded));
  return (
    <div
      className="dataset-library-cover"
      onPointerEnter={(event) => {
        if (event.pointerType === 'mouse') playPreview();
      }}
      onPointerLeave={stopPreview}
    >
      {sample && !failed && (
        <img
          src={sampleImage(entry.id, sample.path)}
          alt={`${entry.name} · ${sample.camera}`}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
        />
      )}
      {camera && !failed && (
        <video
          ref={video}
          src={`${apiMediaUrl(camera.url)}#t=${camera.start_seconds}`}
          preload="metadata"
          muted
          playsInline
          aria-label={`${entry.name} preview`}
          className={loaded ? 'loaded' : ''}
          onLoadedMetadata={(event) => {
            event.currentTarget.currentTime = camera.start_seconds;
          }}
          onLoadedData={() => {
            if (camera.start_seconds === 0) setLoaded(true);
          }}
          onSeeked={(event) => {
            if (Math.abs(event.currentTarget.currentTime - camera.start_seconds) < 0.05) setLoaded(true);
          }}
          onError={() => setFailed(true)}
        />
      )}
      {!loaded || failed ? (
        <div className="dataset-cover-placeholder">
          <Icon name="database" size={32} />
          <small>
            {entry.status === 'uploading'
              ? 'Upload unfinished'
              : entry.status === 'unavailable'
                ? 'Dataset unavailable'
                : loading
                  ? 'Loading preview…'
                  : 'Preview unavailable'}
          </small>
        </div>
      ) : null}
      <span>
        {entry.example ? 'Synthetic example' : entry.source === 'huggingface' ? 'Hugging Face' : 'Local import'}
      </span>
      {camera && loaded && !failed && (
        <span className="dataset-hover-hint">
          <Icon name="play" size={12} />
          {hovered ? 'Playing preview' : 'Hover to preview'}
        </span>
      )}
    </div>
  );
}

function DatasetCard({ entry, onOpen, active }: { entry: LibraryDataset; onOpen: () => void; active: boolean }) {
  const stats = entry.converted ?? entry.detection;
  const episodes = entry.profile?.total_episodes ?? stats?.episodes;
  const cameras = entry.profile
    ? Object.values(entry.profile.features).filter((feature) =>
        ['video', 'image'].includes(String((feature as { dtype?: string }).dtype)),
      ).length
    : stats?.cameras?.length;
  return (
    <button type="button" className="dataset-library-card" aria-label={`Open dataset ${entry.name}`} onClick={onOpen}>
      <DatasetCover entry={entry} active={active} />
      <div className="dataset-library-copy">
        <h3>{entry.name}</h3>
        <p>{formatName(entry.profile?.format ?? stats?.format)}</p>
        <div className="dataset-card-facts">
          <span>
            {entry.status === 'uploading'
              ? `${entry.uploaded_files ?? 0} files uploaded`
              : episodes != null
                ? `${episodes} episodes`
                : 'Format unverified'}
          </span>
          {!!cameras && (
            <span>
              {cameras} {cameras === 1 ? 'view' : 'views'}
            </span>
          )}
        </div>
        <div className="dataset-card-footer">
          <span>
            {entry.status === 'uploading'
              ? 'Review unfinished upload'
              : entry.status === 'converting'
                ? 'Converting…'
                : entry.status === 'failed' || entry.status === 'unavailable'
                  ? 'Needs attention'
                  : entry.training_copy
                    ? 'Training copy saved'
                    : entry.profile?.source === 'huggingface'
                      ? 'Inspection saved'
                      : entry.status === 'ready'
                        ? 'Files saved'
                        : 'Ready to convert'}
          </span>
          <Icon name="arrow" size={15} />
        </div>
      </div>
    </button>
  );
}

export function LocalDatasetImport({
  projectId,
  value,
  onChange,
  disabled,
}: {
  projectId: string;
  value: LibraryDataset | null;
  onChange: (entry: LibraryDataset | null) => void;
  disabled: boolean;
}) {
  const folder = useRef<HTMLInputElement>(null),
    archive = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState('');
  const [fps, setFps] = useState(30),
    [task, setTask] = useState('Recorded robot task'),
    [robot, setRobot] = useState('unspecified');
  const [stateKeys, setStateKeys] = useState(''),
    [actionKey, setActionKey] = useState('');
  const client = useQueryClient();
  const uploadController = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const detectionInitializedFor = useRef<string | null>(null);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [acceptedConversionId, setAcceptedConversionId] = useState<string | null>(null);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      uploadController.current?.abort();
    };
  }, []);
  const upload = useMutation({
    mutationFn: async (files: File[]) => {
      const controller = new AbortController();
      uploadController.current = controller;
      try {
        return await datasetLibrary.upload(
          projectId,
          files,
          (message) => {
            if (alive.current) setProgress(message);
          },
          controller.signal,
          (entry) => {
            client.setQueryData(['dataset-import', entry.id], entry);
            if (alive.current) onChange(entry);
            void client.invalidateQueries({ queryKey: ['datasets'] });
          },
        );
      } finally {
        if (uploadController.current === controller) uploadController.current = null;
      }
    },
    onSuccess: async (entry) => {
      await publishDatasetReceipt(client, entry);
      if (alive.current) onChange(client.getQueryData<LibraryDataset>(['dataset-import', entry.id]) ?? entry);
    },
    onSettled: () => {
      if (alive.current) setProgress('');
      void client.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
  const saved = useQuery({
    queryKey: ['dataset-import', value?.id],
    queryFn: ({ signal }) => datasetLibrary.get(value!.id, signal),
    enabled: !!value && !value.id.startsWith('inspection:'),
    retry: false,
    refetchInterval: (query) => datasetPollingInterval(query.state.data, acceptedConversionId),
  });
  const current = saved.data ?? value;
  useEffect(() => {
    if (saved.data) onChange(saved.data);
  }, [saved.data, onChange]);
  useEffect(() => {
    detectionInitializedFor.current = null;
    setConfirmDiscard(false);
    setFps(30);
    setStateKeys('');
    setActionKey('');
  }, [current?.id]);
  useEffect(() => {
    // The first upload receipt has an ID but no detection. Initialize the
    // conversion fields when detection arrives, then retain edits on refetch.
    if (!current?.detection || detectionInitializedFor.current === current.id) return;
    detectionInitializedFor.current = current.id;
    setFps(current.detection.fps ?? 30);
    setStateKeys(current.detection.mapping.state?.join(', ') ?? '');
    setActionKey(current.detection.mapping.action ?? '');
  }, [current?.id, current?.detection]);
  const convert = useMutation({
    mutationFn: () =>
      datasetLibrary.convert(current!.id, {
        fps,
        task,
        robot_type: robot,
        mapping: current?.detection?.mapping.action
          ? {
              ...current.detection.mapping,
              action: actionKey,
              state: stateKeys
                .split(',')
                .map((k) => k.trim())
                .filter(Boolean),
            }
          : undefined,
      }),
    onSuccess: async (entry) => {
      if (alive.current && entry.status === 'converting') setAcceptedConversionId(entry.id);
      await publishDatasetReceipt(client, entry);
      if (alive.current) onChange(client.getQueryData<LibraryDataset>(['dataset-import', entry.id]) ?? entry);
      void client.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
  const discard = useMutation({
    mutationFn: () => datasetLibrary.discard(current!.id, projectId),
    onSuccess: (receipt) => {
      if (receipt.discarded !== true || receipt.id !== current?.id)
        throw new Error('Discard could not be confirmed. Refresh the saved upload.');
      client.removeQueries({ queryKey: ['dataset-import', receipt.id] });
      onChange(null);
      setConfirmDiscard(false);
      upload.reset();
      void client.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
  function choose(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (!disabled && !busy && current?.status !== 'uploading' && files.length) upload.mutate(files);
  }
  const conversionObserved =
    !!current && current.id === acceptedConversionId && datasetPollingInterval(current, acceptedConversionId) !== false;
  const busy =
    upload.isPending ||
    current?.status === 'converting' ||
    conversionObserved ||
    convert.isPending ||
    discard.isPending;
  return (
    <div className="local-dataset-import">
      <input
        ref={folder}
        className="visually-hidden"
        type="file"
        multiple
        aria-label="Choose dataset folder"
        disabled={disabled || busy || current?.status === 'uploading'}
        {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
        onChange={choose}
      />
      <input
        ref={archive}
        className="visually-hidden"
        type="file"
        accept=".zip,.h5,.hdf5"
        aria-label="Choose dataset archive or HDF5"
        disabled={disabled || busy || current?.status === 'uploading'}
        onChange={choose}
      />
      <div className="local-upload-zone">
        <Icon name="folder" size={29} />
        <strong>Choose files from your computer</strong>
        <p>Folder, ZIP or HDF5 · up to 2 GB</p>
        <div className="local-upload-buttons">
          <button
            type="button"
            className="secondary-button"
            disabled={disabled || busy || current?.status === 'uploading'}
            onClick={() => folder.current?.click()}
          >
            Choose folder
          </button>
          <button
            type="button"
            className="text-button"
            disabled={disabled || busy || current?.status === 'uploading'}
            onClick={() => archive.current?.click()}
          >
            ZIP / HDF5 file
          </button>
        </div>
      </div>
      {progress && (
        <p className="upload-progress" role="status">
          {progress}
        </p>
      )}
      {upload.isPending && (
        <button type="button" className="secondary-button" onClick={() => uploadController.current?.abort()}>
          Stop upload
        </button>
      )}
      {upload.error && (
        <p className="error-notice" role="alert">
          {upload.error.message}
        </p>
      )}
      {current && (
        <section className="detected-dataset" aria-label="Detected dataset">
          <div>
            <strong>{current.name}</strong>
            <span className="metadata-badge">{formatName(current.detection?.format)}</span>
          </div>
          <p>
            {current.detection?.file_count ?? current.uploaded_files ?? 0} files ·{' '}
            {((current.detection?.total_bytes ?? current.total_bytes ?? 0) / 1024 ** 2).toFixed(1)} MB
          </p>
          {current.status === 'uploading' && (
            <p role="status">This upload is unfinished. Refresh it, or discard it and choose the files again.</p>
          )}
          {current.can_discard === true && (
            <div className="workbench-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={disabled || busy}
                onClick={() => void saved.refetch()}
              >
                Refresh saved upload
              </button>
              {!confirmDiscard ? (
                <button
                  type="button"
                  className="secondary-button"
                  disabled={disabled || busy}
                  onClick={() => setConfirmDiscard(true)}
                >
                  Discard unfinished upload
                </button>
              ) : (
                <>
                  <p>Discard these unfinished files from the app host? Your original files stay unchanged.</p>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={disabled || busy}
                    onClick={() => discard.mutate()}
                  >
                    Confirm discard upload
                  </button>
                  <button
                    type="button"
                    className="text-button"
                    disabled={busy}
                    onClick={() => setConfirmDiscard(false)}
                  >
                    Keep upload
                  </button>
                </>
              )}
            </div>
          )}
          {current.status === 'unavailable' && (
            <p role="alert">This dataset record is unavailable. Its files are preserved.</p>
          )}
          {!current.detection?.convertible && <p role="alert">{current.detection?.reason}</p>}
          {current.status === 'converting' && <p role="status">Converting… You can leave this page.</p>}
          {conversionObserved && current.status !== 'converting' && <p role="status">Checking conversion status…</p>}
          {current.status === 'ready' ? (
            <p className="local-import-ready" role="status">
              <Icon name="check" size={16} /> Ready to inspect ·{' '}
              {formatName(current.converted?.format ?? current.detection?.format)}
            </p>
          ) : (
            current.detection?.convertible &&
            current.status !== 'converting' && (
              <div className="conversion-form">
                <label>
                  Frame rate (FPS)
                  <input
                    type="number"
                    min={1}
                    max={240}
                    value={fps}
                    onChange={(event) => setFps(Number(event.target.value))}
                  />
                </label>
                <label>
                  Task description
                  <input value={task} maxLength={1000} onChange={(event) => setTask(event.target.value)} />
                </label>
                <details>
                  <summary>Field mapping & robot</summary>
                  <label>
                    Robot type
                    <input value={robot} onChange={(event) => setRobot(event.target.value)} />
                  </label>
                  {current.detection.mapping.action && (
                    <>
                      <label>
                        Action field
                        <input value={actionKey} onChange={(event) => setActionKey(event.target.value)} />
                      </label>
                      <label>
                        State fields (in order, comma separated)
                        <input value={stateKeys} onChange={(event) => setStateKeys(event.target.value)} />
                      </label>
                      <p>Camera fields: {current.detection.cameras.join(', ')}</p>
                    </>
                  )}
                </details>
                <button
                  type="button"
                  className="secondary-button"
                  disabled={disabled || busy || !task.trim() || fps < 1 || fps > 240}
                  onClick={() => convert.mutate()}
                >
                  Convert to LeRobot <Icon name="arrow" size={15} />
                </button>
              </div>
            )
          )}
          {(current.error || convert.error || saved.error || discard.error) && (
            <p className="error-notice" role="alert">
              {discard.error?.message || current.error || convert.error?.message || saved.error?.message}
            </p>
          )}
        </section>
      )}
    </div>
  );
}

export function DatasetLabeling({ entry, onInspect }: { entry: LibraryDataset; onInspect: () => void }) {
  const client = useQueryClient();
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
  }, []);
  const saved = useQuery({
    queryKey: ['dataset-import', entry.id],
    queryFn: ({ signal }) => datasetLibrary.get(entry.id, signal),
    retry: false,
    refetchInterval: (query) => (query.state.data?.status === 'converting' ? 1000 : false),
  });
  const current = saved.data ?? entry;
  const samples = useQuery({
    queryKey: ['dataset-samples', entry.id],
    queryFn: ({ signal }) => datasetLibrary.samples(entry.id, signal),
    enabled: current.status === 'ready',
    retry: false,
  });
  const draftKey = labelDraftKey(entry);
  const draftCacheKey = ['dataset-label-draft', entry.project_id, entry.id];
  const [initialDraft] = useState<LabelDraft | null>(() => {
    client.setQueryDefaults(draftCacheKey, { gcTime: Infinity });
    const cached = client.getQueryData<LabelDraft>(draftCacheKey);
    if (cached) return cached;
    try {
      return readLabelDraft(sessionStorage.getItem(draftKey));
    } catch {
      return null;
    }
  });
  const [draftStorageError, setDraftStorageError] = useState(() => {
    try {
      sessionStorage.getItem(draftKey);
      return '';
    } catch {
      return initialDraft ? 'Browser storage is unavailable. Save labels before closing this tab.' : '';
    }
  });
  const [labels, setLabels] = useState<DatasetLabels>(
    initialDraft?.labels ?? entry.annotations ?? { views: {}, frames: {} },
  );
  const [revision, setRevision] = useState(initialDraft?.revision ?? entry.annotation_revision ?? 0),
    [dirty, setDirty] = useState(!!initialDraft),
    [selected, setSelected] = useState(0);
  function clearDraft(expected?: LabelDraft) {
    const cached = client.getQueryData<LabelDraft>(draftCacheKey);
    if (!expected || sameLabelDraft(cached, expected)) client.removeQueries({ queryKey: draftCacheKey, exact: true });
    try {
      const stored = readLabelDraft(sessionStorage.getItem(draftKey));
      if (!expected || sameLabelDraft(stored, expected)) sessionStorage.removeItem(draftKey);
      setDraftStorageError('');
    } catch {
      setDraftStorageError('The browser draft could not be cleared. Reload to check the saved labels.');
    }
  }
  function editLabels(next: DatasetLabels) {
    const draft = { revision, labels: next };
    client.setQueryData(draftCacheKey, draft);
    try {
      sessionStorage.setItem(draftKey, JSON.stringify({ version: 1, ...draft }));
      setDraftStorageError('');
    } catch {
      setDraftStorageError('Browser storage is unavailable. Save labels before closing this tab.');
    }
    setLabels(next);
    setDirty(true);
  }
  useEffect(() => {
    if (!dirty) return;
    const protect = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', protect);
    return () => window.removeEventListener('beforeunload', protect);
  }, [dirty]);
  useEffect(() => {
    if (saved.data && !dirty) {
      setLabels(saved.data.annotations ?? { views: {}, frames: {} });
      setRevision(saved.data.annotation_revision ?? 0);
    }
  }, [saved.data, dirty]);
  const save = useMutation({
    mutationFn: (draft: LabelDraft) => datasetLibrary.annotate(entry.id, draft.revision, draft.labels),
    onSuccess: (value, submitted) => {
      clearDraft(submitted);
      setRevision(value.annotation_revision ?? 0);
      setDirty(false);
      client.setQueryData(['dataset-import', entry.id], value);
      void client.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
  const sample = samples.data?.[selected];
  const sampleKey = sample ? `${sample.episode_index}:${sample.frame_index}:${sample.camera}` : '';
  const cameras = current.converted?.cameras ?? current.detection?.cameras ?? [];
  return (
    <section className="panel dataset-labeling" aria-label="Dataset labeling">
      <div className="library-heading">
        <div>
          <h2 ref={heading} tabIndex={-1}>
            {entry.name}
          </h2>
          <p>
            {current.example ? 'Synthetic example · ' : ' '}
            {formatName(current.converted?.format ?? current.detection?.format)}
          </p>
        </div>
        {current.status === 'ready' && (
          <a
            className="secondary-button"
            href={datasetDownload(entry.id)}
            download
            aria-disabled={dirty || save.isPending}
            onClick={(event) => {
              if (dirty || save.isPending) event.preventDefault();
            }}
          >
            {dirty ? 'Save labels to export' : 'Export dataset'}
          </a>
        )}
      </div>
      {current.status === 'converting' && <p role="status">Preparing camera views…</p>}
      {current.error && (
        <p className="error-notice" role="alert">
          {current.error}
        </p>
      )}
      {dirty && !draftStorageError && (
        <p role="status">Unsaved labels are kept in this tab. Save them before exporting.</p>
      )}
      {draftStorageError && (
        <p role="alert" className="error-notice">
          {draftStorageError}
        </p>
      )}
      {current.status === 'ready' && (
        <>
          <div className="dataset-label-grid">
            <div className="label-image-area">
              {sample ? (
                <>
                  <img
                    src={sampleImage(entry.id, sample.path)}
                    alt={`${labels.views[sample.camera] || sample.camera} · episode ${sample.episode_index + 1} frame ${sample.frame_index}`}
                  />
                  <div className="sample-navigation">
                    <button
                      className="secondary-button"
                      disabled={selected === 0}
                      onClick={() => setSelected((i) => i - 1)}
                    >
                      Previous image
                    </button>
                    <span>
                      {selected + 1} / {samples.data?.length}
                    </span>
                    <button
                      className="secondary-button"
                      disabled={selected === (samples.data?.length ?? 0) - 1}
                      onClick={() => setSelected((i) => i + 1)}
                    >
                      Next image
                    </button>
                  </div>
                  <p>
                    Episode {sample.episode_index + 1} · Frame {sample.frame_index} ·{' '}
                    {labels.views[sample.camera] || sample.camera}
                  </p>
                </>
              ) : (
                <p>
                  {samples.isPending
                    ? 'Loading sample frames…'
                    : current.preview_note || 'No preview image available for this dataset.'}
                </p>
              )}
              {samples.error && <p role="alert">{samples.error.message}</p>}
            </div>
            <div className="dataset-label-fields">
              <h3>Camera views</h3>
              {cameras.map((camera) => (
                <label key={camera}>
                  {camera.split(/[/.]/).at(-1)?.replaceAll('_', ' ')} view
                  <input
                    aria-label={`View name ${camera}`}
                    value={labels.views[camera] ?? ''}
                    maxLength={100}
                    placeholder="e.g. Front camera, left wrist"
                    disabled={save.isPending}
                    onChange={(event) =>
                      editLabels({ ...labels, views: { ...labels.views, [camera]: event.target.value } })
                    }
                  />
                </label>
              ))}
              {sample && (
                <label>
                  Label this image
                  <textarea
                    aria-label="Image label"
                    value={labels.frames[sampleKey] ?? ''}
                    maxLength={1000}
                    placeholder="Object, action, task or a note about this frame"
                    disabled={save.isPending}
                    onChange={(event) =>
                      editLabels({ ...labels, frames: { ...labels.frames, [sampleKey]: event.target.value } })
                    }
                  />
                </label>
              )}
              <button
                className="primary-button"
                disabled={!dirty || save.isPending}
                onClick={() => save.mutate({ revision, labels })}
              >
                {save.isPending ? 'Saving…' : 'Save labels'}
              </button>
              {save.isSuccess && !dirty && <p role="status">Labels saved</p>}
              {save.error && (
                <div role="alert">
                  <p className="error-notice">{save.error.message}</p>
                  <button
                    className="text-button"
                    onClick={() => {
                      clearDraft();
                      setDirty(false);
                      void saved.refetch();
                    }}
                  >
                    Reload saved labels
                  </button>
                </div>
              )}
              <p className="field-help">Save labels before exporting.</p>
            </div>
          </div>
          <div className="dataset-label-actions">
            <button className="secondary-button" onClick={onInspect}>
              Inspect for training <Icon name="arrow" size={15} />
            </button>
          </div>
        </>
      )}
    </section>
  );
}
