import { isProjectDataset } from '@/lib/dataset-library';
import { combinationCameraMapping, datasetCombinationIssue } from '@/lib/dataset-combination';
import { CameraPlayer } from '@/components/dataset-explorer';
import { LocalDatasetPlayer } from './local-dataset-player';
import { Icon } from '@/components/icon';
import { TrainingDatasetPicker } from './training-dataset-picker';
import { cameras, datasetIssue, number } from './training-shared';
import type { TrainingPanelState } from './training-state';

/** Setup step 1: the training datasets, their camera mappings and the cameras to train on. */
export function TrainingDatasetStep({
  panel,
  onChooseDataset,
}: {
  panel: TrainingPanelState;
  onChooseDataset: () => void;
}) {
  const {
    activeCameraKeys,
    activeDataset,
    busy,
    cameraSelections,
    dataset,
    datasetId,
    datasets,
    episodeIndex,
    episodes,
    headingRef,
    jobs,
    library,
    mappings,
    mixtureIssue,
    mutation,
    preview,
    projectId,
    resumeId,
    selectedCameras,
    selectedDatasets,
    setCameraMappings,
    setDatasetId,
    setExtraDatasetIds,
    setPreviewEpisode,
    setStep,
    step,
    toggleCamera,
  } = panel;
  return (
    <section aria-labelledby="training-data-title">
      <div className="training-heading">
        <h2 id="training-data-title" ref={headingRef} tabIndex={-1}>
          Dataset
        </h2>
        <button type="button" className="text-button" onClick={onChooseDataset}>
          <Icon name="plus" size={14} /> Import
        </button>
      </div>
      {!resumeId && datasetId && datasetId !== '__none__' && !dataset && !jobs.isPending && (
        <p className="warning-box" role="alert">
          Selected dataset {datasetId} is unavailable in this project. Another dataset has not been substituted. Refresh
          or select a dataset explicitly.
        </p>
      )}
      {jobs.isPending && projectId ? (
        <p role="status">Loading datasets…</p>
      ) : (
        <TrainingDatasetPicker
          datasets={datasets}
          library={(library.data ?? []).filter((entry) => isProjectDataset(entry, projectId))}
          selectedIds={resumeId ? (activeDataset ? [activeDataset.id] : []) : selectedDatasets.map((item) => item.id)}
          disabled={!!resumeId || busy}
          libraryLoading={library.isPending}
          libraryError={library.error?.message}
          onRetryLibrary={() => {
            void library.refetch();
          }}
          issue={(job, ids) => {
            const selected = ids.includes(job.id);
            const reference = datasets.find((item) => item.id === ids[0]);
            const views =
              reference && cameraSelections[reference.id] !== undefined
                ? cameras(reference.result).filter((key) => cameraSelections[reference.id].includes(key))
                : cameras(reference?.result);
            return (
              datasetIssue(job.result) ??
              (!selected && reference
                ? ids.length >= 8
                  ? 'Select up to eight datasets per training job.'
                  : datasetCombinationIssue(reference.result, job.result, views)
                : null)
            );
          }}
          onChange={(ids) => {
            const reference = datasets.find((item) => item.id === ids[0]);
            const views =
              reference && cameraSelections[reference.id] !== undefined
                ? cameras(reference.result).filter((key) => cameraSelections[reference.id].includes(key))
                : cameras(reference?.result);
            setCameraMappings((previous) =>
              Object.fromEntries(
                ids.slice(1).map((id) => {
                  const candidate = datasets.find((item) => item.id === id)!;
                  return [
                    id,
                    reference?.id === dataset?.id && previous[id]
                      ? previous[id]
                      : combinationCameraMapping(reference!.result, candidate.result, views),
                  ];
                }),
              ),
            );
            setDatasetId(ids[0] ?? '__none__');
            setExtraDatasetIds(ids.slice(1));
            mutation.reset();
          }}
          onImport={onChooseDataset}
        />
      )}
      {!resumeId &&
        selectedDatasets.slice(1).map((item) => (
          <fieldset key={item.id} className="training-camera-mapping">
            <legend>Camera mapping · {item.result.repo_id}</legend>
            {selectedCameras.map((key) => (
              <label key={key}>
                {key.replace(/^observation\.images\./, '')}
                <select
                  aria-label={`${item.result.repo_id}: ${key}`}
                  value={mappings[item.id][key]}
                  disabled={busy}
                  onChange={(event) =>
                    setCameraMappings((previous) => ({
                      ...previous,
                      [item.id]: { ...previous[item.id], [key]: event.target.value },
                    }))
                  }
                >
                  <option value="">Choose matching camera</option>
                  {cameras(item.result).map((source) => (
                    <option key={source} value={source}>
                      {source.replace(/^observation\.images\./, '')}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </fieldset>
        ))}
      {mixtureIssue && (
        <p className="error-notice" role="alert">
          {mixtureIssue}
        </p>
      )}
      {activeDataset && (
        <>
          <div className="training-data-facts">
            <span>{number(activeDataset.result.total_episodes)} episodes</span>
            <span>{number(activeDataset.result.total_frames)} frames</span>
            <span>{activeDataset.result.fps} fps</span>
            <span>
              {cameras(activeDataset.result).length} {cameras(activeDataset.result).length === 1 ? 'camera' : 'cameras'}
            </span>
            <span>{activeDataset.result.robot_type}</span>
          </div>
          <div className="training-heading training-camera-heading">
            <h3>Cameras</h3>
            {episodes.data && episodes.data.episodes.length > 1 && (
              <label className="training-episode-picker">
                <span className="visually-hidden">Preview episode</span>
                <select
                  value={episodeIndex ?? ''}
                  onChange={(event) =>
                    setPreviewEpisode({
                      datasetId: activeDataset.id,
                      index: Number(event.target.value),
                    })
                  }
                >
                  {episodes.data.episodes.map((item) => (
                    <option key={item.episode_index} value={item.episode_index}>
                      Episode {item.episode_index}
                    </option>
                  ))}
                </select>
                <svg
                  className="training-select-arrow"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  aria-hidden="true"
                >
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </label>
            )}
          </div>
          {activeDataset.result.source === 'local' && activeDataset.request.library_id ? (
            <LocalDatasetPlayer
              key={activeDataset.id}
              job={activeDataset}
              active={step === 0}
              selection={{ keys: activeCameraKeys, onToggle: toggleCamera, disabled: !!resumeId || busy }}
            />
          ) : preview.data?.cameras.length ? (
            <CameraPlayer
              key={`${activeDataset.id}-${episodeIndex}`}
              preview={preview.data}
              active={step === 0}
              selection={{
                keys: activeCameraKeys,
                onToggle: toggleCamera,
                disabled: !!resumeId || busy,
              }}
            />
          ) : (
            <fieldset className="training-camera-fallback">
              <legend className="visually-hidden">Training cameras</legend>
              {cameras(activeDataset.result).map((key) => (
                <label key={key}>
                  <span className="training-camera-placeholder">
                    <Icon name="play" size={22} />
                    <small>
                      {episodes.isFetching || preview.isFetching ? 'Loading preview…' : 'Preview unavailable'}
                    </small>
                  </span>
                  <span>
                    <input
                      type="checkbox"
                      aria-label={key}
                      checked={activeCameraKeys.includes(key)}
                      disabled={!!resumeId || busy}
                      onChange={() => toggleCamera(key)}
                    />
                    {key.replace(/^observation\.images\./, '')}
                  </span>
                </label>
              ))}
            </fieldset>
          )}
          {!!preview.data?.cameras.length &&
            cameras(activeDataset.result)
              .filter((key) => !preview.data.cameras.some((item) => item.key === key))
              .map((key) => (
                <label key={key} className="training-extra-camera">
                  <input
                    type="checkbox"
                    aria-label={key}
                    checked={activeCameraKeys.includes(key)}
                    disabled={!!resumeId || busy}
                    onChange={() => toggleCamera(key)}
                  />
                  {key.replace(/^observation\.images\./, '')}
                  <small>Preview unavailable</small>
                </label>
              ))}
          {(episodes.isError || preview.isError) && (
            <button
              type="button"
              className="text-button"
              onClick={() => {
                void episodes.refetch();
                if (episodeIndex !== undefined) void preview.refetch();
              }}
            >
              Retry preview
            </button>
          )}
        </>
      )}
      <div className="training-footer">
        <span>{activeDataset && !activeCameraKeys.length ? 'Select at least one camera.' : ''}</span>
        <button
          type="button"
          className="primary-button"
          disabled={
            !activeDataset ||
            !activeCameraKeys.length ||
            (!resumeId && (!!datasetIssue(activeDataset.result) || !!mixtureIssue))
          }
          onClick={() => setStep(1)}
        >
          Next <Icon name="arrow" size={15} />
        </button>
      </div>
    </section>
  );
}
