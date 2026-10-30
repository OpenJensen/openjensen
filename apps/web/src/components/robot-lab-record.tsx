'use client';

import { RobotLabControlDevices } from './robot-lab-devices';
import { Icon } from '@/components/icon';
import { importRobotDataset, robotApi, robotJoints } from '@/lib/robot-api';
import { Heading, Notice, JointControl, DatasetRow } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabRecord({ lab }: { lab: RobotLabState }) {
  const {
    availableCameras,
    busy,
    cameraIndex,
    canAct,
    centering,
    client,
    controlling,
    datasetId,
    datasetList,
    datasets,
    fps,
    imported,
    jog,
    jogging,
    navigateStage,
    ownsSession,
    partialControl,
    pending,
    perform,
    projectId,
    recordCamera,
    recording,
    session,
    sessionJoints,
    setDatasetId,
    setFps,
    setImported,
    setRecordCamera,
    setStep,
    setTask,
    state,
    targetJoint,
    task,
  } = lab;
  return (
    <>
      <Heading title="Recording" />
      <RobotLabControlDevices lab={lab} />
      <div className="robot-lab-form">
        {partialControl && <Notice>Dataset recording requires all six joints.</Notice>}
        {!recording?.active ? (
          <>
            {!controlling && (
              <Notice>
                <strong>No control session</strong>Start teleoperation before recording.
                <button className="text-button" onClick={() => setStep(3)}>
                  Open teleoperation
                </button>
              </Notice>
            )}
            <div className="robot-lab-fields">
              <label>
                Dataset name
                <input
                  value={datasetId}
                  aria-label="Dataset name"
                  placeholder="pick-up-block"
                  onChange={(event) => setDatasetId(event.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '-'))}
                  maxLength={64}
                  disabled={busy}
                />
                {datasetList.some((dataset) => dataset.id === datasetId) && (
                  <small role="alert">Name already used.</small>
                )}
              </label>
              <label>
                Frames per second
                <select
                  value={fps}
                  aria-label="Recording frame rate"
                  onChange={(event) => setFps(Number(event.target.value))}
                  disabled={busy}
                >
                  {[5, 10, 15, 20].map((value) => (
                    <option key={value} value={value}>
                      {value} fps
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label>
              Task instruction
              <textarea
                value={task}
                aria-label="Task instruction"
                placeholder="Pick up the blue block and place it in the tray."
                onChange={(event) => setTask(event.target.value)}
                disabled={busy}
                maxLength={500}
              />
            </label>
            <label className="robot-lab-check">
              <input
                type="checkbox"
                checked={recordCamera}
                disabled={busy || !availableCameras.length}
                onChange={(event) => setRecordCamera(event.target.checked)}
              />
              <span>
                <strong>Record camera</strong>
                {availableCameras.length ? '' : 'No camera available; state and actions only.'}
              </span>
            </label>
            <button
              className="primary-button"
              disabled={
                !canAct ||
                !controlling ||
                !ownsSession ||
                centering ||
                partialControl ||
                !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(datasetId) ||
                !task.trim() ||
                datasetList.some((dataset) => dataset.id === datasetId) ||
                !state?.capabilities?.recording
              }
              onClick={() =>
                void perform('Dataset created. Start an episode when you are ready.', () =>
                  robotApi.startRecording({
                    dataset_id: datasetId,
                    task: task.trim(),
                    fps,
                    camera_index: recordCamera && availableCameras.length ? cameraIndex : null,
                  }),
                )
              }
            >
              {busy ? 'Creating dataset…' : 'Create recording dataset'}
            </button>
          </>
        ) : (
          <>
            <div className={`robot-lab-recording-status${recording.episode_active ? ' recording' : ''}`}>
              <span className="robot-lab-recording-light" />
              <div>
                <strong>{recording.dataset_id}</strong>
                <small>
                  {recording.episode_active
                    ? 'Recording an episode'
                    : recording.frames > 0
                      ? 'Episode has pending frames'
                      : 'Ready for the next episode'}{' '}
                  · {recording.episodes} saved
                </small>
              </div>
              <output>{recording.frames} frames</output>
            </div>
            {session?.mode === 'handguide' && <Notice>Action source: observed joint positions (hand guiding).</Notice>}
            {session?.mode === 'manual' && <Notice>Action source: commanded joint targets.</Notice>}
            <div className="robot-lab-actions">
              <button
                className="primary-button"
                disabled={
                  !canAct ||
                  !controlling ||
                  !ownsSession ||
                  centering ||
                  partialControl ||
                  recording.episode_active ||
                  recording.frames > 0
                }
                onClick={() => void perform('Episode recording started.', robotApi.startEpisode)}
              >
                <span aria-hidden="true">●</span>Start episode
              </button>
              <button
                className="secondary-button"
                disabled={!canAct || !recording.frames}
                onClick={() => void perform('Episode saved.', robotApi.saveEpisode)}
              >
                Save episode
              </button>
              <button
                className="secondary-button"
                disabled={!canAct || (!recording.episode_active && !recording.frames)}
                onClick={() => void perform('Episode discarded.', robotApi.discardEpisode)}
              >
                Discard episode
              </button>
            </div>
            {session?.mode === 'manual' && (
              <div className="robot-lab-joints" aria-busy={jogging}>
                {robotJoints
                  .filter((joint) => sessionJoints.includes(joint.name))
                  .map((joint) => (
                    <JointControl
                      key={joint.name}
                      joint={joint}
                      position={session.positions[joint.name]}
                      target={session.targets?.[joint.name]}
                      sliderTarget={session.slider_targets?.[joint.name]}
                      limits={session.joint_limits?.[joint.name]}
                      sliders={!!state?.capabilities?.joint_sliders}
                      disabled={!canAct || !ownsSession || jogging || centering}
                      step={1}
                      recording
                      onJog={jog}
                      onTarget={targetJoint}
                    />
                  ))}
              </div>
            )}
            <div className="robot-lab-actions">
              <button
                className="secondary-button"
                disabled={!canAct || recording.episode_active || recording.frames > 0 || recording.episodes === 0}
                onClick={() => void perform('Dataset finalized.', robotApi.finishRecording)}
              >
                Finish dataset
              </button>
              {recording.episodes === 0 && (
                <button
                  className="text-button"
                  disabled={!canAct}
                  onClick={() =>
                    void perform(
                      'Unsaved recording closed; files retained on the robot computer.',
                      robotApi.abandonRecording,
                    )
                  }
                >
                  Close unsaved recording
                </button>
              )}
            </div>
            {recording.episodes === 0 && (
              <p className="robot-lab-subtle">
                Closing retains unsaved files. Create a new dataset to change the frame rate.
              </p>
            )}
          </>
        )}
        <hr className="robot-lab-divider" />
        <div>
          <h3>Datasets on the robot computer</h3>
        </div>
        {datasets.error && (
          <p className="robot-lab-subtle" role="alert">
            {datasets.error.message}
          </p>
        )}
        {!datasetList.length ? (
          <div className="robot-lab-empty">
            <Icon name="database" size={26} />
            No datasets
          </div>
        ) : (
          <div className="robot-lab-datasets">
            {datasetList.map((dataset) => (
              <DatasetRow
                key={dataset.id}
                dataset={dataset}
                canImport={canAct && !!projectId}
                pending={pending === `Imported ${dataset.id} into this project.`}
                onImport={() => {
                  const owner = projectId;
                  void perform(
                    `Imported ${dataset.id} into this project.`,
                    () => importRobotDataset(owner, dataset.id),
                    () => {
                      setImported({ dataset: dataset.id, projectId: owner });
                      void client.invalidateQueries({ queryKey: ['datasets', owner] });
                      void client.invalidateQueries({ queryKey: ['jobs', owner] });
                    },
                  );
                }}
              />
            ))}
          </div>
        )}
        {imported && imported.projectId === projectId && (
          <Notice kind="success">
            <div className="robot-lab-imported">
              <span>{imported.dataset} is in this project’s dataset library.</span>
              <button className="text-button" onClick={() => navigateStage('datasets')}>
                View datasets
              </button>
            </div>
          </Notice>
        )}
      </div>
    </>
  );
}
