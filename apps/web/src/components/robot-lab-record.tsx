'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
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
  const [guided, setGuided] = useState(true);
  const [episodeSeconds, setEpisodeSeconds] = useState(30);
  const [resetSeconds, setResetSeconds] = useState(10);
  const [episodeCount, setEpisodeCount] = useState(10);
  const [countdownSeconds, setCountdownSeconds] = useState(3);
  const [voiceCues, setVoiceCues] = useState(true);
  const lastCue = useRef<string | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const primeAudio = useCallback(() => {
    try {
      if (!audioContext.current && typeof window.AudioContext === 'function')
        audioContext.current = new window.AudioContext();
      void audioContext.current?.resume().catch(() => undefined);
    } catch {
      // Visual cues remain available when browser audio is blocked.
    }
  }, []);
  const playCue = useCallback((text: string) => {
    try {
      const speech = 'speechSynthesis' in window ? window.speechSynthesis : null;
      if (
        speech &&
        typeof SpeechSynthesisUtterance !== 'undefined' &&
        (typeof speech.getVoices !== 'function' || speech.getVoices().length > 0)
      ) {
        speech.cancel();
        speech.speak(new SpeechSynthesisUtterance(text));
        return;
      }
    } catch {
      // Use a local tone when no speech engine or voice is available.
    }
    const context = audioContext.current;
    if (!context || context.state === 'closed') return;
    try {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.frequency.value = 660;
      gain.gain.setValueAtTime(0, context.currentTime);
      gain.gain.linearRampToValueAtTime(0.12, context.currentTime + 0.01);
      gain.gain.linearRampToValueAtTime(0, context.currentTime + 0.18);
      oscillator.start();
      oscillator.stop(context.currentTime + 0.2);
      oscillator.onended = () => {
        oscillator.disconnect();
        gain.disconnect();
      };
    } catch {
      // Capture and visual countdowns never depend on audio support.
    }
  }, []);
  const autoAvailable = !!state?.capabilities?.auto_recording_fps;
  const guidedAvailable = !!state?.capabilities?.guided_recording;
  const collection = recording?.collection?.enabled ? recording.collection : null;
  const collecting = !!collection && ['countdown', 'recording', 'reset', 'saving'].includes(collection.phase);
  const saving = !!recording?.saving || !!collection?.saving || collection?.phase === 'saving';
  const measuring = recording?.rate_status === 'measuring';
  const rateFailed = recording?.rate_status === 'error';
  const rateReady = !recording?.rate_status || recording.rate_status === 'ready';
  const canCollect = canAct && controlling && ownsSession && !centering && !partialControl && rateReady;
  const validSchedule =
    Number.isInteger(episodeCount) &&
    episodeCount >= 1 &&
    episodeCount <= 1000 &&
    Number.isInteger(episodeSeconds) &&
    episodeSeconds >= 1 &&
    episodeSeconds <= 300 &&
    Number.isInteger(resetSeconds) &&
    resetSeconds >= 0 &&
    resetSeconds <= 300 &&
    Number.isInteger(countdownSeconds) &&
    countdownSeconds >= 0 &&
    countdownSeconds <= 30;
  const cueKey = recording?.active && collection ? `${recording.dataset_id}:${collection.cue_id}` : null;
  const cueText = collection?.cue;
  useEffect(() => {
    if (!cueKey || cueKey === lastCue.current) return;
    const previous = lastCue.current;
    lastCue.current = cueKey;
    if (!previous || !voiceCues || !cueText) return;
    playCue(cueText);
  }, [cueKey, cueText, voiceCues, playCue]);
  useEffect(
    () => () => {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      void audioContext.current?.close().catch(() => undefined);
    },
    [],
  );
  function speakPreview() {
    primeAudio();
    playCue('Voice cues ready. Keep the workspace clear.');
  }
  const collectionLabels = {
    idle: 'Ready to collect demonstrations',
    countdown: 'Get ready',
    recording: 'Recording demonstration',
    saving: 'Saving episode',
    reset: 'Reset the scene',
    paused: 'Collection paused',
    complete: 'Collection complete',
  };
  return (
    <>
      <Heading title="Recording" />
      <RobotLabControlDevices lab={lab} />
      <div className="robot-lab-form">
        {guidedAvailable && (
          <div className="robot-lab-actions">
            <label className="robot-lab-check">
              <input
                type="checkbox"
                checked={voiceCues}
                onChange={(event) => {
                  setVoiceCues(event.target.checked);
                  if (!event.target.checked && 'speechSynthesis' in window) window.speechSynthesis.cancel();
                }}
              />
              Voice cues
            </label>
            <button className="text-button" onClick={speakPreview}>
              Test voice
            </button>
            <small>Uses a tone if no speech voice is installed. Visual countdowns always remain available.</small>
          </div>
        )}
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
            {guidedAvailable && (
              <section className="robot-lab-form" aria-label="Collection plan">
                <label>
                  Collection mode
                  <select
                    aria-label="Collection mode"
                    value={guided ? 'guided' : 'manual'}
                    disabled={busy}
                    onChange={(event) => setGuided(event.target.value === 'guided')}
                  >
                    <option value="guided">Timed demonstrations</option>
                    <option value="manual">Manual episodes</option>
                  </select>
                </label>
                {guided && (
                  <>
                    <p className="robot-lab-subtle">
                      Record a demonstration, reset the scene, then repeat. Completed episodes save automatically.
                      Teleoperation stays under your control during resets.
                    </p>
                    <div className="robot-lab-fields">
                      <label>
                        Episode length (seconds)
                        <input
                          type="number"
                          aria-label="Episode length in seconds"
                          min={1}
                          max={300}
                          value={episodeSeconds}
                          disabled={busy}
                          onChange={(e) => setEpisodeSeconds(Number(e.target.value))}
                        />
                      </label>
                      <label>
                        Reset time (seconds)
                        <input
                          type="number"
                          aria-label="Reset time in seconds"
                          min={0}
                          max={300}
                          value={resetSeconds}
                          disabled={busy}
                          onChange={(e) => setResetSeconds(Number(e.target.value))}
                        />
                      </label>
                      <label>
                        Number of episodes
                        <input
                          type="number"
                          aria-label="Number of episodes"
                          min={1}
                          max={1000}
                          value={episodeCount}
                          disabled={busy}
                          onChange={(e) => setEpisodeCount(Number(e.target.value))}
                        />
                      </label>
                    </div>
                  </>
                )}
              </section>
            )}
            {autoAvailable && (
              <p className="robot-lab-subtle">
                Frame rate:{' '}
                {fps === null
                  ? 'Auto — measures the camera and robot before recording, up to 40 fps.'
                  : `${fps} fps (manual)`}
              </p>
            )}
            <details className="robot-lab-details" open={!autoAvailable}>
              <summary>Advanced recording settings</summary>
              <div className="robot-lab-form">
                <label>
                  Frames per second
                  <select
                    value={fps === null && autoAvailable ? 'auto' : String(fps ?? 10)}
                    aria-label="Recording frame rate"
                    onChange={(event) => setFps(event.target.value === 'auto' ? null : Number(event.target.value))}
                    disabled={busy}
                  >
                    {autoAvailable && <option value="auto">Auto — highest sustainable rate</option>}
                    {(autoAvailable ? [1, 5, 10, 15, 20, 25, 30, 40] : [5, 10, 15, 20]).map((value) => (
                      <option key={value} value={value}>
                        {value} fps
                      </option>
                    ))}
                  </select>
                </label>
                {guidedAvailable && guided && (
                  <label>
                    Countdown before recording (seconds)
                    <input
                      type="number"
                      aria-label="Recording countdown in seconds"
                      min={0}
                      max={30}
                      value={countdownSeconds}
                      disabled={busy}
                      onChange={(e) => setCountdownSeconds(Number(e.target.value))}
                    />
                  </label>
                )}
                {!autoAvailable && (
                  <p className="robot-lab-subtle">
                    Update the robot service for automatic frame-rate selection and timed collection.
                  </p>
                )}
              </div>
            </details>
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
                !state?.capabilities?.recording ||
                saving ||
                (guidedAvailable && guided && !validSchedule)
              }
              onClick={() =>
                void perform('Dataset created. Prepare the scene, then start recording.', () =>
                  robotApi.startRecording({
                    dataset_id: datasetId,
                    task: task.trim(),
                    fps: autoAvailable ? fps : (fps ?? 10),
                    ...(guidedAvailable && guided
                      ? {
                          collection: {
                            episode_time_s: episodeSeconds,
                            reset_time_s: resetSeconds,
                            num_episodes: episodeCount,
                            countdown_s: countdownSeconds,
                          },
                        }
                      : {}),
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
            {rateFailed && (
              <Notice kind="warning">
                <strong>Capture rate could not be measured.</strong>
                {recording.rate_message || recording.error || 'Check the camera connection and teleoperation session.'}
                <p>Close this unsaved recording and create a new dataset to retry. Existing files are retained.</p>
              </Notice>
            )}
            {recording.rate_status && !rateFailed && (
              <p className="robot-lab-subtle" role="status">
                {measuring
                  ? 'Measuring sustainable capture rate…'
                  : `${recording.fps} fps · ${recording.fps_mode === 'auto' ? 'selected automatically' : 'manual'}`}{' '}
                {recording.rate_message}
              </p>
            )}
            {collection && (
              <section className="robot-lab-collection robot-lab-form" aria-label="Guided collection">
                <div className="robot-lab-mapping-heading">
                  <h3>{saving ? 'Saving episode' : collectionLabels[collection.phase]}</h3>
                  <span>
                    {collection.completed}/{collection.num_episodes} episodes collected
                  </span>
                </div>
                <progress aria-label="Collection progress" value={collection.completed} max={collection.num_episodes} />
                {['countdown', 'recording', 'reset'].includes(collection.phase) && !saving && (
                  <output className="robot-lab-collection-countdown" aria-label="Time remaining">
                    {Math.max(0, Math.ceil(collection.remaining_s))}s
                  </output>
                )}
                <p role="status">
                  {collection.cue ||
                    (collection.phase === 'reset'
                      ? 'Return the object and arm to the starting scene using teleoperation.'
                      : 'The robot computer manages episode timing.')}
                </p>
                {collection.phase === 'paused' && recording.frames > 0 && (
                  <Notice>
                    These frames are preserved. Save or discard this partial episode before resuming, or choose
                    Re-record episode to try it again. Partial episodes do not count toward the timed collection.
                  </Notice>
                )}
                <div className="robot-lab-actions">
                  {collection.phase === 'idle' && (
                    <button
                      className="primary-button"
                      disabled={!canCollect || saving || recording.frames > 0}
                      onClick={() => {
                        primeAudio();
                        void perform('Collection started.', () => robotApi.collectionAction('start'));
                      }}
                    >
                      Start collection
                    </button>
                  )}
                  {collecting && (
                    <button
                      className="secondary-button"
                      disabled={!canAct}
                      onClick={() => void perform('Collection paused.', () => robotApi.collectionAction('pause'))}
                    >
                      Pause collection
                    </button>
                  )}
                  {collection.phase === 'paused' && (
                    <button
                      className="primary-button"
                      disabled={!canCollect || saving || recording.frames > 0}
                      onClick={() => {
                        primeAudio();
                        void perform('Collection resumed.', () => robotApi.collectionAction('resume'));
                      }}
                    >
                      Resume collection
                    </button>
                  )}
                  {collection.phase === 'reset' && (
                    <button
                      className="secondary-button"
                      disabled={!canCollect || saving}
                      onClick={() =>
                        void perform('Reset finished. Preparing next episode.', () => robotApi.collectionAction('next'))
                      }
                    >
                      Ready for next episode
                    </button>
                  )}
                  {['recording', 'paused'].includes(collection.phase) &&
                    (recording.episode_active || recording.frames > 0) && (
                      <button
                        className="secondary-button"
                        disabled={!canCollect || saving}
                        onClick={() =>
                          void perform('Current attempt discarded. Preparing to re-record.', () =>
                            robotApi.collectionAction('rerecord'),
                          )
                        }
                      >
                        Re-record episode
                      </button>
                    )}
                </div>
              </section>
            )}
            {session?.mode === 'handguide' && <Notice>Action source: observed joint positions (hand guiding).</Notice>}
            {session?.mode === 'manual' && <Notice>Action source: commanded joint targets.</Notice>}
            <div className="robot-lab-actions">
              {!collection && (
                <button
                  className="primary-button"
                  disabled={
                    !canAct ||
                    !controlling ||
                    !ownsSession ||
                    centering ||
                    partialControl ||
                    recording.episode_active ||
                    !!collection ||
                    !rateReady ||
                    saving ||
                    recording.frames > 0
                  }
                  onClick={() => void perform('Episode recording started.', robotApi.startEpisode)}
                >
                  <span aria-hidden="true">●</span>Start episode
                </button>
              )}
              <button
                className="secondary-button"
                disabled={!canAct || saving || collecting || !recording.frames}
                onClick={() => void perform('Episode saved.', robotApi.saveEpisode)}
              >
                Save episode
              </button>
              <button
                className="secondary-button"
                disabled={!canAct || saving || collecting || (!recording.episode_active && !recording.frames)}
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
                disabled={
                  !canAct ||
                  collecting ||
                  saving ||
                  recording.episode_active ||
                  recording.frames > 0 ||
                  recording.episodes === 0
                }
                onClick={() => void perform('Dataset finalized.', robotApi.finishRecording)}
              >
                Finish dataset
              </button>
              {recording.episodes === 0 && (
                <button
                  className="text-button"
                  disabled={!canAct || collecting || saving}
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
