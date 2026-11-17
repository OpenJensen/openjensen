'use client';

import { useState, type CSSProperties } from 'react';
import { RobotLabCameraFrame } from './robot-lab-camera-frame';
import { viewName } from '@/lib/policy-cameras';
import { CameraIcon } from './robot-lab-shared';
import { RobotLabCameraSettings } from './robot-lab-camera-settings';
import type { RobotLabState } from './robot-lab-state';

/**
 * The cameras beside every Robot Lab step. Each camera is one card: its name, an
 * On/Off switch, and its live picture while it is on. Cameras that are on are the
 * ones shown here and saved with a recording.
 */
export function RobotLabCamera({ lab }: { lab: RobotLabState }) {
  const {
    availableCameras,
    cameraError,
    cameraIndex,
    selectedCameraIndices,
    setAdditionalCameraIndices,
    online,
    preview,
    previewStamp,
    recording,
    setCameraError,
    setCameraIndex,
    setPreview,
    setPreviewStamp,
    state,
  } = lab;
  const [failedCameras, setFailedCameras] = useState<number[]>([]);
  const [expandedCamera, setExpandedCamera] = useState<number | null>(null);
  const large = lab.cameraWorkspace && (lab.step === 3 || lab.step === 4);
  const multipleAvailable = !!state?.capabilities?.multi_camera_recording;
  // On Run policy the preview shows the policy's views, each named as the policy names it.
  const policyViews = lab.step === 5 ? lab.policyViews : [];
  const indices = policyViews.length
    ? policyViews.map((item) => item.index)
    : multipleAvailable
      ? recording?.active && recording.camera_indices?.length
        ? recording.camera_indices
        : selectedCameraIndices.length
          ? selectedCameraIndices
          : [cameraIndex]
      : [cameraIndex];
  const viewOf = (index: number) => policyViews.find((item) => item.index === index)?.view;
  // Record chooses its cameras in "Cameras for this dataset" and Run policy uses the
  // policy's views; every other step switches cameras here.
  const switchable = lab.step !== 4 && !policyViews.length;
  const cards = switchable && !large ? availableCameras.map((camera) => camera.index) : indices;
  const canPreview = online && availableCameras.length > 0 && !!state?.capabilities?.camera;
  const failed = (index: number) => cameraError && (failedCameras.includes(index) || !failedCameras.length);

  function resetErrors() {
    setCameraError(false);
    setFailedCameras([]);
  }
  function switchCamera(index: number, on: boolean) {
    // Without multi-camera support one camera is on at a time.
    const next = !multipleAvailable ? [index] : on ? [...indices, index] : indices.filter((item) => item !== index);
    if (!next.length || next.length > 4) return;
    setCameraIndex(next[0]);
    setAdditionalCameraIndices(next.slice(1));
    resetErrors();
  }

  return (
    <aside className="robot-lab-side">
      <section className="robot-lab-card robot-lab-cameras" aria-label="Camera preview">
        {(!large || expandedCamera !== null) && (
          <div className="robot-lab-card-header">
            <h3>{large ? 'Camera view' : 'Cameras'}</h3>
            {large && expandedCamera !== null && (
              <button className="secondary-button" onClick={() => setExpandedCamera(null)}>
                Show all cameras
              </button>
            )}
          </div>
        )}
        {!large && (
          <div className="robot-lab-camera-actions">
            <button
              className="secondary-button"
              disabled={!canPreview}
              onClick={() => {
                if (cameraError) {
                  resetErrors();
                  setPreviewStamp(Date.now());
                  setPreview(true);
                } else setPreview(!preview);
              }}
            >
              {cameraError
                ? indices.length > 1
                  ? 'Retry cameras'
                  : 'Retry camera'
                : preview
                  ? 'Pause preview'
                  : 'Start preview'}
            </button>
            {(lab.step === 3 || lab.step === 4) && (
              <button
                className="secondary-button"
                disabled={!canPreview}
                onClick={() => {
                  resetErrors();
                  if (!preview || cameraError || failedCameras.length) setPreviewStamp(Date.now());
                  setPreview(true);
                  lab.setCameraWorkspace(true);
                }}
              >
                Large camera view
              </button>
            )}
          </div>
        )}
        {!availableCameras.length && !policyViews.length && (
          <p className="robot-lab-camera-none">No camera found. Plug one in and press Restart devices.</p>
        )}
        <div
          className="robot-lab-camera-views"
          data-single={cards.length === 1 || (expandedCamera !== null && indices.includes(expandedCamera))}
          data-count={Math.min(4, cards.length)}
          data-recording={!!recording?.episode_active}
        >
          {cards.map((index) => {
            const on = indices.includes(index);
            const name = lab.cameraName(index);
            const size = availableCameras.find((camera) => camera.index === index);
            // In the large view each picture keeps its own shape, so no black bands.
            const shape =
              large && size?.width && size.height
                ? ({ '--camera-ratio': `${size.width} / ${size.height}` } as CSSProperties)
                : undefined;
            return (
              <div
                key={index}
                className={`robot-lab-camera-tile${on ? '' : ' off'}`}
                hidden={
                  large && expandedCamera !== null && indices.includes(expandedCamera) && expandedCamera !== index
                }
              >
                {!large && (
                  <div className="robot-lab-camera-tile-heading">
                    <strong>
                      {viewOf(index) ? `${viewName(viewOf(index)!)} · ` : ''}
                      {name}
                    </strong>
                    {switchable && !large && availableCameras.length > 1 && (
                      <button
                        type="button"
                        role="switch"
                        aria-checked={on}
                        aria-label={`Use ${name}`}
                        className="robot-lab-camera-switch"
                        disabled={
                          !online ||
                          !!recording?.active ||
                          (on && indices.length === 1) ||
                          (!on && multipleAvailable && indices.length >= 4)
                        }
                        onClick={() => switchCamera(index, !on)}
                      >
                        <span aria-hidden="true" />
                        {on ? 'On' : 'Off'}
                      </button>
                    )}
                  </div>
                )}
                {on && (
                  <div className="robot-lab-camera" style={shape}>
                    {preview && online && !failed(index) ? (
                      <RobotLabCameraFrame
                        key={`${index}/${previewStamp}/${indices.join(',')}`}
                        index={index}
                        indices={indices}
                        stamp={previewStamp}
                        large={large && (expandedCamera === null || expandedCamera === index)}
                        alt={
                          index === indices[0]
                            ? 'Live view from the selected robot camera'
                            : `Live view from camera ${index}`
                        }
                        onError={() => {
                          setFailedCameras((current) => (current.includes(index) ? current : [...current, index]));
                          setCameraError(true);
                        }}
                      />
                    ) : (
                      <div className="robot-lab-camera-empty">
                        <CameraIcon size={28} />
                        {failed(index) ? (
                          <span>
                            <strong>This camera won’t open.</strong>
                            {indices.length === 1
                              ? 'Close other apps using it and check camera permission on the robot computer.'
                              : 'Close other apps using it. Cameras on one USB hub may need separate ports.'}
                          </span>
                        ) : (
                          <span>Preview paused</span>
                        )}
                      </div>
                    )}
                    {large && (
                      <>
                        <strong className="robot-lab-camera-name">
                          {viewOf(index) ? `${viewName(viewOf(index)!)} · ` : ''}
                          {name}
                        </strong>
                        {indices.length > 1 && expandedCamera !== index && (
                          <button
                            className="robot-lab-camera-enlarge"
                            aria-label={`Enlarge camera ${index}`}
                            onClick={() => setExpandedCamera(index)}
                          >
                            Enlarge
                          </button>
                        )}
                        {lab.step === 4 && !recording?.active && (
                          <label className="robot-lab-camera-record-toggle">
                            <input
                              type="checkbox"
                              aria-label={`Record view from camera ${index}`}
                              checked={lab.recordingCameraIndices.includes(index)}
                              disabled={lab.busy || !online}
                              onChange={(event) => lab.toggleRecordingCamera(index, event.target.checked)}
                            />
                            {lab.recordingCameraIndices.includes(index) ? 'Included in dataset' : 'Preview only'}
                          </label>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {lab.step === 4 && !large && (
          <p className="robot-lab-camera-note">Choose the cameras to save in “Cameras for this dataset”.</p>
        )}
        {!large && online && state?.capabilities?.camera_mode_controls && availableCameras.length > 0 && (
          <RobotLabCameraSettings
            indices={indices}
            name={lab.cameraName}
            locked={state.session?.mode !== 'idle' || !!recording?.active}
            preview={preview}
            setPreview={setPreview}
            onApplied={() => {
              resetErrors();
              setPreviewStamp(Date.now());
            }}
          />
        )}
      </section>
    </aside>
  );
}
