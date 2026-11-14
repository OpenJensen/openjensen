'use client';

import { useState } from 'react';
import { RobotLabCameraFrame } from './robot-lab-camera-frame';
import { viewName } from '@/lib/policy-cameras';
import { CameraIcon } from './robot-lab-shared';
import { RobotLabCameraSettings } from './robot-lab-camera-settings';
import type { RobotLabState } from './robot-lab-state';
import { Segmented } from './ui/choice';

export function RobotLabCamera({ lab }: { lab: RobotLabState }) {
  const {
    availableCameras,
    cameraError,
    cameraIndex,
    additionalCameraIndices,
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
  const primaryIndex = recording?.active ? (indices[0] ?? cameraIndex) : cameraIndex;
  function resetErrors() {
    setCameraError(false);
    setFailedCameras([]);
  }
  return (
    <aside className="robot-lab-side">
      <section className="robot-lab-card" aria-label="Camera preview">
        <div className="robot-lab-card-header">
          <h3>{large ? 'Camera view' : 'Camera preview'}</h3>
          {(lab.step === 3 || lab.step === 4) && !large && (
            <button
              className="secondary-button"
              disabled={!online || !availableCameras.length || !state?.capabilities?.camera}
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
          {large && expandedCamera !== null && (
            <button className="secondary-button" onClick={() => setExpandedCamera(null)}>
              Show all cameras
            </button>
          )}
        </div>
        <div
          className="robot-lab-camera-views"
          data-single={indices.length === 1 || (expandedCamera !== null && indices.includes(expandedCamera))}
        >
          {indices.map((index, ordinal) => (
            <div
              key={index}
              className="robot-lab-camera-tile"
              hidden={large && expandedCamera !== null && indices.includes(expandedCamera) && expandedCamera !== index}
            >
              {(indices.length > 1 || large || policyViews.length > 0) && (
                <div className="robot-lab-camera-tile-heading">
                  <strong>
                    {viewOf(index) ? `${viewName(viewOf(index)!)} · ` : ''}
                    {lab.cameraName(index)}
                  </strong>
                  {large && indices.length > 1 && expandedCamera !== index && (
                    <button
                      className="text-button"
                      aria-label={`Enlarge camera ${index}`}
                      onClick={() => setExpandedCamera(index)}
                    >
                      Enlarge
                    </button>
                  )}
                </div>
              )}
              {large && lab.step === 4 && (
                <label className="robot-lab-camera-record-toggle">
                  <input
                    type="checkbox"
                    aria-label={`Record view from camera ${index}`}
                    checked={lab.recordingCameraIndices.includes(index)}
                    disabled={lab.busy || !!recording?.active || !online}
                    onChange={(event) => lab.toggleRecordingCamera(index, event.target.checked)}
                  />
                  {lab.recordingCameraIndices.includes(index) ? 'Included in dataset' : 'Preview only'}
                </label>
              )}
              <div className="robot-lab-camera">
                {preview && online && !(cameraError && (failedCameras.includes(index) || !failedCameras.length)) ? (
                  <RobotLabCameraFrame
                    key={`${index}/${previewStamp}/${indices.join(',')}`}
                    index={index}
                    indices={indices}
                    stamp={previewStamp}
                    large={large && (expandedCamera === null || expandedCamera === index)}
                    alt={ordinal === 0 ? 'Live view from the selected robot camera' : `Live view from camera ${index}`}
                    onError={() => {
                      setFailedCameras((failed) => (failed.includes(index) ? failed : [...failed, index]));
                      setCameraError(true);
                    }}
                  />
                ) : (
                  <div className="robot-lab-camera-empty">
                    <CameraIcon size={27} />
                    <span>
                      {cameraError && (failedCameras.includes(index) || !failedCameras.length)
                        ? indices.length === 1
                          ? 'Camera could not open. Check camera permissions on the robot computer or close another app using it.'
                          : `Camera ${index} could not open. Check camera permissions or close another app using it. Shared USB hubs may need cameras on separate ports.`
                        : 'Preview stopped'}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        <details className="robot-lab-camera-options" open={!large}>
          <summary>Camera selection and settings</summary>
          {online && state?.capabilities?.camera_mode_controls && availableCameras.length > 0 && (
            <RobotLabCameraSettings
              indices={indices}
              locked={state.session?.mode !== 'idle' || !!recording?.active}
              preview={preview}
              setPreview={setPreview}
              onApplied={() => {
                resetErrors();
                setPreviewStamp(Date.now());
              }}
            />
          )}
          <div className="robot-lab-camera-controls">
            {lab.step !== 4 && !policyViews.length && (
              <div className="robot-lab-field">
                <span className="robot-lab-field-label">Camera</span>
                {availableCameras.length ? (
                  <Segmented
                    label="Preview camera"
                    value={primaryIndex}
                    disabled={!online || !!recording?.active}
                    onChange={(index) => {
                      setCameraIndex(index);
                      resetErrors();
                    }}
                    options={availableCameras.map((camera) => ({
                      value: camera.index,
                      label: lab.cameraName(camera.index),
                    }))}
                  />
                ) : (
                  <span className="robot-lab-subtle">No camera available</span>
                )}
              </div>
            )}
            {lab.step !== 4 && !policyViews.length && multipleAvailable && availableCameras.length > 1 && (
              <fieldset className="robot-lab-form">
                <legend>Also preview and record</legend>
                {availableCameras
                  .filter((camera) => camera.index !== primaryIndex)
                  .map((camera) => (
                    <label className="robot-lab-check" key={camera.index}>
                      <input
                        type="checkbox"
                        aria-label={`Also use ${camera.name}`}
                        checked={indices.includes(camera.index)}
                        disabled={
                          !online ||
                          !!recording?.active ||
                          (!additionalCameraIndices.includes(camera.index) && indices.length >= 4)
                        }
                        onChange={(event) => {
                          setAdditionalCameraIndices(
                            event.target.checked
                              ? [...additionalCameraIndices.filter((index) => index !== cameraIndex), camera.index]
                              : additionalCameraIndices.filter((index) => index !== camera.index),
                          );
                          resetErrors();
                        }}
                      />
                      {lab.cameraName(camera.index)}
                    </label>
                  ))}
                <small>Selected views are saved together when Record camera is enabled.</small>
              </fieldset>
            )}
            {lab.step === 4 && (
              <p className="robot-lab-subtle">Choose which cameras to save in “Cameras for this dataset”.</p>
            )}
            <button
              className="secondary-button"
              disabled={!online || !availableCameras.length || !state?.capabilities?.camera}
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
          </div>
        </details>
      </section>
    </aside>
  );
}
