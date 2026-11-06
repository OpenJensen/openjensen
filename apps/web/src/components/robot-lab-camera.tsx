'use client';

import { useState } from 'react';
import { robotApi } from '@/lib/robot-api';
import { CameraIcon } from './robot-lab-shared';
import { RobotLabCameraSettings } from './robot-lab-camera-settings';
import type { RobotLabState } from './robot-lab-state';

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
  const multipleAvailable = !!state?.capabilities?.multi_camera_recording;
  const indices = multipleAvailable
    ? recording?.active && recording.camera_indices?.length
      ? recording.camera_indices
      : selectedCameraIndices.length
        ? selectedCameraIndices
        : [cameraIndex]
    : [cameraIndex];
  const primaryIndex = recording?.active ? (indices[0] ?? cameraIndex) : cameraIndex;
  function resetErrors() {
    setCameraError(false);
    setFailedCameras([]);
  }
  return (
    <aside className="robot-lab-side">
      <section className="robot-lab-card" aria-label="Camera preview">
        <div className="robot-lab-card-header">
          <h3>Camera preview</h3>
        </div>
        {indices.map((index, ordinal) => (
          <div key={index}>
            {indices.length > 1 && <strong>{availableCameras.find((camera) => camera.index === index)?.name}</strong>}
            <div className="robot-lab-camera">
              {preview && online && !(cameraError && (failedCameras.includes(index) || !failedCameras.length)) ? (
                <>
                  <img
                    alt={ordinal === 0 ? 'Live view from the selected robot camera' : `Live view from camera ${index}`}
                    src={robotApi.cameraUrl(index, previewStamp, indices)}
                    onError={() => {
                      setFailedCameras((failed) => (failed.includes(index) ? failed : [...failed, index]));
                      setCameraError(true);
                    }}
                  />
                  <span className="robot-lab-camera-live">LIVE</span>
                </>
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
          <label>
            Camera
            <select
              value={primaryIndex}
              aria-label="Preview camera"
              onChange={(event) => {
                setCameraIndex(Number(event.target.value));
                resetErrors();
              }}
              disabled={!online || !availableCameras.length || recording?.active}
            >
              {!availableCameras.length && <option value={0}>No camera available</option>}
              {availableCameras.map((camera) => (
                <option key={camera.index} value={camera.index}>
                  {camera.name} · {camera.width}×{camera.height}
                </option>
              ))}
            </select>
          </label>
          {multipleAvailable && availableCameras.length > 1 && (
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
                    {camera.name}
                  </label>
                ))}
              <small>Selected views are saved together when Record camera is enabled.</small>
            </fieldset>
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
      </section>
    </aside>
  );
}
