'use client';

import { robotApi } from '@/lib/robot-api';
import { CameraIcon } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabCamera({ lab }: { lab: RobotLabState }) {
  const {
    availableCameras,
    cameraError,
    cameraIndex,
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
  return (
    <aside className="robot-lab-side">
      <section className="robot-lab-card" aria-label="Camera preview">
        <div className="robot-lab-card-header">
          <h3>Camera preview</h3>
        </div>
        <div className="robot-lab-camera">
          {preview && online && !cameraError ? (
            <>
              <img
                alt="Live view from the selected robot camera"
                src={robotApi.cameraUrl(cameraIndex, previewStamp)}
                onError={() => setCameraError(true)}
              />
              <span className="robot-lab-camera-live">LIVE</span>
            </>
          ) : (
            <div className="robot-lab-camera-empty">
              <CameraIcon size={27} />
              <span>
                {cameraError
                  ? 'Camera could not open. Check camera permissions on the robot computer or close another app using it.'
                  : 'Preview stopped'}
              </span>
            </div>
          )}
        </div>
        <div className="robot-lab-camera-controls">
          <label>
            Camera
            <select
              value={cameraIndex}
              aria-label="Preview camera"
              onChange={(event) => {
                setCameraIndex(Number(event.target.value));
                setCameraError(false);
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
          <button
            className="secondary-button"
            disabled={!online || !availableCameras.length || !state?.capabilities?.camera}
            onClick={() => {
              if (cameraError) {
                setCameraError(false);
                setPreviewStamp(Date.now());
                setPreview(true);
              } else setPreview(!preview);
            }}
          >
            {cameraError ? 'Retry camera' : preview ? 'Pause preview' : 'Start preview'}
          </button>
        </div>
      </section>
    </aside>
  );
}
