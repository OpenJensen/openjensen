'use client';

import { CameraIcon } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabRecordingCameras({ lab }: { lab: RobotLabState }) {
  const { availableCameras, busy, online, recording, recordingCameraIndices, toggleRecordingCamera } = lab;
  return (
    <section className="robot-lab-recording-cameras" aria-label="Cameras for this dataset">
      <div className="robot-lab-calibration-heading">
        <h3>Cameras for this dataset</h3>
        <span>{recordingCameraIndices.length} selected</span>
      </div>
      <label className="robot-lab-check">
        <input
          type="checkbox"
          checked={recording?.active ? !!recordingCameraIndices.length : lab.recordCamera}
          disabled={busy || !!recording?.active || !availableCameras.length || !online}
          onChange={(event) => lab.setRecordCamera(event.target.checked)}
        />
        <span>
          <strong>Record camera</strong>
          <small>
            {recordingCameraIndices.length
              ? 'Selected views are saved together with the robot data.'
              : 'Robot state and actions only.'}
          </small>
        </span>
      </label>
      <div className="robot-lab-recording-camera-list">
        {availableCameras.map((camera) => {
          const checked = recordingCameraIndices.includes(camera.index);
          return (
            <label
              key={camera.index}
              className={checked ? 'robot-lab-recording-camera selected' : 'robot-lab-recording-camera'}
            >
              <input
                type="checkbox"
                aria-label={`Record ${camera.name}`}
                checked={checked}
                disabled={!online || busy || !!recording?.active || (!checked && recordingCameraIndices.length >= 4)}
                onChange={(event) => toggleRecordingCamera(camera.index, event.target.checked)}
              />
              <CameraIcon size={20} />
              <span>
                <strong>{camera.name}</strong>
                <small>{checked ? 'Included in dataset' : 'Not recorded'}</small>
              </span>
            </label>
          );
        })}
      </div>
      <p className="robot-lab-subtle">
        {recording?.active
          ? 'Camera choices are fixed for this dataset. Finish it to choose a different set.'
          : availableCameras.length
            ? 'Tick the cameras you want to record. The preview follows your selection.'
            : 'Connect a camera and refresh devices to include images.'}
      </p>
    </section>
  );
}
