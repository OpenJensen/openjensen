'use client';

import { useEffect, useState } from 'react';
import { robotApi, type CameraCaptureMode, type CameraCaptureSettings } from '@/lib/robot-api';

function modeKey(mode: CameraCaptureMode) {
  return `${mode.fourcc}:${mode.width}:${mode.height}:${mode.fps}`;
}
function chooseMode(camera: CameraCaptureSettings) {
  const configured = camera.configured ?? camera.actual;
  return (
    camera.modes.find((mode) => configured && modeKey(mode) === modeKey(configured)) ??
    camera.modes.find((mode) => mode.fourcc === 'MJPG' && mode.width === 640 && mode.height === 480) ??
    camera.modes[0]
  );
}

export function RobotLabCameraSettings({
  indices,
  locked,
  preview,
  setPreview,
  onApplied,
}: {
  indices: number[];
  locked: boolean;
  preview: boolean;
  setPreview: (value: boolean) => void;
  onApplied: () => void;
}) {
  const [cameras, setCameras] = useState<CameraCaptureSettings[]>([]);
  const [drafts, setDrafts] = useState<Record<number, CameraCaptureMode>>({});
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const selection = indices.join(',');
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setCameras([]);
    setError(null);
    setNotice(null);
    void robotApi
      .cameraSettings(selection.split(',').map(Number))
      .then((result) => {
        if (cancelled) return;
        setCameras(result.cameras);
        setDrafts(
          Object.fromEntries(
            result.cameras.flatMap((camera) => {
              const mode = chooseMode(camera);
              return mode ? [[camera.index, mode]] : [];
            }),
          ),
        );
      })
      .catch((caught: unknown) => {
        if (!cancelled) setError(caught instanceof Error ? caught.message : 'Could not read camera modes.');
      });
    return () => {
      cancelled = true;
    };
  }, [open, selection]);

  async function apply() {
    setPending(true);
    setError(null);
    setNotice(null);
    setPreview(false);
    try {
      const result = await robotApi.configureCameras(indices.map((index) => ({ index, mode: drafts[index] })));
      setCameras(result.cameras);
      setNotice('Camera settings saved. All selected cameras opened together.');
      onApplied();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not apply camera settings.');
    } finally {
      setPending(false);
      setPreview(preview);
    }
  }
  return (
    <details className="robot-lab-form" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Resolution and frame rate</summary>
      <p className="robot-lab-subtle">
        Modes reported by each camera. Applying checks that the selected cameras can open together. Dataset frame rate
        is set separately in recording settings.
      </p>
      {error && (
        <p className="robot-lab-error" role="alert">
          {error}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
      {open && !cameras.length && !error && <p role="status">Reading camera modes…</p>}
      {cameras.map((camera) => {
        const draft = drafts[camera.index];
        if (!draft) return <p key={camera.index}>No selectable native modes reported for camera {camera.index}.</p>;
        const formats = Array.from(new Set(camera.modes.map((mode) => mode.fourcc)));
        const resolutions = Array.from(
          new Set(
            camera.modes.filter((mode) => mode.fourcc === draft.fourcc).map((mode) => `${mode.width}×${mode.height}`),
          ),
        );
        const rates = camera.modes.filter(
          (mode) => mode.fourcc === draft.fourcc && mode.width === draft.width && mode.height === draft.height,
        );
        return (
          <fieldset key={camera.index} disabled={locked || pending}>
            <legend>Camera {camera.index}</legend>
            <label>
              Format
              <select
                aria-label={`Camera ${camera.index} format`}
                value={draft.fourcc}
                onChange={(event) => {
                  const mode =
                    camera.modes.find(
                      (mode) =>
                        mode.fourcc === event.target.value &&
                        mode.width === draft.width &&
                        mode.height === draft.height,
                    ) ?? camera.modes.find((mode) => mode.fourcc === event.target.value);
                  if (mode) setDrafts((values) => ({ ...values, [camera.index]: mode }));
                }}
              >
                {formats.map((value) => (
                  <option key={value} value={value}>
                    {value === 'MJPG' ? 'MJPEG (compressed)' : value === 'YUYV' ? 'YUYV (uncompressed)' : value}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Resolution
              <select
                aria-label={`Camera ${camera.index} resolution`}
                value={`${draft.width}×${draft.height}`}
                onChange={(event) => {
                  const mode = camera.modes.find(
                    (mode) => mode.fourcc === draft.fourcc && `${mode.width}×${mode.height}` === event.target.value,
                  );
                  if (mode) setDrafts((values) => ({ ...values, [camera.index]: mode }));
                }}
              >
                {resolutions.map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
            <label>
              Capture frame rate
              <select
                aria-label={`Camera ${camera.index} frame rate`}
                value={modeKey(draft)}
                onChange={(event) => {
                  const mode = rates.find((mode) => modeKey(mode) === event.target.value);
                  if (mode) setDrafts((values) => ({ ...values, [camera.index]: mode }));
                }}
              >
                {rates.map((mode) => (
                  <option key={modeKey(mode)} value={modeKey(mode)}>
                    {mode.fps} fps
                  </option>
                ))}
              </select>
            </label>
            {camera.actual && (
              <small>
                Negotiated: {camera.actual.width}×{camera.actual.height} · {camera.actual.fps} fps ·{' '}
                {camera.actual.fourcc}
              </small>
            )}
            {camera.measured_fps !== null && <small>Measured capture: {camera.measured_fps} fps</small>}
          </fieldset>
        );
      })}
      {locked && (
        <p className="robot-lab-subtle">
          Stop teleoperation or policy control and finish the current dataset before changing camera modes.
        </p>
      )}
      <button
        className="secondary-button"
        disabled={
          locked ||
          pending ||
          !indices.length ||
          cameras.length !== indices.length ||
          indices.some((index) => !drafts[index])
        }
        onClick={() => void apply()}
      >
        {pending ? 'Applying camera settings…' : 'Apply camera settings'}
      </button>
    </details>
  );
}
