'use client';

import { useEffect, useState } from 'react';
import { robotApi, type CameraCaptureMode, type CameraCaptureSettings } from '@/lib/robot-api';
import { Segmented } from './ui/choice';

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
  name,
  locked,
  preview,
  setPreview,
  onApplied,
}: {
  indices: number[];
  name: (index: number) => string;
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
      <p className="robot-lab-subtle">Applying opens the cameras that are on together to check they work.</p>
      {error && (
        <p className="robot-lab-error" role="alert">
          {error}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
      {open && !cameras.length && !error && <p role="status">Reading camera modes…</p>}
      {cameras.map((camera) => {
        const draft = drafts[camera.index];
        if (!draft) return <p key={camera.index}>{name(camera.index)} reports no modes to choose from.</p>;
        const formats = Array.from(new Set(camera.modes.map((mode) => mode.fourcc)));
        const resolutions = Array.from(
          new Set(
            camera.modes.filter((mode) => mode.fourcc === draft.fourcc).map((mode) => `${mode.width}×${mode.height}`),
          ),
        );
        const rates = camera.modes.filter(
          (mode) => mode.fourcc === draft.fourcc && mode.width === draft.width && mode.height === draft.height,
        );
        const pick = (mode: CameraCaptureMode | undefined) =>
          mode && setDrafts((values) => ({ ...values, [camera.index]: mode }));
        const disabled = locked || pending;
        return (
          <fieldset key={camera.index} disabled={disabled}>
            <legend>{name(camera.index)}</legend>
            <div className="choice-field">
              <span>Format</span>
              <Segmented
                label={`Camera ${camera.index} format`}
                value={draft.fourcc}
                disabled={disabled}
                options={formats.map((value) => ({
                  value,
                  label: value === 'MJPG' ? 'MJPEG (compressed)' : value === 'YUYV' ? 'YUYV (uncompressed)' : value,
                }))}
                onChange={(value) =>
                  pick(
                    camera.modes.find(
                      (mode) => mode.fourcc === value && mode.width === draft.width && mode.height === draft.height,
                    ) ?? camera.modes.find((mode) => mode.fourcc === value),
                  )
                }
              />
            </div>
            <div className="choice-field">
              <span>Resolution</span>
              <Segmented
                label={`Camera ${camera.index} resolution`}
                value={`${draft.width}×${draft.height}`}
                disabled={disabled}
                options={resolutions.map((value) => ({ value, label: value }))}
                onChange={(value) =>
                  pick(
                    camera.modes.find(
                      (mode) => mode.fourcc === draft.fourcc && `${mode.width}×${mode.height}` === value,
                    ),
                  )
                }
              />
            </div>
            <div className="choice-field">
              <span>Capture frame rate</span>
              <Segmented
                label={`Camera ${camera.index} frame rate`}
                value={modeKey(draft)}
                disabled={disabled}
                options={rates.map((mode) => ({ value: modeKey(mode), label: `${mode.fps} fps` }))}
                onChange={(value) => pick(rates.find((mode) => modeKey(mode) === value))}
              />
            </div>
            {camera.actual && (
              <p className="robot-lab-subtle">
                Running at {camera.actual.width}×{camera.actual.height} · {camera.actual.fps} fps ·{' '}
                {camera.actual.fourcc}
                {camera.measured_fps !== null && ` · measured ${camera.measured_fps} fps`}
              </p>
            )}
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
