'use client';

import { useRef, useState, type ChangeEvent } from 'react';
import {
  robotApi,
  robotJoints,
  type CalibrationDocument,
  type MotorCalibration,
  type RobotJoint,
} from '@/lib/robot-api';
import type { RobotLabState } from './robot-lab-state';

const maximumFileBytes = 16_384;

type ImportFile = {
  name: string;
  document: unknown;
  source: string;
  calibration: Record<RobotJoint, MotorCalibration>;
};

export function RobotLabCalibrationTransfer({ lab }: { lab: RobotLabState }) {
  const { active, busy, online, perform, port, portConnected, robotId, role, selectedProfile, setProbe, state } = lab;
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<ImportFile | null>(null);
  const [replace, setReplace] = useState(false);
  const [reading, setReading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const supported = !!state?.capabilities?.calibration_transfer;
  const unavailable = active || busy || reading || !online || !supported || !/^[A-Za-z0-9_-]{1,64}$/.test(robotId);

  async function exportFile() {
    setError(null);
    setReading(true);
    try {
      const exported = await robotApi.exportCalibration(robotId, role);
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(exported, null, 2) + '\n'], { type: 'application/json' }),
      );
      const link = document.createElement('a');
      link.href = url;
      link.download = `${robotId}-${role}-calibration.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not export calibration.');
    } finally {
      setReading(false);
    }
  }

  async function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0];
    event.target.value = '';
    if (!chosen) return;
    setFile(null);
    setReplace(false);
    setError(null);
    setReading(true);
    try {
      if (chosen.size > maximumFileBytes) throw new Error('Choose a calibration JSON file smaller than 16 KiB.');
      const document: unknown = JSON.parse(await chosen.text());
      if (!document || typeof document !== 'object' || Array.isArray(document))
        throw new Error('Choose an OpenJensen export or LeRobot calibration JSON file.');
      const wrapped = 'format' in document;
      const exported = document as CalibrationDocument;
      if (wrapped && (exported.format !== 'openjensen.so101.calibration' || exported.version !== 1))
        throw new Error('This calibration export format is not supported.');
      if (wrapped && exported.role !== role)
        throw new Error(`This file is for a ${exported.role} arm. Select that arm before importing.`);
      const calibration = (wrapped ? exported.calibration : document) as Record<RobotJoint, MotorCalibration>;
      if (!calibration || typeof calibration !== 'object' || Object.keys(calibration).length !== 6)
        throw new Error('The calibration file must contain all six SO-101 joints.');
      for (const joint of robotJoints) {
        const motor = calibration[joint.name];
        if (
          !motor ||
          typeof motor !== 'object' ||
          !['id', 'drive_mode', 'homing_offset', 'range_min', 'range_max'].every((key) =>
            Number.isInteger(motor[key as keyof MotorCalibration]),
          ) ||
          motor.id !== joint.id ||
          motor.drive_mode !== 0 ||
          motor.homing_offset < -2047 ||
          motor.homing_offset > 2047 ||
          motor.range_min < 0 ||
          motor.range_min >= motor.range_max ||
          motor.range_max > 4095
        )
          throw new Error(`Invalid calibration values for ${joint.label.toLowerCase()}.`);
      }
      setFile({
        name: chosen.name,
        document,
        calibration,
        source: wrapped ? `${exported.robot_id} · ${exported.role}` : 'LeRobot calibration file',
      });
    } catch (caught) {
      setError(
        caught instanceof SyntaxError
          ? 'This file is not valid JSON.'
          : caught instanceof Error
            ? caught.message
            : 'Could not read calibration file.',
      );
    } finally {
      setReading(false);
    }
  }

  return (
    <section className="robot-lab-transfer" aria-label="Transfer calibration">
      <h3>Transfer calibration</h3>
      <p className="robot-lab-subtle">
        Move saved calibration between computers or profile names for the same physical arm.
      </p>
      <div className="robot-lab-actions">
        <button
          className="secondary-button"
          disabled={unavailable || !selectedProfile || selectedProfile.valid === false}
          onClick={() => void exportFile()}
        >
          Export calibration
        </button>
        <button className="secondary-button" disabled={unavailable} onClick={() => input.current?.click()}>
          Import calibration
        </button>
        <input
          ref={input}
          type="file"
          accept=".json,application/json"
          aria-label="Calibration JSON file"
          hidden
          disabled={unavailable}
          onChange={(event) => void chooseFile(event)}
        />
      </div>
      {!supported && online && (
        <p className="robot-lab-subtle">Update the robot companion to transfer calibration files.</p>
      )}
      {error && <p role="alert">{error}</p>}
      {file && (
        <div role="region" aria-label="Review calibration import" className="robot-lab-import-review">
          <h4>Review calibration import</h4>
          <p>
            <strong>{file.name}</strong>
            <br />
            From {file.source}
            <br />
            Save to{' '}
            <strong>
              {robotId} · {role}
            </strong>
          </p>
          <div className="robot-lab-table-wrap">
            <table className="robot-lab-table">
              <thead>
                <tr>
                  <th>Joint</th>
                  <th>Zero offset</th>
                  <th>Min</th>
                  <th>Max</th>
                </tr>
              </thead>
              <tbody>
                {robotJoints.map((joint) => (
                  <tr key={joint.name}>
                    <td>{joint.label}</td>
                    <td>{file.calibration[joint.name].homing_offset}</td>
                    <td>{file.calibration[joint.name].range_min}</td>
                    <td>{file.calibration[joint.name].range_max}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="robot-lab-subtle">
            {portConnected
              ? 'The connected motors will be checked before saving. Import leaves motor settings unchanged.'
              : 'The profile will be saved on the robot computer. Connect and check the same arm before control.'}
          </p>
          {selectedProfile && (
            <label className="robot-lab-check">
              <input
                type="checkbox"
                checked={replace}
                disabled={unavailable}
                onChange={(event) => setReplace(event.target.checked)}
              />
              Replace this profile’s saved calibration. Keep a backup of the previous file.
            </label>
          )}
          <div className="robot-lab-actions">
            <button
              className="primary-button"
              disabled={unavailable || (!!selectedProfile && !replace)}
              onClick={() => {
                setError(null);
                void perform(
                  portConnected
                    ? 'Calibration imported. Connected motor settings match.'
                    : 'Calibration imported. Check the connected arm before control.',
                  () =>
                    robotApi.importCalibration({
                      robot_id: robotId,
                      role,
                      document: file.document,
                      ...(portConnected ? { port } : {}),
                      replace_existing: replace,
                    }),
                  () => {
                    setFile(null);
                    setReplace(false);
                    setProbe(null);
                  },
                );
              }}
            >
              Save imported calibration
            </button>
            <button
              className="secondary-button"
              disabled={busy}
              onClick={() => {
                setFile(null);
                setReplace(false);
              }}
            >
              Cancel import
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
