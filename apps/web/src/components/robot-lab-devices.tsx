'use client';

import { useState } from 'react';
import { Icon } from './icon';
import { ChoiceCards } from './ui/choice';
import { Notice } from './robot-lab-shared';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabDevices({ lab }: { lab: RobotLabState }) {
  const { catalog, chooseDevice, online, devices } = lab;
  return (
    <section aria-label="Devices" className="robot-lab-device-library">
      <div className="robot-lab-library-heading">
        <div>
          <h2>Your devices</h2>
          <p>Connect a USB controller to add an arm. Configured devices stay here when disconnected.</p>
        </div>
        <span>{catalog.length} devices</span>
      </div>
      {online && !lab.state?.capabilities?.device_registry && (
        <Notice kind="warning">
          Install the updated robot connection service to save device names and roles, then refresh devices.
        </Notice>
      )}
      {!online && (
        <Notice kind="warning">
          Device connection service unavailable. Saved configurations return when the service reconnects.
        </Notice>
      )}
      {devices.isError && <Notice kind="warning">Device discovery failed. Refresh devices to retry.</Notice>}
      {!catalog.length && (
        <div className="robot-lab-device-empty">
          <Icon name="robot" size={32} />
          <h3>No devices yet</h3>
          <p>Connect an arm’s USB cable, then refresh devices.</p>
        </div>
      )}
      <div className="robot-lab-device-tiles">
        {catalog.map((device) => (
          <button
            key={device.id}
            data-testid={`device-${device.id}`}
            className="robot-lab-device-tile"
            onClick={() => chooseDevice(device)}
            disabled={lab.busy || lab.active}
          >
            <div className="robot-lab-device-tile-top">
              <span className="robot-lab-device-icon">
                <Icon name="robot" size={22} />
              </span>
              <span className={device.connected && online ? 'robot-lab-pass' : 'robot-lab-subtle'}>
                {device.connected && online ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <strong>{device.name || device.port || device.robot_id}</strong>
            <span className="robot-lab-device-address">{device.port || 'USB port not assigned'}</span>
            <span className="robot-lab-device-role">
              {device.role ? `Default ${device.role}` : 'Choose a role to set up'}
            </span>
            <div className="robot-lab-device-badges">
              <span>Motor IDs · {device.motor_ids ? 'Saved' : 'Needs setup'}</span>
              <span>
                Calibration ·{' '}
                {device.calibration === 'saved'
                  ? 'Saved'
                  : device.calibration === 'needs_attention'
                    ? 'Needs attention'
                    : 'Not saved'}
              </span>
            </div>
            <span className="robot-lab-device-open">
              Open device <Icon name="arrow" size={14} />
            </span>
          </button>
        ))}
      </div>
      {!!devices.data?.removed_devices?.length && (
        <details className="robot-lab-removed-devices">
          <summary>Removed devices ({devices.data.removed_devices.length})</summary>
          <p>Saved calibration files are kept. Restore a device to use its saved name and role again.</p>
          {devices.data.removed_devices.map((device) => (
            <div className="robot-lab-removed-device" key={device.id}>
              <span>{device.name || device.port || device.robot_id}</span>
              <button
                className="secondary-button"
                aria-label={`Restore ${device.name || device.port || device.robot_id}`}
                disabled={!online || lab.busy || lab.active || !!lab.recording?.active}
                onClick={() => void lab.restoreDevice(device.id)}
              >
                Restore device
              </button>
            </div>
          ))}
        </details>
      )}
    </section>
  );
}

export function RobotLabDeviceDetail({ lab, children }: { lab: RobotLabState; children: React.ReactNode }) {
  const device = lab.selectedDevice;
  const [removingId, setRemovingId] = useState<string | null>(null);
  if (!device) return null;
  return (
    <>
      <button
        className="text-button robot-lab-back"
        disabled={lab.busy || lab.active}
        onClick={() => lab.setSelectedDeviceId(null)}
      >
        ← All devices
      </button>
      <header className="robot-lab-device-detail-heading">
        <div>
          <h2>{device.name || device.port || device.robot_id}</h2>
          <p>
            {device.port || 'USB port not assigned'} · {device.connected && lab.online ? 'Connected' : 'Disconnected'}
          </p>
        </div>
      </header>
      {!!device.role && lab.state?.capabilities?.device_removal && (
        <div className="robot-lab-device-removal">
          {removingId === device.id ? (
            <section aria-label="Remove saved device">
              <p>
                Remove {device.name || device.port || device.robot_id} from your saved devices? Calibration files and
                recordings are kept. You can restore it from Removed devices.
              </p>
              {device.connected && <p>This connected controller will still appear by its USB address.</p>}
              <div className="robot-lab-actions">
                <button
                  className="secondary-button"
                  disabled={!lab.online || lab.busy || lab.active || !!lab.recording?.active}
                  onClick={() => void lab.removeDevice()}
                >
                  Remove saved device
                </button>
                <button className="secondary-button" disabled={lab.busy} onClick={() => setRemovingId(null)}>
                  Cancel
                </button>
              </div>
            </section>
          ) : (
            <button
              className="text-button"
              disabled={!lab.online || lab.busy || lab.active || !!lab.recording?.active}
              onClick={() => setRemovingId(device.id)}
            >
              Remove device
            </button>
          )}
        </div>
      )}
      {!device.role ? (
        <section className="robot-lab-role-onboarding" aria-label="Choose device role">
          <h3>How will you usually use this arm?</h3>
          <p>
            This sets the setup guide and default teleoperation assignment. You can change it here or assign either role
            in teleoperation.
          </p>
          <RoleButtons lab={lab} />
        </section>
      ) : (
        <>
          <section className="robot-lab-device-settings" aria-label="Device settings">
            <label>
              Device name
              <div className="robot-lab-name-controls">
                <input
                  aria-label="Device name"
                  value={lab.deviceName}
                  placeholder={device.port || 'Name this device'}
                  maxLength={64}
                  disabled={lab.busy || lab.active}
                  onChange={(e) => lab.setDeviceName(e.target.value)}
                />
                <button
                  className="secondary-button"
                  disabled={lab.busy || lab.active}
                  onClick={() => void lab.saveDeviceConfiguration()}
                >
                  {device.port ? 'Save name' : 'Save device'}
                </button>
              </div>
            </label>
            {!device.port && (
              <label>
                USB connection
                <select
                  aria-label="Device USB connection"
                  value={lab.port}
                  disabled={lab.busy || lab.active}
                  onChange={(e) => lab.updatePort(e.target.value)}
                >
                  <option value="">Choose connected USB device</option>
                  {lab.devices.data?.ports.map((p) => (
                    <option key={p.device} value={p.device}>
                      {p.device} · {p.description}
                    </option>
                  ))}
                </select>
                <small>Save the device to bind this profile to the selected connection.</small>
              </label>
            )}
            <div>
              <span className="robot-lab-settings-label">Default role</span>
              <RoleButtons lab={lab} />
            </div>
          </section>
          <nav className="robot-lab-device-sections" aria-label="Device sections">
            {['Overview', 'Motor IDs', 'Calibration'].map((title, index) => (
              <button
                key={title}
                className={lab.step === index ? 'selected' : ''}
                aria-current={lab.step === index ? 'page' : undefined}
                disabled={lab.busy}
                onClick={() => lab.openStep(index)}
              >
                {title}
              </button>
            ))}
          </nav>
          <section className="robot-lab-device-readiness" aria-label="Arm readiness">
            <span>Motor responses · {lab.stepStates[0].value}</span>
            <span>Motor IDs · {lab.mappingLabel}</span>
            <span>Calibration · {lab.calibrationLabel}</span>
            {!lab.active && (
              <button className="text-button" disabled={!lab.canUseArm || lab.busy} onClick={lab.nextAction.action}>
                {lab.nextAction.label}
              </button>
            )}
          </section>
          <section aria-label={lab.step === 2 ? 'Calibrate' : lab.step === 1 ? 'Motor IDs' : 'Device overview'}>
            {children}
          </section>
        </>
      )}
    </>
  );
}
function RoleButtons({ lab }: { lab: RobotLabState }) {
  return (
    <div className="robot-lab-role-buttons" role="group" aria-label="Default device role">
      {(['follower', 'leader'] as const).map((role) => (
        <button
          key={role}
          className="secondary-button"
          aria-pressed={lab.selectedDevice?.role === role}
          disabled={lab.busy || lab.active}
          onClick={() => void lab.saveDeviceConfiguration(role)}
        >
          {role === 'leader' ? 'Leader' : 'Follower'}
        </button>
      ))}
    </div>
  );
}

/**
 * The arm, or arms, a session uses. Every saved arm is shown as a card, so the choice
 * and each arm's state are visible at once; an arm that cannot be used says why.
 */
export function RobotLabControlDevices({
  lab,
  leader = lab.mode === 'leader',
  hideLeaders = false,
}: {
  lab: RobotLabState;
  leader?: boolean;
  /** Leave out arms set up as leaders: a policy drives the follower, and a leader only adds a wrong choice. */
  hideLeaders?: boolean;
}) {
  const locked = lab.active || lab.busy || !!lab.recording?.active;
  const card = (device: RobotLabState['catalog'][number]) => ({
    value: device.id,
    label: device.name || device.port || device.robot_id,
    detail: !device.connected
      ? 'Unplugged'
      : !device.role
        ? 'Set it up under Devices first'
        : device.calibration === 'saved'
          ? 'Connected · calibrated'
          : 'Connected · needs calibration',
    disabled: !device.connected || !device.role,
  });
  const follower = lab.catalog.find((d) => d.port === lab.port && d.robot_id === lab.robotId)?.id ?? null;
  // Only arms that are plugged in can be used, so only those are offered; one in use stays listed.
  const usable = (device: RobotLabState['catalog'][number], chosen: string | null) =>
    device.connected || (locked && device.id === chosen);
  const followers = lab.catalog.filter(
    (d) => usable(d, follower) && (!hideLeaders || d.role !== 'leader' || d.id === follower),
  );
  const leaderId = lab.catalog.find((d) => d.port === lab.leaderPort && d.robot_id === lab.leaderId)?.id ?? null;
  const leaders = lab.catalog.filter((d) => d.port !== lab.port && usable(d, leaderId));
  return (
    <div className="robot-lab-arms" role="group" aria-label="Session devices">
      <div className="robot-lab-arm-slot">
        <span className="robot-lab-field-label">{leader ? 'Follower arm · the one that moves' : 'Arm'}</span>
        {followers.length ? (
          <ChoiceCards
            label="Follower device"
            value={follower}
            options={followers.map(card)}
            disabled={locked}
            onChange={(id) => lab.selectControlDevice(id, 'follower')}
          />
        ) : (
          <p className="robot-lab-subtle" role="status">
            {lab.online
              ? 'No arm is plugged in. Connect its USB cable and power; it appears here by itself.'
              : 'The robot service is not reachable, so no arm can be used.'}
          </p>
        )}
      </div>
      {leader && (
        <div className="robot-lab-arm-slot">
          <span className="robot-lab-field-label">Leader arm · the one you hold</span>
          {leaders.length ? (
            <ChoiceCards
              label="Leader device"
              value={leaderId}
              options={leaders.map(card)}
              disabled={locked}
              onChange={(id) => lab.selectControlDevice(id, 'leader')}
            />
          ) : (
            <p className="robot-lab-subtle" role="status">
              Plug in a second arm to lead with.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
