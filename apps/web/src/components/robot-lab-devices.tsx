'use client';

import { Icon } from './icon';
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
    </section>
  );
}

export function RobotLabDeviceDetail({ lab, children }: { lab: RobotLabState; children: React.ReactNode }) {
  const device = lab.selectedDevice;
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

export function RobotLabControlDevices({ lab }: { lab: RobotLabState }) {
  const option = (device: RobotLabState['catalog'][number]) => (
    <option key={device.id} value={device.id} disabled={!device.connected || !device.role}>
      {device.name || device.port || device.robot_id}
      {!device.connected ? ' · Disconnected' : !device.role ? ' · Set up first' : ''}
    </option>
  );
  return (
    <div className="robot-lab-fields" role="group" aria-label="Session devices">
      <label>
        Follower device
        <select
          aria-label="Follower device"
          value={lab.catalog.find((d) => d.port === lab.port && d.robot_id === lab.robotId)?.id ?? ''}
          disabled={lab.active || lab.busy || !!lab.recording?.active}
          onChange={(e) => lab.selectControlDevice(e.target.value, 'follower')}
        >
          <option value="">Choose a device</option>
          {lab.catalog.map(option)}
        </select>
        <small>The arm controlled in this session.</small>
      </label>
      {lab.mode === 'leader' && (
        <label>
          Leader device
          <select
            aria-label="Leader device"
            value={lab.catalog.find((d) => d.port === lab.leaderPort && d.robot_id === lab.leaderId)?.id ?? ''}
            disabled={lab.active || lab.busy || !!lab.recording?.active}
            onChange={(e) => lab.selectControlDevice(e.target.value, 'leader')}
          >
            <option value="">Choose a different device</option>
            {lab.catalog.filter((d) => d.port !== lab.port).map(option)}
          </select>
          <small>The arm providing movement targets.</small>
        </label>
      )}
    </div>
  );
}
