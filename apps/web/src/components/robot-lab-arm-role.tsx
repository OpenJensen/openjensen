'use client';

import { Icon } from '@/components/icon';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabArmRole({ lab }: { lab: RobotLabState }) {
  const { active, armLabel, busy, port, role, selectArmRole } = lab;
  return (
    <section className="robot-lab-arm-role" aria-labelledby="setup-arm-heading">
      <div className="robot-lab-arm-heading">
        <h2 id="setup-arm-heading">Which arm are you setting up?</h2>
        <span>Set up one arm at a time</span>
      </div>
      <div className="robot-lab-arm-choices" role="group" aria-label="Arm to set up">
        {(
          [
            {
              role: 'leader',
              title: 'Leader arm',
              description: 'You move this arm by hand.',
              detail: 'It guides the follower during teleoperation.',
              icon: 'sliders',
            },
            {
              role: 'follower',
              title: 'Follower arm',
              description: 'This arm performs the task.',
              detail: 'It follows the leader or software controls.',
              icon: 'robot',
            },
          ] as const
        ).map((arm) => (
          <button
            key={arm.role}
            type="button"
            className={`robot-lab-arm-choice${role === arm.role ? ' selected' : ''}`}
            aria-pressed={role === arm.role}
            disabled={active || busy}
            onClick={() => selectArmRole(arm.role)}
          >
            <Icon name={arm.icon} size={22} />
            <span className="robot-lab-arm-copy">
              <strong>{arm.title}</strong>
              <span>{arm.description}</span>
              <small>{arm.detail}</small>
            </span>
            <span className="robot-lab-arm-selected" aria-hidden="true">
              {role === arm.role ? <Icon name="check" size={16} /> : null}
            </span>
          </button>
        ))}
      </div>
      <div className="robot-lab-arm-context" role="status">
        <strong>Setting up: {armLabel} arm</strong>
        <span>{port || `Choose the ${role}’s USB port below.`}</span>
      </div>
      <p>
        Choose the role yourself; USB detection cannot tell the arms apart. Use a separate name and calibration for each
        arm.
      </p>
      {active && <p>Finish or stop the current session before switching arms.</p>}
    </section>
  );
}
