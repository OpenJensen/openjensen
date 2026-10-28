'use client';

import type { RobotRole } from '@/lib/robot-api';
import type { RobotLabState } from './robot-lab-state';

export function RobotLabArmRole({ lab }: { lab: RobotLabState }) {
  const { active, busy, role, selectArmRole } = lab;
  return (
    <div className="robot-lab-device-picker">
      <h2 id="setup-device-heading">Devices</h2>
      <select
        aria-labelledby="setup-device-heading"
        value={role}
        disabled={active || busy}
        title={active ? 'Finish or stop the current session before switching arms.' : 'Choose the arm to set up.'}
        onChange={(event) => selectArmRole(event.target.value as RobotRole)}
      >
        <option value="leader">Leader arm</option>
        <option value="follower">Follower arm</option>
      </select>
    </div>
  );
}
