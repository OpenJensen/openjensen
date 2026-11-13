/** Robot Lab's tabs, by the name used in links such as /robot-lab/#record. */
export const robotLabTabs = { devices: 0, teleoperation: 3, record: 4, 'run-policy': 5, 'ai-collection': 6 } as const;
export type RobotLabTab = keyof typeof robotLabTabs;

let requested: RobotLabTab | null = null;

/** Another page asks Robot Lab to open on one of its tabs. */
export function requestRobotLabTab(tab: RobotLabTab) {
  requested = tab;
}

/** The tab Robot Lab should open on: the one a page asked for, or the one named in the address. */
export function takeRobotLabTab(): RobotLabTab | null {
  const hash = typeof window === 'undefined' ? '' : window.location.hash.slice(1);
  const tab = requested ?? (hash in robotLabTabs ? (hash as RobotLabTab) : null);
  requested = null;
  return tab;
}
