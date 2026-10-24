import type { WorkspaceModule } from '../../apps/web/src/lib/api';

/** Every optional module installed, so mocked workspaces list each page in the sidebar. */
export const installedModules: WorkspaceModule[] = (
  ['robot-lab', 'teaching', 'augmentation', 'decision-lab', 'cloud-runs'] as const
).map((id) => ({ id, name: id, installed: true, setup: 'Installed by the browser fixture.' }));
