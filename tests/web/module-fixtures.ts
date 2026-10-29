import type { WorkspaceModule } from '../../apps/web/src/lib/api';

/** Explicit full installation for suites that exercise every feature and integration. */
const catalog = [
  ['robot-lab', 'Robot Lab'],
  ['teaching', 'Robot Lab'],
  ['datasets', 'Data'],
  ['augmentation', 'Data'],
  ['real-to-sim', 'Data'],
  ['training', 'Train'],
  ['distillation', 'Train'],
  ['quantization', 'Train'],
  ['evaluation', 'Test'],
  ['simulation', 'Test'],
  ['decision-lab', 'Test'],
  ['models', 'Workspace'],
  ['cloud-runs', 'Workspace'],
  ['google-cloud', 'Cloud & APIs'],
  ['openai', 'Cloud & APIs'],
  ['gemini', 'Cloud & APIs'],
  ['huggingface', 'Cloud & APIs'],
] as const;

export const installedModules: WorkspaceModule[] = catalog.map(([id, group]) => ({
  id,
  name: id === 'robot-lab' ? 'Initial setup' : id === 'teaching' ? 'AI teleoperation' : id,
  group,
  default_installed: id === 'datasets' || id === 'models',
  installed: true,
  available: true,
  enabled: true,
  configured: true,
  managed: false,
  setup: 'Installed by the browser fixture.',
  setup_url: '/guide/',
}));
