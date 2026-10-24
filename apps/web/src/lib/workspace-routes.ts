import type { IconName } from '@/components/icon';
import type { WorkspaceModule } from '@/lib/api';
import { basePath } from '@/lib/base-path';

export type ModuleId = WorkspaceModule['id'];
export type StageId =
  | 'dashboard'
  | 'datasets'
  | 'training'
  | 'distillation'
  | 'quantization'
  | 'evaluation'
  | 'simulation'
  | 'models'
  | 'settings'
  | ModuleId;
export type WorkspaceRoute = { name: string; icon: IconName; path: string; module?: true };

// Route keys match the URL segment. Optional modules use the same id as
// GET /api/v1/modules and only appear in navigation once installed.
export const workspaceRoutes: Record<StageId, WorkspaceRoute> = {
  dashboard: { name: 'Dashboard', icon: 'home', path: '/dashboard/' },
  'robot-lab': { name: 'Robot Lab', icon: 'robot', path: '/robot-lab/', module: true },
  teaching: { name: 'Teaching', icon: 'record', path: '/teaching/', module: true },
  datasets: { name: 'Datasets', icon: 'database', path: '/datasets/' },
  augmentation: { name: 'Augmentation', icon: 'spark', path: '/augmentation/', module: true },
  training: { name: 'Fine-tune', icon: 'sliders', path: '/training/' },
  distillation: { name: 'Distill', icon: 'funnel', path: '/distillation/' },
  quantization: { name: 'Quantize', icon: 'compress', path: '/quantization/' },
  evaluation: { name: 'Evaluate', icon: 'chart', path: '/evaluation/' },
  simulation: { name: 'Run', icon: 'play', path: '/simulation/' },
  'decision-lab': { name: 'Decision lab', icon: 'branch', path: '/decision-lab/', module: true },
  models: { name: 'Models', icon: 'box', path: '/models/' },
  'cloud-runs': { name: 'Cloud runs', icon: 'cloud', path: '/cloud-runs/', module: true },
  settings: { name: 'Settings', icon: 'gear', path: '/settings/' },
};

export const isModule = (id: StageId): id is ModuleId => !!workspaceRoutes[id].module;

export const navigationGroups: { name?: string; items: StageId[] }[] = [
  { items: ['dashboard'] },
  { name: 'Robot', items: ['robot-lab', 'teaching'] },
  { name: 'Data', items: ['datasets', 'augmentation'] },
  { name: 'Train', items: ['training', 'distillation', 'quantization'] },
  { name: 'Test', items: ['evaluation', 'simulation', 'decision-lab'] },
  { name: 'Workspace', items: ['models', 'cloud-runs', 'settings'] },
];

export function workspaceStage(pathname: string): StageId {
  const local =
    basePath && (pathname === basePath || pathname.startsWith(`${basePath}/`))
      ? pathname.slice(basePath.length)
      : pathname;
  const path = `${local.replace(/\/+$/, '')}/`;
  const match = (Object.keys(workspaceRoutes) as StageId[]).find((id) => workspaceRoutes[id].path === path);
  return match ?? 'dashboard';
}
