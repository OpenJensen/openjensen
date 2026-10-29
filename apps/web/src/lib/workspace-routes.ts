import type { IconName } from '@/components/icon';
import type { WorkspaceModule } from '@/lib/api';
import { basePath } from '@/lib/base-path';

export type ModuleId = WorkspaceModule['id'];
export type IntegrationModuleId = 'google-cloud' | 'openai' | 'gemini' | 'huggingface';
export type FeatureModuleId = Exclude<ModuleId, IntegrationModuleId>;
export type StageId = 'jobs' | 'dashboard' | 'modules' | 'cloud-apis' | 'settings' | FeatureModuleId;
export type WorkspaceRoute = { name: string; icon: IconName; path: string; module?: true };

// Stage keys preserve saved module IDs. Canonical addresses may differ from
// those IDs; legacy addresses stay mapped to the same workspace state.
export const workspaceRoutes: Record<StageId, WorkspaceRoute> = {
  jobs: { name: 'Jobs', icon: 'layers', path: '/jobs/' },
  dashboard: { name: 'Dashboard', icon: 'home', path: '/dashboard/' },
  'robot-lab': { name: 'Initial setup', icon: 'robot', path: '/robot-lab/', module: true },
  teaching: { name: 'AI teleoperation', icon: 'record', path: '/ai-teleoperation/', module: true },
  datasets: { name: 'Datasets', icon: 'database', path: '/datasets/', module: true },
  augmentation: { name: 'Augmentation', icon: 'spark', path: '/augmentation/', module: true },
  'real-to-sim': { name: 'Real to sim', icon: 'box', path: '/real-to-sim/', module: true },
  training: { name: 'Fine-tune', icon: 'sliders', path: '/training/', module: true },
  distillation: { name: 'Distill', icon: 'funnel', path: '/distillation/', module: true },
  quantization: { name: 'Quantize', icon: 'compress', path: '/quantization/', module: true },
  evaluation: { name: 'Evaluate', icon: 'chart', path: '/evaluation/', module: true },
  simulation: { name: 'Run', icon: 'play', path: '/simulation/', module: true },
  'decision-lab': { name: 'Decision Lab', icon: 'branch', path: '/decision-lab/', module: true },
  models: { name: 'Models', icon: 'box', path: '/models/', module: true },
  'cloud-runs': { name: 'Cloud runs', icon: 'cloud', path: '/cloud-runs/', module: true },
  modules: { name: 'Modules', icon: 'layers', path: '/modules/' },
  'cloud-apis': { name: 'Cloud & APIs', icon: 'cloud', path: '/cloud-apis/' },
  settings: { name: 'Settings', icon: 'gear', path: '/settings/' },
};

export const isModule = (id: StageId): id is FeatureModuleId => !!workspaceRoutes[id].module;

export const navigationGroups: { name?: string; items: StageId[] }[] = [
  { items: ['dashboard', 'jobs'] },
  { name: 'Robot Lab', items: ['robot-lab', 'teaching'] },
  { name: 'Data', items: ['datasets', 'augmentation', 'real-to-sim'] },
  { name: 'Train', items: ['training', 'distillation', 'quantization'] },
  { name: 'Test', items: ['evaluation', 'simulation', 'decision-lab'] },
  { name: 'Workspace', items: ['models', 'cloud-runs', 'modules', 'cloud-apis', 'settings'] },
];

export function moduleStage(id: ModuleId): StageId {
  return ['google-cloud', 'openai', 'gemini', 'huggingface'].includes(id) ? 'cloud-apis' : (id as FeatureModuleId);
}

export function workspaceStage(pathname: string): StageId {
  const local =
    basePath && (pathname === basePath || pathname.startsWith(`${basePath}/`))
      ? pathname.slice(basePath.length)
      : pathname;
  const path = `${local.replace(/\/+$/, '')}/`;
  if (path === '/teaching/') return 'teaching';
  const match = (Object.keys(workspaceRoutes) as StageId[]).find((id) => workspaceRoutes[id].path === path);
  return match ?? 'dashboard';
}
