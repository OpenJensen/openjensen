import { basePath } from '@/lib/base-path';

export const workspaceRoutes = [
  { name: 'Dataset', icon: 'database', path: '/datasets/' },
  { name: 'Fine-tune', icon: 'sliders', path: '/training/' },
  { name: 'Distill', icon: 'layers', path: '/distillation/' },
  { name: 'Quantize', icon: 'compress', path: '/quantization/' },
  { name: 'Evaluate', icon: 'chart', path: '/evaluation/' },
  { name: 'Run', icon: 'play', path: '/simulation/' },
  { name: 'Settings & diagnostics', icon: 'sliders', path: '/settings/' },
  { name: 'Cloud runs', icon: 'clock', path: '/cloud-runs/' },
  { name: 'Augmentation', icon: 'spark', path: '/augmentation/' },
  { name: 'Teaching', icon: 'plus', path: '/teaching/' },
  { name: 'Decision lab', icon: 'chart', path: '/decision-lab/' },
  { name: 'My models', icon: 'layers', path: '/models/' },
  { name: 'Dashboard', icon: 'layers', path: '/dashboard/' },
] as const;

export const navigationGroups = [
  { name: 'Overview', items: [12] },
  { name: 'Data', items: [0, 8, 9] },
  { name: 'Train', items: [1, 2, 3] },
  { name: 'Test', items: [4, 5, 10] },
  { name: 'Workspace', items: [11, 7, 6] },
];

export function workspaceStage(pathname: string) {
  const local = basePath && (pathname === basePath || pathname.startsWith(`${basePath}/`))
    ? pathname.slice(basePath.length) : pathname;
  const path = `${local.replace(/\/+$/, '')}/`;
  const index = workspaceRoutes.findIndex(route => route.path === path);
  return index < 0 ? 12 : index;
}
