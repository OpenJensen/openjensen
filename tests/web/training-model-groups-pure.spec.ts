import { expect, test } from '@playwright/test';
import { groupedTrainingModels, type TrainingModel } from '../../apps/web/src/lib/training-models';
const model = (id: string, status: TrainingModel['status'] = 'ready') => ({ id, status } as TrainingModel);

test('policy families stay separate, unknown policies remain visible, and coming-soon entries are last', () => {
  const groups = groupedTrainingModels([model('openvla', 'coming_soon'), model('act'), model('smolvla', 'setup_required'), model('vqbet'), model('multi_task_dit'), model('diffusion'), model('vla_jepa'), model('psi0'), model('custom')]);
  expect(groups.map(group => [group.id, group.models.map(item => item.id)])).toEqual([
    ['vla', ['smolvla', 'psi0']], ['action-sequence', ['act', 'vqbet']], ['diffusion', ['multi_task_dit', 'diffusion']], ['world-model', ['vla_jepa']], ['other', ['custom']], ['coming-soon', ['openvla']],
  ]);
  expect(groupedTrainingModels([model('pi0'), model('openvla_oft', 'ready')]).map(group => group.id)).toEqual(['vla']);
});
