import { expect, test } from '@playwright/test';
import { trainingProgress } from '../../apps/web/src/lib/training-progress';
import type { JobEvent } from '../../apps/web/src/lib/api';
const event = (stage: string, message: string) => ({ stage, message, timestamp: '2026-10-08T11:00:00Z', sequence: 1 } as JobEvent);

test('setup observations distinguish compute, environment, and downloads without claiming readiness', () => {
  for (const [message, active] of [['Provisioning A100:1', 'compute'], ['Installing worker dependencies', 'environment'], ['Downloading the pinned model from Hugging Face', 'inputs']]) {
    const stages = trainingProgress('running', 'preparing', message, [event('preparing', message)]);
    expect(stages.find(stage => stage.state === 'active')?.id).toBe(active);
    expect(stages.at(-1)?.state).toBe('pending');
  }
});
test('repeated validation and checkpoint events stay in training; terminal failure never means ready', () => {
  const events = [event('training', 'Optimizer step 10'), event('checkpoint', 'Checkpoint 10 saved locally'), event('validation', 'Validating')];
  expect(trainingProgress('running', 'validation', 'Validating', events).find(stage => stage.state === 'active')?.id).toBe('training');
  const stopped = trainingProgress('failed', 'failed', 'CUDA out of memory', events);
  expect(stopped.find(stage => stage.state === 'stopped')?.id).toBe('training');
  expect(stopped.at(-1)?.state).toBe('pending');
  expect(trainingProgress('succeeded', 'completed', 'Complete', events).every(stage => stage.state === 'complete')).toBe(true);
});

test('shutdown and completion messages cannot replace allocation or receipt milestones', () => {
  const events = [event('preparing', 'Provisioning A100:1'), event('preparing', 'Verifying the dataset snapshot'), event('training', 'Optimizer step 2'), event('verifying', 'Checkpoint reload verified'), event('operation', 'Stopping Google Cloud resources'), event('operation', 'Completed policy.finetune')];
  const stages = trainingProgress('succeeded', 'completed', 'Training completed', events);
  expect(stages.find(stage => stage.id === 'accepted')?.message).toBeUndefined();
  expect(stages.find(stage => stage.id === 'compute')?.message).toBe('Provisioning A100:1');
  expect(stages.find(stage => stage.id === 'inputs')?.message).toBe('Verifying the dataset snapshot');
  expect(stages.find(stage => stage.id === 'verification')?.message).toBe('Checkpoint reload verified');
});
