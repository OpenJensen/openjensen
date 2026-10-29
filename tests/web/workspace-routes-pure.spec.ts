import { expect, test } from '@playwright/test';
import { navigationGroups, workspaceRoutes, workspaceStage } from '../../apps/web/src/lib/workspace-routes';
import { guideSectionId } from '../../apps/web/src/lib/guide-section';

test('robot navigation uses canonical AI teleoperation without changing saved module IDs', () => {
  expect(workspaceRoutes['robot-lab'].name).toBe('Initial setup');
  expect(workspaceRoutes.teaching.name).toBe('AI teleoperation');
  expect(workspaceRoutes.teaching.path).toBe('/ai-teleoperation/');
  expect(navigationGroups.find((group) => group.name === 'Robot Lab')?.items).toEqual(['robot-lab', 'teaching']);
  expect(workspaceStage('/ai-teleoperation/')).toBe('teaching');
  expect(workspaceStage('/teaching/')).toBe('teaching');
  expect(workspaceStage('/teaching')).toBe('teaching');
  expect(workspaceStage('/robot-lab/')).toBe('robot-lab');
});

test('renamed robot pages retain shared guide anchors', () => {
  expect(guideSectionId('AI teleoperation')).toBe('teaching');
  expect(guideSectionId('Initial setup')).toBe('robot-lab');
  expect(guideSectionId('Fine-tune')).toBe('fine-tune');
});

test('real to sim belongs to Data and has its own route', () => {
  expect(workspaceStage('/real-to-sim/')).toBe('real-to-sim');
  expect(navigationGroups.find((group) => group.name === 'Data')?.items).toContain('real-to-sim');
  expect(navigationGroups.find((group) => group.name === 'Test')?.items).not.toContain('real-to-sim');
  expect(guideSectionId('Real to sim')).toBe('real-to-sim');
});
