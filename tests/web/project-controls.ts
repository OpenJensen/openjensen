import { expect, type Page } from '@playwright/test';

export const projectsDialog = (page: Page) => page.getByRole('dialog', { name: 'Projects', exact: true });

export async function selectProject(page: Page, id: string) {
  await page.getByRole('button', { name: 'Current project', exact: true }).click();
  await projectsDialog(page).locator(`[role="radio"][data-project-id="${id}"]`).click();
  await expect(projectsDialog(page)).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Current project', exact: true })).toHaveAttribute(
    'data-project-id',
    id,
  );
}

export async function openCreateProject(page: Page) {
  await page.getByRole('button', { name: 'Current project', exact: true }).click();
  await projectsDialog(page).getByRole('button', { name: 'New project', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Create project', exact: true })).toBeVisible();
}
