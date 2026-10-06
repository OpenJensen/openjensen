import { expect, type Page } from '@playwright/test';

export async function selectProject(page: Page, id: string) {
  await page.getByRole('button', { name: 'Current project', exact: true }).click();
  await page.getByRole('menu', { name: 'Projects', exact: true }).locator(`[role="menuitemradio"][data-project-id="${id}"]`).click();
  await expect(page.getByRole('button', { name: 'Current project', exact: true })).toHaveAttribute('data-project-id', id);
}

export async function openCreateProject(page: Page) {
  await page.getByRole('button', { name: 'Current project', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Create project…', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Create project', exact: true })).toBeVisible();
}
