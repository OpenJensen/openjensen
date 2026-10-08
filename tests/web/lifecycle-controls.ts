import { expect, type Page } from '@playwright/test';

async function transformationReady(page: Page, operation: 'distillation' | 'quantization') {
  await expect(page).toHaveURL(new RegExp(`/${operation}/(?:[?#].*)?$`));
  const destination = operation === 'distillation'
    ? page.getByRole('region', { name: 'Distillation workspace', exact: true })
      .or(page.getByRole('region', { name: 'ACT distillation', exact: true }))
    : page.getByRole('region', { name: 'Quantization workspace', exact: true })
      .or(page.getByRole('region', { name: 'Native ACT quantization', exact: true }))
      .or(page.locator('.workflow-panel'));
  await expect(destination.first()).toBeVisible();
}

export async function transformationJobs(page: Page, operation: 'distillation' | 'quantization') {
  await transformationReady(page, operation);
  const history=page.getByRole('region',{name:`${operation === 'distillation' ? 'Distillation' : 'Quantization'} jobs`,exact:true});
  const back=page.getByRole('button',{name:/^(← )?(Back to jobs|All quantization jobs)$/});
  // Navigation may still be committing the destination when the helper starts.
  await expect.poll(async()=>await history.isVisible() || await back.isVisible()).toBe(true);
  if(!await history.isVisible()) await back.click();
  await expect(history).toBeVisible();
  return history;
}
export async function openTransformationJob(page:Page,operation:'distillation'|'quantization',id:string) {
  const history=await transformationJobs(page,operation);
  await history.locator(`[data-job-id="${id}"]`).click();
}
export async function chooseTransformationModel(page:Page,operation:'distillation'|'quantization',name:string) {
  await transformationReady(page, operation);
  const choice=page.getByRole('button',{name,exact:true,includeHidden:true});
  const existingGroup=choice.locator('xpath=ancestor::details[contains(@class,"model-run-group")]');
  if(await existingGroup.count() && !await choice.isVisible()) await existingGroup.locator('summary').click();
  if(!await choice.isVisible()) {
    const history=await transformationJobs(page,operation);
    await history.getByRole('button',{name:`Start a new ${operation}`,exact:true}).click();
  }
  const group = choice.locator('xpath=ancestor::details[contains(@class,"model-run-group")]');
  if(await group.count() && !await choice.isVisible()) await group.locator('summary').click();
  await choice.click();
}
