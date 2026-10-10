import { expect, type Page } from '@playwright/test';

export function selectedQuantizationModel(page: Page) {
  return page.getByLabel('Checkpoint', { exact: true })
    .or(page.getByRole('group', { name: /^(Policy|My model)$/ }).locator('input:checked'));
}

export function quantizationModelOption(page: Page, id: string) {
  return page.getByLabel('Checkpoint', { exact: true }).locator(`option[value="${id}"]`)
    .or(page.getByRole('group', { name: /^(Policy|My model)$/ }).locator(`input[value="${id}"]`));
}

export async function selectQuantizationModel(page: Page, id: string) {
  const checkpoint = page.getByLabel('Checkpoint', { exact: true });
  if (await checkpoint.isVisible()) await checkpoint.selectOption(id);
  else await quantizationModelOption(page, id).check();
  await expect(selectedQuantizationModel(page)).toHaveValue(id);
}

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
  const back=page.getByRole('button',{name:/^(← )?(Back to jobs|All quantization jobs|Back to models)$/});
  // Navigation may still be committing the destination when the helper starts.
  await expect.poll(async()=>await history.isVisible() || await back.isVisible()).toBe(true);
  if(!await history.isVisible()) await back.click();
  if(!await history.isVisible()) await page.getByRole('button',{name:/^(← )?Back to jobs$/}).click();
  await expect(history).toBeVisible();
  return history;
}
export async function openTransformationJob(page:Page,operation:'distillation'|'quantization',id:string) {
  const history=await transformationJobs(page,operation);
  await history.locator(`[data-job-id="${id}"]`).click();
}
export async function chooseTransformationModel(page:Page,operation:'distillation'|'quantization',name:string,runId?:string) {
  await transformationReady(page, operation);
  if(operation === 'quantization') {
    const id=name.split(' · ').at(-1)!;
    const checkpoint=page.getByLabel('Checkpoint',{exact:true});
    async function selectCheckpoint() {
      if (await checkpoint.inputValue() !== id) await checkpoint.selectOption(id);
      await expect(checkpoint).toHaveValue(id);
    }
    if(await checkpoint.count() && await checkpoint.locator(`option[value="${id}"]`).count()) {await selectCheckpoint();return;}
    const card=page.locator(runId ? `.model-overview-card[data-model-run="${runId}"]` : `.model-overview-card[data-artifact-id="${id}"]`);
    if(!await card.isVisible()) {
      const modelsBack=page.getByRole('button',{name:'← Back to models',exact:true});
      if(await modelsBack.isVisible()) await modelsBack.click();
      else {const history=await transformationJobs(page,operation);await history.getByRole('button',{name:'Start a new quantization',exact:true}).click();}
    }
    await card.click();
    await expect(checkpoint).toBeVisible();
    await selectCheckpoint();
    return;
  }
  const choice=page.getByRole('button',{name,exact:true,includeHidden:true});
  const existingGroup=choice.locator('xpath=ancestor::details[contains(@class,"model-run-group")]');
  if(await existingGroup.count() && !await choice.isVisible()) await existingGroup.locator('summary').click();
  if(!await choice.isVisible()) {
    const history=await transformationJobs(page,operation);
    await history.getByRole('button',{name:`Start a new ${operation}`,exact:true}).click();
  }
  await choice.waitFor({ state: 'attached' });
  const group = choice.locator('xpath=ancestor::details[contains(@class,"model-run-group")]');
  if(await group.count() && !await choice.isVisible()) await group.locator('summary').click();
  await choice.click();
}
