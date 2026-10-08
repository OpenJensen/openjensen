import { expect, type Page } from '@playwright/test';

export async function transformationJobs(page: Page, operation: 'distillation' | 'quantization') {
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
export async function chooseTransformationModel(page:Page,operation:'distillation'|'quantization',name:string) {
  if(operation === 'quantization') {
    const id=name.split(' · ').at(-1)!;
    const checkpoint=page.getByLabel('Checkpoint',{exact:true});
    if(await checkpoint.count() && await checkpoint.locator(`option[value="${id}"]`).count()) {await checkpoint.selectOption(id);return;}
    const card=page.locator(`.model-overview-card[data-artifact-id="${id}"]`);
    if(!await card.isVisible()) {
      const modelsBack=page.getByRole('button',{name:'← Back to models',exact:true});
      if(await modelsBack.isVisible()) await modelsBack.click();
      else {const history=await transformationJobs(page,operation);await history.getByRole('button',{name:'Start a new quantization',exact:true}).click();}
    }
    await card.click();return;
  }
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
