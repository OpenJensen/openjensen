import { expect, type Page } from '@playwright/test';

export async function transformationJobs(page: Page, operation: 'distillation' | 'quantization') {
  const history=page.getByRole('region',{name:`${operation === 'distillation' ? 'Distillation' : 'Quantization'} jobs`,exact:true});
  if(!await history.isVisible()) await page.getByRole('button',{name:/^(← )?(Back to jobs|All quantization jobs)$/}).click();
  await expect(history).toBeVisible();
  return history;
}
export async function openTransformationJob(page:Page,operation:'distillation'|'quantization',id:string) {
  const history=await transformationJobs(page,operation);
  await history.locator(`[data-job-id="${id}"]`).click();
}
export async function chooseTransformationModel(page:Page,operation:'distillation'|'quantization',name:string) {
  const choice=page.getByRole('button',{name,exact:true});
  if(!await choice.isVisible()) {
    const history=await transformationJobs(page,operation);
    await history.getByRole('button',{name:`Start a new ${operation}`,exact:true}).click();
  }
  await choice.click();
}
