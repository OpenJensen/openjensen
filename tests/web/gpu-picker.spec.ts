import { expect, test, type Page } from '@playwright/test';

async function openGpuPicker(page: Page, local = false) {
  const timestamp = '2026-09-26T12:00:00Z';
  const writes: string[] = [];
  const unexpected: string[] = [];
  const runtimes = ['L4', 'T4', 'A100'].map(accelerator => ({
    id: `skypilot-gcp-${accelerator}`, accelerator, label: accelerator,
    execution: 'skypilot', provider: 'gcp', region: 'us-central1',
    enabled: true, device: 'cuda', training: true, simulation: false,
    training_model_ids: ['smolvla', 'pi05'],
  }));
  const localRuntime = {
    id: 'local-3070', label: 'Robotics lab RTX 3070', provider: 'local', execution: 'native',
    enabled: true, device: 'cuda', training: true, simulation: false, training_model_ids: ['smolvla', 'pi05'], gpu_memory_mib: 8192,
  };
  await page.route('**/api/v1/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() !== 'GET') {
      writes.push(`${request.method()} ${path}`);
      await route.fulfill({ status: 405, json: { detail: 'GPU picker tests never submit compute.' } });
      return;
    }
    const replies: Record<string, unknown> = {
      '/api/v1/health': { status: 'ok', version: 'gpu-picker-fixture' },
      '/api/v1/capabilities': [],
      '/api/v1/projects': [{ id: 'picker', name: 'GPU selection', created_at: timestamp }],
      '/api/v1/projects/picker/jobs': [],
      '/api/v1/projects/picker/artifacts': [],
      '/api/v1/policy-options': {
        runtimes: local ? [...runtimes, localRuntime] : runtimes,
        sources: [], training_models: [{ id: 'pi05', label: 'π₀.₅', model_id: 'lerobot/pi05_base', model_revision: 'a'.repeat(40), description: 'Generalist policy', backend: 'lerobot', methods: ['full'], minimum_gpu_memory_gb: 40, runtime_ids: runtimes.map(item => item.id) }],
        training_methods: [{ id: 'lora', label: 'LoRA', description: 'Train adapters.' }, { id: 'full', label: 'Full training', description: 'Train policy weights.' }],
        default_training_method: 'lora',
        compute: { local: { enabled: local, label: 'Robotics lab' }, gcp: { enabled: true, default_gpu: 'A100', disk_size_gb: 200, idle_minutes: 10 } },
        quantization_defaults: { cuda: { language: 'Q8_0', vision: null }, cpu: { language: 'Q8_0', vision: null }, note: '' },
      },
    };
    if (path in replies) await route.fulfill({ json: replies[path] });
    else {
      unexpected.push(path);
      await route.fulfill({ status: 405, json: { detail: 'Unexpected picker fixture request.' } });
    }
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Fine-tune', exact: true }).click();
  await page.getByRole('navigation', { name: 'Training setup' }).getByRole('button', { name: 'Compute', exact: true }).click();
  const picker = page.getByRole('combobox', { name: 'GPU', exact: true });
  await expect(picker).toBeVisible();
  await expect(picker).toHaveAttribute('value', 'A100');
  return { picker, writes, unexpected };
}

test('GPU picker supports keyboard navigation, explicit selection and Escape without submitting', async ({ page }) => {
  const { picker, writes, unexpected } = await openGpuPicker(page);
  await picker.focus();
  await page.keyboard.press('ArrowDown');
  const menu = page.getByRole('listbox', { name: 'GPU', exact: true });
  await expect(menu).toBeVisible();
  await expect(menu.getByRole('option')).toHaveCount(3);
  await expect(menu.getByRole('option', { name: 'A100', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Home');
  await expect(picker).toHaveAttribute('aria-activedescendant', (await menu.getByRole('option', { name: 'L4', exact: true }).getAttribute('id'))!);
  await expect(picker).toHaveAttribute('value', 'A100');
  await page.keyboard.press('ArrowDown');
  await expect(picker).toHaveAttribute('aria-activedescendant', (await menu.getByRole('option', { name: 'T4', exact: true }).getAttribute('id'))!);
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('End');
  await expect(picker).toHaveAttribute('aria-activedescendant', (await menu.getByRole('option', { name: 'A100', exact: true }).getAttribute('id'))!);
  await page.keyboard.press('Home');
  await page.keyboard.press('Enter');
  await expect(menu).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  await expect(picker).toBeFocused();
  await page.keyboard.press('Space');
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Escape');
  await expect(menu).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  await expect(picker).toBeFocused();
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('changing to a large model removes undersized GPUs and clears the previous T4 selection', async ({ page }) => {
  const { picker, writes, unexpected } = await openGpuPicker(page, true);
  await picker.click();
  await page.getByRole('option', { name: 'T4', exact: true }).click();
  const setup = page.getByRole('navigation', { name: 'Training setup' });
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await page.getByRole('radio', { name: 'π₀.₅', exact: true }).locator('..').click();
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await expect(picker).toHaveAttribute('value', 'A100');
  await expect(page.getByText(/requires at least 40 GB of GPU memory/)).toBeVisible();
  await picker.click();
  const menu = page.getByRole('listbox', { name: 'GPU', exact: true });
  await expect(menu.getByRole('option')).toHaveCount(1);
  await expect(menu.getByRole('option', { name: 'A100', exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('4');
  await page.getByRole('spinbutton', { name: 'Batch size', exact: true }).fill('3');
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await page.getByRole('radio', { name: 'SmolVLA', exact: true }).locator('..').click();
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('64');
  await setup.getByRole('button', { name: 'Model', exact: true }).click();
  await page.getByRole('radio', { name: 'π₀.₅', exact: true }).locator('..').click();
  await setup.getByRole('button', { name: 'Compute', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Training settings/ }).click();
  await expect(page.getByRole('spinbutton', { name: 'Batch size', exact: true })).toHaveValue('3');
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

test('GPU picker supports pointer choice, typeahead, Tab and outside dismissal', async ({ page }) => {
  const { picker, writes, unexpected } = await openGpuPicker(page);
  await picker.click();
  await page.getByRole('option', { name: 'T4', exact: true }).click();
  await expect(picker).toHaveAttribute('value', 'T4');
  await expect(picker).toBeFocused();
  await page.keyboard.press('l');
  await page.keyboard.press('Enter');
  await expect(picker).toHaveAttribute('value', 'L4');
  await picker.click();
  await page.keyboard.press('End');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('listbox')).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  await expect(picker).not.toBeFocused();
  await picker.click();
  // The menu can flip above and cover the section heading on mobile.
  await page.locator('body').click({ position: { x: 2, y: 2 } });
  await expect(page.getByRole('listbox')).toHaveCount(0);
  await expect(picker).toHaveAttribute('value', 'L4');
  expect(writes).toEqual([]);
  expect(unexpected).toEqual([]);
});

for (const theme of ['Light', 'Dark']) {
  test(`GPU picker menu fits 320px and shows memory badges in ${theme.toLowerCase()} mode`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 320, height: 740 });
    const { picker, writes, unexpected } = await openGpuPicker(page, true);
    await page.getByRole('button', { name: theme, exact: true }).click();
    await picker.click();
    const menu = page.getByRole('listbox', { name: 'GPU', exact: true });
    await expect(menu.getByRole('option')).toHaveCount(4);
    for (const [gpu, memory] of [['L4', '24 GB'], ['T4', '16 GB'], ['A100', '40 GB']]) {
      await expect(menu.getByRole('option', { name: gpu, exact: true }).locator('.gpu-picker-memory')).toHaveText(memory);
    }
    const bounds = await menu.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(320);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
    const screenshot = testInfo.outputPath(`gpu-picker-${theme.toLowerCase()}-320.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    await testInfo.attach(`gpu-picker-${theme.toLowerCase()}-320`, { path: screenshot, contentType: 'image/png' });
    await menu.getByRole('option', { name: 'Robotics lab RTX 3070', exact: true }).click();
    await expect(picker).toHaveAttribute('value', 'local-3070');
    await expect(picker).toContainText('Robotics lab RTX 3070');
    expect(writes).toEqual([]);
    expect(unexpected).toEqual([]);
  });
}
