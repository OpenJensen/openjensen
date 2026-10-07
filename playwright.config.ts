import { defineConfig, devices } from '@playwright/test';

const python = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
const port = process.env.FIREBIRD_BROWSER_PORT ?? '8765';

export default defineConfig({
  testDir: './tests/web',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 4,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    reducedMotion: 'reduce',
  },
  projects: [
    { name: 'pure', testMatch: '*-pure.spec.ts' },
    { name: 'openapi', testMatch: 'openapi.spec.ts' },
    { name: 'workflow', testMatch: 'workflow.spec.ts', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
    { name: 'desktop', testMatch: ['workspace-routes.spec.ts', 'project-menu.spec.ts', 'model-library.spec.ts', 'dataset-submission.spec.ts', 'recording-preparation.spec.ts', 'distillation-model-choice.spec.ts', 'workspace-ux.spec.ts', 'native-lifecycle-handoff.spec.ts', 'native-replay.spec.ts', 'native-distillation.spec.ts', 'native-quantization.spec.ts', 'native-simulation.spec.ts', 'decision.spec.ts', 'teaching.spec.ts', 'managed-teaching.spec.ts', 'api-reference.spec.ts', 'augmentation.spec.ts', 'cloud-connections.spec.ts', 'gpu-picker.spec.ts', 'dataset-library.spec.ts', 'training-monitor.spec.ts', 'hf-settings.spec.ts', 'quantization.spec.ts', 'cloud-runs.spec.ts'], use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile', testMatch: ['workspace-routes.spec.ts', 'project-menu.spec.ts', 'model-library.spec.ts', 'dataset-submission.spec.ts', 'recording-preparation.spec.ts', 'distillation-model-choice.spec.ts', 'workspace-ux.spec.ts', 'native-lifecycle-handoff.spec.ts', 'native-replay.spec.ts', 'native-distillation.spec.ts', 'native-quantization.spec.ts', 'native-simulation.spec.ts', 'decision.spec.ts', 'teaching.spec.ts', 'managed-teaching.spec.ts', 'api-reference.spec.ts', 'augmentation.spec.ts', 'cloud-connections.spec.ts', 'gpu-picker.spec.ts', 'dataset-library.spec.ts', 'training-monitor.spec.ts', 'hf-settings.spec.ts', 'quantization.spec.ts', 'cloud-runs.spec.ts'], use: { ...devices['Pixel 7'], defaultBrowserType: 'chromium' } },
  ],
  // Exercise the production export and Python routing together. Refuse to reuse
  // a server: the developer's workspace and running application stay untouched.
  webServer: {
    command: process.env.CI ? 'uv run --frozen python tests/web/serve.py' : `${python} tests/web/serve.py`,
    url: `http://127.0.0.1:${port}/api/v1/health`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
