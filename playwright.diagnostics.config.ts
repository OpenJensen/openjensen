import { defineConfig, devices } from '@playwright/test';
import base from './playwright.config';

const python = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
const command = process.env.CI ? 'uv run --frozen python tests/web/serve.py' : `${python} tests/web/serve.py`;

export default defineConfig({
  ...base,
  workers: 2,
  // Import and evaluation can each use the shared 45s worker budget.
  timeout: 120_000,
  use: { ...base.use, baseURL: 'http://127.0.0.1:8767' },
  projects: [
    { name: 'desktop', testMatch: 'diagnostics.spec.ts', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile', testMatch: 'diagnostics.spec.ts', use: { ...devices['Pixel 7'], defaultBrowserType: 'chromium' } },
  ],
  webServer: [
    { command, url: 'http://127.0.0.1:8767/api/v1/health', reuseExistingServer: false,
      env: { FIREBIRD_BROWSER_PORT: '8767', FIREBIRD_BROWSER_EMPTY_RUNTIME: '1' } },
    { command, url: 'http://127.0.0.1:8766/api/v1/health', reuseExistingServer: false,
      env: { FIREBIRD_BROWSER_PORT: '8766', FIREBIRD_BROWSER_NATIVE_FIXTURE: '1' } },
  ],
});
