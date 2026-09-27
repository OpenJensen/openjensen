import { defineConfig, devices } from '@playwright/test';
import { resolve } from 'node:path';

export default defineConfig({
  testDir: '.', testMatch: 'public-demo.spec.ts', workers: 1, reporter: 'list',
  use: { baseURL: 'http://127.0.0.1:18767', trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['Pixel 7'], defaultBrowserType: 'chromium' } },
  ],
  webServer: {
    cwd: resolve(__dirname, '../..'), command: '.venv/bin/python tests/web/base_path_serve.py',
    env: { NEXT_PUBLIC_BASE_PATH: '/firebird', FIREBIRD_PREFIX_SMOKE_PORT: '18767' },
    url: 'http://127.0.0.1:18767/firebird/api/v1/health', reuseExistingServer: false,
  },
});
