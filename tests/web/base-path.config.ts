import { defineConfig, devices } from '@playwright/test';
import { resolve } from 'node:path';

const prefix = (process.env.NEXT_PUBLIC_BASE_PATH ?? '/firebird').replace(/\/+$/, '');
const port = process.env.FIREBIRD_PREFIX_SMOKE_PORT ?? '18766';
const python = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';

export default defineConfig({
  testDir: '.',
  testMatch: 'base-path-smoke.spec.ts',
  workers: 1,
  reporter: 'list',
  use: { ...devices['Desktop Chrome'], baseURL: `http://127.0.0.1:${port}`, screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  webServer: {
    cwd: resolve(__dirname, '../..'),
    command: `${python} tests/web/base_path_serve.py`,
    env: { NEXT_PUBLIC_BASE_PATH: prefix, FIREBIRD_PREFIX_SMOKE_PORT: port },
    url: `http://127.0.0.1:${port}${prefix}/api/v1/health`,
    reuseExistingServer: false,
    timeout: 30000,
  },
});
