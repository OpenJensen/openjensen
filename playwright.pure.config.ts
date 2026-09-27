import { defineConfig } from '@playwright/test';

// Contract and presentation helpers need neither a browser nor a built web app.
// Keep this lane serial and independent of the live developer workspace.
export default defineConfig({
  testDir: './tests/web',
  testMatch: '*-pure.spec.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: Boolean(process.env.CI),
  reporter: 'list',
  timeout: 15_000,
});
