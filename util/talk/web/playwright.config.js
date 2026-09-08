import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:18585',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'node e2e/start-server.mjs',
    url: 'http://127.0.0.1:18585/api/health',
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
