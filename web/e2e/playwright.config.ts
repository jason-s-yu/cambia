// web/e2e/playwright.config.ts
// Playwright config for the end-to-end gameplay pass (cambia-934), Phase 1's exit criterion.
//
// The suite drives the real client against a real Go service, so it needs two processes running
// before it starts:
//
//   1. the game service on :8088, launched detached by tools/e2e/launch-dev-service.sh
//   2. the Vite dev server on :5180, which `webServer` below starts when nothing holds the port
//
// Only the second is startable from here. The service owns a database and a Redis queue and is
// deliberately left to the launch script, which stamps a manifest naming its pid, sha and log;
// globalSetup fails with that script's path when the port is dead, rather than letting every test
// fail on a blank page.
//
// One worker, never more: the seats share one service and one dev database, and two games at once
// makes a queue-order defect indistinguishable from a test-order defect. Pass 1 (note cambia-960)
// lost a run to exactly that.
import { defineConfig, devices } from '@playwright/test';

/** Where the client is served. Overridable so the same suite can run against a staging front. */
export const BASE_URL = process.env.CAMBIA_E2E_BASE_URL ?? 'http://localhost:5180';
/** Where the service answers. Only globalSetup talks to it directly; the client goes through the Vite proxy. */
export const API_URL = process.env.CAMBIA_E2E_API_URL ?? 'http://localhost:8088';

export default defineConfig({
  testDir: '.',
  // A full game runs turn by turn through two browser contexts against a live turn clock, so the
  // per-test budget is minutes, not seconds. The reconnect-cap test spends ~35s in backoff alone.
  timeout: 240_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  outputDir: '.artifacts/test-results',
  reporter: [['list'], ['html', { outputFolder: '.artifacts/report', open: 'never' }]],
  globalSetup: './support/globalSetup.ts',
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    actionTimeout: 15_000,
  },
  projects: [
    {
      name: 'chromium',
      // Tall enough that the action column under the felt is in the viewport without scrolling;
      // the table's own controls sit below the piles at 1280x720.
      use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 900 } },
    },
  ],
  webServer: {
    command: 'npm run dev',
    cwd: '..',
    url: BASE_URL,
    reuseExistingServer: true,
    timeout: 120_000,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
