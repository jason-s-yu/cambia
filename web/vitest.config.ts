// vitest.config.ts
// Render-level test rig for the web client (cambia-1172). Standalone from vite.config.js's dev
// server config (API proxying, remote-mode compression/HMR tuning) since none of that applies to a
// jsdom test run; this only needs the two things every source file already assumes: the JSX
// transform and the `@/` path alias.
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false
  }
});
