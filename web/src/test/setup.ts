// src/test/setup.ts
// Global vitest setup for the render rig (cambia-1172): jest-dom's matchers extend `expect`
// (toBeInTheDocument, toHaveTextContent, ...) and every test unmounts its tree afterward, so a
// component mounted in one test can never leak DOM nodes or store subscriptions into the next.
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});
