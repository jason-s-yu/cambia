// src/lib/appRoot.ts
// The app's mount element, named once. index.html declares <div id="root">,
// main.tsx renders into it, and ds/core/Modal marks it inert behind a dialog.
// Each of those had the literal 'root' inline, so a rename in index.html would
// have failed silently in the modal (cambia-914, DL-8 R3).

export const APP_ROOT_ID = 'root';

/**
 * Resolves the app mount. Returns null and reports it if the element is gone:
 * callers degrade (the modal skips inert, main.tsx mounts nothing), and the
 * console carries the reason rather than the behaviour disappearing without a
 * trace.
 */
export function getAppRoot(): HTMLElement | null {
  const el = document.getElementById(APP_ROOT_ID);
  if (!el) {
    console.error(`app mount #${APP_ROOT_ID} not found; index.html and lib/appRoot must agree`);
    return null;
  }
  return el;
}
