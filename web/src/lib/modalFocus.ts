// src/lib/modalFocus.ts
// Where a dialog puts focus when it opens.
//
// ds/core/Modal used to focus the last control in the action row. That is a
// positional rule standing in for an intent: it lands on the confirm action
// only as long as the confirm action is drawn last, and a footer that ends in a
// destructive control ("Leave", "Delete") opens with that control armed under
// Enter (cambia-935, F6). The choice is a prop now, and the default never arms
// a destructive action.
//
// Kept apart from Modal.tsx so it is testable without a DOM: the picker is
// element-agnostic and takes the already-collected candidates.

/** Named places a dialog can open on. */
export type ModalFocusTarget =
  /** The action the dialog exists to take: the last control in the action row. */
  | 'confirm'
  /** The way out: the first control in the action row. */
  | 'dismiss'
  /** The first control in the body, for a dialog whose point is a form field. */
  | 'body'
  /** The panel itself, so nothing is pre-armed. */
  | 'panel';

/** The focusable controls a dialog holds, in DOM order. */
export interface ModalFocusables<T> {
  /** Controls in the action row. */
  footer: T[];
  /** Controls in the scrolling body. */
  body: T[];
  /** The panel element, focusable as a last resort. */
  panel: T | null;
}

/**
 * Resolves where focus lands when a dialog opens.
 *
 * The default is the first control in the action row that is not destructive,
 * then the first body control, then the panel. A named target that is not
 * present falls back to that same chain rather than dropping focus on the body
 * of the page.
 *
 * `isDestructive` reports whether a control performs a destructive action; the
 * DOM caller answers it from the `data-destructive` marker the danger button
 * variants carry.
 */
export function pickInitialFocus<T>(
  target: ModalFocusTarget | undefined,
  els: ModalFocusables<T>,
  isDestructive: (el: T) => boolean = () => false
): T | null {
  const { footer, body, panel } = els;
  let named: T | undefined;
  if (target === 'confirm') named = footer[footer.length - 1];
  else if (target === 'dismiss') named = footer[0];
  else if (target === 'body') named = body[0];
  else if (target === 'panel') named = panel ?? undefined;
  return named ?? footer.find((el) => !isDestructive(el)) ?? body[0] ?? panel;
}
