// src/lib/devSwitcher.ts
// What the dev identity switcher says and which control it moves to
// (cambia-1149).
//
// Held apart from the component for the same reason lib/modalFocus.ts is:
// there is no React renderer in this repo's test setup, so the decisions the
// panel makes are tested here (web/scripts/test-dev-switcher.mjs) and the
// component is left holding markup.
//
// Imports nothing, like every module web/scripts loads directly: node --test
// strips types but does not resolve the `@/` alias. The dev account name rule
// lives in lib/tabSession.ts, next to the `?as=` parser that enforces the same
// rule, rather than being restated here.

/** Shown in place of the account list when the service has the dev routes off. */
export const NAMED_ACCOUNTS_OFF =
  'Named dev accounts are off on this service: set CAMBIA_DEV_ACCOUNTS=1';

/** Which session an identity came from: this tab's own token, or the browser-wide cookie. */
export type IdentityScope = 'tab' | 'shared';

export interface IdentityInput {
  /** Username from the auth store, once /user/me has answered. */
  username?: string | null;
  /** Signed in at all. */
  authenticated: boolean;
  /** This tab holds its own token. */
  pinned: boolean;
  /** Name recorded with the pin, used until the store catches up. */
  label?: string | null;
}

export interface IdentityLine {
  /** Who this tab is: a username, or "signed out". */
  name: string;
  scope: IdentityScope;
  /** One line for the pill: `alice (tab)`. */
  pill: string;
}

/**
 * The identity the pill states.
 *
 * The store's username wins over the pinned label: the label is what the tab
 * asked for, the username is who the service says it got, and a pin that
 * resolved to a different account should say so. The label still covers the
 * window between pinning and the /user/me answer.
 */
export function identityLine({ username, authenticated, pinned, label }: IdentityInput): IdentityLine {
  const scope: IdentityScope = pinned ? 'tab' : 'shared';
  const name = (authenticated && username) || (pinned && label) || 'signed out';
  return { name, scope, pill: `${name} (${scope})` };
}

/** The panel's own state, one flag per block, so the component just renders it. */
export interface PanelModel {
  /** The named-account block is usable. */
  namedAccounts: boolean;
  /** Hint shown instead of that block. */
  namedAccountsHint: string | null;
  /** A handoff link needs a token to hand over. */
  canOpenNewTab: boolean;
  /** Only a pinned tab has a pin to drop. */
  canUnpin: boolean;
}

export function panelModel(input: { devRoutesEnabled: boolean; pinned: boolean; authenticated: boolean }): PanelModel {
  return {
    namedAccounts: input.devRoutesEnabled,
    namedAccountsHint: input.devRoutesEnabled ? null : NAMED_ACCOUNTS_OFF,
    // The fragment carries a token, and only a pinned tab holds one: the shared
    // cookie is HttpOnly and cannot be read out to hand on.
    canOpenNewTab: input.pinned,
    canUnpin: input.pinned
  };
}

/**
 * Next index for a focus trap: Tab moves forward, Shift+Tab back, both wrap.
 * An index that is out of range (a control removed while focused) restarts at
 * the near end rather than moving off the panel.
 */
export function trapIndex(count: number, current: number, back: boolean): number {
  if (count <= 0) return -1;
  if (current < 0 || current >= count) return back ? count - 1 : 0;
  return back ? (current - 1 + count) % count : (current + 1) % count;
}
