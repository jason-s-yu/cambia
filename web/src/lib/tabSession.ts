// src/lib/tabSession.ts
// The identity this browser tab is pinned to (cambia-1149).
//
// Identity is normally one host-wide HttpOnly cookie, so every tab on an
// origin is the same player. A tab that pins an identity holds its JWT in
// sessionStorage instead and sends it explicitly: an Authorization header on
// REST calls, an extra subprotocol on WebSocket handshakes. The server
// resolves an explicit token first and falls back to the cookie, so an
// unpinned tab behaves exactly as before.
//
// sessionStorage is what makes this per tab: a duplicated tab inherits the
// pin, a fresh tab starts empty.
//
// This module deliberately imports nothing. It is the seam the socket hooks,
// the axios instance and the auth store all read, it runs before React does,
// and it has to be loadable in `node --test` with a fake location and storage
// (see web/scripts/test-tab-session.mjs). Everything it touches on the host
// (sessionStorage, location, history, atob) is read through globalThis at
// call time and wrapped, since storage access throws outright in some browser
// configurations.

/** sessionStorage keys. Namespaced so they never collide with app state. */
export const TAB_TOKEN_KEY = 'cambia.tab_token';
export const TAB_LABEL_KEY = 'cambia.tab_label';

/** The subprotocol every Cambia socket offers, pinned or not. */
export const WS_PROTOCOL = 'cambia';
/** Prefix of the handshake entry that carries a tab token. Never selected by the server. */
export const WS_TOKEN_PREFIX = 'cambia-token.';
/** Request header that asks an auth endpoint to answer with a token and set no cookie. */
export const TAB_SESSION_HEADER = 'X-Cambia-Session';

/**
 * Names a dev account may take, matching the server's own rule. Kept here so
 * the switcher rejects a bad name before the round trip and so both halves of
 * the contract are stated in one place.
 */
export const DEV_ACCOUNT_NAME_RE = /^[a-z0-9_-]{1,32}$/;

/** Stated in one place, since both `?as=` and the switcher's field enforce it. */
export const DEV_ACCOUNT_NAME_RULE = 'Use 1 to 32 characters: a-z, 0-9, underscore or dash.';

/** A checked dev account name, or the reason it was refused. */
export type NameCheck = { ok: true; name: string } | { ok: false; message: string };

/**
 * Checks a dev account name before the round trip.
 *
 * Trims and lowercases first: a name typed with a capital or a trailing space
 * is the name the person meant, and the alternative is a 400 from the service
 * for something the field could have fixed.
 */
export function checkAccountName(raw: string): NameCheck {
  const name = raw.trim().toLowerCase();
  if (name === '') return { ok: false, message: 'Enter a name.' };
  if (!DEV_ACCOUNT_NAME_RE.test(name)) return { ok: false, message: DEV_ACCOUNT_NAME_RULE };
  return { ok: true, name };
}

/** What a boot URL asked this tab to become. */
export type BootIntent =
  /** `?as=<name>`: mint (or reuse) the named dev account. */
  | { kind: 'dev-account'; name: string }
  /** `?as=guest`: mint a fresh guest for this tab alone. */
  | { kind: 'guest' }
  /**
   * `#tab=<jwt>`: pin a token handed over by another tab.
   * `label` rides a second fragment field (`&label=<name>`) written by the
   * tab that made the handoff; null when the fragment carried none, which
   * falls back to the token's short subject rather than losing the name.
   */
  | { kind: 'token'; token: string; label: string | null };

/** A pinned identity: the token that proves it and the name to show for it. */
export interface TabIdentity {
  token: string;
  label: string;
}

/** Turns a boot intent into a token. Supplied by the caller so this module stays dependency-free. */
export type TabMinter = (intent: BootIntent) => Promise<TabIdentity | null>;

function storage(): Storage | null {
  try {
    return (globalThis as { sessionStorage?: Storage }).sessionStorage ?? null;
  } catch {
    // Storage access itself throws when site data is blocked.
    return null;
  }
}

function read(key: string): string | null {
  try {
    return storage()?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    storage()?.setItem(key, value);
  } catch {
    // A tab that cannot hold a token stays on the shared cookie identity.
  }
}

function drop(key: string): void {
  try {
    storage()?.removeItem(key);
  } catch {
    // Nothing to do: the getters below already fail closed.
  }
}

// --- Change notification -------------------------------------------------
//
// A pin or unpin changes what every open socket should have handshaked with,
// so the hooks subscribe here rather than polling. The counter is the value
// useSyncExternalStore compares; the listeners are what fire.

let epoch = 0;
const listeners = new Set<() => void>();

function bump(): void {
  epoch += 1;
  for (const listener of [...listeners]) {
    try {
      listener();
    } catch (error) {
      console.error('[tabSession] listener failed:', error);
    }
  }
}

/** Subscribes to pin/unpin. Returns the unsubscribe function. */
export function subscribeTabSession(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Counter bumped on every pin or unpin, for useSyncExternalStore. */
export function getTabSessionEpoch(): number {
  return epoch;
}

// --- The pin -------------------------------------------------------------

/** The JWT this tab is pinned to, or null when it rides the shared cookie. */
export function getTabToken(): string | null {
  const token = read(TAB_TOKEN_KEY);
  return token && token.length > 0 ? token : null;
}

/** The display name recorded with the pin. Falls back to the token's short subject. */
export function getTabLabel(): string | null {
  const token = getTabToken();
  if (!token) return null;
  return read(TAB_LABEL_KEY) || shortSubject(token) || 'pinned';
}

/** True when this tab carries its own identity. */
export function isPinned(): boolean {
  return getTabToken() !== null;
}

/** Pins this tab to `token`, displayed as `label`. */
export function pinTab(token: string, label: string): void {
  if (typeof token !== 'string' || token.length === 0) return;
  write(TAB_TOKEN_KEY, token);
  write(TAB_LABEL_KEY, label || shortSubject(token) || 'pinned');
  bump();
}

/**
 * Drops the pin, returning the tab to the shared cookie identity. Idempotent
 * and silent when nothing was pinned: the 401 path in lib/axios.ts calls this
 * on every rejected request, and a notification per call would restart every
 * socket on a tab that was never pinned.
 */
export function unpinTab(): void {
  if (!isPinned()) return;
  drop(TAB_TOKEN_KEY);
  drop(TAB_LABEL_KEY);
  bump();
}

/** The subprotocol list for `new WebSocket`. The token entry rides second and is never selected. */
export function wsProtocols(): string[] {
  const token = getTabToken();
  return token ? [WS_PROTOCOL, WS_TOKEN_PREFIX + token] : [WS_PROTOCOL];
}

/** The Authorization header for a pinned tab, or nothing at all. */
export function authHeader(): Record<string, string> {
  const token = getTabToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** The `sub` claim of a JWT, or null when it cannot be read. Never throws. */
export function tokenSubject(token: string | null = getTabToken()): string | null {
  if (!token) return null;
  const payload = token.split('.')[1];
  if (!payload) return null;
  try {
    const base64 = payload.replace(/-/g, '+').replace(/_/g, '/');
    const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4);
    const decoded = (globalThis as { atob?: (s: string) => string }).atob?.(padded);
    if (!decoded) return null;
    const claims = JSON.parse(decoded) as { sub?: unknown };
    return typeof claims.sub === 'string' ? claims.sub : null;
  } catch {
    return null;
  }
}

/** First segment of a token's subject, short enough to sit in a pill. */
export function shortSubject(token: string | null = getTabToken()): string | null {
  const sub = tokenSubject(token);
  return sub ? sub.slice(0, 8) : null;
}

// --- Boot notice ---------------------------------------------------------
//
// There is no toast system in this client, so a boot failure (the dev routes
// are off, the network is down) is recorded here and rendered by the dev
// switcher, which is the only surface that exists for it.

let notice: string | null = null;

/** Records a message for the dev switcher to show. */
export function setTabNotice(message: string | null): void {
  notice = message;
  bump();
}

/** The last boot or minting failure, if any. */
export function getTabNotice(): string | null {
  return notice;
}

// --- Boot -----------------------------------------------------------------

/** What a URL asked for, and the URL with that request taken back out. */
export interface BootRequest {
  intent: BootIntent | null;
  /** The URL to leave in the address bar, or null when nothing needs stripping. */
  cleanedHref: string | null;
  /** Set when the URL carried a request that could not be honoured. */
  error: string | null;
}

/**
 * Reads `?as=<name>`, `?as=guest` and `#tab=<jwt>` off a URL.
 *
 * `?as=` wins when a URL carries both: it is the explicit request to mint,
 * and a handoff fragment is only ever written by this app on a URL that has
 * no query of its own.
 */
export function readBootIntent(href: string): BootRequest {
  let url: URL;
  try {
    url = new URL(href);
  } catch {
    return { intent: null, cleanedHref: null, error: null };
  }

  const as = url.searchParams.get('as');
  if (as !== null) {
    url.searchParams.delete('as');
    const cleanedHref = url.pathname + url.search + url.hash;
    const name = as.trim().toLowerCase();
    if (name === 'guest' || name === '') {
      return { intent: { kind: 'guest' }, cleanedHref, error: null };
    }
    if (!DEV_ACCOUNT_NAME_RE.test(name)) {
      return {
        intent: null,
        cleanedHref,
        error: `Ignored ?as=${as}: a dev account name is 1 to 32 characters of a-z, 0-9, underscore or dash.`
      };
    }
    return { intent: { kind: 'dev-account', name }, cleanedHref, error: null };
  }

  const hash = url.hash.startsWith('#') ? url.hash.slice(1) : url.hash;
  if (hash.startsWith('tab=')) {
    const fields = hash.split('&');
    const token = decodeURIComponent(fields[0].slice('tab='.length));
    const cleanedHref = url.pathname + url.search;
    if (!token) {
      return { intent: null, cleanedHref, error: 'Ignored an empty #tab= handoff.' };
    }
    const labelField = fields.slice(1).find((field) => field.startsWith('label='));
    const label = labelField ? decodeURIComponent(labelField.slice('label='.length)) : null;
    return { intent: { kind: 'token', token, label }, cleanedHref, error: null };
  }

  return { intent: null, cleanedHref: null, error: null };
}

function replaceUrl(href: string): void {
  try {
    (globalThis as { history?: History }).history?.replaceState(null, '', href);
  } catch {
    // Not fatal: the app runs, the address bar just keeps the parameter.
  }
}

/**
 * Consumes a boot URL's identity request, once, before anything probes
 * `/user/me`. A probe that runs first answers for the shared cookie identity
 * and the tab shows the wrong player until something else forces a recheck,
 * so main.tsx awaits this ahead of the first render.
 *
 * `mint` turns an intent into a token (services/devSessionService.ts). It is a
 * parameter rather than an import so this module keeps no dependencies.
 *
 * The request is stripped from the address bar whether or not it succeeds:
 * consumed means consumed, and a token left in the fragment would be handed
 * on by any copied link. A failure leaves the tab unpinned, records a notice
 * for the switcher, and lets the app continue on the shared cookie.
 */
export async function consumeBootIdentity(mint: TabMinter): Promise<BootIntent | null> {
  const href = (globalThis as { location?: Location }).location?.href;
  if (!href) return null;

  const { intent, cleanedHref, error } = readBootIntent(href);
  if (cleanedHref !== null) replaceUrl(cleanedHref);
  if (error) {
    console.warn(`[tabSession] ${error}`);
    setTabNotice(error);
  }
  if (!intent) return null;

  try {
    const identity = intent.kind === 'token'
      ? { token: intent.token, label: intent.label || shortSubject(intent.token) || 'pinned' }
      : await mint(intent);
    if (identity) {
      pinTab(identity.token, identity.label);
      setTabNotice(null);
    } else {
      setTabNotice(describeMintFailure(intent, null));
    }
  } catch (err) {
    console.warn('[tabSession] could not pin this tab from the boot URL:', err);
    setTabNotice(describeMintFailure(intent, err));
  }
  return intent;
}

/** Message shown when a boot request could not be honoured. */
export function describeMintFailure(intent: BootIntent, error: unknown): string {
  const status = (error as { response?: { status?: number } } | null)?.response?.status;
  const who = intent.kind === 'dev-account' ? `dev account "${intent.name}"` : 'guest';
  if (status === 404) {
    return `Could not pin the ${who}: named dev accounts are off on this service. Set CAMBIA_DEV_ACCOUNTS=1.`;
  }
  return `Could not pin the ${who}. Staying on the shared session.`;
}

/**
 * The URL that opens a new tab already pinned to `token`, carrying `label` as
 * a second fragment field so the new tab's pill and stored label read the
 * account name instead of falling back to the token's short subject.
 */
export function handoffHref(origin: string, pathname: string, token: string, label?: string | null): string {
  const base = `${origin}${pathname}#tab=${encodeURIComponent(token)}`;
  return label ? `${base}&label=${encodeURIComponent(label)}` : base;
}
