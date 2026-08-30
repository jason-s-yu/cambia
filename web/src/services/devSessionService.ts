// src/services/devSessionService.ts
// Minting the identities a tab can pin (cambia-1149).
//
// Named dev accounts come from `/dev/session`, which only exists when the
// service runs with CAMBIA_DEV_ACCOUNTS=1; the route is absent otherwise and
// answers 404, which is a state the caller renders rather than an error.
// Tab guests go through the ordinary guest endpoint with the tab-session
// header, so they work on any deployment, dev routes or not.
//
// Every call here returns a token and sets no cookie: pinning a tab must not
// touch the shared cookie jar the other tabs are using.
import api from '@/lib/axios';
import { TAB_SESSION_HEADER, type BootIntent, type TabIdentity } from '@/lib/tabSession';
import type { User } from '@/types';

/** One named dev account, as `GET /dev/session` lists it. */
export interface DevAccount {
  name: string;
  id: string;
}

/** The dev account list, plus whether the service offers them at all. */
export interface DevAccountList {
  enabled: boolean;
  accounts: DevAccount[];
}

const TAB_HEADERS = { [TAB_SESSION_HEADER]: 'tab' };

/**
 * Lists the named dev accounts.
 *
 * A 404 is the documented "off on this service" answer, so it is accepted as
 * a status rather than rejected: letting axios reject would put a console
 * error on every load of the switcher on a service that simply has the flag
 * unset.
 */
export const listDevAccounts = async (): Promise<DevAccountList> => {
  const response = await api.get<Partial<DevAccountList>>('/dev/session', {
    validateStatus: (status) => status === 200 || status === 404
  });
  if (response.status === 404 || response.data?.enabled !== true) {
    return { enabled: false, accounts: [] };
  }
  return { enabled: true, accounts: response.data.accounts ?? [] };
};

/** Mints or reuses the named dev account and returns its token. Rejects with 404 when the flag is unset. */
export const mintDevAccount = async (name: string): Promise<TabIdentity> => {
  const response = await api.post<{ token: string; user?: User }>('/dev/session', { name });
  return { token: response.data.token, label: response.data.user?.username || name };
};

/** Mints a guest for this tab alone: a token in the body, no Set-Cookie. */
export const mintTabGuest = async (): Promise<TabIdentity & { id: string }> => {
  const response = await api.post<{ id: string; token: string }>(
    '/user/guest',
    {},
    { headers: TAB_HEADERS }
  );
  return { id: response.data.id, token: response.data.token, label: 'guest' };
};

/** Logs a real account in without touching the shared cookie, for a pinned tab. */
export const loginToTab = async (email: string, password: string): Promise<TabIdentity> => {
  const response = await api.post<{ token: string }>(
    '/user/login',
    { email, password },
    { headers: TAB_HEADERS }
  );
  return { token: response.data.token, label: email };
};

/** Turns a boot URL's intent into a token. Passed to consumeBootIdentity at startup. */
export const mintForIntent = async (intent: BootIntent): Promise<TabIdentity | null> => {
  if (intent.kind === 'dev-account') return mintDevAccount(intent.name);
  if (intent.kind === 'guest') return mintTabGuest();
  // Unreached in practice: consumeBootIdentity resolves a `token` intent
  // itself and never calls this mint (lib/tabSession.ts). Kept correct so a
  // future caller does not silently drop the handed-over label.
  return { token: intent.token, label: intent.label ?? 'pinned' };
};
