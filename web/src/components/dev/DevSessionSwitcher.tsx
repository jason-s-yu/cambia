// src/components/dev/DevSessionSwitcher.tsx
// The dev identity switcher (cambia-1149): what tab this is, and how to make
// it somebody else.
//
// Development wants several players in one browser window. Identity is a
// host-wide cookie, so without this the second seat needs a second hostname.
// The switcher pins this tab to its own token (see lib/tabSession.ts), which
// every REST call and socket handshake then carries.
//
// Dev builds only. main.tsx mounts it behind `import.meta.env.DEV`, which
// Rollup folds to false for `vite build`, so neither this component nor its
// copy reaches a production bundle; the guard here is the second half of that,
// for any other call site.
import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useTabSession } from '@/hooks/useTabSession';
import {
  checkAccountName,
  describeMintFailure,
  getTabToken,
  handoffHref,
  pinTab,
  setTabNotice,
  shortSubject,
  unpinTab,
  type BootIntent
} from '@/lib/tabSession';
import { identityLine, panelModel, trapIndex } from '@/lib/devSwitcher';
import { listDevAccounts, mintDevAccount, mintTabGuest, type DevAccount } from '@/services/devSessionService';

/** Controls the panel can hold, in the order Tab walks them. */
const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), a[href]';

const PILL: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  height: 26,
  padding: '0 10px',
  fontFamily: 'var(--font-sans)',
  fontSize: 'var(--ds-text-xs)',
  fontVariantNumeric: 'tabular-nums',
  color: 'var(--text-secondary)',
  background: 'var(--surface-2)',
  border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-pill)',
  cursor: 'pointer'
};

const PANEL: React.CSSProperties = {
  width: 280,
  marginBottom: 8,
  padding: 'var(--space-3)',
  display: 'flex',
  flexDirection: 'column',
  gap: 'var(--space-3)',
  fontFamily: 'var(--font-sans)',
  fontSize: 'var(--ds-text-xs)',
  color: 'var(--text-primary)',
  background: 'var(--surface-1)',
  border: '1px solid var(--border-default)',
  borderRadius: 'var(--ds-radius-lg)',
  boxShadow: 'var(--shadow-raised)'
};

const ROW: React.CSSProperties = {
  display: 'block',
  width: '100%',
  textAlign: 'left',
  padding: '5px 8px',
  fontFamily: 'inherit',
  fontSize: 'inherit',
  color: 'var(--text-primary)',
  background: 'var(--surface-2)',
  border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--ds-radius-sm)',
  cursor: 'pointer'
};

const HEADING: React.CSSProperties = {
  margin: 0,
  fontSize: 'var(--ds-text-xs)',
  fontWeight: 'var(--weight-bold)' as React.CSSProperties['fontWeight'],
  color: 'var(--text-secondary)'
};

const DevSessionSwitcher: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [accounts, setAccounts] = useState<DevAccount[]>([]);
  const [devRoutesEnabled, setDevRoutesEnabled] = useState(false);
  const [newName, setNewName] = useState('');
  const [nameError, setNameError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const panelRef = useRef<HTMLDivElement | null>(null);
  const pillRef = useRef<HTMLButtonElement | null>(null);

  const panelId = useId();
  const headingId = useId();

  const { pinned, label, notice } = useTabSession();
  const user = useAuthStore((state) => state.user);
  const authenticated = useAuthStore((state) => state.isAuthenticated);
  const checkAuth = useAuthStore((state) => state.checkAuth);

  const identity = identityLine({ username: user?.username, authenticated, pinned, label });
  const model = panelModel({ devRoutesEnabled, pinned, authenticated });
  const subject = shortSubject();

  // The account list is only worth asking for while the panel is on screen,
  // and it is asked for again on each open so an account pinned in another tab
  // shows up here without a reload.
  useEffect(() => {
    if (!open) return;
    let live = true;
    listDevAccounts()
      .then((list) => {
        if (!live) return;
        setDevRoutesEnabled(list.enabled);
        setAccounts(list.accounts);
      })
      .catch(() => {
        if (live) setDevRoutesEnabled(false);
      });
    return () => {
      live = false;
    };
  }, [open]);

  const close = useCallback(() => {
    setOpen(false);
    pillRef.current?.focus();
  }, []);

  // Tab cycles inside the panel. A switcher that swallowed focus would be
  // worse than no switcher: it sits over a game table that is played from the
  // keyboard. Escape is handled at the document level below, not here: this
  // handler only fires when focus is already inside the panel, and a
  // programmatic focus move (a mint, an unpin) can leave it elsewhere while
  // the panel is still open.
  const onPanelKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab') return;
    const panel = panelRef.current;
    if (!panel) return;
    const items = Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE));
    const next = trapIndex(items.length, items.indexOf(document.activeElement as HTMLElement), event.shiftKey);
    if (next < 0) return;
    event.preventDefault();
    items[next]?.focus();
  };

  // Opening moves focus into the panel so the trap has something to hold.
  useEffect(() => {
    if (!open) return;
    const first = panelRef.current?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();
  }, [open]);

  // Escape closes the panel and hands focus back to the pill, no matter where
  // focus is: bound to the document rather than the panel, on capture like
  // Modal's own Escape handler (ds/core/Modal.tsx), so a focus move away from
  // the panel does not leave it stuck open. Removed the moment the panel closes.
  useEffect(() => {
    if (!open) return;
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.stopPropagation();
      close();
    };
    document.addEventListener('keydown', onEscape, true);
    return () => document.removeEventListener('keydown', onEscape, true);
  }, [open, close]);

  /**
   * Pins the token an action produced, then re-reads who this tab now is.
   * The store's checkAuth is what repaints the app around the new identity;
   * the sockets redial on their own, off the pin itself (hooks/useTabSession).
   */
  const applyPin = useCallback(
    async (mint: () => Promise<{ token: string; label: string }>, intent: BootIntent) => {
      setBusy(true);
      try {
        const identityMinted = await mint();
        pinTab(identityMinted.token, identityMinted.label);
        setTabNotice(null);
        await checkAuth();
      } catch (error) {
        console.warn('[devSwitcher] could not pin this tab:', error);
        setTabNotice(describeMintFailure(intent, error));
      } finally {
        setBusy(false);
      }
    },
    [checkAuth]
  );

  const pinAccount = (name: string) => applyPin(() => mintDevAccount(name), { kind: 'dev-account', name });

  const pinNewName = () => {
    const check = checkAccountName(newName);
    if (!check.ok) {
      setNameError(check.message);
      return;
    }
    setNameError(null);
    setNewName('');
    void pinAccount(check.name);
  };

  const newGuest = () => applyPin(mintTabGuest, { kind: 'guest' });

  const openNewTab = () => {
    const token = getTabToken();
    if (!token) return;
    window.open(handoffHref(window.location.origin, window.location.pathname, token, label), '_blank');
  };

  const unpin = async () => {
    unpinTab();
    setTabNotice(null);
    await checkAuth();
  };

  if (!import.meta.env.DEV) return null;

  return (
    <div
      style={{
        position: 'fixed',
        // Clear of the table's own bottom row: the hand and the action column
        // are centred inside the felt (DsGameTable, justifyContent center),
        // whose left edge is already 26px in (16px page padding plus the 10px
        // rail), so a pill in the corner sits beside that row rather than over
        // it. The safe-area inset is added for a phone with a home bar.
        left: 'var(--space-3)',
        bottom: 'calc(var(--space-3) + env(safe-area-inset-bottom, 0px))',
        // Under Modal and the results overlay, both of which sit at 100: a
        // dialog is modal and this is not.
        zIndex: 90,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start'
      }}
      data-testid='dev-session-switcher'
    >
      {open && (
        <div
          ref={panelRef}
          id={panelId}
          role='dialog'
          aria-labelledby={headingId}
          onKeyDown={onPanelKeyDown}
          style={PANEL}
        >
          <div>
            <h2 id={headingId} style={HEADING}>Dev session</h2>
            <p style={{ margin: '4px 0 0', color: 'var(--text-primary)' }}>
              {identity.name}
              <span style={{ color: 'var(--text-tertiary)' }}>
                {' '}
                ({identity.scope === 'tab' ? 'this tab' : 'shared session'})
              </span>
            </p>
            {subject && (
              <p style={{ margin: '2px 0 0', fontFamily: 'var(--ds-font-mono)', color: 'var(--text-tertiary)' }}>
                sub {subject}
              </p>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <h3 style={HEADING}>Pin a dev account</h3>
            {model.namedAccountsHint && (
              <p style={{ margin: 0, color: 'var(--text-tertiary)' }}>{model.namedAccountsHint}</p>
            )}
            {model.namedAccounts && accounts.length === 0 && (
              <p style={{ margin: 0, color: 'var(--text-tertiary)' }}>No dev accounts yet.</p>
            )}
            {model.namedAccounts &&
              accounts.map((account) => (
                <button
                  key={account.id}
                  type='button'
                  disabled={busy}
                  style={ROW}
                  onClick={() => void pinAccount(account.name)}
                  data-testid={`dev-pin-${account.name}`}
                >
                  {account.name}
                </button>
              ))}
            {model.namedAccounts && (
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  type='text'
                  value={newName}
                  placeholder='new name'
                  aria-label='New dev account name'
                  disabled={busy}
                  onChange={(event) => setNewName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      pinNewName();
                    }
                  }}
                  style={{
                    flex: 1,
                    minWidth: 0,
                    height: 26,
                    padding: '0 8px',
                    fontFamily: 'inherit',
                    fontSize: 'inherit',
                    color: 'var(--text-primary)',
                    background: 'var(--surface-inset)',
                    border: '1px solid ' + (nameError ? 'var(--status-danger)' : 'var(--border-default)'),
                    borderRadius: 'var(--ds-radius-sm)'
                  }}
                  data-testid='dev-new-name'
                />
                <button type='button' disabled={busy} style={{ ...ROW, width: 'auto' }} onClick={pinNewName} data-testid='dev-pin-new'>
                  Pin
                </button>
              </div>
            )}
            {nameError && <p style={{ margin: 0, color: 'var(--status-danger)' }}>{nameError}</p>}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <button type='button' disabled={busy} style={ROW} onClick={() => void newGuest()} data-testid='dev-new-guest'>
              New guest in this tab
            </button>
            <button
              type='button'
              disabled={busy || !model.canOpenNewTab}
              style={{ ...ROW, opacity: model.canOpenNewTab ? 1 : 0.55, cursor: model.canOpenNewTab ? 'pointer' : 'not-allowed' }}
              onClick={openNewTab}
              data-testid='dev-open-new-tab'
            >
              Open a new tab as this user
            </button>
            {!model.canOpenNewTab && (
              <p style={{ margin: 0, color: 'var(--text-tertiary)' }}>
                Pin this tab first: the shared session is an HttpOnly cookie and cannot be read out of the page.
              </p>
            )}
            {model.canUnpin && (
              <button type='button' disabled={busy} style={ROW} onClick={() => void unpin()} data-testid='dev-unpin'>
                Unpin (use shared session)
              </button>
            )}
          </div>

          {notice && <p style={{ margin: 0, color: 'var(--status-warning)' }}>{notice}</p>}
        </div>
      )}

      <button
        ref={pillRef}
        type='button'
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        aria-label={`Dev session: ${identity.pill}`}
        style={PILL}
        onClick={() => setOpen((was) => !was)}
        data-testid='dev-session-pill'
      >
        <span
          aria-hidden='true'
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: pinned ? 'var(--accent-gold)' : 'var(--text-tertiary)'
          }}
        />
        {identity.pill}
      </button>
    </div>
  );
};

export default DevSessionSwitcher;
