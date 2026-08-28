// src/components/lobby/DsLobbyConnectState.tsx
// Pre-connection states of the lobby route (cambia-847): joining, connecting,
// waiting for the first lobby_state, and the error fallbacks. One centered
// block on the app ground so the join and invite-link flows read the same
// whether the lobby is reachable or not. Presentation only; LobbyPage keeps
// the socket and redirect logic.
import React from 'react';
import Spinner from '@/components/ds/core/Spinner';
import Button from '@/components/ds/core/Button';
import { EYEBROW } from '@/components/ds/eyebrow';

export interface DsLobbyConnectAction {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
}

interface DsLobbyConnectStateProps {
  /** Progress line shown next to the spinner. */
  message?: string;
  /** Error line; replaces the spinner. */
  error?: string | null;
  onClearError?: () => void;
  action?: DsLobbyConnectAction;
}

const DsLobbyConnectState: React.FC<DsLobbyConnectStateProps> = ({ message, error, onClearError, action }) => {
  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: 'var(--space-12) var(--space-5)' }}>
      <div
        role={error ? 'alert' : 'status'}
        style={{
          width: '100%',
          maxWidth: 420,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 'var(--space-4)',
          padding: 'var(--space-6) var(--space-5)',
          background: 'var(--surface-1)',
          border: '1px solid ' + (error ? 'var(--status-danger-border)' : 'var(--border-default)'),
          borderRadius: 'var(--ds-radius-lg)',
          textAlign: 'center'
        }}
      >
        {error ? (
          <>
            {/* Shared eyebrow, danger colour: the local copy dropped wordSpacing and
                'Lobby error' rendered as LOBBYERROR (cambia-892, DL-7 F1). */}
            <span style={{ ...EYEBROW, color: 'var(--status-danger)' }}>
              Lobby error
            </span>
            <span style={{ fontSize: 'var(--text-md)', color: 'var(--text-primary)', lineHeight: 'var(--ds-leading-snug)' }}>{error}</span>
          </>
        ) : (
          <Spinner size={28} />
        )}
        {!error && message && (
          <span style={{ fontSize: 'var(--text-md)', color: 'var(--text-secondary)' }}>{message}</span>
        )}
        {(action || (error && onClearError)) && (
          <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', justifyContent: 'center' }}>
            {action && (
              <Button variant={action.variant ?? 'primary'} onClick={action.onClick}>
                {action.label}
              </Button>
            )}
            {error && onClearError && (
              <Button variant='ghost' onClick={onClearError}>
                Dismiss
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DsLobbyConnectState;
