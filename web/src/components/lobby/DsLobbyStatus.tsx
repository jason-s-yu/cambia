// src/components/lobby/DsLobbyStatus.tsx
// Lobby status strip (cambia-847): one line under the seat list that reports
// the table state (waiting, searching, match found, all ready) and carries the
// start countdown as a tabular readout. Status color reports state and is
// never a call to action; gold is reserved for the two highlight moments
// (match found, starting).
import React from 'react';

export type LobbyStatusTone = 'info' | 'success' | 'gold';

export interface DsLobbyStatusProps {
  tone: LobbyStatusTone;
  text: string;
  /** Live numeric readout, right-aligned in tabular figures (the countdown). Visual
   *  only: it is aria-hidden, so `text` has to stand on its own. */
  value?: string;
  style?: React.CSSProperties;
}

interface ToneSpec {
  bg: string;
  border: string;
  color: string;
}

const TONES: Record<LobbyStatusTone, ToneSpec> = {
  info: { bg: 'var(--status-info-bg)', border: 'var(--status-info-border)', color: 'var(--status-info)' },
  success: { bg: 'var(--status-success-bg)', border: 'var(--status-success-border)', color: 'var(--status-success)' },
  gold: { bg: 'var(--accent-gold-soft)', border: 'var(--border-accent)', color: 'var(--accent-gold-text)' }
};

const DsLobbyStatus: React.FC<DsLobbyStatusProps> = ({ tone, text, value, style }) => {
  const t = TONES[tone] || TONES.info;
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        minHeight: 'var(--control-h-md)',
        padding: '6px 12px',
        borderRadius: 'var(--ds-radius-md)',
        background: t.bg,
        border: '1px solid ' + t.border,
        color: t.color,
        fontSize: 'var(--ds-text-sm)',
        fontWeight: 'var(--weight-medium)',
        lineHeight: 'var(--ds-leading-snug)',
        ...style
      }}
    >
      {/* The live region is the state text, which changes when the state does. The
          value ticks once a second, so it stays out of it: announcing '3s', '2s',
          '1s' is noise, not status (cambia-876, DL-3 review F7). */}
      <span role='status' aria-live='polite'>{text}</span>
      {value !== undefined && (
        <span
          aria-hidden='true'
          style={{
            flex: 'none',
            fontSize: 'var(--ds-text-xl)',
            fontWeight: 'var(--weight-black)',
            fontVariantNumeric: 'tabular-nums',
            letterSpacing: 'var(--ds-tracking-tight)',
            lineHeight: 1
          }}
        >
          {value}
        </span>
      )}
    </div>
  );
};

export default DsLobbyStatus;
