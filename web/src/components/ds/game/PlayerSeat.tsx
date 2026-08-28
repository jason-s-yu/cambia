import React from 'react';

export type PlayerSeatState = 'turn' | 'ready' | 'cambia' | 'disconnected';

export interface PlayerSeatProps {
  username?: string;
  state?: PlayerSeatState;
  isYou?: boolean;
  /** Displayed rating (tabular figures). */
  rating?: number | string;
  /** Card count in hand (tabular figures). */
  handSize?: number;
  compact?: boolean;
  style?: React.CSSProperties;
}

const AVATAR_COLORS = ['var(--gold-500)', 'var(--blue-500)', 'var(--green-500)', 'var(--red-500)', 'var(--tier-platinum)', 'var(--tier-master)'];

function colorFor(name: string): string {
  let h = 0;
  for (let i = 0; i < (name || '').length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}

interface StateSpec {
  label: string;
  color: string;
}

const STATES: Record<PlayerSeatState, StateSpec> = {
  turn: { label: 'Their turn', color: 'var(--accent-gold)' },
  ready: { label: 'Ready', color: 'var(--status-success)' },
  cambia: { label: 'Called Cambia', color: 'var(--status-danger)' },
  disconnected: { label: 'Reconnecting…', color: 'var(--text-tertiary)' }
};

/** Player chip: initial avatar + name + state line. Gold border = their turn; danger border = called Cambia. */
const PlayerSeat: React.FC<PlayerSeatProps> = ({ username = 'Player', state, isYou = false, rating, handSize, compact = false, style }) => {
  const s = state ? STATES[state] : undefined;
  const isTurn = state === 'turn';
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 10,
        padding: compact ? '5px 12px 5px 6px' : '7px 14px 7px 8px',
        background: isTurn ? 'var(--interactive-selected)' : 'var(--surface-2)',
        border: '1px solid ' + (isTurn ? 'var(--accent-gold)' : state === 'cambia' ? 'var(--accent-danger)' : 'var(--border-default)'),
        borderRadius: 'var(--radius-pill)',
        opacity: state === 'disconnected' ? 0.6 : 1,
        transition: 'background var(--dur-med) var(--ds-ease-out), border-color var(--dur-med) var(--ds-ease-out)',
        ...style
      }}
    >
      <span
        style={{
          width: compact ? 26 : 32,
          height: compact ? 26 : 32,
          flex: 'none',
          borderRadius: '50%',
          background: colorFor(username),
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-on-gold)',
          fontWeight: 'var(--weight-black)',
          fontSize: compact ? 12 : 14
        }}
      >
        {(username[0] || '?').toUpperCase()}
      </span>
      <span style={{ lineHeight: 1.2, minWidth: 0 }}>
        <span style={{ display: 'block', fontWeight: 'var(--weight-medium)', fontSize: compact ? 'var(--ds-text-sm)' : 'var(--text-md)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {username}
          {isYou ? ' (you)' : ''}
        </span>
        <span style={{ display: 'flex', gap: 8, alignItems: 'baseline', fontSize: 'var(--text-2xs)', fontVariantNumeric: 'tabular-nums' }}>
          {s && <span style={{ color: s.color, fontWeight: 'var(--weight-bold)' }}>{s.label}</span>}
          {rating !== undefined && <span style={{ color: 'var(--text-tertiary)' }}>{rating}</span>}
          {handSize !== undefined && <span style={{ color: 'var(--text-tertiary)' }}>{handSize} cards</span>}
        </span>
      </span>
    </div>
  );
};

export default PlayerSeat;
