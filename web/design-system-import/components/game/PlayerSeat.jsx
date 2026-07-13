import React from 'react';

const AVATAR_COLORS = ['var(--ember-500)', 'var(--dusk-500)', 'var(--moss-500)', 'var(--honey-500)', 'var(--berry-500)', 'var(--tier-platinum)'];
function colorFor(name) {
  let h = 0;
  for (let i = 0; i < (name || '').length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}

const STATES = {
  turn:         { label: 'Their turn', color: 'var(--honey-400)' },
  ready:        { label: 'Ready', color: 'var(--moss-400)' },
  cambia:       { label: 'Called Cambia', color: 'var(--berry-400)' },
  disconnected: { label: 'Reconnecting…', color: 'var(--text-tertiary)' },
};

export function PlayerSeat({ username = 'Player', state, isYou = false, rating, handSize, compact = false, style }) {
  const s = STATES[state];
  const isTurn = state === 'turn';
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 10,
      padding: compact ? '5px 12px 5px 6px' : '7px 14px 7px 8px',
      background: 'var(--surface-raised)',
      border: 'var(--line-thick) solid ' + (isTurn ? 'var(--honey-500)' : state === 'cambia' ? 'var(--berry-500)' : 'var(--border-default)'),
      borderRadius: 'var(--radius-pill)',
      boxShadow: isTurn ? '0 2px 0 var(--outline-ink), 0 0 14px rgba(223,174,71,0.25)' : 'var(--shadow-piece)',
      opacity: state === 'disconnected' ? 0.6 : 1,
      ...style,
    }}>
      <span style={{
        width: compact ? 26 : 32, height: compact ? 26 : 32, flex: 'none', borderRadius: '50%',
        background: colorFor(username), border: '1.5px solid var(--outline-ink)',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-on-ember)', fontWeight: 800, fontSize: compact ? 12 : 14,
      }}>{(username[0] || '?').toUpperCase()}</span>
      <span style={{ lineHeight: 1.2 }}>
        <span style={{ display: 'block', fontWeight: 'var(--weight-bold)', fontSize: compact ? 'var(--text-sm)' : 'var(--text-md)' }}>
          {username}{isYou ? ' (you)' : ''}
        </span>
        <span style={{ display: 'flex', gap: 8, alignItems: 'baseline', fontSize: 'var(--text-2xs)' }}>
          {s && <span style={{ color: s.color, fontWeight: 800 }}>{s.label}</span>}
          {rating !== undefined && <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{rating}</span>}
          {handSize !== undefined && <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>{handSize} cards</span>}
        </span>
      </span>
    </div>
  );
}
