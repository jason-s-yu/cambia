import React from 'react';
import { Button } from '../core/Button.jsx';
import { Badge } from '../core/Badge.jsx';

export function QueueCard({ name = 'H2H Rapid', tagline, players = 2, rounds = 8, minutes = 40, pool, primary = false, ranked = true, onPlay, style }) {
  return (
    <div style={{
      background: 'var(--surface-card)',
      border: primary ? 'var(--line-thick) solid var(--honey-600)' : '1.5px solid var(--border-default)',
      borderRadius: 'var(--radius-lg)',
      boxShadow: 'var(--shadow-card)',
      padding: '16px 18px',
      display: 'flex', flexDirection: 'column', gap: 10,
      ...style,
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-xl)', lineHeight: 1.15 }}>{name}</div>
          {tagline && <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', marginTop: 2 }}>{tagline}</div>}
        </div>
        {ranked ? <Badge tone={primary ? 'gold' : 'warning'}>{primary ? 'PRIMARY' : 'Ranked'}</Badge> : <Badge>Casual</Badge>}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', minWidth: 0, gap: '4px 14px', fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
        <span>{players}p</span>
        <span>{rounds} {rounds === 1 ? 'round' : 'rounds'}</span>
        <span>~{minutes} min</span>
        {pool && <span style={{ color: 'var(--text-tertiary)' }}>{pool}</span>}
      </div>
      <Button variant={primary ? 'primary' : 'secondary'} fullWidth onClick={onPlay}>Play</Button>
    </div>
  );
}
