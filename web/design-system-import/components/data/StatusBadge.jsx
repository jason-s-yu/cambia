import React from 'react';

const MAP = {
  running:   { color: 'var(--moss-400)', bg: 'rgba(79,138,94,0.18)', border: 'var(--moss-600)' },
  succeeded: { color: 'var(--moss-400)', bg: 'rgba(79,138,94,0.18)', border: 'var(--moss-600)' },
  created:   { color: 'var(--dusk-400)', bg: 'rgba(92,127,163,0.16)', border: 'var(--dusk-600)' },
  queued:    { color: 'var(--dusk-400)', bg: 'rgba(92,127,163,0.16)', border: 'var(--dusk-600)' },
  starting:  { color: 'var(--honey-400)', bg: 'rgba(223,174,71,0.14)', border: 'var(--honey-600)' },
  stopping:  { color: 'var(--honey-400)', bg: 'rgba(223,174,71,0.14)', border: 'var(--honey-600)' },
  stopped:   { color: 'var(--text-secondary)', bg: 'var(--surface-raised)', border: 'var(--border-strong)' },
  crashed:   { color: 'var(--berry-400)', bg: 'rgba(179,58,53,0.16)', border: 'var(--berry-600)' },
  failed:    { color: 'var(--berry-400)', bg: 'rgba(179,58,53,0.16)', border: 'var(--berry-600)' },
};

export function StatusBadge({ status = 'created', style }) {
  const m = MAP[status] || MAP.created;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '2px 10px', borderRadius: 'var(--radius-pill)',
      background: m.bg, border: '1.5px solid ' + m.border, color: m.color,
      fontFamily: 'var(--font-mono)', fontSize: 'var(--text-2xs)', fontWeight: 700,
      textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap',
      ...style,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor', flex: 'none' }}></span>
      {status}
    </span>
  );
}
