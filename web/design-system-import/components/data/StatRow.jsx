import React from 'react';

export function StatRow({ label, value, unit, delta, deltaTone = 'moss', style }) {
  const dc = deltaTone === 'berry' ? 'var(--berry-400)' : 'var(--moss-400)';
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, padding: '7px 0', borderBottom: '1px solid var(--border-subtle)', ...style }}>
      <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', fontWeight: 'var(--weight-medium)' }}>{label}</span>
      <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 6 }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 'var(--text-md)' }}>{value}</span>
        {unit && <span style={{ fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)' }}>{unit}</span>}
        {delta && <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: dc }}>{delta}</span>}
      </span>
    </div>
  );
}
