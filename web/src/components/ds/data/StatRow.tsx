import React from 'react';

export interface StatRowProps {
  label?: string;
  value?: React.ReactNode;
  unit?: string;
  /** e.g. "+12" / "-8"; colored by deltaTone. */
  delta?: string;
  deltaTone?: 'moss' | 'berry';
  style?: React.CSSProperties;
}

/** Label/value stat line (mono value), for profiles and dashboards. */
const StatRow: React.FC<StatRowProps> = ({ label, value, unit, delta, deltaTone = 'moss', style }) => {
  const dc = deltaTone === 'berry' ? 'var(--berry-400)' : 'var(--moss-400)';
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, padding: '7px 0', borderBottom: '1px solid var(--border-subtle)', ...style }}>
      <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', fontWeight: 'var(--weight-medium)' }}>{label}</span>
      <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 6 }}>
        <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 700, fontSize: 'var(--text-md)' }}>{value}</span>
        {unit && <span style={{ fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)' }}>{unit}</span>}
        {delta && <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: dc }}>{delta}</span>}
      </span>
    </div>
  );
};

export default StatRow;
