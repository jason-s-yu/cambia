import React from 'react';

export interface StatRowProps {
  label?: string;
  value?: React.ReactNode;
  unit?: string;
  /** e.g. "+12" / "-8"; colored by deltaTone. 'moss'/'berry' are the legacy tone names. */
  delta?: string;
  deltaTone?: 'moss' | 'berry';
  style?: React.CSSProperties;
}

/** Label/value stat line with tabular figures, for profiles and dashboards. */
const StatRow: React.FC<StatRowProps> = ({ label, value, unit, delta, deltaTone = 'moss', style }) => {
  const dc = deltaTone === 'berry' ? 'var(--status-danger)' : 'var(--status-success)';
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, padding: '9px 0', borderBottom: '1px solid var(--border-subtle)', ...style }}>
      <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', fontWeight: 'var(--weight-regular)' }}>{label}</span>
      {/* The number outranks its label: --ds-text-lg against the label's
          --ds-text-sm, with the unit a tier under both (cambia-1097). */}
      <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 6, fontVariantNumeric: 'tabular-nums' }}>
        <span style={{ fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-lg)', letterSpacing: 'var(--ds-tracking-tight)', color: 'var(--text-primary)' }}>{value}</span>
        {unit && <span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>{unit}</span>}
        {delta && <span style={{ fontSize: 'var(--ds-text-xs)', fontWeight: 'var(--weight-medium)', color: dc }}>{delta}</span>}
      </span>
    </div>
  );
};

export default StatRow;
