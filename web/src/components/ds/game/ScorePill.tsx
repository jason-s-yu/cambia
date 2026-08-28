import React from 'react';

export interface ScorePillProps {
  /** Uppercase micro-label, e.g. "ROUND", "TOTAL". */
  label?: string;
  value?: React.ReactNode;
  /** 'success' and 'danger' carry a delta's sign; 'gold' marks the headline number. */
  tone?: 'neutral' | 'gold' | 'success' | 'danger';
  big?: boolean;
  style?: React.CSSProperties;
}

interface ToneSpec {
  color: string;
  border: string;
}

const TONES: Record<NonNullable<ScorePillProps['tone']>, ToneSpec> = {
  neutral: { color: 'var(--text-primary)', border: 'var(--border-default)' },
  gold: { color: 'var(--accent-gold)', border: 'var(--border-accent)' },
  success: { color: 'var(--status-success)', border: 'var(--status-success-border)' },
  danger: { color: 'var(--status-danger)', border: 'var(--status-danger-border)' }
};

/** Tabular number in an inset pill: scores, round counters, ratings, timers. */
const ScorePill: React.FC<ScorePillProps> = ({ label, value, tone = 'neutral', big = false, style }) => {
  const t = TONES[tone] || TONES.neutral;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'baseline',
        gap: 8,
        padding: big ? '7px 15px' : '3px 11px',
        background: 'var(--surface-inset)',
        border: '1px solid ' + t.border,
        borderRadius: 'var(--radius-pill)',
        whiteSpace: 'nowrap',
        ...style
      }}
    >
      {label && (
        <span style={{ fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>{label}</span>
      )}
      <span style={{ fontWeight: 'var(--weight-black)', fontVariantNumeric: 'tabular-nums', fontSize: big ? 'var(--ds-text-xl)' : 'var(--text-md)', color: t.color }}>{value}</span>
    </span>
  );
};

export default ScorePill;
