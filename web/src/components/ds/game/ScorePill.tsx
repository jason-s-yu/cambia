import React from 'react';

export interface ScorePillProps {
  /** Uppercase micro-label, e.g. "ROUND", "TOTAL". */
  label?: string;
  value?: React.ReactNode;
  tone?: 'neutral' | 'gold' | 'berry' | 'moss';
  big?: boolean;
  style?: React.CSSProperties;
}

interface ToneSpec {
  color: string;
  border: string;
}

const TONES: Record<NonNullable<ScorePillProps['tone']>, ToneSpec> = {
  neutral: { color: 'var(--text-primary)', border: 'var(--border-strong)' },
  gold: { color: 'var(--honey-400)', border: 'var(--honey-600)' },
  berry: { color: 'var(--berry-400)', border: 'var(--berry-600)' },
  moss: { color: 'var(--moss-400)', border: 'var(--moss-600)' }
};

/** Mono number in an inset pill: scores, round counters, ratings, timers. */
const ScorePill: React.FC<ScorePillProps> = ({ label, value, tone = 'neutral', big = false, style }) => {
  const t = TONES[tone] || TONES.neutral;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'baseline',
        gap: 8,
        padding: big ? '8px 16px' : '4px 12px',
        background: 'var(--surface-inset)',
        border: '1.5px solid ' + t.border,
        borderRadius: 'var(--radius-pill)',
        ...style
      }}
    >
      {label && (
        <span style={{ fontSize: 'var(--text-2xs)', fontWeight: 800, letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>{label}</span>
      )}
      <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 700, fontSize: big ? 'var(--ds-text-xl)' : 'var(--text-md)', color: t.color }}>{value}</span>
    </span>
  );
};

export default ScorePill;
