import React from 'react';

export interface BadgeProps {
  tone?: 'neutral' | 'success' | 'danger' | 'warning' | 'info' | 'gold' | 'ember';
  /** Leading status dot. */
  dot?: boolean;
  /** Space Mono contents (counts, ratings). */
  mono?: boolean;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

interface Tone {
  bg: string;
  color: string;
  border: string;
}

const TONES: Record<NonNullable<BadgeProps['tone']>, Tone> = {
  neutral: { bg: 'var(--surface-raised)', color: 'var(--text-secondary)', border: 'var(--border-strong)' },
  success: { bg: 'rgba(79,138,94,0.18)', color: 'var(--moss-400)', border: 'var(--moss-600)' },
  danger: { bg: 'rgba(179,58,53,0.16)', color: 'var(--berry-400)', border: 'var(--berry-600)' },
  warning: { bg: 'rgba(223,174,71,0.14)', color: 'var(--honey-400)', border: 'var(--honey-600)' },
  info: { bg: 'rgba(92,127,163,0.16)', color: 'var(--dusk-400)', border: 'var(--dusk-600)' },
  gold: { bg: 'var(--honey-500)', color: 'var(--text-on-honey)', border: 'var(--outline-ink)' },
  ember: { bg: 'var(--ember-500)', color: 'var(--text-on-ember)', border: 'var(--outline-ink)' }
};

/** Pill badge for statuses, counts and labels. */
const Badge: React.FC<BadgeProps> = ({ tone = 'neutral', dot = false, mono = false, children, style }) => {
  const t = TONES[tone] || TONES.neutral;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '3px 10px',
        borderRadius: 'var(--radius-pill)',
        background: t.bg,
        color: t.color,
        border: '1.5px solid ' + t.border,
        fontFamily: mono ? 'var(--ds-font-mono)' : 'var(--font-ui)',
        fontSize: 'var(--ds-text-xs)',
        fontWeight: 'var(--weight-bold)',
        lineHeight: 1.4,
        whiteSpace: 'nowrap',
        ...style
      }}
    >
      {dot && <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'currentColor', flex: 'none' }}></span>}
      {children}
    </span>
  );
};

export default Badge;
