import React from 'react';

export interface BadgeProps {
  /** 'ember' is a deprecated alias of 'gold'; call sites migrate in DL-2..5. */
  tone?: 'neutral' | 'success' | 'danger' | 'warning' | 'info' | 'gold' | 'ember';
  /** Leading status dot. */
  dot?: boolean;
  /** Monospace contents (ids, seeds, codes). Numeric readouts stay in the UI face with tabular figures. */
  mono?: boolean;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

interface Tone {
  bg: string;
  color: string;
  border: string;
}

const GOLD: Tone = { bg: 'var(--accent-gold-soft)', color: 'var(--accent-gold-text)', border: 'var(--border-accent)' };

const TONES: Record<NonNullable<BadgeProps['tone']>, Tone> = {
  neutral: { bg: 'var(--surface-2)', color: 'var(--text-secondary)', border: 'var(--border-default)' },
  success: { bg: 'var(--status-success-bg)', color: 'var(--status-success)', border: 'var(--status-success-border)' },
  danger: { bg: 'var(--status-danger-bg)', color: 'var(--status-danger)', border: 'var(--status-danger-border)' },
  warning: { bg: 'var(--status-warning-bg)', color: 'var(--status-warning)', border: 'var(--status-warning-border)' },
  info: { bg: 'var(--status-info-bg)', color: 'var(--status-info)', border: 'var(--status-info-border)' },
  gold: GOLD,
  ember: GOLD
};

/** Flat pill badge for statuses, counts and labels. Tinted fill, 1px border. */
const Badge: React.FC<BadgeProps> = ({ tone = 'neutral', dot = false, mono = false, children, style }) => {
  const t = TONES[tone] || TONES.neutral;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '2px 9px',
        borderRadius: 'var(--radius-pill)',
        background: t.bg,
        color: t.color,
        border: '1px solid ' + t.border,
        fontFamily: mono ? 'var(--ds-font-mono)' : 'var(--font-sans)',
        fontSize: 'var(--ds-text-xs)',
        fontWeight: 'var(--weight-bold)',
        fontVariantNumeric: 'tabular-nums',
        lineHeight: 1.5,
        whiteSpace: 'nowrap',
        ...style
      }}
    >
      {dot && <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor', flex: 'none' }}></span>}
      {children}
    </span>
  );
};

export default Badge;
