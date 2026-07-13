import React, { useState } from 'react';

export interface ButtonProps {
  /** 'primary' ember (default) · 'cambia' berry red (Cambia call / destructive) · 'gold' honey (ranked/win) · 'secondary' · 'ghost' */
  variant?: 'primary' | 'secondary' | 'ghost' | 'cambia' | 'gold';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  fullWidth?: boolean;
  onClick?: () => void;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

interface SizeSpec {
  height: string;
  padding: string;
  fontSize: string;
}

interface VariantSpec {
  bg: string;
  hover: string;
  color: string;
  border: string;
  shadow: string;
}

const SIZES: Record<NonNullable<ButtonProps['size']>, SizeSpec> = {
  sm: { height: 'var(--control-h-sm)', padding: '0 12px', fontSize: 'var(--ds-text-sm)' },
  md: { height: 'var(--control-h-md)', padding: '0 18px', fontSize: 'var(--text-md)' },
  lg: { height: 'var(--control-h-lg)', padding: '0 24px', fontSize: 'var(--ds-text-lg)' }
};

const VARIANTS: Record<NonNullable<ButtonProps['variant']>, VariantSpec> = {
  primary: { bg: 'var(--ember-500)', hover: 'var(--ember-400)', color: 'var(--text-on-ember)', border: 'var(--line-thick) solid var(--outline-ink)', shadow: 'var(--shadow-piece)' },
  cambia: { bg: 'var(--berry-500)', hover: 'var(--berry-400)', color: 'var(--text-on-ember)', border: 'var(--line-thick) solid var(--outline-ink)', shadow: 'var(--shadow-piece)' },
  gold: { bg: 'var(--honey-500)', hover: 'var(--honey-400)', color: 'var(--text-on-honey)', border: 'var(--line-thick) solid var(--outline-ink)', shadow: 'var(--shadow-piece)' },
  secondary: { bg: 'var(--surface-raised)', hover: 'var(--bark-700)', color: 'var(--text-primary)', border: 'var(--line-thick) solid var(--border-strong)', shadow: 'var(--shadow-piece)' },
  ghost: { bg: 'transparent', hover: 'rgba(243,236,218,0.07)', color: 'var(--text-secondary)', border: 'var(--line-thick) solid transparent', shadow: 'none' }
};

/** Tabletop-piece action button: ink outline, hard offset shadow, presses DOWN. */
const Button: React.FC<ButtonProps> = ({ variant = 'primary', size = 'md', disabled = false, fullWidth = false, onClick, children, style }) => {
  const [hover, setHover] = useState(false);
  const [press, setPress] = useState(false);
  const v = VARIANTS[variant] || VARIANTS.primary;
  const s = SIZES[size] || SIZES.md;
  const down = press && !disabled;
  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => {
        setHover(false);
        setPress(false);
      }}
      onMouseDown={() => setPress(true)}
      onMouseUp={() => setPress(false)}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        width: fullWidth ? '100%' : undefined,
        height: s.height,
        padding: s.padding,
        fontSize: s.fontSize,
        fontFamily: 'var(--font-ui)',
        fontWeight: 'var(--weight-bold)',
        color: v.color,
        background: hover && !disabled ? v.hover : v.bg,
        border: v.border,
        borderRadius: 'var(--ds-radius-md)',
        boxShadow: down ? 'none' : v.shadow,
        transform: down ? 'translateY(2px)' : 'none',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.45 : 1,
        transition: 'background var(--dur-fast) var(--ds-ease-out), transform var(--dur-fast) var(--ds-ease-out), box-shadow var(--dur-fast) var(--ds-ease-out)',
        ...style
      }}
    >
      {children}
    </button>
  );
};

export default Button;
