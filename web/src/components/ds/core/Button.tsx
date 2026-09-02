import React, { useState } from 'react';

export interface ButtonProps {
  /** 'primary' gold CTA (default) · 'cambia' red (Cambia call / destructive) · 'gold' alias of primary, kept for existing call sites · 'secondary' surface + border · 'ghost' */
  variant?: 'primary' | 'secondary' | 'ghost' | 'cambia' | 'gold';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  fullWidth?: boolean;
  onClick?: () => void;
  children?: React.ReactNode;
  /** Stable e2e hook, e.g. `action-snap` (cambia-959). */
  testId?: string;
  /** Marks this control as where a dialog opens focus (`data-autofocus`). ds/core/Modal picks its
   *  own target from lib/modalFocus; this is for a hand-rolled dialog whose primary action can be
   *  absent, where "the first button" is not the same thing as "the primary action". */
  autoFocus?: boolean;
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
  active: string;
  color: string;
  border: string;
}

const SIZES: Record<NonNullable<ButtonProps['size']>, SizeSpec> = {
  sm: { height: 'var(--control-h-sm)', padding: '0 12px', fontSize: 'var(--ds-text-sm)' },
  md: { height: 'var(--control-h-md)', padding: '0 16px', fontSize: 'var(--text-md)' },
  lg: { height: 'var(--control-h-lg)', padding: '0 22px', fontSize: 'var(--ds-text-lg)' }
};

const GOLD: VariantSpec = {
  bg: 'var(--accent-gold)',
  hover: 'var(--accent-gold-hover)',
  active: 'var(--accent-gold-active)',
  color: 'var(--text-on-gold)',
  border: '1px solid transparent'
};

const VARIANTS: Record<NonNullable<ButtonProps['variant']>, VariantSpec> = {
  primary: GOLD,
  // Deprecated alias: the previous language had a separate honey "ranked"
  // button next to the ember primary. The flat language has one accent
  // family, so both resolve to the gold CTA. Call sites migrate in DL-2..5.
  gold: GOLD,
  cambia: {
    bg: 'var(--accent-danger)',
    hover: 'var(--accent-danger-hover)',
    active: 'var(--accent-danger)',
    color: 'var(--text-on-danger)',
    border: '1px solid transparent'
  },
  secondary: {
    bg: 'var(--surface-2)',
    hover: 'var(--surface-3)',
    active: 'var(--surface-1)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-default)'
  },
  ghost: {
    bg: 'transparent',
    hover: 'var(--interactive-hover)',
    active: 'var(--interactive-active)',
    color: 'var(--text-secondary)',
    border: '1px solid transparent'
  }
};

// Disabled is its own fill, not a faded accent. Gold at 50% opacity put the
// label at 1.15:1 in light and 1.84:1 in dark, and a washed CTA still reads as
// the CTA; a neutral surface with the standard border reads as "not now"
// (cambia-876, DL-3 review F12). The fill is --surface-disabled, not
// --surface-2: in light --surface-2 is white, lighter than the card the button
// sits on, so the disabled control read as a hole (cambia-914, DL-8 R7).
// Ghost keeps its transparent shell so a disabled ghost control does not grow
// a chip on a bare surface.
const DISABLED: VariantSpec = {
  bg: 'var(--surface-disabled)',
  hover: 'var(--surface-disabled)',
  active: 'var(--surface-disabled)',
  color: 'var(--text-disabled)',
  border: '1px solid var(--border-default)'
};

const DISABLED_GHOST: VariantSpec = { ...DISABLED, bg: 'transparent', hover: 'transparent', active: 'transparent', border: '1px solid transparent' };

/** Flat action button: solid fill or 1px border, no offset shadow. */
const Button: React.FC<ButtonProps> = ({ variant = 'primary', size = 'md', disabled = false, fullWidth = false, onClick, children, testId, autoFocus = false, style }) => {
  const [hover, setHover] = useState(false);
  const [press, setPress] = useState(false);
  const base = VARIANTS[variant] || VARIANTS.primary;
  const v = disabled ? (variant === 'ghost' ? DISABLED_GHOST : DISABLED) : base;
  const s = SIZES[size] || SIZES.md;
  const down = press && !disabled;
  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      // The marker a dialog reads to keep initial focus off a destructive
      // action (lib/modalFocus, cambia-935 F6). On the element itself, so the
      // rule holds wherever the button is placed.
      data-destructive={variant === 'cambia' ? '' : undefined}
      data-autofocus={autoFocus ? '' : undefined}
      data-testid={testId}
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
        fontFamily: 'var(--font-sans)',
        fontSize: s.fontSize,
        fontWeight: 'var(--weight-bold)',
        letterSpacing: 'var(--ds-tracking-tight)',
        whiteSpace: 'nowrap',
        color: v.color,
        background: down ? v.active : hover && !disabled ? v.hover : v.bg,
        border: v.border,
        borderRadius: 'var(--ds-radius-md)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background var(--dur-fast) var(--ds-ease-out), color var(--dur-fast) var(--ds-ease-out)',
        ...style
      }}
    >
      {children}
    </button>
  );
};

export default Button;
