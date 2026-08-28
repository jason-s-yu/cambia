import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
}

/**
 * Attribute-passthrough button for forms and legacy call sites (type=submit,
 * aria-*, className). Same visual spec as ds/core/Button, expressed through
 * the token utilities: solid gold primary, bordered secondary, red danger,
 * transparent ghost; 1px borders, no offset shadow, color-only press state.
 * New non-form markup composes ds/core/Button directly.
 */
const VARIANT_CLASS: Record<NonNullable<ButtonProps['variant']>, string> = {
  primary: 'bg-accent-gold text-text-on-gold border-transparent hover:bg-accent-gold-hover active:bg-[var(--accent-gold-active)]',
  secondary: 'bg-surface-2 text-text-primary border-border-default hover:bg-surface-3 active:bg-surface-1',
  danger: 'bg-accent-danger text-[var(--text-on-danger)] border-transparent hover:bg-[var(--accent-danger-hover)]',
  ghost: 'bg-transparent text-text-secondary border-transparent hover:bg-[var(--interactive-hover)] hover:text-text-primary active:bg-[var(--interactive-active)]'
};

// Disabled is its own fill, not a faded accent: opacity-50 washed the gold
// primary down to a 1.65:1 label in light (measured on the login form's
// in-flight submit), and LoginForm, RegisterForm and CreateRunModal all render
// their in-flight submit through it. Same neutral fill as ds/core/Button, and
// ghost keeps its transparent shell so a disabled ghost control does not grow a
// chip (cambia-892, DL-7 F2).
const DISABLED_CLASS = 'bg-surface-2 text-text-disabled border-border-default';
const DISABLED_GHOST_CLASS = 'bg-transparent text-text-disabled border-transparent';

const SIZE_CLASS: Record<NonNullable<ButtonProps['size']>, string> = {
  sm: 'h-[var(--control-h-sm)] px-3 text-ds-sm',
  md: 'h-[var(--control-h-md)] px-4 text-ds-md',
  lg: 'h-[var(--control-h-lg)] px-[22px] text-ds-lg'
};

const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  className = '',
  disabled,
  ...props
}) => {
  const base = 'inline-flex items-center justify-center gap-2 border rounded-ds-md font-sans font-ds-bold tracking-ds-tight whitespace-nowrap cursor-pointer transition-colors duration-[var(--dur-fast)] disabled:cursor-not-allowed';
  // The variant classes carry :hover fills that still match on a disabled
  // button, so the disabled state replaces them rather than layering over them.
  const isDisabled = disabled || isLoading;
  const stateClass = isDisabled
    ? (variant === 'ghost' ? DISABLED_GHOST_CLASS : DISABLED_CLASS)
    : VARIANT_CLASS[variant];

  return (
    <button
      className={`${base} ${stateClass} ${SIZE_CLASS[size]} ${className}`}
      disabled={isDisabled}
      aria-busy={isLoading || undefined}
      {...props}
    >
      {isLoading && (
        <span
          aria-hidden='true'
          className='inline-block w-3.5 h-3.5 rounded-full border-2 border-current border-t-transparent opacity-70'
          style={{ animation: 'ds-spin 0.9s linear infinite' }}
        />
      )}
      {children}
    </button>
  );
};

export default Button;
