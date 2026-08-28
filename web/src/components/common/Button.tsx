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
  const base = 'inline-flex items-center justify-center gap-2 border rounded-ds-md font-sans font-ds-bold tracking-ds-tight whitespace-nowrap cursor-pointer transition-colors duration-[var(--dur-fast)] disabled:opacity-50 disabled:cursor-not-allowed';

  return (
    <button
      className={`${base} ${VARIANT_CLASS[variant]} ${SIZE_CLASS[size]} ${className}`}
      disabled={disabled || isLoading}
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
