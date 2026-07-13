import React, { useState } from 'react';

export interface IconButtonProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'secondary' | 'ghost';
  disabled?: boolean;
  onClick?: () => void;
  /** Accessible label (also tooltip). */
  title?: string;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

const SIZE_PX: Record<NonNullable<IconButtonProps['size']>, number> = { sm: 30, md: 40, lg: 48 };

/** Square icon-only button; pass a Lucide icon or inline SVG as children. */
const IconButton: React.FC<IconButtonProps> = ({ size = 'md', variant = 'secondary', disabled = false, onClick, title, children, style }) => {
  const [hover, setHover] = useState(false);
  const [press, setPress] = useState(false);
  const px = SIZE_PX[size] || SIZE_PX.md;
  const solid = variant !== 'ghost';
  const down = press && !disabled;
  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
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
        width: px,
        height: px,
        flex: 'none',
        color: 'var(--text-primary)',
        background: solid ? (hover ? 'var(--bark-700)' : 'var(--surface-raised)') : hover ? 'rgba(243,236,218,0.07)' : 'transparent',
        border: solid ? 'var(--line-thick) solid var(--border-strong)' : 'var(--line-thick) solid transparent',
        borderRadius: 'var(--ds-radius-md)',
        boxShadow: solid && !down ? 'var(--shadow-piece)' : 'none',
        transform: down ? 'translateY(2px)' : 'none',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.45 : 1,
        transition: 'background var(--dur-fast) var(--ds-ease-out), transform var(--dur-fast) var(--ds-ease-out)',
        ...style
      }}
    >
      {children}
    </button>
  );
};

export default IconButton;
