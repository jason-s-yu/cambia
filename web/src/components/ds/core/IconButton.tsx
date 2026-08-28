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

const SIZE_PX: Record<NonNullable<IconButtonProps['size']>, number> = { sm: 28, md: 36, lg: 44 };

/** Flat square icon-only button; pass a Lucide icon or inline SVG as children. */
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
        color: disabled ? 'var(--text-disabled)' : 'var(--text-primary)',
        background: down
          ? 'var(--interactive-active)'
          : solid
            ? hover
              ? 'var(--surface-3)'
              : 'var(--surface-2)'
            : hover
              ? 'var(--interactive-hover)'
              : 'transparent',
        border: '1px solid ' + (solid ? 'var(--border-default)' : 'transparent'),
        borderRadius: 'var(--ds-radius-md)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.55 : 1,
        transition: 'background var(--dur-fast) var(--ds-ease-out), color var(--dur-fast) var(--ds-ease-out)',
        ...style
      }}
    >
      {children}
    </button>
  );
};

export default IconButton;
