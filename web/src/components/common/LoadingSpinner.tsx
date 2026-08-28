import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  /** Tailwind text color class for the arc; defaults to the gold accent. */
  color?: string;
  className?: string;
}

const SIZE_PX: Record<NonNullable<LoadingSpinnerProps['size']>, number> = { sm: 20, md: 32, lg: 48 };

/**
 * Loading spinner: a 1px-track ring with a gold arc, the same drawing as
 * ds/core/Spinner, kept under this legacy name and class-based API for the
 * route guards and pages that already use it. Uses the ds-spin keyframe
 * declared in styles/design-tokens.css.
 */
const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ size = 'md', color = 'text-accent-gold', className = '' }) => {
  const px = SIZE_PX[size] || SIZE_PX.md;
  return (
    <span
      role='status'
      aria-label='Loading'
      className={`inline-block rounded-full border-border-default border-t-current ${color} ${className}`}
      style={{
        width: px,
        height: px,
        borderWidth: Math.max(2, Math.round(px / 10)),
        animation: 'ds-spin 0.9s linear infinite'
      }}
    />
  );
};

export default LoadingSpinner;
