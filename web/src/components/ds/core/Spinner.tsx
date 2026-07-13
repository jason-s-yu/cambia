import React from 'react';

export interface SpinnerProps {
  /** Diameter in px. Default 24. */
  size?: number;
  label?: string;
  style?: React.CSSProperties;
}

/**
 * Loading spinner (ember arc) with optional label.
 *
 * The source design-system-import/components/core/Spinner.jsx injects its
 * @keyframes rule into document.head at module scope on first render, which
 * is unsafe under React StrictMode's double-invoke and has no SSR story.
 * This adaptation instead references the `ds-spin` keyframe declared once
 * in src/styles/design-tokens.css.
 */
const Spinner: React.FC<SpinnerProps> = ({ size = 24, label, style }) => {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10, ...style }}>
      <span
        style={{
          width: size,
          height: size,
          flex: 'none',
          borderRadius: '50%',
          border: Math.max(2, Math.round(size / 9)) + 'px solid var(--border-default)',
          borderTopColor: 'var(--ember-500)',
          animation: 'ds-spin 0.9s linear infinite'
        }}
      ></span>
      {label && <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', fontWeight: 'var(--weight-bold)' }}>{label}</span>}
    </span>
  );
};

export default Spinner;
