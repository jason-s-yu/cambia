import React from 'react';

export interface SpinnerProps {
  /** Diameter in px. Default 24. */
  size?: number;
  label?: string;
  style?: React.CSSProperties;
}

/**
 * Loading spinner (gold arc) with optional label. References the `ds-spin`
 * keyframe declared once in src/styles/design-tokens.css rather than
 * injecting a <style> tag at module scope.
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
          border: Math.max(2, Math.round(size / 10)) + 'px solid var(--border-default)',
          borderTopColor: 'var(--accent-gold)',
          animation: 'ds-spin 0.9s linear infinite'
        }}
      ></span>
      {label && <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', fontWeight: 'var(--weight-medium)' }}>{label}</span>}
    </span>
  );
};

export default Spinner;
