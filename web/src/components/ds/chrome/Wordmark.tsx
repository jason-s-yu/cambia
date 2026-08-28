import React from 'react';

export interface WordmarkProps {
  /** Wordmark font size in px. The suit glyph scales to 0.62x. */
  size?: number;
  style?: React.CSSProperties;
}

/**
 * Cambia wordmark: the UI sans at black weight with tight tracking, plus a
 * gold diamond. Presentational only.
 */
const Wordmark: React.FC<WordmarkProps> = ({ size = 22, style }) => {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 6, ...style }}>
      <span
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: size,
          fontWeight: 'var(--weight-black)',
          letterSpacing: 'var(--ds-tracking-tight)',
          lineHeight: 1
        }}
      >
        Cambia
      </span>
      <span style={{ color: 'var(--accent-gold)', fontSize: size * 0.62, lineHeight: 1 }}>&#9670;</span>
    </span>
  );
};

export default Wordmark;
