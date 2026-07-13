import React from 'react';

export interface WordmarkProps {
  /** Wordmark font size in px. The heart glyph scales to 0.6x. */
  size?: number;
  style?: React.CSSProperties;
}

/**
 * Cambia wordmark: display-serif name with a berry heart.
 *
 * Adapted from design-system-import/ui_kits/platform/shared.jsx Wordmark for
 * the additive DS preview (cambia-438). Presentational only.
 */
const Wordmark: React.FC<WordmarkProps> = ({ size = 26, style }) => {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 7, ...style }}>
      <span style={{ fontFamily: 'var(--font-display)', fontSize: size, lineHeight: 1 }}>Cambia</span>
      <span style={{ color: 'var(--berry-400)', fontSize: size * 0.6 }}>&#9829;</span>
    </span>
  );
};

export default Wordmark;
