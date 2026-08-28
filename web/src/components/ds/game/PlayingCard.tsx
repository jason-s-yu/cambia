import React from 'react';

export type PlayingCardSuit = 'spades' | 'hearts' | 'diamonds' | 'clubs';

export interface PlayingCardProps {
  /** 'A','2'-'10','J','Q','K' or 'JOKER'. Ignored when faceDown. */
  rank?: string;
  suit?: PlayingCardSuit;
  faceDown?: boolean;
  size?: 'sm' | 'md' | 'lg';
  /** Lifted with a gold focus ring (targeting / chosen). */
  selected?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
}

const GLYPHS: Record<PlayingCardSuit, string> = { spades: '♠', hearts: '♥', diamonds: '♦', clubs: '♣' };
const RED: Partial<Record<PlayingCardSuit, boolean>> = { hearts: true, diamonds: true };

interface Dims {
  w: string;
  h: string;
  idx: number;
  pip: number;
  star: number;
}

const DIMS: Record<NonNullable<PlayingCardProps['size']>, Dims> = {
  sm: { w: 'var(--card-w-sm)', h: 'var(--card-h-sm)', idx: 12, pip: 20, star: 18 },
  md: { w: 'var(--card-w-md)', h: 'var(--card-h-md)', idx: 15, pip: 30, star: 26 },
  lg: { w: 'var(--card-w-lg)', h: 'var(--card-h-lg)', idx: 20, pip: 44, star: 38 }
};

/** Playing card: flat cream face with ink/red pips, or a green back with a gold hairline frame. */
const PlayingCard: React.FC<PlayingCardProps> = ({ rank, suit, faceDown = false, size = 'md', selected = false, onClick, style }) => {
  const d = DIMS[size] || DIMS.md;
  const joker = rank === 'JOKER';
  const glyph = joker ? '★' : suit ? GLYPHS[suit] || '' : '';
  const color = joker ? 'var(--gold-600)' : suit && RED[suit] ? 'var(--suit-red)' : 'var(--suit-black)';
  const base: React.CSSProperties = {
    width: d.w,
    height: d.h,
    flex: 'none',
    position: 'relative',
    borderRadius: 'var(--radius-playing-card)',
    boxShadow: selected ? 'var(--focus-ring), var(--shadow-playing-card)' : 'var(--shadow-playing-card)',
    transform: selected ? 'translateY(-6px)' : 'none',
    transition: 'transform var(--dur-med) var(--ease-snap), box-shadow var(--dur-fast) var(--ds-ease-out)',
    cursor: onClick ? 'pointer' : 'default',
    userSelect: 'none',
    ...style
  };
  if (faceDown) {
    return (
      <div
        onClick={onClick}
        style={{
          ...base,
          background: 'var(--card-back)',
          border: '1px solid var(--border-strong)',
          boxSizing: 'border-box'
        }}
      >
        <div style={{ position: 'absolute', inset: 4, borderRadius: 4, border: '1px solid var(--card-back-line)', pointerEvents: 'none' }}></div>
      </div>
    );
  }
  return (
    <div
      onClick={onClick}
      style={{
        ...base,
        background: 'var(--card-face)',
        border: '1px solid var(--card-face-edge)',
        boxSizing: 'border-box',
        color,
        fontFamily: 'var(--font-sans)',
        fontVariantNumeric: 'tabular-nums'
      }}
    >
      <div style={{ position: 'absolute', top: 4, left: 6, textAlign: 'center', lineHeight: 1, fontWeight: 'var(--weight-black)', fontSize: d.idx }}>
        <div>{joker ? '★' : rank}</div>
        {!joker && <div style={{ fontSize: d.idx - 2 }}>{glyph}</div>}
      </div>
      <div style={{ position: 'absolute', bottom: 4, right: 6, textAlign: 'center', lineHeight: 1, fontWeight: 'var(--weight-black)', fontSize: d.idx, transform: 'rotate(180deg)' }}>
        <div>{joker ? '★' : rank}</div>
        {!joker && <div style={{ fontSize: d.idx - 2 }}>{glyph}</div>}
      </div>
      <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
        <span style={{ fontSize: joker ? d.star : d.pip, lineHeight: 1 }}>{glyph}</span>
        {joker && <span style={{ fontSize: Math.max(8, d.idx - 5), fontWeight: 'var(--weight-black)', letterSpacing: 'var(--ds-tracking-wide)' }}>JOKER</span>}
      </div>
    </div>
  );
};

export default PlayingCard;
