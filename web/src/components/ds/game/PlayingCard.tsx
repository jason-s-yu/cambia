import React from 'react';

export type PlayingCardSuit = 'spades' | 'hearts' | 'diamonds' | 'clubs';

export interface PlayingCardProps {
  /** 'A','2'-'10','J','Q','K' or 'JOKER'. Ignored when faceDown. */
  rank?: string;
  suit?: PlayingCardSuit;
  faceDown?: boolean;
  size?: 'sm' | 'md' | 'lg';
  /** Lifted with honey ring (targeting / chosen). */
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

/** Playing card: parchment face with ink/berry pips, or ember lattice back. */
const PlayingCard: React.FC<PlayingCardProps> = ({ rank, suit, faceDown = false, size = 'md', selected = false, onClick, style }) => {
  const d = DIMS[size] || DIMS.md;
  const joker = rank === 'JOKER';
  const glyph = joker ? '★' : suit ? GLYPHS[suit] || '' : '';
  const color = joker ? 'var(--honey-600)' : suit && RED[suit] ? 'var(--suit-red)' : 'var(--suit-black)';
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
          background: 'repeating-linear-gradient(45deg, var(--card-back), var(--card-back) 5px, transparent 5px, transparent 9px), var(--ember-600)',
          backgroundColor: 'var(--card-back)',
          backgroundImage:
            'repeating-linear-gradient(45deg, transparent, transparent 5px, var(--card-back-line) 5px, var(--card-back-line) 6.5px), repeating-linear-gradient(-45deg, transparent, transparent 5px, var(--card-back-line) 5px, var(--card-back-line) 6.5px)',
          border: 'var(--line-thick) solid var(--outline-ink)',
          boxSizing: 'border-box'
        }}
      >
        <div style={{ position: 'absolute', inset: 3, borderRadius: 6, border: '1.5px solid var(--card-back-line)', pointerEvents: 'none' }}></div>
      </div>
    );
  }
  return (
    <div
      onClick={onClick}
      style={{
        ...base,
        background: 'var(--card-face)',
        border: '1.5px solid var(--card-face-edge)',
        color,
        fontFamily: 'var(--font-ui)'
      }}
    >
      <div style={{ position: 'absolute', top: 4, left: 6, textAlign: 'center', lineHeight: 1, fontWeight: 800, fontSize: d.idx }}>
        <div>{joker ? '★' : rank}</div>
        {!joker && <div style={{ fontSize: d.idx - 2 }}>{glyph}</div>}
      </div>
      <div style={{ position: 'absolute', bottom: 4, right: 6, textAlign: 'center', lineHeight: 1, fontWeight: 800, fontSize: d.idx, transform: 'rotate(180deg)' }}>
        <div>{joker ? '★' : rank}</div>
        {!joker && <div style={{ fontSize: d.idx - 2 }}>{glyph}</div>}
      </div>
      <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
        <span style={{ fontSize: joker ? d.star : d.pip, lineHeight: 1 }}>{glyph}</span>
        {joker && <span style={{ fontSize: Math.max(8, d.idx - 5), fontWeight: 800, letterSpacing: '0.12em' }}>JOKER</span>}
      </div>
    </div>
  );
};

export default PlayingCard;
