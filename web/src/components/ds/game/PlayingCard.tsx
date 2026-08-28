import React from 'react';

export type PlayingCardSuit = 'spades' | 'hearts' | 'diamonds' | 'clubs';

export interface PlayingCardProps {
  /** 'A','2'-'10','J','Q','K' or 'JOKER'. Ignored when faceDown. */
  rank?: string;
  suit?: PlayingCardSuit;
  faceDown?: boolean;
  size?: 'sm' | 'md' | 'lg';
  /** Lifted with a gold focus ring: the chosen card, or the card being shown. */
  selected?: boolean;
  /** Gold 1px edge without the lift: a legal target or an actionable pile. */
  highlight?: boolean;
  /** Faded: not a legal target right now. */
  dimmed?: boolean;
  onClick?: () => void;
  /** Accessible name for the button role when clickable. */
  label?: string;
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

/**
 * Playing card: flat cream face with ink/red pips, or a green back with a gold
 * hairline frame. 1px border, one hairline lift. `selected` raises the card and
 * rings it in gold; `highlight` only recolors the edge, for a target that has
 * not been chosen yet. A clickable card is a button for the keyboard.
 */
const PlayingCard: React.FC<PlayingCardProps> = ({ rank, suit, faceDown = false, size = 'md', selected = false, highlight = false, dimmed = false, onClick, label, style }) => {
  const d = DIMS[size] || DIMS.md;
  const joker = rank === 'JOKER';
  const glyph = joker ? '★' : suit ? GLYPHS[suit] || '' : '';
  const color = joker ? 'var(--accent-gold)' : suit && RED[suit] ? 'var(--suit-red)' : 'var(--suit-black)';
  const edge = faceDown ? 'var(--border-strong)' : 'var(--card-face-edge)';
  const base: React.CSSProperties = {
    width: d.w,
    height: d.h,
    flex: 'none',
    position: 'relative',
    boxSizing: 'border-box',
    borderRadius: 'var(--radius-playing-card)',
    border: '1px solid ' + (selected || highlight ? 'var(--border-accent)' : edge),
    boxShadow: selected ? 'var(--focus-ring), var(--shadow-playing-card)' : 'var(--shadow-playing-card)',
    transform: selected ? 'translateY(-6px)' : 'none',
    opacity: dimmed ? 0.55 : 1,
    transition:
      'transform var(--dur-med) var(--ease-snap), box-shadow var(--dur-fast) var(--ds-ease-out), border-color var(--dur-fast) var(--ds-ease-out), opacity var(--dur-med) var(--ds-ease-out)',
    cursor: onClick ? 'pointer' : 'default',
    userSelect: 'none',
    ...style
  };
  const interactive = onClick
    ? {
        role: 'button' as const,
        tabIndex: 0,
        'aria-label': label,
        onClick,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onClick();
          }
        }
      }
    // A label on a bare div names nothing: a generic element takes no accessible
    // name, so a non-clickable card that carries one gets role='img' to be
    // announced at all (cambia-876, DL-4 review F13).
    : label
      ? { role: 'img' as const, 'aria-label': label }
      : {};
  if (faceDown) {
    return (
      <div {...interactive} style={{ ...base, background: 'var(--card-back)' }}>
        <div style={{ position: 'absolute', inset: 4, borderRadius: 4, border: '1px solid var(--card-back-line)', pointerEvents: 'none' }}></div>
      </div>
    );
  }
  return (
    <div
      {...interactive}
      style={{
        ...base,
        background: 'var(--card-face)',
        color,
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
