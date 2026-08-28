import React from 'react';

export type PlayingCardSuit = 'spades' | 'hearts' | 'diamonds' | 'clubs';

export interface PlayingCardProps {
  /** 'A','2'-'10','J','Q','K' or 'JOKER'. Ignored when faceDown. */
  rank?: string;
  suit?: PlayingCardSuit;
  faceDown?: boolean;
  size?: 'sm' | 'md' | 'lg';
  /** Lifted with a gold ring: the chosen card, or the card being shown. */
  selected?: boolean;
  /** Soft gold outline without the lift: a legal target or an actionable pile. */
  highlight?: boolean;
  /** Faded: not a legal target right now. */
  dimmed?: boolean;
  onClick?: () => void;
  /** Accessible name. Names the button, or the card as an image when it takes no click. */
  label?: string;
  /**
   * Toggle state for a card that is picked and unpicked (own-hand selection, an
   * opponent card picked for a snap). Left undefined for a card whose click
   * commits an action, where aria-pressed would report a state that does not exist.
   */
  pressed?: boolean;
  /** Stable e2e hook, e.g. `card-0-2`, `pile-stock`. */
  testId?: string;
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
 * hairline frame. 1px border, one hairline lift.
 *
 * Four states, kept apart so a chosen card never looks like a merely legal one
 * (cambia-959): idle is the plain edge; `highlight` (targetable) recolors the
 * edge and adds a soft gold outline, no lift; `selected` rings the card in gold
 * and raises it; `dimmed` fades a card that is not a target right now.
 *
 * A card that takes a click is a real <button>, so Enter and Space activate it
 * natively and it takes the global :focus-visible ring. A card that takes no
 * click is not focusable: it carries role='img' so a screen reader still reads
 * its name, since a generic element takes no accessible name from aria-label
 * (cambia-876, DL-4 review F13).
 */
const PlayingCard: React.FC<PlayingCardProps> = ({ rank, suit, faceDown = false, size = 'md', selected = false, highlight = false, dimmed = false, onClick, label, pressed, testId, style }) => {
  const d = DIMS[size] || DIMS.md;
  const joker = rank === 'JOKER';
  const glyph = joker ? '★' : suit ? GLYPHS[suit] || '' : '';
  const color = joker ? 'var(--accent-gold)' : suit && RED[suit] ? 'var(--suit-red)' : 'var(--suit-black)';
  const edge = faceDown ? 'var(--border-strong)' : 'var(--card-face-edge)';
  const base: React.CSSProperties = {
    // Button reset: the card owns its box, so the UA padding, font and fill go.
    appearance: 'none',
    margin: 0,
    padding: 0,
    font: 'inherit',
    textAlign: 'left',
    display: 'block',
    width: d.w,
    height: d.h,
    flex: 'none',
    position: 'relative',
    boxSizing: 'border-box',
    borderRadius: 'var(--radius-playing-card)',
    border: '1px solid ' + (selected || highlight ? 'var(--border-accent)' : edge),
    boxShadow: selected
      ? 'var(--focus-ring), var(--shadow-playing-card)'
      : highlight
        ? '0 0 0 2px var(--accent-gold-soft), var(--shadow-playing-card)'
        : 'var(--shadow-playing-card)',
    transform: selected ? 'translateY(-6px)' : 'none',
    opacity: dimmed ? 0.55 : 1,
    transition:
      'transform var(--dur-med) var(--ease-snap), box-shadow var(--dur-fast) var(--ds-ease-out), border-color var(--dur-fast) var(--ds-ease-out), opacity var(--dur-med) var(--ds-ease-out)',
    cursor: onClick ? 'pointer' : 'default',
    userSelect: 'none',
    ...style
  };

  const body = faceDown ? (
    <span style={{ position: 'absolute', inset: 4, borderRadius: 4, border: '1px solid var(--card-back-line)', pointerEvents: 'none' }}></span>
  ) : (
    <>
      <span style={{ position: 'absolute', top: 4, left: 6, textAlign: 'center', lineHeight: 1, fontWeight: 'var(--weight-black)', fontSize: d.idx }}>
        <span style={{ display: 'block' }}>{joker ? '★' : rank}</span>
        {!joker && <span style={{ display: 'block', fontSize: d.idx - 2 }}>{glyph}</span>}
      </span>
      <span style={{ position: 'absolute', bottom: 4, right: 6, textAlign: 'center', lineHeight: 1, fontWeight: 'var(--weight-black)', fontSize: d.idx, transform: 'rotate(180deg)' }}>
        <span style={{ display: 'block' }}>{joker ? '★' : rank}</span>
        {!joker && <span style={{ display: 'block', fontSize: d.idx - 2 }}>{glyph}</span>}
      </span>
      <span style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
        <span style={{ fontSize: joker ? d.star : d.pip, lineHeight: 1 }}>{glyph}</span>
        {joker && <span style={{ fontSize: Math.max(8, d.idx - 5), fontWeight: 'var(--weight-black)', letterSpacing: 'var(--ds-tracking-wide)' }}>JOKER</span>}
      </span>
    </>
  );

  const face: React.CSSProperties = faceDown
    ? { background: 'var(--card-back)' }
    : { background: 'var(--card-face)', color, fontVariantNumeric: 'tabular-nums' };

  if (onClick) {
    return (
      <button
        type='button'
        aria-label={label}
        aria-pressed={pressed}
        data-testid={testId}
        onClick={onClick}
        style={{ ...base, ...face }}
      >
        {body}
      </button>
    );
  }
  return (
    <div role={label ? 'img' : undefined} aria-label={label} data-testid={testId} style={{ ...base, ...face }}>
      {body}
    </div>
  );
};

export default PlayingCard;
