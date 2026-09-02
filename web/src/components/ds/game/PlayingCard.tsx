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
  /** Gold edge and ring without the lift: a legal target or an actionable pile. */
  highlight?: boolean;
  /** Faded: not a legal target right now. */
  dimmed?: boolean;
  onClick?: () => void;
  /**
   * A control whose click is unavailable right now, for a card that is a control in
   * every state: the piles, which stay the same control whether or not it is this
   * player's turn. Keeps the role and the tab stop, drops the click (cambia-1242).
   */
  disabled?: boolean;
  /** Accessible name. Names the button, or the card as an image when it takes no click. */
  label?: string;
  /**
   * Toggle state for a card that is picked and unpicked (own-hand selection, an
   * opponent card picked for a snap). Left undefined only for a card that holds no
   * pick at all, where aria-pressed would report a state that does not exist; a card
   * drawn as `selected` because it is picked passes the same value here, so the lift
   * and the pressed state cannot part. Reported only where the card takes a click,
   * since only a live control has a pressed state to report.
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
  /** Inset of the back's hairline frame from the card edge. */
  frame: number;
  /** Pitch of the back's crosshatch, in px. Scaled so the weave stays the same
   *  visual density at every card size instead of packing up at sm. */
  weave: number;
}

const DIMS: Record<NonNullable<PlayingCardProps['size']>, Dims> = {
  sm: { w: 'var(--card-w-sm)', h: 'var(--card-h-sm)', idx: 12, pip: 20, star: 18, frame: 3, weave: 6 },
  md: { w: 'var(--card-w-md)', h: 'var(--card-h-md)', idx: 15, pip: 30, star: 26, frame: 4, weave: 7 },
  lg: { w: 'var(--card-w-lg)', h: 'var(--card-h-lg)', idx: 20, pip: 44, star: 38, frame: 6, weave: 10 }
};

/**
 * Crosshatch of the card back: 1px --card-back-pattern lines on both diagonals,
 * hard stops so it stays a flat weave and never reads as a gradient sheen.
 */
const crosshatch = (pitch: number): string =>
  `repeating-linear-gradient(45deg, var(--card-back-pattern) 0, var(--card-back-pattern) 1px, transparent 1px, transparent ${pitch}px), ` +
  `repeating-linear-gradient(-45deg, var(--card-back-pattern) 0, var(--card-back-pattern) 1px, transparent 1px, transparent ${pitch}px)`;

/**
 * Playing card: flat cream face with ink/red pips, or a face-down back. 1px
 * border, one hairline lift.
 *
 * The back is a plate, not an outline (cambia-1096). It used to be filled with
 * --card-back, which resolves to the felt's own green, so a face-down card came
 * out as a hollow rectangle indistinguishable from DsGameTable's EmptySlot and
 * a four-card deal read as two cards and two empty slots. It is now three
 * layers on a --card-back-fill ground that sits under the felt: a gold
 * crosshatch, the --card-back-line frame inset from the edge, and a light
 * neutral --card-back-edge silhouette. Solid and filled against the empty
 * slot's dashed and transparent, which stays the marker for a genuinely empty
 * pile.
 *
 * Four states, kept apart so a chosen card never looks like a merely legal one
 * (cambia-959): idle is the plain edge; `highlight` (targetable) draws a solid
 * --card-targetable-ring edge and a 2px ring in the same gold, no lift;
 * `selected` rings the card in gold and raises it; `dimmed` fades a card that is
 * not a target right now. The targetable ring is opaque because the tint it
 * replaced measured 1.03:1 on the felt (review F1). A dimmed back fades less
 * far than a dimmed face: opacity blends the card toward the felt behind it,
 * and the cream face has the contrast to spend while the back does not. The
 * back's 1.99:1 on the felt falls to 1.51:1 at 0.55 and slides under the
 * 1.73:1 the empty slot's dashed line holds, which is the read this ticket
 * exists to fix; 0.7 keeps it at 1.67:1 (dark; 2.47 -> 1.93 light).
 *
 * The element is always a <button>, in every state, and only its attributes
 * change (cambia-1242). It used to be a <button> where it took a click and a
 * <div> where it did not, and React reconciles a changed element type by
 * unmounting the old node: a card that stopped being clickable while the
 * keyboard user had it focused sent that focus to <body>, which is nowhere.
 * Three states on the one node:
 *   - takes a click: a real button, so Enter and Space activate it natively and
 *     it takes the global :focus-visible ring.
 *   - `disabled`: still a control and still a tab stop, but unavailable. Marked
 *     with aria-disabled rather than the disabled attribute, since the attribute
 *     drops the focus it holds, which is the churn this exists to avoid.
 *   - neither: not a control. It carries role='img' so a screen reader still
 *     reads its name (cambia-876, DL-4 review F13) and tabIndex -1 so it leaves
 *     the tab sequence while still able to hold focus it already has.
 */
const PlayingCard: React.FC<PlayingCardProps> = ({ rank, suit, faceDown = false, size = 'md', selected = false, highlight = false, dimmed = false, onClick, disabled = false, label, pressed, testId, style }) => {
  const d = DIMS[size] || DIMS.md;
  // A disabled card keeps the control's role and swallows its click; a card with no click at all
  // is not a control in the first place.
  const interactive = !!onClick && !disabled;
  const control = interactive || disabled;
  const joker = rank === 'JOKER';
  const glyph = joker ? '★' : suit ? GLYPHS[suit] || '' : '';
  const color = joker ? 'var(--accent-gold)' : suit && RED[suit] ? 'var(--suit-red)' : 'var(--suit-black)';
  const edge = faceDown ? 'var(--card-back-edge)' : 'var(--card-face-edge)';
  const base: React.CSSProperties = {
    // Button reset: the card owns its box, so the UA padding, font and fill go.
    // The font reset is spelled out in longhands rather than `font: inherit`, and
    // fontVariantNumeric is set here rather than on the face alone, because a card
    // flips between the two `face` objects below on the same element: the `font`
    // shorthand covers font-variant-numeric, so the longhand disappearing on the
    // flip to a back left React removing it while the shorthand was still set, which
    // it reports as a style conflict on every reveal and every deal (cambia-1124).
    appearance: 'none',
    margin: 0,
    padding: 0,
    fontFamily: 'inherit',
    fontSize: 'inherit',
    fontStyle: 'inherit',
    fontWeight: 'inherit',
    lineHeight: 'inherit',
    fontVariantNumeric: 'tabular-nums',
    textAlign: 'left',
    display: 'block',
    width: d.w,
    height: d.h,
    flex: 'none',
    position: 'relative',
    boxSizing: 'border-box',
    borderRadius: 'var(--radius-playing-card)',
    border: '1px solid ' + (selected ? 'var(--border-accent)' : highlight ? 'var(--card-targetable-ring)' : edge),
    boxShadow: selected
      ? 'var(--focus-ring), var(--shadow-playing-card)'
      : highlight
        ? '0 0 0 2px var(--card-targetable-ring), var(--shadow-playing-card)'
        : 'var(--shadow-playing-card)',
    transform: selected ? 'translateY(-6px)' : 'none',
    opacity: dimmed ? (faceDown ? 0.7 : 0.55) : 1,
    transition:
      'transform var(--dur-med) var(--ease-snap), box-shadow var(--dur-fast) var(--ds-ease-out), border-color var(--dur-fast) var(--ds-ease-out), opacity var(--dur-med) var(--ds-ease-out)',
    cursor: interactive ? 'pointer' : 'default',
    userSelect: 'none',
    ...style
  };

  const body = faceDown ? (
    <span
      aria-hidden='true'
      style={{
        position: 'absolute',
        inset: d.frame,
        borderRadius: `calc(var(--radius-playing-card) - ${d.frame}px)`,
        border: '1px solid var(--card-back-line)',
        pointerEvents: 'none'
      }}
    ></span>
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

  // Two longhands on both branches, never the `background` shorthand: the shorthand
  // resets background-image, so pairing it with the crosshatch would leave the back
  // dependent on the order the two keys happen to be emitted in. The two branches
  // also carry the same keys as each other, so flipping a card sets every property
  // rather than dropping one, which is what keeps the flip clear of the shorthand
  // collision React reports on a removed longhand (cambia-1124).
  const face: React.CSSProperties = faceDown
    ? { backgroundColor: 'var(--card-back-fill)', backgroundImage: crosshatch(d.weave), color: 'inherit' }
    : { backgroundColor: 'var(--card-face)', backgroundImage: 'none', color };

  return (
    <button
      type='button'
      role={control ? undefined : label ? 'img' : 'presentation'}
      tabIndex={control ? undefined : -1}
      aria-label={label}
      aria-disabled={disabled || undefined}
      aria-pressed={interactive ? pressed : undefined}
      data-testid={testId}
      onClick={interactive ? onClick : undefined}
      style={{ ...base, ...face }}
    >
      {body}
    </button>
  );
};

export default PlayingCard;
