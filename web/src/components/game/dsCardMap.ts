// src/components/game/dsCardMap.ts
// Maps engine card encoding onto the DS PlayingCard props for the live table
// re-skin (cambia-484). Engine ranks: A,2-9,T,J,Q,K,O (T=Ten, O=Joker); engine
// suits: H,D,C,S plus R/B for the two jokers. The DS PlayingCard wants
// '10' for tens, 'JOKER' for jokers, and long suit names.
import type { PlayingCardSuit } from '@/components/ds/game/PlayingCard';
import type { ObfCard } from '@/types/game';

const SUIT_MAP: Record<string, PlayingCardSuit> = {
  H: 'hearts',
  D: 'diamonds',
  C: 'clubs',
  S: 'spades'
};

export interface DsCardFace {
  rank: string;
  suit?: PlayingCardSuit;
}

/**
 * Converts a revealed ObfCard into DS PlayingCard face props. Returns null when
 * the card carries no rank (unknown / obfuscated), so callers render it face
 * down. `known` is not required: the discard top is always dealt with a rank.
 */
export function toDsCardFace(card: ObfCard | null | undefined): DsCardFace | null {
  if (!card || !card.rank) return null;
  const rank = card.rank.toUpperCase();
  if (rank === 'O') return { rank: 'JOKER' };
  const displayRank = rank === 'T' ? '10' : rank;
  const suit = card.suit ? SUIT_MAP[card.suit.toUpperCase()] : undefined;
  return { rank: displayRank, suit };
}

/** Spoken rank: a screen reader reads the printed 'K' as the letter, not the card (cambia-959). */
const SPOKEN_RANK: Record<string, string> = { A: 'ace', J: 'jack', Q: 'queen', K: 'king', JOKER: 'joker' };

/**
 * The face as an accessible name fragment: '9 of clubs', 'king of hearts',
 * 'joker'. Used inside the card and pile names, never drawn on screen.
 */
export function cardFaceName(face: DsCardFace | null | undefined): string | null {
  if (!face) return null;
  const rank = SPOKEN_RANK[face.rank] ?? face.rank;
  return face.suit ? `${rank} of ${face.suit}` : rank;
}
