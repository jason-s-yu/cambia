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

/**
 * Spoken suffix for a card in the hand LockCallerHand has frozen (cambia-1069). A locked card is
 * not a button any more, and an element that simply stops taking clicks says nothing to a screen
 * reader, so the name has to carry the reason it went inert.
 */
export const LOCKED_SUFFIX = ', locked after calling Cambia';

/**
 * Spoken suffix for a card in a seat whose reconnect window closed (cambia-955). The seat keeps
 * its cards on the felt to watch the round out with, and nothing about a slot says the score
 * behind it stopped counting, so a screen reader read a forfeited hand as an ordinary live one
 * (cambia-1468, carried from cambia-1237).
 */
export const FORFEITED_SUFFIX = ', seat forfeited, not scored';

/**
 * What the seat is doing to the slot right now, spoken as one suffix. 'forfeited' outranks
 * 'locked': a seat that is not scored at all has nothing left for the Cambia lock to say about it.
 */
export type CardSlotState = 'live' | 'locked' | 'forfeited';

const SLOT_SUFFIX: Record<CardSlotState, string> = {
  live: '',
  locked: LOCKED_SUFFIX,
  forfeited: FORFEITED_SUFFIX
};

/**
 * Accessible name for one hand slot: who owns it, which slot it is, the face it is showing right
 * now, and what its seat is doing to it. `owner` is 'Your' on the player's own side and the
 * opponent's username across the table; `slot` is the engine slot index, spoken 1-based
 * (cambia-1095).
 *
 * The name takes the same face the slot is drawing, so a transient reveal is named while it is
 * up and the slot goes back to 'face down' the moment the hold ends (cambia-1094, cambia-1124).
 * Callers pass `null` for a slot showing a back. Keeping the composition here is what stops the
 * two from drifting: a caller that named a card off its own state could leave a face-up card
 * announced as face down.
 */
export function cardSlotName(owner: string, slot: number, face: DsCardFace | null | undefined, state: CardSlotState = 'live'): string {
  const spoken = cardFaceName(face);
  const named = spoken ? `${owner} card ${slot + 1}: ${spoken}` : `${owner} card ${slot + 1}, face down`;
  return named + SLOT_SUFFIX[state];
}
