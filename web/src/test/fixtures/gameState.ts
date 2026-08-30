// src/test/fixtures/gameState.ts
// Fixture ObfGameState for the render rig (cambia-1172): a two-seat table, mid-round, SELF_ID to
// act with a full four-card hand and a live discard pile. Every own/opponent hand slot arrives the
// way the wire actually sends it - a card id and its index, never a face, since sync_state
// obfuscates every hand in every phase (cambia-1094, see ObfPlayerState.revealedHand).
import type { HouseRules } from '@/types';
import type { ObfCard, ObfGameState, ObfPlayerState } from '@/types/game';

export const SELF_ID = 'fixture-self';
export const OPP_ID = 'fixture-opp';

const DEFAULT_HOUSE_RULES: HouseRules = {
  allowDrawFromDiscardPile: true,
  allowReplaceAbilities: true,
  allowOpponentSnapping: true,
  snapRace: false,
  forfeitOnDisconnect: true,
  penaltyDrawCount: 2,
  // Off by default so a fixture render never has to account for TimerBar's deadline math; a test
  // that cares about the turn clock sets this itself.
  turnTimerSec: 0,
  cardsPerPlayer: 4,
  numJokers: 2,
  numDecks: 1,
  initialViewCount: 2
};

function handSlot(ownerId: string, idx: number): ObfCard {
  return { id: `${ownerId}-card-${idx}`, known: false, idx };
}

function buildPlayer(playerId: string, overrides: Partial<ObfPlayerState> = {}): ObfPlayerState {
  const handSize = overrides.handSize ?? 4;
  return {
    playerId,
    username: playerId === SELF_ID ? 'You' : 'Rival',
    handSize,
    hasCalledCambia: false,
    connected: true,
    isCurrentTurn: playerId === SELF_ID,
    revealedHand: Array.from({ length: handSize }, (_, i) => handSlot(playerId, i)),
    drawnCard: null,
    ...overrides
  };
}

export interface GameStateOverrides extends Partial<ObfGameState> {
  /** Patches applied to the fixture's own seat (SELF_ID). */
  self?: Partial<ObfPlayerState>;
  /** Patches applied to the fixture's single opponent seat (OPP_ID). */
  opponent?: Partial<ObfPlayerState>;
}

/**
 * Baseline two-seat table: SELF_ID to act on turn 3, a five-card discard pile with a live top
 * card, nothing pending. `self`/`opponent` patch one seat; every other ObfGameState field
 * (`currentPlayerId`, `cambiaCalled`, `specialAction`, ...) overrides directly.
 */
export function buildGameState(overrides: GameStateOverrides = {}): ObfGameState {
  const { self, opponent, ...rest } = overrides;
  const selfPlayer = buildPlayer(SELF_ID, self);
  const oppPlayer = buildPlayer(OPP_ID, opponent);

  return {
    gameId: 'fixture-game-1',
    preGameActive: false,
    started: true,
    gameOver: false,
    currentPlayerId: SELF_ID,
    turnId: 2,
    stockpileSize: 30,
    discardSize: 5,
    discardTop: { id: 'fixture-discard-top', known: true, rank: '7', suit: 'H', value: 7 },
    players: [selfPlayer, oppPlayer],
    cambiaCalled: false,
    cambiaCallerId: null,
    houseRules: DEFAULT_HOUSE_RULES,
    specialAction: null,
    snapMoves: [],
    turnDeadline: null,
    serverNow: Date.now(),
    ...rest
  };
}
